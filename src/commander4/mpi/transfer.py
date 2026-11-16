"""Moving data between the TOD-processing and component-separation communicators each iteration.

TOD processing sends `DetectorMap`s (one per band) and receives back a `CompList`. Both directions
go band master to band master, since the two sides may split their bands over different numbers of
ranks; the receiving master then distributes within its own band communicator.
"""
import numpy as np
import logging
from collections.abc import Mapping
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.data_models.detector_map import DetectorMap
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.pixel_domain import DistributedMap
from commander4.file_io.map_reader import read_data_map_from_file
from commander4.sky.comp_list import CompList

logger = logging.getLogger(__name__)


def _get_compsep_sender_id_for_tod_band(todproc_my_band_id: str,
                                        senders: Mapping[str, int]) -> str:
    """Return the CompSep execution-view ID that sends the sky model back to a TOD band.

    For bands that have both intensity and QU execution views, the intensity rank sends the
    full component list. QU-only bands fall back to their `_QU` identifier.
    """
    for view_name in (f"{todproc_my_band_id}_I", f"{todproc_my_band_id}_QU"):
        if view_name in senders:
            return view_name
    raise KeyError(f"No CompSep sender found for TOD band '{todproc_my_band_id}'.")


def _should_send_compsep_result(compsep_my_band_id: str,
                                destinations: Mapping[str, int] | None) -> bool:
    """Return whether this CompSep execution view should send the realized sky model to TOD."""
    if destinations is None or compsep_my_band_id not in destinations:
        return False
    if compsep_my_band_id.endswith("_QU"):
        paired_intensity_id = compsep_my_band_id.removesuffix("_QU") + "_I"
        if paired_intensity_id in destinations:
            return False
    return True


###########################################################
# ON TOD SIDE
###########################################################

def _realize_and_distribute_sky(comp_list: CompList | None, experiment_data: DetectorGroupTOD,
                                band_comm) -> DistributedMap:
    """Realize the band sky on the master and hand each rank only the pixels it needs.

    The band master realizes the full-sky ``(npols, npix)`` detector map from `comp_list` and keeps
    it, for the chain output and the far-sidelobe transform. Every rank, the master included, then
    receives through the band's ``PixelDomain`` the sky values at its local pixels, the same pixels
    its map buffers hold. Only the master needs the component list (None elsewhere), so it is never
    broadcast, avoiding a full-sky realization and a copy of the components on every other rank
    (the dominant TOD-side memory cost at high nside).
    """
    is_band_master = band_comm.Get_rank() == 0
    pols = experiment_data.pols
    ncomp = {"I": 1, "QU": 2, "IQU": 3}[pols]
    if is_band_master:
        # This detectors FWHM is used to realize the sky model, unless the per-pixel solver was
        # used with common-resolution smoothing, in which case the sky carries an intrinsic
        # resolution `amp_fwhm_rad`, and we simply have to use that, even though it's wrong.
        fwhm = max([np.deg2rad(experiment_data.fwhm/60.0)]
                   + [comp.amp_fwhm_rad for comp in comp_list])
        full_map = comp_list.get_sky_at_nu(experiment_data.nu, experiment_data.nside, pols, fwhm,
                                           lmax=experiment_data.lmax)
    else:
        full_map = None
    local_map = experiment_data.pixel_domain.scatter_from_full(full_map, ncomp, dtype=np.float32)
    return DistributedMap(local=local_map, full=full_map)


def receive_compsep(mpi_info: Bunch, experiment_data: DetectorGroupTOD, todproc_my_band_id: str,
                    senders: dict[str, int]) -> DistributedMap:
    """Receive the CompSep components and distribute the realized band map within TOD.

    The band master receives the ``CompList`` from the CompSep side and realizes it at the band
    frequency/resolution; the result is distributed to all band ranks (see
    ``_realize_and_distribute_sky``).

    Args:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        experiment_data (DetectorGroupTOD): The experiment TOD data, used to determine band frequency,
            resolution, and polarization for evaluating the sky model.
        todproc_my_band_id (str): The string uniquely indentifying the band of this rank
                                  (example: '30GHz').
        senders (dict[str, int]): A dictionary mapping the string in 'todproc_my_band_id' to the
            world rank of the sender task (on the CompSep side), keyed by execution-view band ID.

    Returns:
        The realized band sky map, as this rank holds it.
    """
    world_comm = mpi_info.world.comm
    band_comm = mpi_info.band.comm
    if mpi_info.band.is_master:
        source_band_id = _get_compsep_sender_id_for_tod_band(todproc_my_band_id, senders)
        comp_list = world_comm.recv(source=senders[source_band_id])
    else:
        comp_list = None
    return _realize_and_distribute_sky(comp_list, experiment_data, band_comm)


def get_local_initial_sky(mpi_info: Bunch, experiment_data: DetectorGroupTOD,
                          params: Bunch) -> DistributedMap:
    """Build the initial components locally and realize them at this TOD band.

    Used when there are no CompSep ranks: the band master builds the component list from the
    component parameters and init files and realizes it; the realized map is distributed to all
    band ranks (see `_realize_and_distribute_sky`). Mirrors `receive_compsep`, minus the
    cross-world receive.
    """
    comp_list = None
    if mpi_info.band.is_master:
        comp_list = CompList.init_from_params(params)
        for comp in comp_list:
            # Nothing samples beta in a TOD-only run, so beta is that of the parameter file.
            comp.load_initial_state(None, apply_init_from=True, restore_beta=False)
    return _realize_and_distribute_sky(comp_list, experiment_data, mpi_info.band.comm)


def send_tod(mpi_info: Bunch, tod_map_dict: dict[DetectorMap], todproc_my_band_id: str,
             receivers: Bunch) -> None:
    """ MPI-send the results from a single band TOD processing to a task on the CompSep side
        (used in conjunction with receive_tod).

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        tod_map_dict (dict[str, DetectorMap]): The output maps keyed by polarization component
            (e.g. 'I', 'QU') from process_tod for the band belonging to this process.
        todproc_my_band_id (str): The string uniquely indentifying the band of this rank,
                                  regardless of polarization (example: '30GHz').
        receivers (Bunch): Maps a band identifier to the band master on the compsep side.
    """
    if mpi_info.tod.is_master:
        logger.debug(f"CompSep band masters: {mpi_info.world.compsep_band_masters}")
    if mpi_info.band.is_master:
        for pol, detector_map in tod_map_dict.items():
            target_band = f"{todproc_my_band_id}_{pol}"
            if target_band in receivers.keys():
                mpi_info.world.comm.send(detector_map, dest=receivers[target_band])
            else:
                logger.debug(f"Pol-{pol} TOD-processing result discarded because band "
                             f"{todproc_my_band_id} does not require it on the CompSep side.")


###########################################################
# ON COMPSEP SIDE
###########################################################

def receive_tod(mpi_info: Bunch, band_map: DetectorMap|None, params: Bunch) -> DetectorMap:
    """Return this CompSep rank's band map of the next iteration (the counterpart of `send_tod`).

    A band with `get_from: file` reads its map on the first call, and keeps it after that. Any
    other band receives its map from the TOD side.

    Args:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        band_map (DetectorMap): This rank's band map of the previous iteration. None before
                 the first iteration.
        params (Bunch): The full parameter file.
    """
    band = mpi_info.band
    my_band = params.compsep.bands[band.name]
    if my_band.get_from == "file":
        if band_map is None:
            logger.info(f"CompSep: Rank {mpi_info.compsep.rank} reading static map data from file.")
            band_map = read_data_map_from_file(my_band, band.polarization, params)
    else:
        sender = mpi_info.world.tod_band_masters[band.identifier]
        logger.debug(f"CompSep: Rank {mpi_info.compsep.rank} receiving TOD data for "
                     f"{band.identifier} from world rank {sender}.")
        band_map = mpi_info.world.comm.recv(source=sender)
    return band_map


def send_compsep(mpi_info: Bunch, comp_list: CompList) -> None:
    """Send the CompSep components back to TOD when this execution view owns the return path.

    Before this boundary, ``process_compsep`` broadcasts every polarization view solved by a
    sampling group across the CompSep communicator. Every rank therefore holds the same complete
    component list, with both I and QU amplitudes; this function performs no additional QU-to-I
    transfer.

    An IQU TOD band needs only one copy of that complete list. Its `_I` execution view is chosen as
    the sender and the paired `_QU` view is suppressed. A QU-only band has no intensity sender, so
    its `_QU` execution view sends directly.

    Args:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        comp_list: The components of this Gibbs sample.
    """
    # Maps a view name (example: '30GHz_I') to the world rank of the band's master on the TOD
    # side. Empty when the run has no TOD side.
    destinations = mpi_info.world.tod_band_masters
    if _should_send_compsep_result(mpi_info.band.identifier, destinations):
        mpi_info.world.comm.send(comp_list, dest=destinations[mpi_info.band.identifier])
