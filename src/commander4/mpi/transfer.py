"""Moving data between the TOD-processing and component-separation communicators each iteration.

TOD processing sends `DetectorMap`s (one per band) and receives back a `CompList`. Both directions
go band master to band master, since the two sides may split their bands over different numbers of
ranks; the receiving master then distributes within its own band communicator. The world rank of
the other side's partner is in `mpi_info` (see `mpi.setup.init_mpi`).
"""
import numpy as np
import logging
from pixell.bunch import Bunch

from commander4.data_models.detector_map import DetectorMap
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.pixel_domain import DistributedMap
from commander4.file_io.map_reader import read_data_map_from_file
from commander4.mpi.setup import MPIInfo
from commander4.sky.comp_list import CompList

logger = logging.getLogger(__name__)


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
        fwhm_rad = max([np.deg2rad(experiment_data.fwhm/60.0)]
                       + [comp.amp_fwhm_rad for comp in comp_list])
        full_map = comp_list.get_sky_at_nu(experiment_data.nu, experiment_data.nside, pols,
                                           fwhm_rad, lmax=experiment_data.lmax)
    else:
        full_map = None
    local_map = experiment_data.pixel_domain.scatter_from_full(full_map, ncomp, dtype=np.float32)
    return DistributedMap(local=local_map, full=full_map)


def receive_compsep(mpi_info: MPIInfo, experiment_data: DetectorGroupTOD) -> DistributedMap:
    """Receive the CompSep components and distribute the realized band map within TOD.

    The band master receives the ``CompList`` from the band's CompSep sender and realizes it at the
    band frequency and resolution of `experiment_data`; the result is distributed to all band ranks
    (see ``_realize_and_distribute_sky``).

    Returns:
        The realized band sky map, as this rank holds it.
    """
    comp_list = None
    if mpi_info.band.rank == 0:
        comp_list = mpi_info.world.comm.recv(source=mpi_info.tod.compsep_sender)
    return _realize_and_distribute_sky(comp_list, experiment_data, mpi_info.band.comm)


def get_local_initial_sky(mpi_info: MPIInfo, experiment_data: DetectorGroupTOD,
                          params: Bunch) -> DistributedMap:
    """Build the initial components locally and realize them at this TOD band.

    Used when there are no CompSep ranks: the band master builds the component list from the
    component parameters and init files and realizes it; the realized map is distributed to all
    band ranks (see `_realize_and_distribute_sky`). Mirrors `receive_compsep`, minus the
    cross-world receive.
    """
    comp_list = None
    if mpi_info.band.rank == 0:
        comp_list = CompList.init_from_params(params)
        for comp in comp_list:
            # Nothing samples beta in a TOD-only run, so beta is that of the parameter file.
            comp.load_initial_state(None, apply_init_from=True, restore_beta=False)
    return _realize_and_distribute_sky(comp_list, experiment_data, mpi_info.band.comm)


def send_tod(mpi_info: MPIInfo, tod_map_dict: dict[str, DetectorMap]) -> None:
    """Send this band's maps to its CompSep ranks, one per polarization (see `receive_tod`).

    `tod_map_dict` holds the band's maps from `process_tod`, by polarization ("I", "QU"). A map in
    a polarization that the band has no CompSep polband of is not sent.
    """
    if mpi_info.band.rank == 0:
        for pol, detector_map in tod_map_dict.items():
            if pol in mpi_info.tod.compsep_receivers:
                mpi_info.world.comm.send(detector_map, dest=mpi_info.tod.compsep_receivers[pol])
            else:
                logger.debug(f"Pol-{pol} TOD-processing result discarded because band "
                             f"{mpi_info.tod.band_name} does not require it on the CompSep side.")


###########################################################
# ON COMPSEP SIDE
###########################################################

def receive_tod(mpi_info: MPIInfo, band_map: DetectorMap|None, params: Bunch) -> DetectorMap:
    """Return this CompSep rank's band map of the next iteration (the counterpart of `send_tod`).

    A band with `get_from: file` reads its map on the first call, and keeps it after that. Any
    other band receives its map from the TOD side.

    Args:
        mpi_info (MPIInfo): Where this rank sits in the run.
        band_map (DetectorMap): This rank's band map of the previous iteration. None before
                 the first iteration.
        params (Bunch): The full parameter file.
    """
    compsep = mpi_info.compsep
    if compsep.tod_sender is None:  # A band read from file.
        if band_map is None:
            logger.info(f"CompSep: Rank {compsep.rank} reading static map data from file.")
            band_map = read_data_map_from_file(params.compsep.bands[compsep.band_name],
                                               compsep.pol, params)
    else:
        logger.debug(f"CompSep: Rank {compsep.rank} receiving TOD data for {compsep.band_name} "
                     f"{compsep.pol} from world rank {compsep.tod_sender}.")
        band_map = mpi_info.world.comm.recv(source=compsep.tod_sender)
    return band_map


def send_compsep(mpi_info: MPIInfo, comp_list: CompList) -> None:
    """Send the CompSep components back to TOD when this rank's polband owns the return path.

    Before this boundary, ``sample_amplitudes`` broadcasts every polarization part solved by a
    sampling group across the CompSep communicator. Every rank therefore holds the same complete
    component list, with both I and QU amplitudes; this function performs no additional QU-to-I
    transfer.

    An IQU TOD band needs only one copy of that complete list, so only its I rank sends it. A
    QU-only band has no I rank, so its QU rank sends. A band read from file sends nothing.
    (`mpi_info.compsep.tod_receiver` is None on the ranks that do not send.)

    Args:
        mpi_info (MPIInfo): Where this rank sits in the run.
        comp_list: The components of this Gibbs sample.
    """
    if mpi_info.compsep.tod_receiver is not None:
        mpi_info.world.comm.send(comp_list, dest=mpi_info.compsep.tod_receiver)
