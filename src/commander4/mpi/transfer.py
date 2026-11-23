"""Moving data between the TOD-processing and component-separation communicators each iteration.

TOD processing sends `BandMap`s (one per band) and receives back a `CompList`. Both directions
go band master to band master, and the receiving master distributes within its band communicator.
"""
import numpy as np
import logging
from pixell.bunch import Bunch

from commander4.data_models.band_map import BandMap
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
    """Realize the band sky on the band master, and give each rank the sky at its local pixels.

    `comp_list` is needed on the band master only (None elsewhere).
    """
    # The component list is not broadcast: a copy of it and a full-sky realization on every rank
    # would be the largest TOD-side memory cost at high nside. The master keeps the full map, for
    # the chain output and the far-sidelobe transform.
    is_band_master = band_comm.Get_rank() == 0
    pols = experiment_data.pols
    npol = {"I": 1, "QU": 2, "IQU": 3}[pols]
    if is_band_master:
        # The band beam, or the beam the amplitudes already carry if it is larger (after the
        # per-pixel solver with common-resolution smoothing). The sky is then too smooth for the
        # band, but it cannot be sharpened.
        fwhm_rad = max([np.deg2rad(experiment_data.fwhm_arcmin/60.0)]
                       + [comp.amp_fwhm_rad for comp in comp_list])
        full_map = comp_list.get_sky_at_nu(experiment_data.nu, experiment_data.nside, pols,
                                           fwhm_rad, lmax=experiment_data.lmax)
    else:
        full_map = None
    local_map = experiment_data.pixel_domain.scatter_from_full(full_map, npol, dtype=np.float32)
    return DistributedMap(local=local_map, full=full_map)


def receive_compsep(mpi_info: MPIInfo, experiment_data: DetectorGroupTOD) -> DistributedMap:
    """Receive the components from CompSep, and return this band's sky as this rank holds it.

    The counterpart of `send_compsep`. Only the band master receives the `CompList`.
    """
    comp_list = None
    if mpi_info.band.rank == 0:
        comp_list = mpi_info.world.comm.recv(source=mpi_info.tod.compsep_sender)
    return _realize_and_distribute_sky(comp_list, experiment_data, mpi_info.band.comm)


def get_local_initial_sky(mpi_info: MPIInfo, experiment_data: DetectorGroupTOD,
                          params: Bunch) -> DistributedMap:
    """Return this band's initial sky, built from the parameter file, as this rank holds it.

    Used instead of `receive_compsep` in a run without CompSep ranks. The components start from
    their `init_from` files.
    """
    comp_list = None
    if mpi_info.band.rank == 0:
        comp_list = CompList.init_from_params(params)
        for comp in comp_list:
            # Nothing samples beta in a TOD-only run.
            comp.load_initial_state(None, apply_init_from=True, restore_beta=False)
    return _realize_and_distribute_sky(comp_list, experiment_data, mpi_info.band.comm)


def send_tod(mpi_info: MPIInfo, band_maps: dict[str, BandMap]) -> None:
    """Send this band's maps, by polarization ("I", "QU"), from the band master to CompSep.

    The counterpart of `receive_tod`. A map is dropped if CompSep does not use that polarization of
    the band.
    """
    if mpi_info.band.rank == 0:
        for pol, band_map in band_maps.items():
            if pol in mpi_info.tod.compsep_receivers:
                mpi_info.world.comm.send(band_map, dest=mpi_info.tod.compsep_receivers[pol])
            else:
                logger.debug(f"Pol-{pol} TOD-processing result discarded because band "
                             f"{mpi_info.tod.band_name} does not require it on the CompSep side.")


###########################################################
# ON COMPSEP SIDE
###########################################################

def receive_tod(mpi_info: MPIInfo, band_map: BandMap|None, params: Bunch) -> BandMap:
    """Return this CompSep rank's band map for the next iteration.

    The counterpart of `send_tod`. A band with `get_from: file` reads its map on the first call
    (`band_map` is None), and keeps `band_map` after that. Any other band receives a new map from
    the TOD side.
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
    """Send the components to the TOD side. The counterpart of `receive_compsep`.

    Every CompSep rank must hold the full, updated `comp_list`. Only the ranks with a TOD receiver
    send it.
    """
    # The I and QU ranks of a band hold the same components, so only one of them (the I rank, if
    # the band has one) has a `tod_receiver`.
    if mpi_info.compsep.tod_receiver is not None:
        mpi_info.world.comm.send(comp_list, dest=mpi_info.compsep.tod_receiver)
