from mpi4py import MPI
from mpi4py.MPI import Comm
import numpy as np
import logging
from pixell.bunch import Bunch
import healpy as hp
from numpy.typing import NDArray

from src.python.data_models.detector_map import DetectorMap
from src.python.data_models.detector_TOD import DetectorTOD
from src.python.maps_from_file import read_sim_map_from_file, read_data_map_from_file


def send_compsep(mpi_info: Bunch, my_band_identifier: str, band_sky_map: NDArray[np.floating], destinations: dict[str, int]|None) -> None:
    """ MPI-send the results from compsep to a destinations on the TOD processing side (used in conjunction with receive_compsep).
    Assumes the COMM_WORLD communicator.

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        my_band_identifier (str): The string uniquely indentifying the experiment+band of this rank
                                  (example: 'PlanckLFI$$$30GHz').
        band_sky_map (np.array[float]): A sky realization at a given band frequency.
        destinations (dict[str->int]): A dictionary mapping the string in 'my_band_identifier' to
                                       the world rank of the destination task (on the TOD side)
                                       (This is the same as is found in mpi_info)
    """
    if destinations is not None:
        if my_band_identifier in destinations:  # If the band our rank is holding is not in
                                                # "destinations", it means it did not come from
                                                # TOD-processing, and should not be sent
                                                #back either.
            if my_band_identifier.endswith('_I'): #Currently the polarization ranks send over to the Intensity ones which then in turn broadcast and evaluate the sky.
                mpi_info.world.comm.send(band_sky_map, dest=destinations[my_band_identifier])
 

def receive_compsep(mpi_info: Bunch, experiment_data: DetectorTOD, my_band_identifier: str,
                    senders: dict[str, int]) -> NDArray[np.floating]:
    """ MPI-receive the results from compsep (used in conjunction with send_compsep).

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        my_band_identifier (str): The string uniquely indentifying the experiment+band of this rank
                                  (example: 'PlanckLFI$$$30GHz').
        senders (dict[str->int]): A dictionary mapping the string in 'my_band_identifier' to the
                                  world rank of the senbder task (on the CompSep side).

    Returns:
        detector_map (np.array): The detector map of a single band, distributed to all processes
                                 belonging to the band communicator.
    """
    world_comm = mpi_info.world.comm
    band_comm = mpi_info.band.comm
    is_band_master = mpi_info.band.is_master
    band_master = mpi_info.band.master
    if is_band_master:
        sky_model = world_comm.recv(source=senders[my_band_identifier])
    else:
        sky_model = None
    sky_model = band_comm.bcast(sky_model, root=0)  # Currently all TOD MPI ranks need a copy of the relevant detector map, which is a little wasteful - a reason for doing OpenMP for mapmaking.
    detector_map = sky_model.get_sky_at_nu(experiment_data.nu, experiment_data.nside,
                                           fwhm=np.deg2rad(experiment_data.fwhm/60.0))
    return detector_map


def send_tod(mpi_info: Bunch, tod_map: DetectorMap, my_band_identifier: str, receivers: Bunch) -> None:
    """ MPI-send the results from a single band TOD processing to a task on the CompSep side (used in conjunction with receive_tod).

    Assumes the COMM_WORLD communicator.

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        tod_map (DetectorMap): The output map from process_tod for the band belonging to this process.
        my_band_identifier (str): The string uniquely indentifying the experiment+band of this rank (example: 'PlanckLFI$$$30GHz').
        receivers (Bunch): Maps a band identifier to the band master on the compsep side.
    """
    logger = logging.getLogger(__name__)
    if mpi_info.tod.is_master:
        logger.info(f"Compsep band masters: {mpi_info.world.compsep_band_masters}")
    if mpi_info.band.is_master:
        mpi_info.world.comm.send(tod_map, dest=receivers[my_band_identifier])


def receive_tod(mpi_info: Bunch, senders: dict[str,int], my_band: Bunch, band_identifier: str, curr_tod_output: DetectorMap|None) -> DetectorMap:
    """ MPI-receive the results from the TOD processing (used in conjunction with send_tod).

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        senders: (dict[str->int]): A dictionary mapping a string uniquely identifying each experiment+band to the world rank of the sender task (on the CompSep side).
        my_band (Bunch): The section of the parameter file corresponding to this CompSep band, as a "Bunch" type.
        band_identifier (str): The string uniquely indentifying the experiment+band of this rank (example: 'PlanckLFI$$$30GHz').
        curr_tod_output (DetectorMap): The current map output from the TOD process. Should be None unless map is read from file already in a previous iteration.

    Returns:
        data (list of DetectorMaps): nbands (DetectorMap)
    """
    logger = logging.getLogger(__name__)
    my_compsep_rank = mpi_info.compsep.rank
    if my_band.get_from == "file":
        if curr_tod_output is None:
            logger.info(f"CompSep: Rank {my_compsep_rank} reading static map data from file.")
            curr_tod_output = read_data_map_from_file(my_band)

            ###
            mpi_info.world.comm.Barrier()
        else:
            logger.info(f"CompSep: Rank {my_compsep_rank} already has static map data. Continuing.")
    else:
        logger.info(f"CompSep: Rank {my_compsep_rank} receiving TOD data ({band_identifier}) from TOD process with global rank {senders[band_identifier]}")
        curr_tod_output = mpi_info.world.comm.recv(source=senders[band_identifier])

    #We split the detector map depending on the polarization of the mpi rank's band.
    # FIXME: this is a bit wasteful, it should be handled more elegantly with the correct pol being sent to the right tasks.
    if curr_tod_output.npol>2: #The detector map holds both I and QU
        if my_band.identifier.endswith("_I"):
            curr_tod_output._map_sky = curr_tod_output._map_sky[0,:].reshape((1,-1))
            curr_tod_output._inv_n_map = curr_tod_output._inv_n_map[0,:].reshape((1,-1))
        elif my_band.identifier.endswith("_QU"):
            curr_tod_output._map_sky = curr_tod_output._map_sky[1:,:]
            curr_tod_output._inv_n_map = curr_tod_output._inv_n_map[1:,:]
        else:
            raise ValueError(f"Polarization undefined in band {my_band.identifier}!")
    
    return curr_tod_output
