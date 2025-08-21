from mpi4py import MPI
from mpi4py.MPI import Comm
import numpy as np
import logging
from pixell.bunch import Bunch
import healpy as hp
from numpy.typing import NDArray

from src.python.data_models.detector_map import DetectorMap


def send_compsep(mpi_info: Bunch, my_band_identifier: str, detector_map: NDArray[np.floating], destinations: dict[str, int]|None) -> None:
    """ MPI-send the results from compsep to a destinations on the TOD processing side (used in conjunction with receive_compsep).
    Assumes the COMM_WORLD communicator.

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        my_band_identifier (str): The string uniquely indentifying the experiment+band of this rank (example: 'PlanckLFI$$$30GHz').
        detector_map (np.array[float]): A sky realization at a given band frequency.
        destinations (dict[str->int]): A dictionary mapping the string in 'my_band_identifier' to the world rank of the destination task (on the TOD side) (This is the same as is found in mpi_info)
    """
    if destinations is not None:
        mpi_info['world']['comm'].send(detector_map, dest=destinations[my_band_identifier])


def receive_compsep(mpi_info: Bunch, my_band_identifier: str, senders: dict[str, int]) -> NDArray[np.floating]:
    """ MPI-receive the results from compsep (used in conjunction with send_compsep).

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        my_band_identifier (str): The string uniquely indentifying the experiment+band of this rank (example: 'PlanckLFI$$$30GHz').
        senders (dict[str->int]): A dictionary mapping the string in 'my_band_identifier' to the world rank of the senbder task (on the CompSep side).

    Returns:
        detector_map (np.array): The detector map of a single band, distributed to all processes belonging to the band communicator.
    """

    world_comm = mpi_info['world']['comm']
    is_band_master = mpi_info['band']['is_master']
    band_master = mpi_info['band']['master']
    if is_band_master:
        detector_map = world_comm.recv(source=senders[my_band_identifier])
    else:
        detector_map = None
    detector_map = band_comm.bcast(detector_map, root=band_master)  # Currently all TOD MPI ranks need a copy of the relevant detector map, which is a little wasteful - a reason for doing OpenMP for mapmaking.
    return detector_map


def send_tod(mpi_info: Bunch, tod_map: DetectorMap, my_band_identifier: str) -> None:
    """ MPI-send the results from a single band TOD processing to a task on the CompSep side (used in conjunction with receive_tod).

    Assumes the COMM_WORLD communicator.

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        tod_map (DetectorMap): The output map from process_tod for the band belonging to this process.
        my_band_identifier (str): The string uniquely indentifying the experiment+band of this rank (example: 'PlanckLFI$$$30GHz').
    """
    if mpi_info['band']['is_master']:
        mpi_info['world']['comm'].send(tod_map, dest=mpi_info['world']['compsep_band_masters_dict'][my_band_identifier])


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
    my_compsep_rank = mpi_info['compsep']['rank']
    if my_band.get_from == "file":
        if curr_tod_output is None:
            logger.info(f"CompSep: Rank {my_compsep_rank} reading static map data from file.")
            curr_tod_output = read_map_from_file(my_band)
        else:
            logger.info(f"CompSep: Rank {my_compsep_rank} already has static map data. Continuing.")
    else:
        logger.info(f"CompSep: Rank {my_compsep_rank} receiving TOD data ({band_identifier}) from TOD process with global rank {senders[band_identifier]}")
        curr_tod_output = mpi_info['world']['comm'].recv(source=senders[band_identifier])

    return curr_tod_output


def read_map_from_file(my_band: Bunch) -> DetectorMap:
    """To be run once before starting TOD processing.

    Determines whether the process is TOD master, creates the band communicator
    and determines whether the process is the band master. Also reads the
    experiment data.

    Input:
        my_band (Bunch): The section of the parameter file corresponding to this CompSep band, as a "Bunch" type.

    Output:
        data (list of DetectorMaps): nbands (DetectorMap)
    """

    map_signal = hp.read_map(my_band.path_signal_map)
    map_rms = hp.read_map(my_band.path_rms_map)
    # map_rms = np.ones_like(map_signal)
    detmap = DetectorMap(map_signal, None, map_rms, my_band.freq, my_band.fwhm)
    return detmap
