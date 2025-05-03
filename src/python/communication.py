from mpi4py import MPI
from mpi4py.MPI import Comm
import numpy as np
import logging
from pixell.bunch import Bunch
import healpy as hp
from numpy.typing import NDArray

from src.python.data_models.detector_map import DetectorMap


def send_compsep(my_band_identifier: str, detector_map: NDArray[np.floating], destinations: dict[str, int]|None) -> None:
    """ MPI-send the results from compsep to a destinations on the TOD processing side (used in conjunction with receive_compsep).
    Assumes the COMM_WORLD communicator.

    Input:
        my_band_identifier (str): The string uniquely indentifying the experiment+band of this rank (example: 'PlanckLFI$$$30GHz').
        detector_map (np.array[float]): A sky realization at a given band frequency.
        destinations (dict[str->int]): A dictionary mapping the string in 'my_band_identifier' to the world rank of the destination task (on the TOD side).
    """
    if destinations is not None:
        MPI.COMM_WORLD.send(detector_map, dest=destinations[my_band_identifier])


def receive_compsep(band_comm: Comm, my_band_identifier: str, band_master: bool, senders: dict[str, int]) -> NDArray[np.floating]:
    """ MPI-receive the results from compsep (used in conjunction with send_compsep).

    Input:
        band_comm (MPI.Comm): The inter-band communicator.
        my_band_identifier (str): The string uniquely indentifying the experiment+band of this rank (example: 'PlanckLFI$$$30GHz').
        senders (dict[str->int]): A dictionary mapping the string in 'my_band_identifier' to the world rank of the senbder task (on the CompSep side).

    Returns:
        detector_map (np.array): The detector map of a single band, distributed to all processes belonging to the band communicator.
    """
    if band_master:
        detector_map = MPI.COMM_WORLD.recv(source=senders[my_band_identifier])
    else:
        detector_map = None
    detector_map = band_comm.bcast(detector_map, root=0)  # Currently all TOD MPI ranks need a copy of the relevant detector map, which is a little wasteful - a reason for doing OpenMP for mapmaking.
    return detector_map


def send_tod(band_master: bool, tod_map: DetectorMap, CompSep_band_masters_dict: dict[str, int], my_band_identifier: str) -> None:
    """ MPI-send the results from a single band TOD processing to a task on the CompSep side (used in conjunction with receive_tod).

    Assumes the COMM_WORLD communicator.

    Input:
        band_master (bool): Whether this is the master band process.
        tod_map (DetectorMap): The output map from process_tod for the band belonging to this process.
        CompSep_band_masters_dict (dict [str -> int]): The world rank of the destination process.
    """
    if band_master:
        MPI.COMM_WORLD.send(tod_map, dest=CompSep_band_masters_dict[my_band_identifier])


def receive_tod(senders: dict[str,int], my_compsep_rank: int, my_band: Bunch, band_identifier: str, curr_tod_output: DetectorMap|None) -> DetectorMap:
    """ MPI-receive the results from the TOD processing (used in conjunction with send_tod).

    Input:
        senders: (dict[str->int]): A dictionary mapping a string uniquely identifying each experiment+band to the world rank of the sender task (on the CompSep side).
        my_compsep_rank (int): Rank of the current process within the CompSep communicator. Only used for prints.
        my_band (Bunch): The section of the parameter file corresponding to this CompSep band, as a "Bunch" type.
        band_identifier (str): The string uniquely indentifying the experiment+band of this rank (example: 'PlanckLFI$$$30GHz').
        curr_tod_output (DetectorMap): The current map output from the TOD process. Should be None unless map is read from file already in a previous iteration.

    Returns:
        data (list of DetectorMaps): nbands (DetectorMap)
    """
    logger = logging.getLogger(__name__)
    if my_band.get_from == "file":
        if curr_tod_output is None:
            logger.info(f"CompSep: Rank {my_compsep_rank} reading static map data from file.")
            curr_tod_output = read_map_from_file(my_band)
        else:
            logger.info(f"CompSep: Rank {my_compsep_rank} already has static map data. Continuing.")
    else:
        logger.info(f"CompSep: Rank {my_compsep_rank} receiving TOD data ({band_identifier}) from TOD process with global rank {senders[band_identifier]}")
        curr_tod_output = MPI.COMM_WORLD.recv(source=senders[band_identifier])

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