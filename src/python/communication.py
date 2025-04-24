from mpi4py import MPI
from mpi4py.MPI import Comm
import numpy as np
import logging
from pixell import bunch
import healpy as hp

from src.python.data_models.detector_map import DetectorMap


def send_compsep(my_band_identifier: int, detector_map: np.array, destinations: list[int]):
    """ MPI-send the results from compsep to a set of other destinations.

    Assumes the COMM_WORLD communicator.

    Input:
        my_band_idx (int): The index of the band for which the current process is responsible.
        detector_map (np.array[float]): A sky realization at a given band frequency.
        destinations (list of ints): The world ranks of the destination processes, one per band.
    """
    if destinations is not None:
        MPI.COMM_WORLD.send(detector_map, dest=destinations[my_band_identifier])


def receive_compsep(band_comm: Comm, my_band_identifier: int, band_master: bool, CompSep_band_masters_dict: list[int]):
    """ MPI-receive the results from compsep (used in conjunction with send_compsep).

    Input:
        band_comm (MPI.Comm): The inter-band communicator.
        my_band_idx (int): The index of the band for which the current process is responsible.
        compsep_band_masters list(int): List of the World ranks of the senders of the compsep information.

    Returns:
        detector_map (np.array): The detector map of a single band,
            distributed to all processes belonging to the band communicator.
    """
    if band_master:
        detector_map = MPI.COMM_WORLD.recv(source=CompSep_band_masters_dict[my_band_identifier])
    else:
        detector_map = None
    detector_map = band_comm.bcast(detector_map, root=0)  # Currently all TOD MPI ranks need a copy of the relevant detector map, which is a little wasteful - a reason for doing OpenMP for mapmaking.
    return detector_map


def send_tod(tod_map: DetectorMap, CompSep_band_masters_dict, my_band_identifier):
    """ MPI-send the results from a single band TOD processing to another
        destination.

    Assumes the COMM_WORLD communicator.

    Input:
        band_master (bool): Whether this is the master band process.
        tod_map (DetectorMap): The output map from process_tod for the band
            belonging to this process.
        destination (ints): The world rank of the destination process.
    """
    MPI.COMM_WORLD.send(tod_map, dest=CompSep_band_masters_dict[my_band_identifier])


def receive_tod(senders, my_compsep_rank, my_band, band_identifier, curr_tod_output):
    """ MPI-receive the results from the TOD processing (used in conjunction
        with send_tod).

    Input:
        proc_master (bool): Whether this is the master of the TOD processing
            communicator.
        proc_comm (MPI.Comm): The TOD processing communicator.
        senders (list of int): The sender world rank of the compsep information.
        num_bands (int): The number of bands to receive from (should be same as
            length of 'senders').

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


def read_map_from_file(my_band: bunch):
    """To be run once before starting TOD processing.

    Determines whether the process is TOD master, creates the band communicator
    and determines whether the process is the band master. Also reads the
    experiment data.

    Input:
        proc_comm (MPI.Comm): Communicator for the TOD processes.
        params (bunch): The parameters from the input parameter file.

    Output:
        proc_master (bool): Whether this process is the master of the TOD process.
        proc_comm (MPI.Comm): The same as the input communicator (just returned for clarity).
        band_master (bool): Whether this process is the master of the inter-band communicator.
        band_comm (MPI.Comm): The inter-band communicator.
        experiment_data (DetectorTOD): THe TOD data for the band of this process.
    """

    map_signal = hp.read_map(my_band.path_signal_map)
    map_rms = hp.read_map(my_band.path_rms_map)  # map_rms = np.ones_like(map_signal)
    detmap = DetectorMap(map_signal, None, map_rms, my_band.freq, my_band.fwhm)
    return detmap
