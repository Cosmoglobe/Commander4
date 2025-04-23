from mpi4py import MPI
from mpi4py.MPI import Comm
import numpy as np
import logging
from pixell import bunch
import healpy as hp

from src.python.data_models.detector_map import DetectorMap


def send_compsep(my_band_idx: int, detector_map: np.array, destinations: list[int]):
    """ MPI-send the results from compsep to a set of other destinations.

    Assumes the COMM_WORLD communicator.

    Input:
        my_band_idx (int): The index of the band for which the current process is responsible.
        detector_map (np.array[float]): A sky realization at a given band frequency.
        destinations (list of ints): The world ranks of the destination processes, one per band.
    """

    MPI.COMM_WORLD.send(detector_map, dest=destinations[my_band_idx])


def receive_compsep(band_comm: Comm, my_band_idx: int, band_master: bool, compsep_band_masters: list[int]):
    """ MPI-receive the results from compsep (used in conjunction with send_compsep).

    Input:
        band_comm (MPI.Comm): The inter-band communicator.
        my_band_idx (int): The index of the band for which the current process is responsible.
        compsep_band_masters list(int): List of the World ranks of the senders of the compsep information.

    Returns:
        detector_map (np.array): The detector map of a single band,
            distributed to all processes belonging to the band communicator.
    """
    # detector_map = None
    if band_master:
        detector_map = MPI.COMM_WORLD.recv(source=compsep_band_masters[my_band_idx])
    else:
        detector_map = None
    detector_map = band_comm.bcast(detector_map, root=0)  # Currently all TOD MPI ranks need a copy of the relevant detector map, which is a little wasteful - a reason for doing OpenMP for mapmaking.
    return detector_map


def send_tod(senders: list[int], destinations: list[int], tod_map: DetectorMap):
    """ MPI-send the results from a single band TOD processing to another
        destination.

    Assumes the COMM_WORLD communicator.

    Input:
        band_master (bool): Whether this is the master band process.
        tod_map (DetectorMap): The output map from process_tod for the band
            belonging to this process.
        destination (ints): The world rank of the destination process.
    """
    logger = logging.getLogger(__name__)
    for i in range(len(senders)):
        if MPI.COMM_WORLD.rank == senders[i]:  # If I am sender nr i, I send the data to destination nr i.
            logger.info(f"Rank {senders[i]} sending TOD data to {destinations[i]}")
            MPI.COMM_WORLD.send(tod_map, dest=destinations[i])


def receive_tod(senders, my_compsep_rank):
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
    logger.info(f"Rank {my_compsep_rank} receiving TOD data from {senders[my_compsep_rank]}")
    data = MPI.COMM_WORLD.recv(source=senders[my_compsep_rank])
    return data