import numpy as np
from mpi4py import MPI
from mpi4py.MPI import Comm
import h5py
import healpy as hp
import math
import time
import logging
from data_models import ScanTOD, DetectorTOD, DetectorMap
from utils import single_det_mapmaker, single_det_map_accumulator
from pixell import bunch
from output import log 

nthreads=1

def get_empty_compsep_output(staticData: list[DetectorTOD], params) -> list[np.array]:
    "Creates a dummy compsep output for a single band"
    return np.zeros(12*params.nside**2,dtype=np.float64)


def tod2map(band_comm, det_static: DetectorTOD, det_cs_map: np.array, params: bunch) -> DetectorMap:
    detmap_signal, detmap_corr_noise, detmap_inv_var = single_det_map_accumulator(det_static, det_cs_map, params)
    map_signal = np.zeros_like(detmap_signal)
    map_corr_noise = np.zeros_like(detmap_corr_noise)
    map_inv_var = np.zeros_like(detmap_inv_var)
    if band_comm.Get_rank() == 0:
        band_comm.Reduce(detmap_signal, map_signal, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_corr_noise, map_corr_noise, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_inv_var, map_inv_var, op=MPI.SUM, root=0)
    else:
        band_comm.Reduce(detmap_signal, None, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_corr_noise, None, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_inv_var, None, op=MPI.SUM, root=0)

    if band_comm.Get_rank() == 0:
        map_signal[map_signal != 0] /= map_inv_var[map_signal != 0]
        map_corr_noise[map_corr_noise != 0] /= map_inv_var[map_corr_noise != 0]
        map_rms = np.zeros_like(map_inv_var) + np.inf
        map_rms[map_inv_var != 0] = 1.0/np.sqrt(map_inv_var[map_inv_var != 0])
        detmap = DetectorMap(map_signal, map_corr_noise, map_rms, det_static.nu)
        return detmap


def read_data(band_idx, scan_idx_start, scan_idx_stop, params: bunch) -> list[ScanTOD]:
    logger = logging.getLogger(__name__)
    h5_filename = params.input_paths.tod_filename
    with h5py.File(h5_filename) as f:
        # for band in bands:
        band = params.bands[band_idx]
        band_formatted = f"{band:04d}"
        scanlist = []
        for iscan in range(scan_idx_start, scan_idx_stop):
            try:
                tod = f[f'{iscan+1:06}/{band_formatted}/tod'][()].astype(np.float64)
                pix = f[f'{iscan+1:06}/{band_formatted}/pix'][()]
                psi = f[f'{iscan+1:06}/{band_formatted}/psi'][()].astype(np.float64)
            except KeyError:
                logger.exception(f"{iscan}\n{band_formatted}\n{list(f)}")
                raise KeyError
            log.logassert(np.max(pix) < 12*params.nside**2, f"Nside is {params.nside}, but found pixel index exceeding 12nside^2 ({np.max(12*params.nside**2)})", logger)
            theta, phi = hp.pix2ang(params.nside, pix)
            scanlist.append(ScanTOD(tod, theta, phi, psi, 0., iscan))
        det = DetectorTOD(scanlist, float(band))
    return det


def init_tod_processing(proc_comm: Comm, params: bunch):
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

    logger = logging.getLogger(__name__)
    num_bands = len(params.bands)

    # am I the master of the TOD communicator?
    MPIsize_tod, MPIrank_tod = proc_comm.Get_size(), proc_comm.Get_rank()
    proc_master = MPIrank_tod == 0
    if proc_master:
        logger.info(f"TOD: {MPIsize_tod} tasks allocated to TOD processing of {num_bands} bands.")
        log.logassert(MPIsize_tod >= num_bands, f"Number of MPI tasks dedicated to TOD processing ({MPIsize_tod}) must be equal to or larger than the number of bands ({num_bands}).", logger)

    MPIcolor_band = MPIrank_tod%num_bands  # Spread the MPI tasks over the different bands.
    band_comm = proc_comm.Split(MPIcolor_band, key=MPIrank_tod)  # Create communicators for each different band.
    MPIsize_band, MPIrank_band = band_comm.Get_size(), band_comm.Get_rank()  # Get my local rank, and the total size of, the band-communicator I'm on.
    logger.info(f"TOD: Hello from TOD-rank {MPIrank_tod}, dedicated to band {MPIcolor_band}, with local rank {MPIrank_band} (local communicator size: {MPIsize_band}).")
    
    band_master = MPIrank_band == 0  # Am I the master of my local band.

    scans_per_rank = math.ceil(params.num_scans/MPIsize_band)
    my_scans_start = scans_per_rank * MPIrank_band
    my_scans_stop = min(scans_per_rank * (MPIrank_band + 1), params.num_scans) # In case the number of scans is not divisible by the number of ranks
#    my_scans_start, my_scans_stop = scans_per_rank*MPIrank_band, scans_per_rank*(MPIrank_band + 1)
    logger.info(f"TOD: Rank {MPIrank_tod} assigned scans {my_scans_start} - {my_scans_stop} on band{MPIcolor_band}.")
    experiment_data = read_data(MPIcolor_band, my_scans_start, my_scans_stop, params)

    return proc_master, proc_comm, band_comm, band_master, experiment_data


def process_tod(band_comm: Comm, experiment_data: DetectorTOD,
                compsep_output: np.array, params: bunch) -> DetectorMap:
    """ Performs a single TOD iteration.

    Input:
        band_comm (MPI.Comm): The inter-band communicator.
        experiment_data (DetectorTOD): The input experiment TOD for the band
            belonging to the current process.
        compsep_output (np.array): The current best estimate of the sky model
            as seen by the band belonging to the current process.
        params (bunch): The parameters from the input parameter file.

    Output:
        DetectorMap instance which represents the correlated noise subtracted
            TOD data for the band belonging to the current process.
    """
    todproc_output = tod2map(band_comm, experiment_data, compsep_output, params)
    return todproc_output


def send_tod(band_master: bool, tod_map: DetectorMap, iter_idx: int,
             chain_idx: int, destination: int):
    """ MPI-send the results from a single band TOD processing to another
        destination.

    Assumes the COMM_WORLD communicator.

    Input:
        band_master (bool): Whether this is the master band process.
        tod_map (DetectorMap): The output map from process_tod for the band
            belonging to this process.
        iter_idx (int): The current Gibbs iteration.
        chain_idx (int): The current chain.
        destination (ints): The world rank of the destination
            process.
    """

    if band_master:
        MPI.COMM_WORLD.send((tod_map, iter_idx, chain_idx), dest=destination)


def receive_tod(proc_master: bool, proc_comm: Comm, senders: list[int],
                num_bands: int):
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
        iter (int): The Gibbs iteration of the data being received.
        chain (int): The chain of the data being received.
    """

    logger = logging.getLogger(__name__)
    data, iter, chain = [], [], []
    if proc_master:
        for i in range(num_bands):
            _data, _iter, _chain = MPI.COMM_WORLD.recv(source=senders[i])
            data.append(_data)
            iter.append(_iter)
            chain.append(_chain)
        log.logassert(np.all([i == iter[0] for i in iter]), "Different CompSep tasks received different Gibbs iteration number from TOD loop!", logger)
        log.logassert(np.all([i == chain[0] for i in chain]), "Different CompSep tasks received different Gibbs chain number from TOD loop!", logger)
        chain = chain[0]
        iter = iter[0]

    data = proc_comm.bcast(data, root=0)
    iter = proc_comm.bcast(iter, root=0)
    chain = proc_comm.bcast(chain, root=0)
    return data, iter, chain
