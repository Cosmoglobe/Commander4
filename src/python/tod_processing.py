import numpy as np
from mpi4py import MPI
import h5py
import healpy as hp
import math
import logging
from pixell.bunch import Bunch
from output import log
from scipy.fft import rfft, irfft, rfftfreq
from numpy.typing import NDArray

from src.python.data_models.detector_map import DetectorMap
from src.python.data_models.detector_TOD import DetectorTOD
from src.python.data_models.scan_TOD import ScanTOD
from src.python.utils.mapmaker import single_det_map_accumulator

nthreads=1

def get_empty_compsep_output(staticData: list[DetectorTOD], params) -> NDArray[np.float64]:
    "Creates a dummy compsep output for a single band"
    return np.zeros(12*params.nside**2,dtype=np.float64)


def tod2map(band_comm: MPI.Comm, det_static: DetectorTOD, det_cs_map: NDArray, params: Bunch) -> DetectorMap:
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
        detmap = DetectorMap(map_signal, map_corr_noise, map_rms, det_static.nu, det_static.fwhm)
        return detmap


def read_TOD_data(h5_filename: str, band: int, scan_idx_start: int, scan_idx_stop: int, nside: int, fwhm: float) -> list[ScanTOD]:
    logger = logging.getLogger(__name__)
    # h5_filename = params.input_paths.tod_filename
    with h5py.File(h5_filename) as f:
        # for band in bands:
        # band = params.bands[band_idx]
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
            log.logassert(np.max(pix) < 12*nside**2, f"Nside is {nside}, but found pixel index exceeding 12nside^2 ({np.max(12*nside**2)})", logger)
            theta, phi = hp.pix2ang(nside, pix)
            scanlist.append(ScanTOD(tod, theta, phi, psi, 0., iscan))
        det = DetectorTOD(scanlist, float(band), fwhm)
    return det


def find_unique_pixels(scanlist: list[ScanTOD], params: Bunch) -> NDArray[np.float64]:
    """Finds the unique pixels in the list of scans.

    Input:
        scanlist (list[ScanTOD]): The list of scans.
        params (Bunch): The parameters from the input parameter file.

    Output:
        unique_pixels (np.array): The unique pixels in the scans.
    """
    logger = logging.getLogger(__name__)
    nside = params.nside
    unique_pixels = np.zeros(12*nside**2, dtype=np.float64)
    for scan in scanlist:
        scan_map, theta, phi, psi = scan.data
        ntod = scan_map.shape[0]
        pix = hp.ang2pix(nside, theta, phi)
        unique_pixels[pix] += 1
    unique_pixels[unique_pixels != 0] = 1
    return unique_pixels


def init_tod_processing(tod_comm: MPI.Comm, params: Bunch) -> tuple[MPI.Comm, MPI.Comm, str, dict[str,int], DetectorTOD]:
    """To be run once before starting TOD processing.

    Determines whether the process is TOD master, creates the band communicator
    and determines whether the process is the band master. Also reads the
    experiment data.

    Input:
        tod_comm (MPI.Comm): Communicator for the TOD processes.
        params (Bunch): The parameters from the input parameter file.

    Output:
        is_band_master (bool): Whether this process is the master of the band communicator.
        band_comm (MPI.Comm): A new communicator for the ranks working on the same band within TOD processing.
        my_band_identifier (str): Unique string identifier for the experiment+band this process is responsible for.
        tod_band_masters_dict (dict[str->int]): Dictionary mapping band identifiers to the global rank of the process responsible for that band.
        experiment_data (DetectorTOD): THe TOD data for the band of this process.
    """

    logger = logging.getLogger(__name__)
    # num_bands = len(params.bands)
    # bands_per_experiment = []
    # experiment_names = []
    global_rank = 0
    for experiment in params.experiments:
        if params.experiments[experiment].enabled:
            # experiment_names.append(experiment)
            # bands_per_experiment.append(len(params.experiments[experiment].bands))
            for band in params.experiments[experiment].bands:
                if params.experiments[experiment].bands[band].enabled:
                    if tod_comm.Get_rank() == global_rank:
                        my_experiment_name = experiment
                        my_band_name = band
                        my_experiment = params.experiments[experiment]
                        my_band = params.experiments[experiment].bands[band]
                        my_num_scans = params.experiments[experiment].num_scans
                    global_rank += 1
    tot_num_bands = global_rank

    # rank_to_assignment = {}
    # current_rank_offset = 0
    # for exp_idx, num_bands in enumerate(bands_per_experiment):
    #     for band_idx in range(num_bands):
    #         global_rank = current_rank_offset + band_idx
    #         # rank_to_assignment[global_rank] = (exp_idx, band_idx)
    #         rank_to_assignment[global_rank] = (exp_idx, band_idx)
    #     current_rank_offset += num_bands
    # my_experiment_idx, my_band_idx = rank_to_assignment[tod_comm.Get_rank()]
    # my_experiment_name = experiment_names[my_experiment_idx]
    # my_experiment = params.experiments[my_experiment_name]
    # my_band = my_experiment.bands[my_band_idx]

    # am I the master of the TOD communicator?
    MPIsize_tod, MPIrank_tod = tod_comm.Get_size(), tod_comm.Get_rank()
    tod_master = MPIrank_tod == 0
    if tod_master:
        logger.info(f"TOD: {MPIsize_tod} tasks allocated to TOD processing of {tot_num_bands} bands.")
        log.logassert(MPIsize_tod >= tot_num_bands, f"Number of MPI tasks dedicated to TOD processing ({MPIsize_tod}) must be equal to or larger than the number of bands ({tot_num_bands}).", logger)

    MPIcolor_band = MPIrank_tod%tot_num_bands  # Spread the MPI tasks over the different bands.
    band_comm = tod_comm.Split(MPIcolor_band, key=MPIrank_tod)  # Create communicators for each different band.
    MPIsize_band, MPIrank_band = band_comm.Get_size(), band_comm.Get_rank()  # Get my local rank, and the total size of, the band-communicator I'm on.
    logger.info(f"TOD: Hello from TOD-rank {MPIrank_tod} (on machine {MPI.Get_processor_name()}), dedicated to band {MPIcolor_band}, with local rank {MPIrank_band} (local communicator size: {MPIsize_band}).")
    
    is_band_master = MPIrank_band == 0  # Am I the master of my local band.

    # Creating "tod_band_masters", an array which maps the band index to the rank of the master of that band.
    my_band_identifier = f"{my_experiment_name}$$${my_band_name}"
    data = (my_band_identifier, MPI.COMM_WORLD.Get_rank()) if band_comm.Get_rank() == 0 else None
    # data = (MPIcolor_band, tod_comm.Get_rank()) if band_comm.Get_rank() == 0 else None
    all_data = tod_comm.allgather(data)
    tod_band_masters_dict = {item[0]: item[1] for item in all_data if item is not None}
    # tod_band_masters = np.array([tod_band_masters_dict[i] for i in range(tot_num_bands)])
    scans_per_rank = math.ceil(my_num_scans/MPIsize_band)
    my_scans_start = scans_per_rank * MPIrank_band
    my_scans_stop = min(scans_per_rank * (MPIrank_band + 1), my_num_scans) # "min" in case the number of scans is not divisible by the number of ranks
#    my_scans_start, my_scans_stop = scans_per_rank*MPIrank_band, scans_per_rank*(MPIrank_band + 1)
    logger.info(f"TOD: Rank {MPIrank_tod} assigned scans {my_scans_start} - {my_scans_stop} on band{MPIcolor_band}.")
    experiment_data = read_TOD_data(my_experiment.data_path, my_band.freq, my_scans_start, my_scans_stop, my_experiment.nside, my_band.fwhm)

    return is_band_master, band_comm, my_band_identifier, tod_band_masters_dict, experiment_data



def subtract_sky_model(experiment_data: DetectorTOD, det_compsep_map: np.array, params: Bunch) -> DetectorTOD:
    """Subtracts the sky model from the TOD data.
    Input:
        experiment_data (DetectorTOD): The experiment TOD object.
        det_compsep_map (np.array): The current estimate of the sky model as seen by the band belonging to the current process.
        params (Bunch): The parameters from the input parameter file.
    Output:
        experiment_data (DetectorTOD): The experiment TOD with the estimated white noise level added to each scan.
    """
    nside = params.nside
    for scan in experiment_data.scans:
        scan_map, theta, phi, psi = scan.data
        ntod = scan_map.shape[0]
        pix = hp.ang2pix(nside, theta, phi)
        scan.sky_subtracted_tod = scan_map - det_compsep_map[pix]
        if params.galactic_mask:
            scan.galactic_mask_array = np.abs(theta - np.pi/2.0) > 5.0*np.pi/180.0
    return experiment_data



def estimate_white_noise(experiment_data: DetectorTOD, params: Bunch) -> DetectorTOD:
    """Estimate the white noise level in the TOD data, add it to the scans, and return the updated experiment data.
    Input:
        experiment_data (DetectorTOD): The experiment TOD object.
        params (Bunch): The parameters from the input parameter file.
    Output:
        experiment_data (DetectorTOD): The experiment TOD with the estimated white noise level added to each scan.
    """
    for scan in experiment_data.scans:
        if params.galactic_mask and np.sum(scan.galactic_mask_array) > 50:  # If we have enough data points to estimate the noise, we use the masked version.
            sigma0 = np.std(scan.sky_subtracted_tod[scan.galactic_mask_array][1:] - scan.sky_subtracted_tod[scan.galactic_mask_array][:-1])/np.sqrt(2)
        else:
            sigma0 = np.std(scan.sky_subtracted_tod[1:] - scan.sky_subtracted_tod[:-1])/np.sqrt(2)
        scan.sigma0 = sigma0
    return experiment_data



def sample_noise(band_comm: MPI.Comm, experiment_data: DetectorTOD, params: Bunch) -> DetectorTOD:
    nside = params.nside
    for scan in experiment_data.scans:
        f_samp = 180 # params.fsamp
        scan_map, theta, phi, psi = scan.data
        ntod = scan_map.shape[0]
        freq = rfftfreq(ntod, d = 1/f_samp)
        fknee = 1.0
        alpha = -1.0
        N = freq.shape[0]

        if params.sample_corr_noise:
            C_wn = scan.sigma0*np.ones(N)
            C_1f_inv = np.zeros(N)  # 1.0/C_1f
            C_1f_inv[1:] = freq[1:]/scan.sigma0

            const = 1.0  # This is the normalization constant for FFT, which I'm unsure what is for scipys FFT, might be wrong!
            w1 = (np.random.normal(0, 1, N) + 1.j*np.random.normal(0, 1, N))/np.sqrt(2)
            w2 = (np.random.normal(0, 1, N) + 1.j*np.random.normal(0, 1, N))/np.sqrt(2)
            # I'm always a bit confused about when it's fine to use rfft as opposed to full fft, so might want to double check this:
            n_corr_est_fft_WF = rfft(scan.sky_subtracted_tod)/(1 + C_wn*C_1f_inv)
            n_corr_est_fft_fluct = (const*(np.sqrt(C_wn)*w1 + C_wn*np.sqrt(C_1f_inv)*w2))/(1 + C_wn*C_1f_inv)
            n_corr_est_fft = n_corr_est_fft_WF + n_corr_est_fft_fluct
            n_corr_est_fft[0] = 0.0
            n_corr_est_fft_WF[0] = 0.0
            # n_corr_est_WF = irfft(n_corr_est_fft_WF, n=scan.sky_subtracted_tod.shape[0])
            scan.n_corr_est = irfft(n_corr_est_fft, n=scan.sky_subtracted_tod.shape[0])
            scan.sky_subtracted_tod -= scan.n_corr_est
    return experiment_data



def process_tod(band_comm: MPI.Comm, experiment_data: DetectorTOD,
                compsep_output: np.array, params: Bunch) -> DetectorMap:
    """ Performs a single TOD iteration.

    Input:
        band_comm (MPI.Comm): The inter-band communicator.
        experiment_data (DetectorTOD): The input experiment TOD for the band
            belonging to the current process.
        compsep_output (np.array): The current best estimate of the sky model
            as seen by the band belonging to the current process.
        params (Bunch): The parameters from the input parameter file.

    Output:
        DetectorMap instance which represents the correlated noise subtracted
            TOD data for the band belonging to the current process.
    """
    experiment_data = subtract_sky_model(experiment_data, compsep_output, params)
    experiment_data = estimate_white_noise(experiment_data, params)
    experiment_data = sample_noise(band_comm, experiment_data, params)
    todproc_output = tod2map(band_comm, experiment_data, compsep_output, params)
    return todproc_output
