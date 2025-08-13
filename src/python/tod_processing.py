import numpy as np
from mpi4py import MPI
import healpy as hp
import math
import logging
from pixell.bunch import Bunch
from output import log
from scipy.fft import rfft, irfft, rfftfreq
import time
from numpy.typing import NDArray
from scipy.sparse.linalg import cg, LinearOperator

from src.python.data_models.detector_map import DetectorMap
from src.python.data_models.detector_TOD import DetectorTOD
from src.python.data_models.scan_TOD import ScanTOD
from src.python.data_models.detector_samples import DetectorSamples
from src.python.data_models.scan_samples import ScanSamples
from src.python.utils.mapmaker import single_det_map_accumulator
from src.python.noise_sampling import corr_noise_realization_with_gaps, sample_noise_PS_params
from src.python.tod_processing_Planck import read_Planck_TOD_data
from src.python.utils.map_utils import get_sky_model_TOD, calculate_s_orb
from src.python.tod_processing_sim import read_TOD_sim_data

nthreads=1

def get_empty_compsep_output(staticData: DetectorTOD) -> NDArray[np.float64]:
    "Creates a dummy compsep output for a single band"
    return np.zeros(12*staticData.nside**2, dtype=np.float64)


def tod2map(band_comm: MPI.Comm, det_static: DetectorTOD, det_cs_map: NDArray, detector_samples, params: Bunch) -> DetectorMap:
    detmap_rawobs, detmap_signal, detmap_orbdipole, detmap_skysub, detmap_corr_noise, detmap_inv_var, detmap_hits = single_det_map_accumulator(det_static, det_cs_map, detector_samples, params)
    map_signal = np.zeros_like(detmap_signal)
    map_orbdipole = np.zeros_like(detmap_orbdipole)
    map_rawobs = np.zeros_like(detmap_rawobs)
    map_skysub = np.zeros_like(detmap_skysub)
    map_corr_noise = np.zeros_like(detmap_corr_noise)
    map_inv_var = np.zeros_like(detmap_inv_var)
    map_hits = np.zeros_like(detmap_hits)
    if band_comm.Get_rank() == 0:
        band_comm.Reduce(detmap_signal, map_signal, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_skysub, map_skysub, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_orbdipole, map_orbdipole, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_rawobs, map_rawobs, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_corr_noise, map_corr_noise, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_inv_var, map_inv_var, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_hits, map_hits, op=MPI.SUM, root=0)
    else:
        band_comm.Reduce(detmap_signal, None, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_skysub, None, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_orbdipole, None, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_rawobs, None, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_corr_noise, None, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_inv_var, None, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_hits, None, op=MPI.SUM, root=0)

    if band_comm.Get_rank() == 0:
        map_orbdipole[map_orbdipole != 0] /= map_hits[map_orbdipole != 0]
        map_rawobs[map_rawobs != 0] /= map_inv_var[map_rawobs != 0]
        map_signal[map_signal != 0] /= map_inv_var[map_signal != 0]
        map_skysub[map_skysub != 0] /= map_inv_var[map_skysub != 0]
        map_corr_noise[map_corr_noise != 0] /= map_hits[map_corr_noise != 0]
        map_rms = np.zeros_like(map_inv_var) + np.inf
        map_rms[map_inv_var != 0] = 1.0/np.sqrt(map_inv_var[map_inv_var != 0])
        detmap = DetectorMap(map_signal, map_corr_noise, map_rms, det_static.nu, det_static.fwhm, det_static.nside)
        detmap.g0 = detector_samples.g0_est
        detmap.gain = detector_samples.scans[0].rel_gain_est + detector_samples.g0_est
        detmap.skysub_map = map_skysub
        detmap.rawobs_map = map_rawobs
        detmap.orbdipole_map = map_orbdipole
        return detmap




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


def init_tod_processing(tod_comm: MPI.Comm, params: Bunch) -> tuple[bool, MPI.Comm, str, dict[str,int], DetectorTOD]:
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

    MPIsize_tod, MPIrank_tod = tod_comm.Get_size(), tod_comm.Get_rank()
    tod_master = MPIrank_tod == 0

    # We now loop over all bands in all experiments, and allocate them to the first ranks of the TOD MPI communicator.
    # These ranks will then become the "band masters" for those bands, handling all communication with CompSep.
    my_experiment_name = None  # All the non-master ranks will have None values, and receive them from master further down.
    my_band_name = None
    my_experiment = None
    my_band = None
    my_num_scans = 0
    TOD_rank = 0
    for experiment in params.experiments:
        if params.experiments[experiment].enabled:
            for band in params.experiments[experiment].bands:
                if params.experiments[experiment].bands[band].enabled:
                    if tod_comm.Get_rank() == TOD_rank:
                        my_experiment_name = experiment
                        my_band_name = band
                        my_experiment = params.experiments[experiment]
                        my_band = params.experiments[experiment].bands[band]
                        my_num_scans = params.experiments[experiment].num_scans
                    TOD_rank += 1
    tot_num_bands = TOD_rank
    if tot_num_bands > MPIsize_tod:
        log.lograise(RuntimeError, f"Total number of experiment bands {tot_num_bands} exceed number of TOD MPI tasks {MPIsize_tod}.", logger)

    if tod_master:
        logger.info(f"TOD: {MPIsize_tod} tasks allocated to TOD processing of {tot_num_bands} bands.")
        log.logassert(MPIsize_tod >= tot_num_bands, f"Number of MPI tasks dedicated to TOD processing ({MPIsize_tod}) must be equal to or larger than the number of bands ({tot_num_bands}).", logger)

    MPIcolor_band = MPIrank_tod%tot_num_bands  # Spread the MPI tasks over the different bands.
    band_comm = tod_comm.Split(MPIcolor_band, key=MPIrank_tod)  # Create communicators for each different band.
    MPIsize_band, MPIrank_band = band_comm.Get_size(), band_comm.Get_rank()  # Get my local rank, and the total size of, the band-communicator I'm on.
    logger.info(f"TOD: Hello from TOD-rank {MPIrank_tod} (on machine {MPI.Get_processor_name()}), dedicated to band {MPIcolor_band}, with local rank {MPIrank_band} (local communicator size: {MPIsize_band}).")
    
    is_band_master = MPIrank_band == 0  # Am I the master of my local band.
    my_experiment_name = band_comm.bcast(my_experiment_name, root=0)  # Surely there is a more elegant way of doing this, but it'll do for now.
    my_band_name = band_comm.bcast(my_band_name, root=0)
    my_experiment = band_comm.bcast(my_experiment, root=0)
    my_band = band_comm.bcast(my_band, root=0)
    my_num_scans = band_comm.bcast(my_num_scans, root=0)

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
    t0 = time.time()
    if my_experiment.is_sim:
        experiment_data, detector_samples = read_TOD_sim_data(my_experiment.data_path, my_band, params, my_scans_start, my_scans_stop)
    else:
        experiment_data, detector_samples = read_Planck_TOD_data(my_experiment.data_path, my_band, params, my_scans_start, my_scans_stop)
    tod_comm.Barrier()
    logger.info(f"TOD: Finished reading all files in {time.time()-t0:.1f}s.")

    return is_band_master, band_comm, my_band_identifier, tod_band_masters_dict, experiment_data, detector_samples



def find_galactic_mask(experiment_data: DetectorTOD, params: Bunch) -> DetectorTOD:
    """Subtracts the sky model from the TOD data.
    Input:
        experiment_data (DetectorTOD): The experiment TOD object.
        params (Bunch): The parameters from the input parameter file.
    Output:
        experiment_data (DetectorTOD): The experiment TOD with the estimated white noise level added to each scan.
    """
    if params.galactic_mask:
        for scan in experiment_data.scans:
            scan_map, theta, phi, psi = scan.data
            scan.galactic_mask_array = np.abs(theta - np.pi/2.0) > 8.0*np.pi/180.0
    return experiment_data



def estimate_white_noise(experiment_data: DetectorTOD, detector_samples: DetectorSamples, det_compsep_map: NDArray, params: Bunch) -> DetectorTOD:
    """Estimate the white noise level in the TOD data, add it to the scans, and return the updated experiment data.
    Input:
        experiment_data (DetectorTOD): The experiment TOD object.
        params (Bunch): The parameters from the input parameter file.
    Output:
        experiment_data (DetectorTOD): The experiment TOD with the estimated white noise level added to each scan.
    """
    for scan, scan_samples in zip(experiment_data.scans, detector_samples.scans):
        raw_TOD = scan.data[0]
        sky_subtracted_tod = raw_TOD - scan_samples.gain_est*get_sky_model_TOD(scan, det_compsep_map)  #TODO: also subtract orbital dipole?
        if params.galactic_mask and np.sum(scan.galactic_mask_array) > 50:  # If we have enough data points to estimate the noise, we use the masked version.
            sigma0 = np.std(sky_subtracted_tod[scan.galactic_mask_array][1:] - sky_subtracted_tod[scan.galactic_mask_array][:-1])/np.sqrt(2)
        else:
            sigma0 = np.std(sky_subtracted_tod[1:] - sky_subtracted_tod[:-1])/np.sqrt(2)
        scan_samples.sigma0 = sigma0/scan_samples.gain_est
    return detector_samples



def sample_noise(band_comm: MPI.Comm, experiment_data: DetectorTOD, detector_samples: DetectorSamples, det_compsep_map: NDArray) -> DetectorTOD:
    for scan, scansamples in zip(experiment_data.scans, detector_samples.scans):
        f_samp = scan.fsamp
        raw_tod, theta, phi, psi = scan.data
        sky_subtracted_TOD = raw_tod - get_sky_model_TOD(scan, det_compsep_map)
        Ntod = raw_tod.shape[0]
        Nfft = Ntod//2 + 1
        freq = rfftfreq(Ntod, d = 1/f_samp)
        fknee = scan.fknee_est
        alpha = scan.alpha_est
        C_1f_inv = np.zeros(Nfft)
        C_1f_inv[1:] = 1.0 / (scan.sigma0**2*(freq[1:]/fknee)**alpha)
        scansamples.n_corr_est = corr_noise_realization_with_gaps(sky_subtracted_TOD,
                                                                  scan.galactic_mask_array,
                                                                  scansamples.gain_est*scansamples.sigma0,
                                                                  C_1f_inv).astype(np.float32)

    return detector_samples



def sample_noise_PS(band_comm: MPI.Comm, experiment_data: DetectorTOD, detector_samples: DetectorSamples, params: Bunch):
    logger = logging.getLogger(__name__)
    alphas = []
    fknees = []
    for scan, scansamples in zip(experiment_data.scans, detector_samples.scans):
        fknee, alpha = sample_noise_PS_params(scansamples.n_corr_est, scansamples.sigma0, scan.fsamp, scansamples.alpha_est, freq_max=2.0, n_grid=150, n_burnin=4)
        scansamples.fknee_est = fknee
        scansamples.alpha_est = alpha
        alphas.append(alpha)
        fknees.append(fknee)
    alphas = band_comm.gather(alphas, root=0)
    fknees = band_comm.gather(fknees, root=0)
    if band_comm.Get_rank() == 0:
        alphas = np.concatenate(alphas)
        fknees = np.concatenate(fknees)
        logger.info(f"{MPI.COMM_WORLD.Get_rank()} fknees {np.min(fknees):.4f} {np.percentile(fknees, 1):.4f} {np.mean(fknees):.4f} {np.percentile(fknees, 99):.4f} {np.max(fknees):.4f}")
        logger.info(f"{MPI.COMM_WORLD.Get_rank()} alphas {np.min(alphas):.4f} {np.percentile(alphas, 1):.4f} {np.mean(alphas):.4f} {np.percentile(alphas, 99):.4f} {np.max(alphas):.4f}")
    return detector_samples


def fill_gaps(TOD, mask, noise_sigma0, window_size=20):
    """ In-place fills the gaps in the provided TOD array with linearly interpolated
        values plus a white noise term.
    Args:
        TOD (np.ndarray): The data array with gaps. This array will changed in-place!
        mask (np.ndarray): A boolean array of same shape as TOD where False indicates a gap to be filled.
        noise_std (float): The standard deviation of the white noise to add.
        window_size (int): The number of points to average on each side of a gap.
    """
    # Find the start and end of each gap by looking at when the masks changes value.
    gap_starts = np.where(np.diff(mask.astype(int)) == -1)[0] + 1
    gap_ends = np.where(np.diff(mask.astype(int)) == 1)[0] + 1

    if not mask[0]:  # Special case: If first sample is masked.
        gap_starts = np.insert(gap_starts, 0, 0)
    if not mask[-1]:  # If last sample is masked.
        gap_ends = np.append(gap_ends, len(mask))
        
    for start, end in zip(gap_starts, gap_ends):
        gap_len = end - start

        # Case 1: Gap is at the beginning of the data: Use only right anchor.
        if start == 0:
            right_window = TOD[end:end + window_size]
            right_mask_window = mask[end:end + window_size]
            anchor = np.mean(right_window[right_mask_window])
            interp_values = np.full(gap_len, anchor)
        
        # Case 2: Gap is at the end of the data: Use only left anchor.
        elif end == len(mask):
            left_window = TOD[max(0, start - window_size):start]
            left_mask_window = mask[max(0, start - window_size):start]
            anchor = np.mean(left_window[left_mask_window])
            interp_values = np.full(gap_len, anchor)

        # Case 3: Gap is not at either end: Linearly interpolate between anchors.
        else:
            left_window = TOD[max(0, start - window_size):start]
            left_mask_window = mask[max(0, start - window_size):start]
            left_anchor = np.mean(left_window[left_mask_window])

            right_window = TOD[end:end + window_size]
            right_mask_window = mask[end:end + window_size]
            right_anchor = np.mean(right_window[right_mask_window])
            
            interp_values = np.linspace(left_anchor, right_anchor, gap_len)
        
        # Add a white noise to the interpolated (or constant) values.
        TOD[start:end] = interp_values + np.random.normal(0, noise_sigma0, gap_len)



def sample_absolute_gain(TOD_comm: MPI.Comm, experiment_data: DetectorTOD, detector_samples, det_compsep_map: NDArray):
    """ Function for drawing a realization of the absolute gain term, g0, which is constant across both all bands and all scans.
        Args:
            TOD_comm (MPI.Comm): The full TOD communicator, since we will calculate a single g0 value across all bands.
            experiment_data (DetectorTOD): The object holding all the scan data. Will be changed in-place to update g0.
            params (Bunch): Parameters from parameter file.
    """
    logger = logging.getLogger(__name__)

    sum_s_T_N_inv_d = 0  # Accumulators for the numerator and denominator of eqn 16.
    sum_s_T_N_inv_s = 0
    for scan, scan_samples in zip(experiment_data.scans, detector_samples.scans):
        # --- Setup ---
        tod, theta, phi, _ = scan.data

        s_orb = calculate_s_orb(scan, experiment_data)
        sky_model_TOD = get_sky_model_TOD(scan, det_compsep_map)

        tod_residual = tod - scan_samples.gain_est*(sky_model_TOD + s_orb)  # Subtracting sky signal.
        tod_residual += detector_samples.g0_est*s_orb  # Now we can add back in the orbital dipole.

        sigma0 = scan_samples.gain_est*scan_samples.sigma0  # scan.sigma0 is in tempearture units.
        Ntod = tod.shape[0]
        Nrfft = Ntod//2+1
        freqs = rfftfreq(Ntod, 1.0/scan.fsamp)
        inv_power_spectrum = np.zeros(Nrfft)
        inv_power_spectrum[1:] = 1.0/(sigma0**2*(1 + (freqs[1:]/scan_samples.fknee_est)**scan_samples.alpha_est))

        ### Solving Equation 16 from BP7 ###
        s_fft = rfft(s_orb)
        d_fft = rfft(tod_residual)
        N_inv_s_fft = s_fft * inv_power_spectrum
        N_inv_d_fft = d_fft * inv_power_spectrum
        N_inv_s = irfft(N_inv_s_fft, n=Ntod)
        N_inv_d = irfft(N_inv_d_fft, n=Ntod)
        # We now exclude the time-samples hitting the masked area. We don't want to do this before now, because it would mess up the FFT stuff.

        sum_s_T_N_inv_d += np.dot(s_orb[scan.galactic_mask_array], N_inv_d[scan.galactic_mask_array])  # Add to the numerator and denominator.
        sum_s_T_N_inv_s += np.dot(s_orb[scan.galactic_mask_array], N_inv_s[scan.galactic_mask_array])

    # The g0 term is fully global, so we reduce across both all scans and all bands:
    sum_s_T_N_inv_d = TOD_comm.reduce(sum_s_T_N_inv_d, op=MPI.SUM, root=0)
    sum_s_T_N_inv_s = TOD_comm.reduce(sum_s_T_N_inv_s, op=MPI.SUM, root=0)
    g_sampled = 0.0
    if TOD_comm.Get_rank() == 0:  # Rank 0 draws a sample of g0 from eq (16) from BP6, and bcasts it to the other ranks.
        eta = np.random.randn()
        g_mean = sum_s_T_N_inv_d / sum_s_T_N_inv_s
        g_std = 1.0 / np.sqrt(sum_s_T_N_inv_s)

        g_sampled = g_mean + eta * g_std
        logger.info(f"Previous g0:   {detector_samples.g0_est*1e9:10.5f}.")
        logger.info(f"New g0 mean:   {g_mean*1e9:10.5f}.")
        logger.info(f"New g0 std:    {g_std*1e9:10.5f}.")
        logger.info(f"New g0 sample: {g_sampled*1e9:10.5f}.")
    g_sampled = TOD_comm.bcast(g_sampled, root=0)

    detector_samples.g0_est = g_sampled

    return detector_samples


def sample_relative_gain(TOD_comm: MPI.Comm, band_comm: MPI.Comm, experiment_data: DetectorTOD, detector_samples, det_compsep_map: NDArray):
    """Samples the detector-dependent relative gain (Delta g_i).
    This function implements the logic from Sec. 3.4 of the BP7.
    It uses a two-stage MPI communication:
    1. A reduction over the `band_comm` to accumulate sums for a single detector
       that is distributed across multiple ranks.
    2. A gather over the global `TOD_comm` to collect the final sums from each
       unique detector on the root rank for solving the global system.

    Args:
        TOD_comm (MPI.Comm): The global MPI communicator for all bands.
        band_comm (MPI.Comm): The communicator for ranks sharing the same detector.
        experiment_data (DetectorTOD): The object holding scan data for a single
                                       detector on the calling rank.
        params (Bunch): Parameters from the parameter file.
    """
    logger = logging.getLogger(__name__)
    global_rank = TOD_comm.Get_rank()
    band_rank = band_comm.Get_rank()

    #### 1. Local Calculation (on each rank) ###
    # Each rank calculates the sum of terms for its local subset of scans.
    local_s_T_N_inv_s = 0.0
    local_r_T_N_inv_s = 0.0

    for scan, scan_samples in zip(experiment_data.scans, detector_samples.scans):
        # Define the residual for this sampling step, as per Eq. (17)
        s_orb = calculate_s_orb(scan, experiment_data)
        sky_model_TOD = get_sky_model_TOD(scan, det_compsep_map)
        raw_tod, theta, phi, psi = scan.data
        s_tot = sky_model_TOD + s_orb

        residual_tod = raw_tod - (detector_samples.g0_est + scan_samples.time_dep_rel_gain_est)*s_tot

        # Setup FFT-based calculation for N^-1 operations
        Ntod = residual_tod.shape[0]
        Nrfft = Ntod // 2 + 1
        sigma0 = scan_samples.gain_est*scan_samples.sigma0  # scan.sigma0 is in tempearture units.
        freqs = rfftfreq(Ntod, 1.0 / scan.fsamp)
        inv_power_spectrum = np.zeros(Nrfft)
        inv_power_spectrum[1:] = 1.0 / (sigma0**2 * (1 + (freqs[1:] / scan_samples.fknee_est)**scan_samples.alpha_est))

        s_fft = rfft(s_tot)
        N_inv_s_fft = s_fft * inv_power_spectrum
        N_inv_s = irfft(N_inv_s_fft, n=Ntod)
        
        mask = scan.galactic_mask_array
        s_T_N_inv_s_scan = np.dot(s_tot[mask], N_inv_s[mask])
        r_T_N_inv_s_scan = np.dot(residual_tod[mask], N_inv_s[mask])

        # Add the contribution from this scan to the local sum
        local_s_T_N_inv_s += s_T_N_inv_s_scan
        local_r_T_N_inv_s += r_T_N_inv_s_scan

    ### 2. Intra-Detector Reduction ###
    # Sum the local values across all ranks that share the same detector using band_comm.
    # After this, every rank in the band_comm will have the total sum for their detector.
    total_s_for_detector = band_comm.allreduce(local_s_T_N_inv_s, op=MPI.SUM)
    total_r_for_detector = band_comm.allreduce(local_r_T_N_inv_s, op=MPI.SUM)

    ### 3. Gather all unique detector sums on the global Rank 0 ###
    # To avoid redundant communication, only the root of each detector group (band_rank 0)
    # sends its data to the global root. Other ranks send None.
    if band_rank == 0:
        data_to_send = (experiment_data.detector_id, total_s_for_detector, total_r_for_detector)
    else:
        data_to_send = None
    
    gathered_data = TOD_comm.gather(data_to_send, root=0)

    ### 4. Solve Global System on Rank 0 and Broadcast Result ###
    result_to_bcast = None
    if global_rank == 0:
        # Filter out the None entries from non-sending ranks
        valid_data = [data for data in gathered_data if data is not None]
        
        # Aggregate results into dictionaries.
        s_map = {}
        r_map = {}
        for det_id, s_val, r_val in valid_data:
            s_map[det_id] = s_val
            r_map[det_id] = r_val
        
        detector_ids = sorted(s_map.keys())
        n_detectors = len(detector_ids)

        if n_detectors > 1:
            s_T_N_inv_s_list = [s_map[did] for did in detector_ids]
            r_T_N_inv_s_list = [r_map[did] for did in detector_ids]

            A = np.zeros((n_detectors + 1, n_detectors + 1))
            b = np.zeros(n_detectors + 1)
            diagonal = np.array(s_T_N_inv_s_list)
            A[:n_detectors, :n_detectors] = np.diag(diagonal)
            A[:n_detectors, n_detectors] = 0.5
            A[n_detectors, :n_detectors] = 1.0
            eta = np.random.randn(n_detectors)
            fluctuation_term = np.sqrt(diagonal) * eta
            
            b[:n_detectors] = np.array(r_T_N_inv_s_list) + fluctuation_term
            
            try:
                solution = np.linalg.solve(A, b)
                delta_g_samples = solution[:n_detectors]
                result_to_bcast = {'dids': detector_ids, 'dgs': delta_g_samples}
                logger.info(f"Solved global relative gains for {n_detectors} detectors.")
            except np.linalg.LinAlgError:
                logger.error("Failed to solve global linear system for relative gain. Skipping update.")
                result_to_bcast = None
        else:
             logger.warning(f"Relative gain sampling requires > 1 detector, but only {n_detectors} found across all ranks. Skipping.")

    # Broadcast the result to all ranks in the global communicator
    result_to_bcast = TOD_comm.bcast(result_to_bcast, root=0)

    ### 5. Update Gain on All Ranks ###
    if result_to_bcast:
        my_did = experiment_data.detector_id
        try:
            my_dg_index = result_to_bcast['dids'].index(my_did)
            my_delta_g = result_to_bcast['dgs'][my_dg_index]
            for scan_samples in detector_samples.scans:
                scan_samples.rel_gain_est = my_delta_g
            if band_comm.Get_rank() == 0:
                logging.info(f"Relative gain for detector {experiment_data.nu}GHz: {detector_samples.g0_est*1e9+my_delta_g*1e9:8.3f} ({detector_samples.g0_est*1e9:8.3f} + {my_delta_g*1e9:8.3f})")
        except ValueError:
            logger.error(f"Rank {global_rank} with detector {my_did} not found in solved gain list.")
    else:
        logger.warning(f"No valid relative gain solution was broadcast. Not updating gains on rank {global_rank}.")

    return detector_samples



def process_tod(TOD_comm: MPI.Comm, band_comm: MPI.Comm, experiment_data: DetectorTOD, detector_samples,
                compsep_output: NDArray, params: Bunch, chain, iter) -> DetectorMap:
    """ Performs a single TOD iteration.

    Input:
        TOD_comm (MPI.Comm): The full TOD communicator between all TOD-processing ranks.
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
    # Steps:
    # 1. Find galactic mask (currently hard-coded).
    # 2. Create scan.sky_subtracted_tod by subtracting the sky model we get from CompSep (Skipped on iter==1 because we don't have any compsep data). Note that this relies on the gain from the previous iteration.
    # 3. Sample the gain from the sky-subtracted TOD (Skipped on iter==1 because we don't have a reliable sky-subtracted TOD).
    # 4. Estimate White noise from the sky-subtracted TOD.
    # 5. Sample correlated noise (skipped on iter==1).
    # 6. Sample correlated noise PS parameters (skipped on iter==1).
    # 7. Mapmaking on (TOD - corr_noise_TOD - orb_dipole_TOD.
    # In other words, on iteration 1 we do just do 1. White noise estimation -> 2. Mapmaking.

    logger = logging.getLogger(__name__)
    if iter == 1:
        for scan, scan_samples in zip(experiment_data.scans, detector_samples.scans):
            scan_samples.n_corr_est = np.zeros_like(scan.data[0], dtype=np.float32)
            scan_samples.alpha_est = params.noise_alpha  # These should be sampled params!
            scan_samples.fknee_est = params.noise_fknee

    experiment_data = find_galactic_mask(experiment_data, params)

    t0 = time.time()
    if iter >= params.sample_gain_from_iter_num:
        detector_samples = sample_absolute_gain(TOD_comm, experiment_data, detector_samples, compsep_output)
        if TOD_comm.Get_rank() == 0:
            logger.info(f"Rank {MPI.COMM_WORLD.Get_rank()} chain {chain} iter{iter}: Finished absolute gain estimation in {time.time()-t0:.1f}s."); t0 = time.time()
        detector_samples = sample_relative_gain(TOD_comm, band_comm, experiment_data, detector_samples, compsep_output)
        if TOD_comm.Get_rank() == 0:
            logger.info(f"Rank {MPI.COMM_WORLD.Get_rank()} chain {chain} iter{iter}: Finished relative gain estimation in {time.time()-t0:.1f}s."); t0 = time.time()
                
        # Update total gain from all new components and re-subtract sky model
        for scan_samples in detector_samples.scans:
            scan_samples.gain_est = detector_samples.g0_est + scan_samples.rel_gain_est + scan_samples.time_dep_rel_gain_est

    detector_samples = estimate_white_noise(experiment_data, detector_samples, compsep_output, params)
    if TOD_comm.Get_rank() == 0:
        logger.info(f"Rank {MPI.COMM_WORLD.Get_rank()} chain {chain} iter{iter}: Finished white noise estimation in {time.time()-t0:.1f}s."); t0 = time.time()

    if iter >= params.sample_corr_noise_from_iter_num:
        detector_samples = sample_noise(band_comm, experiment_data, detector_samples, compsep_output)
        if TOD_comm.Get_rank() == 0:
            logger.info(f"Rank {MPI.COMM_WORLD.Get_rank()} chain {chain} iter{iter}: Finished corr noise realizations in {time.time()-t0:.1f}s."); t0 = time.time()
        detector_samples = sample_noise_PS(band_comm, experiment_data, detector_samples, params)
        if TOD_comm.Get_rank() == 0:
            logger.info(f"Rank {MPI.COMM_WORLD.Get_rank()} chain {chain} iter{iter}: Finished corr noise PS parameter sampling in {time.time()-t0:.1f}s."); t0 = time.time()
    todproc_output = tod2map(band_comm, experiment_data, compsep_output, detector_samples, params)
    if TOD_comm.Get_rank() == 0:
        logger.info(f"Rank {MPI.COMM_WORLD.Get_rank()} chain {chain} iter{iter}: Finished mapmaking in {time.time()-t0:.1f}s."); t0 = time.time()

    return todproc_output, detector_samples