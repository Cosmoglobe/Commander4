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
import fits

from src.python.data_models.detector_map import DetectorMap
from src.python.data_models.detector_TOD import DetectorTOD
from src.python.data_models.scan_TOD import ScanTOD
from src.python.data_models.detector_samples import DetectorSamples
from src.python.data_models.scan_samples import ScanSamples
from src.python.utils.mapmaker import single_det_map_accumulator
from src.python.utils.mapmaker import single_det_map_accumulator_IQU
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
    logger = logging.getLogger(__name__)
    # detmap_rawobs, detmap_signal, detmap_orbdipole, detmap_skysub, detmap_corr_noise, detmap_inv_var = single_det_map_accumulator_IQU(det_static, det_cs_map, params)
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
        # Pre-IQU stuff:
        #map_orbdipole[map_orbdipole != 0] /= map_hits[map_orbdipole != 0]
        #map_rawobs[map_rawobs != 0] /= map_inv_var[map_rawobs != 0]
        #map_signal[map_signal != 0] /= map_inv_var[map_signal != 0]
        #map_skysub[map_skysub != 0] /= map_inv_var[map_skysub != 0]
        #map_corr_noise[map_corr_noise != 0] /= map_hits[map_corr_noise != 0]
        #map_rms = np.zeros_like(map_inv_var) + np.inf
        #map_rms[map_inv_var != 0] = 1.0/np.sqrt(map_inv_var[map_inv_var != 0])
        #detmap = DetectorMap(map_signal, map_corr_noise, map_rms, det_static.nu, det_static.fwhm, det_static.nside)
        #detmap.g0 = detector_samples.g0_est
        #detmap.gain = detector_samples.scans[0].rel_gain_est + detector_samples.g0_est
        #detmap.skysub_map = map_skysub
        #detmap.rawobs_map = map_rawobs
        #detmap.orbdipole_map = map_orbdipole
        #return detmap

        if map_signal.ndim == 1:
            # Intensity mapmaking
            map_orbdipole[map_orbdipole != 0] /= map_inv_var[map_orbdipole != 0]
            map_rawobs[map_rawobs != 0] /= map_inv_var[map_rawobs != 0]
            map_signal[map_signal != 0] /= map_inv_var[map_signal != 0]
            map_skysub[map_skysub != 0] /= map_inv_var[map_skysub != 0]
            map_corr_noise[map_corr_noise != 0] /= map_inv_var[map_corr_noise != 0]
            map_rms = np.zeros_like(map_inv_var) + np.inf
            map_rms[map_inv_var != 0] = 1.0/np.sqrt(map_inv_var[map_inv_var != 0])
        elif map_signal.ndim == 2:
            if (map_signal.shape[0] == 3) and (map_inv_var.shape[0] == 6):
                # Standard IQU mapmaking
                A = np.zeros((map_inv_var.shape[1], 3, 3), dtype=map_inv_var.dtype)
                
                A[:,0,0] = map_inv_var[0]

                A[:,0,1] = map_inv_var[1]
                A[:,1,0] = map_inv_var[1]

                A[:,0,2] = map_inv_var[2]
                A[:,2,0] = map_inv_var[2]

                A[:,1,1] = map_inv_var[3]

                A[:,1,2] = map_inv_var[4]
                A[:,2,1] = map_inv_var[4]

                A[:,2,2] = map_inv_var[5]

                for m in [map_orbdipole, map_rawobs, map_signal, map_skysub, map_corr_noise]:
                    m[:] = np.linalg.solve(A, m.T).T

                map_rms = np.zeros_like(map_rawobs)
                A_inv = np.linalg.inv(A)
                map_rms[0] = A_inv[:,0,0]**0.5
                map_rms[1] = A_inv[:,1,1]**0.5
                map_rms[2] = A_inv[:,2,2]**0.5

            else:
                # Improperly formatted IQU map, and/or right format not yet implemented.
                log.lograise(RuntimeError, "Maps did not have expected shape: "
                             f"({map_signal.shape[0]} != 3 or {map_inv_var.shape[0]} != 6", logger)
        else:
            log.lograise(RuntimeError, "Maps did not have ndim 1 or 2 expected for total intensity"
                         f"and polarization, respectively. (ndim = {map_signal.ndim})", logger)

        logger.info(f"Temporarily setting everything to intensity only")
        detmap = DetectorMap(map_signal[0], map_corr_noise[0], map_rms[0], det_static.nu, det_static.fwhm, det_static.nside)
        detmap.g0 = det_static.scans[0].g0_est
        detmap.skysub_map = map_skysub[0]
        detmap.rawobs_map = map_rawobs[0]
        detmap.orbdipole_map = map_orbdipole[0]
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
        fknee = scansamples.fknee_est
        alpha = scansamples.alpha_est
        C_1f_inv = np.zeros(Nfft)
        C_1f_inv[1:] = 1.0 / (scansamples.sigma0**2*(freq[1:]/fknee)**alpha)
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



def sample_temporal_gain_variations(band_comm: MPI.Comm, experiment_data: DetectorTOD, detector_samples, det_compsep_map: NDArray, chain: int, iter: int):
    """
    Samples the time-dependent relative gain variations (delta g_qi).
    This function implements the logic from Sec. 3.5 of the BP7 paper,
    using a Wiener filter to smooth the gain solution over time (PIDs).
    It solves a global system for all scans of a given detector, which are
    distributed across the ranks of the band_comm.

    Args:
        band_comm (MPI.Comm): The communicator for ranks sharing the same detector.
        experiment_data (DetectorTOD): The object holding scan data.
        params (Bunch): Parameters from the parameter file.
    """
    logger = logging.getLogger(__name__)
    band_rank = band_comm.Get_rank()
    band_size = band_comm.Get_size()

    # Local calculations on each rank
    A_qq_local = []
    b_q_local = []

    for scan, scan_samples in zip(experiment_data.scans, detector_samples.scans):
        # Per Eq. (26), the residual is d - (g0 + Delta_g)*s
        s_orb = calculate_s_orb(scan, experiment_data)
        sky_model_TOD = get_sky_model_TOD(scan, det_compsep_map)
        s_tot = sky_model_TOD + s_orb
        residual_tod = scan.data[0] - (detector_samples.g0_est + scan_samples.rel_gain_est)*s_tot

        # FFT-based N^-1 operation setup
        Ntod = residual_tod.shape[0]
        Nrfft = Ntod // 2 + 1
        freqs = rfftfreq(Ntod, 1.0 / scan.fsamp)
        sigma0 = scan_samples.gain_est*scan_samples.sigma0
        inv_power_spectrum = np.zeros(Nrfft)
        inv_power_spectrum[1:] = 1.0 / (sigma0**2 * (1 + (freqs[1:] / scan_samples.fknee_est)**scan_samples.alpha_est))

        # Calculate N^-1 * s_tot and N^-1 * residual_tod
        N_inv_s = irfft(rfft(s_tot) * inv_power_spectrum, n=Ntod)
        N_inv_r = irfft(rfft(residual_tod) * inv_power_spectrum, n=Ntod)

        mask = scan.galactic_mask_array
        
        # Calculate elements for the linear system
        A_qq = np.dot(s_tot[mask], N_inv_s[mask])
        b_q = np.dot(s_tot[mask], N_inv_r[mask])
        
        A_qq_local.append(A_qq)
        b_q_local.append(b_q)

    A_qq_local = np.array(A_qq_local, dtype=np.float64)
    b_q_local = np.array(b_q_local, dtype=np.float64)

    # Gather all data on the root rank of the band communicator
    scan_counts = band_comm.gather(len(A_qq_local), root=0)
    all_A_qq = band_comm.gather(A_qq_local, root=0)
    all_b_q = band_comm.gather(b_q_local, root=0)
    
    delta_g_sample = None
    if band_rank == 0:
        # Concatenate gathered arrays into single flat arrays
        A_diag = np.concatenate(all_A_qq)
        b = np.concatenate(all_b_q)
        
        n_scans_total = len(A_diag)
        if n_scans_total > 1:
            # Define Prior (Wiener Filter) based on Eq. (31)
            alpha_gain = -2.5
            fknee_gain = 1.0  # Hour (which equals 1 scan)
            # sigma0_sq_gain_V2_per_K2 = 3e-4  # V^2/K^2
            # sigma0_sq_gain = 1e12 * sigma0_sq_gain_V2_per_K2  # V^2/K^2 -> V^2/uK^2
            sigma0_sq_gain = 1e-25  # Randomly set value since I can't figure out the one from the paper.

            gain_freqs = rfftfreq(n_scans_total, d=1.0)
            prior_ps = np.zeros_like(gain_freqs)
            prior_ps[1:] = sigma0_sq_gain * (np.abs(gain_freqs[1:]) / fknee_gain)**alpha_gain
            
            prior_ps_inv = np.zeros_like(gain_freqs)
            prior_ps_inv[prior_ps > 0] = 1.0 / prior_ps[prior_ps > 0]
            prior_ps_inv_sqrt = np.sqrt(prior_ps_inv)

            # Define Linear Operator for Conjugate Gradient Solver
            def matvec(v):
                g_inv_v = irfft(rfft(v) * prior_ps_inv, n=n_scans_total).real
                diag_v = A_diag * v
                return g_inv_v + diag_v

            # Construct RHS of the sampling equation (Eq. 30)
            eta1 = np.random.randn(n_scans_total)
            fluctuation1 = np.sqrt(np.maximum(A_diag, 0)) * eta1

            eta2 = np.random.randn(n_scans_total)
            fluctuation2 = irfft(rfft(eta2) * prior_ps_inv_sqrt, n=n_scans_total).real

            logger.info(f"DATA TERM (A_diag) MEAN: {np.mean(A_diag)}")
            logger.info(f"PRIOR TERM (G^-1) MEAN: {np.mean(prior_ps_inv)}")
            logger.info(f"RHS = {b} + {fluctuation1} + {fluctuation2}")
            logger.info(f"RHS (means) = {np.nanmean(b)} + {np.nanmean(fluctuation1)} + {np.nanmean(fluctuation2)}")
            RHS = b + fluctuation1 + fluctuation2

            ### Simpler sanity check solution  ##
            epsilon = 1e-12             
            # The mean value is simply b / A
            g_mean = b / (A_diag + epsilon)
            # The standard deviation is 1 / sqrt(A)
            g_std = 1.0 / np.sqrt(np.maximum(A_diag, 0) + epsilon)
            logger.info(f"Sanity check solution A, b = {A_diag} {b}")
            logger.info(f"Sanity check solution: {g_mean} {g_std}")
            logger.info(f"Sanity check solution(means): {np.mean(g_mean)} {np.mean(g_std)}")

            from pixell import utils
            CG_solver = utils.CG(matvec, RHS, x0=g_mean)
            for i in range(1000):
                CG_solver.step()
                if CG_solver.err < 1e-8:
                    break
            logging.info(f"CG solver for gain fluctuations (nu={experiment_data.nu})"
                        f"finished after {i} iterations (residual = {CG_solver.err})")
            delta_g_sample = CG_solver.x
            logger.info(f"delta_g_sample mean = {np.mean(delta_g_sample)}")
            delta_g_sample -= np.mean(delta_g_sample)
            logger.info(f"delta_g: {delta_g_sample}")
            logging.info(f"Band {experiment_data.nu}GHz time-dependent gain: min={np.min(delta_g_sample)*1e9:14.4f} mean={np.mean(delta_g_sample)*1e9:14.4f} std={np.std(delta_g_sample)*1e9:14.4f} max={np.max(delta_g_sample)*1e9:14.4f}")

            if False: #debug stuff
                def matvec_noprior(v):
                    diag_v = A_diag * v
                    return diag_v
                CG_solver = utils.CG(matvec_noprior, RHS, x0=g_mean)
                for i in range(1, 2000):
                    CG_solver.step()
                    if CG_solver.err < 1e-8:
                        logging.info("Warning: CG for time-relative gain did not converge.")
                        break
                logging.info(f"CG solver (noprior) for gain fluctuations (nu={experiment_data.nu})"
                            f"finished after {i} iterations (residual = {CG_solver.err})")

                delta_g_sample_noprior = CG_solver.x
                logger.info(f"delta_g_sample mean (noprior) = {np.mean(delta_g_sample_noprior)}")
                logger.info(f"delta_g (no_prior): {delta_g_sample_noprior}")

                import matplotlib.pyplot as plt
                plt.figure(figsize=(16,9))
                other_gain = detector_samples.g0_est + detector_samples.scans[0].rel_gain_est
                plt.plot(1e9*(other_gain + delta_g_sample))
                plt.ylim(0, 1.5*1e9*other_gain)
                plt.xlabel("PID")
                plt.ylabel("Gain [mV/K]")
                plt.savefig(f"chain{chain}_iter{iter}_{experiment_data.nu}.png")
                plt.close()
        else:
            delta_g_sample = np.zeros(n_scans_total)
            
    # Scatter the results back to all ranks
    if band_size > 1:
        displacements = None
        if band_rank == 0:
            scan_counts = np.array(scan_counts, dtype=int)
            displacements = np.insert(np.cumsum(scan_counts), 0, 0)[:-1]
        
        delta_g_local = np.empty(len(A_qq_local), dtype=np.float64)
        band_comm.Scatterv([delta_g_sample, scan_counts, displacements, MPI.DOUBLE], delta_g_local, root=0)
    else:
        delta_g_local = delta_g_sample if delta_g_sample is not None else np.array([])

    # Update local scan objects
    if delta_g_local.size == len(experiment_data.scans):
        gain_per_PID = np.zeros_like(delta_g_local)
        for i, scan_samples in enumerate(detector_samples.scans):
            scan_samples.time_dep_rel_gain_est = delta_g_local[i]
            gain_per_PID[i] = detector_samples.g0_est + scan_samples.rel_gain_est + delta_g_local[i]
        logging.info(f"Rank {band_rank} {experiment_data.nu} time-dependent gain: min={np.min(delta_g_local)*1e9:14.4f} mean={np.mean(delta_g_local)*1e9:14.4f} std={np.std(delta_g_local)*1e9:14.4f} max={np.max(delta_g_local)*1e9:14.4f}")
    else:
        logger.warning(f"Rank {band_rank} received mismatched number of gain samples. Expected {len(experiment_data.scans)}, got {delta_g_local.size}.")

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
        detector_samples = sample_temporal_gain_variations(band_comm, experiment_data, detector_samples, compsep_output, chain, iter)
        if TOD_comm.Get_rank() == 0:
            logger.info(f"Rank {MPI.COMM_WORLD.Get_rank()} chain {chain} iter{iter}: Finished time-dependent gain estimation in {time.time()-t0:.1f}s."); t0 = time.time()            
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
