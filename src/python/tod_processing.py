import numpy as np
from mpi4py import MPI
import h5py
import healpy as hp
import math
import logging
from pixell.bunch import Bunch
from output import log
from scipy.fft import rfft, irfft, rfftfreq
import time
from numpy.typing import NDArray
import pysm3.units as pysm3_u

from src.python.data_models.detector_map import DetectorMap
from src.python.data_models.detector_TOD import DetectorTOD
from src.python.data_models.scan_TOD import ScanTOD
from src.python.utils.mapmaker import single_det_map_accumulator
from src.python.noise_sampling import corr_noise_realization_with_gaps, sample_noise_PS_params

nthreads=1

def get_empty_compsep_output(staticData: DetectorTOD) -> NDArray[np.float64]:
    "Creates a dummy compsep output for a single band"
    return np.zeros(12*staticData.nside**2, dtype=np.float64)


def tod2map(band_comm: MPI.Comm, det_static: DetectorTOD, det_cs_map: NDArray, params: Bunch) -> DetectorMap:
    detmap_rawobs, detmap_signal, detmap_orbdipole, detmap_skysub, detmap_corr_noise, detmap_inv_var = single_det_map_accumulator(det_static, det_cs_map, params)
    map_signal = np.zeros_like(detmap_signal)
    map_orbdipole = np.zeros_like(detmap_orbdipole)
    map_rawobs = np.zeros_like(detmap_rawobs)
    map_skysub = np.zeros_like(detmap_skysub)
    map_corr_noise = np.zeros_like(detmap_corr_noise)
    map_inv_var = np.zeros_like(detmap_inv_var)
    if band_comm.Get_rank() == 0:
        band_comm.Reduce(detmap_signal, map_signal, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_skysub, map_skysub, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_orbdipole, map_orbdipole, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_rawobs, map_rawobs, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_corr_noise, map_corr_noise, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_inv_var, map_inv_var, op=MPI.SUM, root=0)
    else:
        band_comm.Reduce(detmap_signal, None, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_skysub, None, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_orbdipole, None, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_rawobs, None, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_corr_noise, None, op=MPI.SUM, root=0)
        band_comm.Reduce(detmap_inv_var, None, op=MPI.SUM, root=0)

    if band_comm.Get_rank() == 0:
        map_orbdipole[map_orbdipole != 0] /= map_inv_var[map_orbdipole != 0]
        map_rawobs[map_rawobs != 0] /= map_inv_var[map_rawobs != 0]
        map_signal[map_signal != 0] /= map_inv_var[map_signal != 0]
        map_skysub[map_skysub != 0] /= map_inv_var[map_skysub != 0]
        map_corr_noise[map_corr_noise != 0] /= map_inv_var[map_corr_noise != 0]
        map_rms = np.zeros_like(map_inv_var) + np.inf
        map_rms[map_inv_var != 0] = 1.0/np.sqrt(map_inv_var[map_inv_var != 0])
        detmap = DetectorMap(map_signal, map_corr_noise, map_rms, det_static.nu, det_static.fwhm, det_static.nside)
        detmap.g0 = det_static.scans[0].g0_est
        detmap.skysub_map = map_skysub
        detmap.rawobs_map = map_rawobs
        detmap.orbdipole_map = map_orbdipole
        return detmap


def read_TOD_sim_data(h5_filename: str, band: int, scan_idx_start: int, scan_idx_stop: int, nside: int, fwhm: float) -> DetectorTOD:
    logger = logging.getLogger(__name__)
    with h5py.File(h5_filename) as f:
        band_formatted = f"{band:04d}"
        scanlist = []
        for iscan in range(scan_idx_start, scan_idx_stop):
            try:
                tod = f[f"{iscan+1:06}/{band_formatted}/tod"][()]
                theta = f[f"{iscan+1:06}/{band_formatted}/theta"][()]
                phi = f[f"{iscan+1:06}/{band_formatted}/phi"][()]
                psi = f[f"{iscan+1:06}/{band_formatted}/psi"][()]
                Ntod = tod.size
                # We assume the orbital velocity to be constant over the duration of a scan, so we read the half-way point.
                # This orbital velociy is relative to the Sun, but is in **Galactic coordinates**, to be easily compatible with other quantities.
                orb_dir_vec = f[f"{iscan+1:06}/{band_formatted}/orbital_dir_vec"][Ntod//2]
            except KeyError:
                logger.exception(f"{iscan}\n{band_formatted}\n{list(f)}")
                raise KeyError
            scanlist.append(ScanTOD(tod, theta, phi, psi, 0., iscan))
            scanlist[-1].orb_dir_vec = orb_dir_vec
        det = DetectorTOD(scanlist, float(band), fwhm, nside)
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
    experiment_data = read_TOD_sim_data(my_experiment.data_path, my_band.freq, my_scans_start, my_scans_stop, my_band.nside, my_band.fwhm)

    return is_band_master, band_comm, my_band_identifier, tod_band_masters_dict, experiment_data



def subtract_sky_model(experiment_data: DetectorTOD, det_compsep_map: NDArray, params: Bunch) -> DetectorTOD:
    """Subtracts the sky model from the TOD data.
    Input:
        experiment_data (DetectorTOD): The experiment TOD object.
        det_compsep_map (np.array): The current estimate of the sky model as seen by the band belonging to the current process.
        params (Bunch): The parameters from the input parameter file.
    Output:
        experiment_data (DetectorTOD): The experiment TOD with the estimated white noise level added to each scan.
    """
    nside = experiment_data.nside
    for scan in experiment_data.scans:
        scan_map, theta, phi, psi = scan.data
        ntod = scan_map.shape[0]
        pix = hp.ang2pix(nside, theta, phi)
        scan.sky_subtracted_tod = scan_map - scan.g0_est*det_compsep_map[pix]
        # if params.galactic_mask:
        #     scan.galactic_mask_array = np.abs(theta - np.pi/2.0) > 5.0*np.pi/180.0
    return experiment_data


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
        scan.sigma0 = sigma0/scan.g0_est
    return experiment_data



def sample_noise(band_comm: MPI.Comm, experiment_data: DetectorTOD, params: Bunch) -> DetectorTOD:
    nside = experiment_data.nside
    for scan in experiment_data.scans:
        f_samp = params.samp_freq
        scan_map, theta, phi, psi = scan.data
        ntod = scan_map.shape[0]
        freq = rfftfreq(ntod, d = 1/f_samp)
        fknee = scan.fknee_est
        alpha = scan.alpha_est
        N = freq.shape[0]

        if params.sample_corr_noise:
            C_1f_inv = np.zeros(N)
            C_1f_inv[1:] = 1.0 / (scan.sigma0**2*(freq[1:]/fknee)**alpha)
            scan.n_corr_est = corr_noise_realization_with_gaps(scan.sky_subtracted_tod/scan.g0_est, scan.galactic_mask_array, scan.sigma0, C_1f_inv)

    return experiment_data



def sample_noise_PS(band_comm: MPI.Comm, experiment_data: DetectorTOD, params: Bunch):
    logger = logging.getLogger(__name__)
    alphas = []
    fknees = []
    for scan in experiment_data.scans:
        fknee, alpha = sample_noise_PS_params(scan.n_corr_est, scan.sigma0, 6.0, scan.alpha_est, freq_max=2.0, n_grid=150, n_burnin=4)
        scan.fknee_est = fknee
        scan.alpha_est = alpha
        alphas.append(alpha)
        fknees.append(fknee)
    alphas = band_comm.gather(alphas, root=0)
    fknees = band_comm.gather(fknees, root=0)
    if band_comm.Get_rank() == 0:
        alphas = np.concatenate(alphas)
        fknees = np.concatenate(fknees)
        logger.info(f"{MPI.COMM_WORLD.Get_rank()} fknees {np.min(fknees):.4f} {np.percentile(fknees, 1):.4f} {np.mean(fknees):.4f} {np.percentile(fknees, 99):.4f} {np.max(fknees):.4f}")
        logger.info(f"{MPI.COMM_WORLD.Get_rank()} alphas {np.min(alphas):.4f} {np.percentile(alphas, 1):.4f} {np.mean(alphas):.4f} {np.percentile(alphas, 99):.4f} {np.max(alphas):.4f}")
    return experiment_data


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


def sample_absolute_gain(TOD_comm: MPI.Comm, experiment_data: DetectorTOD, params: Bunch):
    """ Function for drawing a realization of the absolute gain term, g0, which is constant across both all bands and all scans.
        Args:
            TOD_comm (MPI.Comm): The full TOD communicator, since we will calculate a single g0 value across all bands.
            experiment_data (DetectorTOD): The object holding all the scan data. Will be changed in-place to update g0.
            params (Bunch): Parameters from parameter file.
    """
    logger = logging.getLogger(__name__)
    T_CMB = 2.725  # K (Cosmic Microwave Background temperature)
    C = 299792458  # m/s (Speed of light)

    T_CMB = T_CMB * pysm3_u.K_CMB

    sum_s_T_N_inv_d = 0  # Accumulators for the numerator and denominator of eqn 16.
    sum_s_T_N_inv_s = 0
    for scan in experiment_data.scans:
        tod = scan.sky_subtracted_tod

        # --- Setup ---
        _, theta, phi, _ = scan.data
        vec_galactic = hp.ang2vec(theta, phi)
        dot_product = np.sum(scan.orb_dir_vec * vec_galactic, axis=-1)  # How much do the LOS and orbital velocity align?

        s_orb = T_CMB * dot_product / C  # The orbital dipole in units of uK_CMB.
        s_orb = s_orb.to("uK_RJ", equivalencies=pysm3_u.cmb_equivalencies(experiment_data.nu*pysm3_u.GHz))  # Converting to uK_RJ
        scan.s_orb = s_orb.value 

        Ntod = tod.shape[0]
        Nrfft = Ntod//2+1
        sigma0 = np.std(tod[1:] - tod[:-1])/np.sqrt(2)
        freqs = rfftfreq(Ntod, 1.0/params.samp_freq)
        inv_power_spectrum = np.zeros(Nrfft)
        inv_power_spectrum[1:] = 1.0/(sigma0**2*(1 + (freqs[1:]/scan.fknee_est)**scan.alpha_est))

        # --- Solving Equation 16 from BP7 ---
        s_fft = rfft(scan.s_orb)
        d_fft = rfft(tod)
        N_inv_s_fft = s_fft * inv_power_spectrum
        N_inv_d_fft = d_fft * inv_power_spectrum
        N_inv_s = irfft(N_inv_s_fft, n=Ntod)
        N_inv_d = irfft(N_inv_d_fft, n=Ntod)
        # We now exclude the time-samples hitting the masked area. We don't want to do this before now, because it would mess up the FFT stuff.
        s_T_N_inv_d = np.dot(scan.s_orb[scan.galactic_mask_array], N_inv_d[scan.galactic_mask_array])
        s_T_N_inv_s = np.dot(scan.s_orb[scan.galactic_mask_array], N_inv_s[scan.galactic_mask_array])
        sum_s_T_N_inv_d += s_T_N_inv_d  # Add to the numerator and denominator.
        sum_s_T_N_inv_s += s_T_N_inv_s

    # The g0 term is fully global, so we reduce across both all scans and all bands:
    sum_s_T_N_inv_d = TOD_comm.reduce(sum_s_T_N_inv_d, op=MPI.SUM, root=0)
    sum_s_T_N_inv_s = TOD_comm.reduce(sum_s_T_N_inv_s, op=MPI.SUM, root=0)
    g_sampled = 0.0
    if TOD_comm.Get_rank() == 0:  # Rank 0 draws a sample of g0 from eq (16) from BP6, and bcasts it to the other ranks.
        eta = np.random.randn()
        g_mean = sum_s_T_N_inv_d / sum_s_T_N_inv_s
        g_std = 1.0 / np.sqrt(sum_s_T_N_inv_s)
        g_sampled = g_mean + eta * g_std
        logger.info(f"Absolute gain (g0) mean: {g_mean:.5e}.")
        logger.info(f"Absolute gain (g0) std: {g_std:.5e}.")
        logger.info(f"Absolute gain (g0) sample: {g_sampled:.5e}.")
    g_sampled = TOD_comm.bcast(g_sampled, root=0)

    for scan in experiment_data.scans:
        scan.g0_est = g_sampled
        _, theta, phi, _ = scan.data
        vec_galactic = hp.ang2vec(theta, phi)
        dot_product = np.sum(scan.orb_dir_vec * vec_galactic, axis=-1)  # How much do the LOS and orbital velocity align?
        scan.sky_subtracted_tod -= scan.g0_est*scan.s_orb
        scan.orbital_dipole = scan.g0_est*scan.s_orb

    return experiment_data


def process_tod(TOD_comm: MPI.Comm, band_comm: MPI.Comm, experiment_data: DetectorTOD,
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
        for scan in experiment_data.scans:
            scan.n_corr_est = np.zeros_like(scan.data[0])
            scan.alpha_est = params.noise_alpha
            scan.fknee_est = params.noise_fknee
            scan.g0_est = params.initial_g0
            scan.orbital_dipole = np.zeros_like(scan.data[0])

    experiment_data = find_galactic_mask(experiment_data, params)

    experiment_data = subtract_sky_model(experiment_data, compsep_output, params)
    t0 = time.time()
    logger.info(f"Rank {MPI.COMM_WORLD.Get_rank()} chain {chain} iter{iter}: Finished sky model subtraction in {time.time()-t0:.1f}s."); t0 = time.time()
    if iter > 1:
        experiment_data = sample_absolute_gain(TOD_comm, experiment_data, params)
        logger.info(f"Rank {MPI.COMM_WORLD.Get_rank()} Finished gain estimation in {time.time()-t0:.1f}s."); t0 = time.time()
    experiment_data = estimate_white_noise(experiment_data, params)
    logger.info(f"Rank {MPI.COMM_WORLD.Get_rank()} chain {chain} iter{iter}: Finished white noise estimation in {time.time()-t0:.1f}s."); t0 = time.time()

    if iter > 6:
        experiment_data = sample_noise(band_comm, experiment_data, params)
        logger.info(f"Rank {MPI.COMM_WORLD.Get_rank()} chain {chain} iter{iter}: Finished corr noise realizations in {time.time()-t0:.1f}s."); t0 = time.time()
        experiment_data = sample_noise_PS(band_comm, experiment_data, params)
        logger.info(f"Rank {MPI.COMM_WORLD.Get_rank()} chain {chain} iter{iter}: Finished corr noise PS parameter sampling in {time.time()-t0:.1f}s."); t0 = time.time()
    todproc_output = tod2map(band_comm, experiment_data, compsep_output, params)
    logger.info(f"Rank {MPI.COMM_WORLD.Get_rank()} chain {chain} iter{iter}: Finished mapmaking in {time.time()-t0:.1f}s."); t0 = time.time()

    return todproc_output