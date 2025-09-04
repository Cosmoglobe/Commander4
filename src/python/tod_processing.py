import numpy as np
from mpi4py import MPI
import logging
from pixell.bunch import Bunch
from scipy.fft import rfft, irfft, rfftfreq
import time
from numpy.typing import NDArray

from src.python.output import log
from src.python.data_models.detector_map import DetectorMap
from src.python.data_models.detector_TOD import DetectorTOD
from src.python.data_models.detector_samples import DetectorSamples
from src.python.data_models.scan_samples import ScanSamples
from src.python.utils.mapmaker import MapmakerIQU, WeightsMapmakerIQU
from src.python.noise_sampling import corr_noise_realization_with_gaps, sample_noise_PS_params
from src.python.tod_processing_Planck import read_Planck_TOD_data
from src.python.utils.map_utils import get_static_sky_TOD, get_s_orb_TOD
from src.python.tod_processing_sim import read_TOD_sim_data

nthreads=1

def get_empty_compsep_output(staticData: DetectorTOD) -> NDArray[np.float32]:
    "Creates a dummy compsep output for a single band"
    return np.zeros((3, 12*staticData.nside**2), dtype=np.float32)


def tod2map(band_comm: MPI.Comm, experiment_data: DetectorTOD, compsep_output: NDArray,
            detector_samples:DetectorSamples, params: Bunch,
            mapmaker_corrnoise:MapmakerIQU = None) -> DetectorMap:
    mapmaker = MapmakerIQU(band_comm, experiment_data.nside)
    mapmaker_orbdipole = MapmakerIQU(band_comm, experiment_data.nside)
    mapmaker_skymodel = MapmakerIQU(band_comm, experiment_data.nside)
    mapmaker_invvar = WeightsMapmakerIQU(band_comm, experiment_data.nside)
    for scan, scan_samples in zip(experiment_data.scans, detector_samples.scans):
        pix = scan.pix
        psi = scan.psi
        mapmaker_invvar.accumulate_to_map(scan_samples.sigma0, pix, psi)
        mapmaker.accumulate_to_map(scan.tod/scan_samples.gain_est, scan_samples.sigma0, pix, psi)
        sky_orb_dipole = get_s_orb_TOD(scan, experiment_data, pix).astype(np.float32)
        mapmaker_orbdipole.accumulate_to_map(sky_orb_dipole, scan_samples.sigma0, pix, psi)
        sky_model = get_static_sky_TOD(compsep_output, pix, psi)
        mapmaker_skymodel.accumulate_to_map(sky_model, scan_samples.sigma0, pix, psi)
    mapmaker_invvar.gather_map()
    mapmaker.gather_map()
    mapmaker_orbdipole.gather_map()
    mapmaker_skymodel.gather_map()
    mapmaker_invvar.normalize_map()
    map_invvar = mapmaker_invvar.final_map
    map_cov = mapmaker_invvar.final_cov_map
    mapmaker.normalize_map(map_cov)
    map_signal = mapmaker.final_map
    mapmaker_orbdipole.normalize_map(map_cov)
    mapmaker_skymodel.normalize_map(map_cov)
    map_orbdipole = mapmaker_orbdipole.final_map
    map_skymodel = mapmaker_skymodel.final_map
    if mapmaker_corrnoise is not None:
        mapmaker_corrnoise.normalize_map(map_cov)
        map_corrnoise = mapmaker_corrnoise.final_map
    if band_comm.Get_rank() == 0:
        detmap = DetectorMap(map_signal, 1.0/map_invvar**2, experiment_data.nu,
                             experiment_data.fwhm, experiment_data.nside)
        detmap.g0 = detector_samples.g0_est
        detmap.gain = detector_samples.scans[0].rel_gain_est + detector_samples.g0_est
        detmap.map_skymodel = map_skymodel
        detmap.map_orbdipole = map_orbdipole
        if mapmaker_corrnoise is not None:
            detmap.map_corrnoise = map_corrnoise
    else:
        detmap = None
    return detmap


def init_tod_processing(tod_comm: MPI.Comm, params: Bunch) -> tuple[bool, MPI.Comm, MPI.Comm, str, dict[str,int], DetectorTOD, DetectorSamples]:
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
    my_band_id = None
    my_detector_id = None
    my_scans_start = None
    my_scans_stop = None
    TOD_rank = 0  # A counter for how many TOD ranks have been allocated so far.
    current_detector_id = 0  # A unique number identifying every detector of every band.
    for exp_name in params.experiments:
        experiment = params.experiments[exp_name]
        if not experiment.enabled:
            continue
        for iband, band_name in enumerate(experiment.bands):
            band = experiment.bands[band_name]
            if not band.enabled:
                continue
            this_band_TOD_ranks = np.arange(TOD_rank, TOD_rank + band.num_MPI_tasks)
            if MPIrank_tod in this_band_TOD_ranks:
                # Splitting TOD ranks evenly among the detectors of this band.
                TOD_ranks_per_detector = np.array_split(this_band_TOD_ranks, len(band.detectors))
                for idet, det_name in enumerate(band.detectors):
                    num_ranks_this_detector = len(TOD_ranks_per_detector[idet])
                    detector = band.detectors[det_name]
                    # Check if our rank belongs to this detector
                    if tod_comm.Get_rank() in TOD_ranks_per_detector[idet]:
                        # What is my rank number among the ranks processing this detector?
                        my_local_rank = tod_comm.Get_rank() - np.min(TOD_ranks_per_detector[idet])
                        my_experiment_name = exp_name
                        my_band_name = band_name
                        my_experiment = experiment
                        my_band = band
                        my_det = detector
                        my_detector_name = det_name
                        # Setting our unique detector id. Note that this is a global, not per band.
                        my_detector_id = current_detector_id
                        my_band_id = iband
                        tot_num_scans = experiment.num_scans
                        scans = np.arange(tot_num_scans)
                        my_scans = np.array_split(scans, num_ranks_this_detector)[my_local_rank]
                        my_scans_start = my_scans[0]
                        my_scans_stop = my_scans[-1]
                    current_detector_id += 1  # Update detector counter.
            else:
                # Update detector counter for ranks not assigned to current band.
                current_detector_id += len(band.detectors)
            TOD_rank += band.num_MPI_tasks
    tot_num_bands = TOD_rank
    tod_comm.Barrier()

    if tot_num_bands != MPIsize_tod:
        log.lograise(RuntimeError, f"Total number of bands dedicated to the various experiments "
                     f"({tot_num_bands}) differs from the total number of tasks dedicated to "
                     f"TOD processing ({MPIsize_tod}).", logger)

    if tod_master:
        logger.info(f"TOD: {MPIsize_tod} tasks allocated to TOD processing of {tot_num_bands} bands.")
        log.logassert(MPIsize_tod >= tot_num_bands, f"Number of MPI tasks dedicated to TOD processing ({MPIsize_tod}) must be equal to or larger than the number of bands ({tot_num_bands}).", logger)

    tod_comm.barrier()
    time.sleep(tod_comm.Get_rank()*1e-3)  # Small sleep to get prints in nice order.
    # MPIcolor_band = MPIrank_tod%tot_num_bands  # Spread the MPI tasks over the different bands.
    band_comm = tod_comm.Split(my_band_id, key=MPIrank_tod)  # Create communicators for each different band.
    MPIsize_band, MPIrank_band = band_comm.Get_size(), band_comm.Get_rank()  # Get my local rank, and the total size of, the band-communicator I'm on.
    logger.info(f"TOD-rank {MPIrank_tod:4} (on machine {MPI.Get_processor_name()}), dedicated to band {my_band_id:4}, with local rank {MPIrank_band:4} (local communicator size: {MPIsize_band:4}).")
    is_band_master = MPIrank_band == 0  # Am I the master of my local band.
    
    det_comm = band_comm.Split(my_detector_id, key=MPIrank_tod)  # Create communicators for each different band.
    MPIsize_det, MPIrank_det = det_comm.Get_size(), band_comm.Get_rank()  # Get my local rank, and the total size of, the band-communicator I'm on.
    logger.info(f"TOD-rank {MPIrank_tod:4} (on machine {MPI.Get_processor_name()}), dedicated to detector {my_detector_id:4}, with local rank {MPIrank_det:4} (local communicator size: {MPIsize_det:4}).")

    # Creating "tod_band_masters", an array which maps the band index to the rank of the master of that band.
    my_band_identifier = f"{my_experiment_name}$$${my_band_name}"
    band_data = (my_band_identifier, MPI.COMM_WORLD.Get_rank()) if band_comm.Get_rank() == 0 else None
    all_band_data = tod_comm.allgather(band_data)
    tod_band_masters_dict = {item[0]: item[1] for item in all_band_data if item is not None}
    logger.info(f"TOD: Rank {MPIrank_tod:4} assigned scans {my_scans_start:6} - {my_scans_stop:6} on band {my_band_id:4}, det{my_detector_id:4}.")

    t0 = time.time()
    if my_experiment.is_sim:
        experiment_data, detector_samples = read_TOD_sim_data(my_experiment.data_path, my_band, my_det, params, my_detector_id, my_scans_start, my_scans_stop)
    else:
        experiment_data, detector_samples = read_Planck_TOD_data(my_experiment.data_path, my_band, my_det, params, my_detector_id, my_scans_start, my_scans_stop, my_experiment.bad_PIDs_path)
    tod_comm.Barrier()
    if tod_comm.Get_rank() == 0:
        logger.info(f"TOD: Finished reading all files in {time.time()-t0:.1f}s.")

    return is_band_master, band_comm, det_comm, my_band_identifier, tod_band_masters_dict, experiment_data, detector_samples



def estimate_white_noise(experiment_data: DetectorTOD, detector_samples: DetectorSamples, det_compsep_map: NDArray, params: Bunch) -> DetectorTOD:
    """Estimate the white noise level in the TOD data, add it to the scans, and return the updated experiment data.
    Input:
        experiment_data (DetectorTOD): The experiment TOD object.
        params (Bunch): The parameters from the input parameter file.
    Output:
        experiment_data (DetectorTOD): The experiment TOD with the estimated white noise level added to each scan.
    """
    for scan, scan_samples in zip(experiment_data.scans, detector_samples.scans):
        raw_TOD = scan.tod
        pix = scan.pix
        psi = scan.psi
        sky_TOD = get_static_sky_TOD(det_compsep_map, pix, psi) + get_s_orb_TOD(scan, experiment_data, pix)
        sky_subtracted_tod = raw_TOD - scan_samples.gain_est*sky_TOD
        mask = scan.processing_mask_TOD
        if np.sum(mask) > 50:  # If we have enough data points to estimate the noise, we use the masked version.
            sigma0 = np.std(sky_subtracted_tod[mask][1:] - sky_subtracted_tod[mask][:-1])/np.sqrt(2)
        else:
            sigma0 = np.std(sky_subtracted_tod[1:] - sky_subtracted_tod[:-1])/np.sqrt(2)
        scan_samples.sigma0 = sigma0/scan_samples.gain_est
    return detector_samples



def sample_noise(band_comm: MPI.Comm, experiment_data: DetectorTOD,
                 detector_samples: DetectorSamples, det_compsep_map: NDArray) -> DetectorTOD:
    logger = logging.getLogger(__name__)
    num_failed_convergence = 0
    worst_residual = 0.0
    alphas = []
    fknees = []
    mapmaker = MapmakerIQU(band_comm, experiment_data.nside)
    for scan, scansamples in zip(experiment_data.scans, detector_samples.scans):
        f_samp = scan.fsamp
        raw_tod = scan.tod
        pix = scan.pix
        psi = scan.psi
        sky_TOD = get_static_sky_TOD(det_compsep_map, pix, psi) + get_s_orb_TOD(scan, experiment_data, pix)
        sky_subtracted_TOD = raw_tod - scansamples.gain_est*sky_TOD
        Ntod = raw_tod.shape[0]
        Nfft = Ntod//2 + 1
        freq = rfftfreq(Ntod, d = 1/f_samp)
        fknee = scansamples.fknee_est
        alpha = scansamples.alpha_est
        sigma0 = scansamples.gain_est*scansamples.sigma0
        C_1f_inv = np.zeros(Nfft)
        C_1f_inv[1:] = 1.0 / (sigma0**2*(freq[1:]/fknee)**alpha)
        err_tol = 1e-8
        mask = scan.processing_mask_TOD
        n_corr_est, residual = corr_noise_realization_with_gaps(sky_subtracted_TOD,
                                                                mask, sigma0, C_1f_inv,
                                                                err_tol=err_tol)
        mapmaker.accumulate_to_map(n_corr_est/scansamples.gain_est, scansamples.sigma0, pix, psi)
        scansamples.n_corr_est = n_corr_est.astype(np.float32)
        if residual > err_tol:
            num_failed_convergence += 1
            worst_residual = max(worst_residual, residual)

        sigma0 = scansamples.gain_est*scansamples.sigma0
        fknee, alpha = sample_noise_PS_params(n_corr_est, sigma0, scan.fsamp, scansamples.alpha_est,
                                              freq_max=2.0, n_grid=150, n_burnin=4)
        scansamples.fknee_est = fknee
        scansamples.alpha_est = alpha
        alphas.append(alpha)
        fknees.append(fknee)

    mapmaker.gather_map()
    num_failed_convergence = band_comm.reduce(num_failed_convergence, op=MPI.SUM)
    worst_residual = band_comm.reduce(worst_residual, op=MPI.MAX)
    if band_comm.Get_rank() == 0:
        if num_failed_convergence > 0:
            logger.info(f"Band {experiment_data.nu}GHz failed noise CG for {num_failed_convergence}"
                        f"scans. Worst residual = {worst_residual:.3e}.")

    alphas = band_comm.gather(alphas, root=0)
    fknees = band_comm.gather(fknees, root=0)
    if band_comm.Get_rank() == 0:
        alphas = np.concatenate(alphas)
        fknees = np.concatenate(fknees)
        logger.info(f"{MPI.COMM_WORLD.Get_rank()} fknees {np.min(fknees):.4f} {np.percentile(fknees, 1):.4f} {np.mean(fknees):.4f} {np.percentile(fknees, 99):.4f} {np.max(fknees):.4f}")
        logger.info(f"{MPI.COMM_WORLD.Get_rank()} alphas {np.min(alphas):.4f} {np.percentile(alphas, 1):.4f} {np.mean(alphas):.4f} {np.percentile(alphas, 99):.4f} {np.max(alphas):.4f}")

    return detector_samples, mapmaker



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
        pix = scan.pix  # Only decompressing pix once for efficiency.
        psi = scan.psi
        s_orb = get_s_orb_TOD(scan, experiment_data, pix)
        sky_model_TOD = get_static_sky_TOD(det_compsep_map, pix, psi)

        tod_residual = scan.tod - scan_samples.gain_est*(sky_model_TOD + s_orb)  # Subtracting sky signal.
        tod_residual += detector_samples.g0_est*s_orb  # Now we can add back in the orbital dipole.

        sigma0 = scan_samples.gain_est*scan_samples.sigma0  # scan.sigma0 is in tempearture units.
        Ntod = scan.tod.shape[0]
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

        # mask = experiment_data.processing_mask_map[scan.pix]
        mask = scan.processing_mask_TOD
        sum_s_T_N_inv_d += np.dot(s_orb[mask], N_inv_d[mask])  # Add to the numerator and denominator.
        sum_s_T_N_inv_s += np.dot(s_orb[mask], N_inv_s[mask])

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

    t0 = time.time()
    TOD_comm.Barrier()
    wait_time = time.time() - t0
    g_sampled = TOD_comm.bcast(g_sampled, root=0)

    detector_samples.g0_est = g_sampled

    return detector_samples, wait_time


def sample_relative_gain(TOD_comm: MPI.Comm, det_comm: MPI.Comm, experiment_data: DetectorTOD, detector_samples, det_compsep_map: NDArray):
    """Samples the detector-dependent relative gain (Delta g_i).
    This function implements the logic from Sec. 3.4 of the BP7.
    It uses a two-stage MPI communication:
    1. A reduction over the `det_comm` to accumulate sums for a single detector
       that is distributed across multiple ranks.
    2. A gather over the global `TOD_comm` to collect the final sums from each
       unique detector on the root rank for solving the global system.

    Args:
        TOD_comm (MPI.Comm): The global MPI communicator for all bands.
        det_comm (MPI.Comm): The communicator for ranks sharing the same detector.
        experiment_data (DetectorTOD): The object holding scan data for a single
                                       detector on the calling rank.
        params (Bunch): Parameters from the parameter file.
    """
    logger = logging.getLogger(__name__)
    global_rank = TOD_comm.Get_rank()
    band_rank = det_comm.Get_rank()

    #### 1. Local Calculation (on each rank) ###
    # Each rank calculates the sum of terms for its local subset of scans.
    local_s_T_N_inv_s = 0.0
    local_r_T_N_inv_s = 0.0

    for scan, scan_samples in zip(experiment_data.scans, detector_samples.scans):
        # Define the residual for this sampling step, as per Eq. (17)
        pix = scan.pix
        psi = scan.psi
        s_tot = get_static_sky_TOD(det_compsep_map, pix, psi) + get_s_orb_TOD(scan, experiment_data, pix)

        residual_tod = scan.tod - (detector_samples.g0_est + scan_samples.time_dep_rel_gain_est)*s_tot

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
        
        # mask = experiment_data.processing_mask_map[scan.pix]
        mask = scan.processing_mask_TOD
        s_T_N_inv_s_scan = np.dot(s_tot[mask], N_inv_s[mask])
        r_T_N_inv_s_scan = np.dot(residual_tod[mask], N_inv_s[mask])

        # Add the contribution from this scan to the local sum
        local_s_T_N_inv_s += s_T_N_inv_s_scan
        local_r_T_N_inv_s += r_T_N_inv_s_scan

    ### 2. Intra-Detector Reduction ###
    # Sum the local values across all ranks that share the same detector using det_comm.
    # After this, every rank in the det_comm will have the total sum for their detector.
    total_s_for_detector = det_comm.allreduce(local_s_T_N_inv_s, op=MPI.SUM)
    total_r_for_detector = det_comm.allreduce(local_r_T_N_inv_s, op=MPI.SUM)

    ### 3. Gather all unique detector sums on the global Rank 0 ###
    # To avoid redundant communication, only the root of each detector group (band_rank 0)
    # sends its data to the global root. Other ranks send None.
    if band_rank == 0:
        data_to_send = (experiment_data.detector_id, total_s_for_detector, total_r_for_detector)
    else:
        data_to_send = None
    
    t0 = time.time()
    TOD_comm.Barrier()
    wait_time = time.time() - t0
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
            if det_comm.Get_rank() == 0:
                logging.info(f"Relative gain for detector {experiment_data.detector_name} ({experiment_data.nu}GHz): {detector_samples.g0_est*1e9+my_delta_g*1e9:8.3f} ({detector_samples.g0_est*1e9:8.3f} + {my_delta_g*1e9:8.3f})")
        except ValueError:
            logger.error(f"Rank {global_rank} with detector {my_did} not found in solved gain list.")
    else:
        logger.warning(f"No valid relative gain solution was broadcast. Not updating gains on rank {global_rank}.")

    return detector_samples, wait_time



def sample_temporal_gain_variations(det_comm: MPI.Comm, experiment_data: DetectorTOD, detector_samples, det_compsep_map: NDArray, chain: int, iter: int, params: Bunch):
    """
    Samples the time-dependent relative gain variations (delta g_qi).
    This function implements the logic from Sec. 3.5 of the BP7 paper,
    using a Wiener filter to smooth the gain solution over time (PIDs).
    It solves a global system for all scans of a given detector, which are
    distributed across the ranks of the det_comm.

    Args:
        det_comm (MPI.Comm): The communicator for ranks sharing the same detector.
        experiment_data (DetectorTOD): The object holding scan data.
        params (Bunch): Parameters from the parameter file.
    """
    logger = logging.getLogger(__name__)
    band_rank = det_comm.Get_rank()
    band_size = det_comm.Get_size()

    # Local calculations on each rank
    A_qq_local = []
    b_q_local = []

    for scan, scan_samples in zip(experiment_data.scans, detector_samples.scans):
        # I'm still not sure what way of dealing with the masked samples are best:
        # 1. Replace masked values with 0s before FFT.
        # 2. Replace masked values with n_corr realizations before FFT.
        # 3. Remove masked values by reducing TOD size before FFTs.
        # (simply passing the full data through the FFTs seems like a bad idea because of
        # ringing from the large residual in the galactic plane).

        # Per Eq. (26), the residual is d - (g0 + Delta_g)*s
        pix = scan.pix
        psi = scan.psi
        s_tot = get_static_sky_TOD(det_compsep_map, pix, psi) + get_s_orb_TOD(scan, experiment_data, pix)

        residual_tod = scan.tod - (detector_samples.g0_est + scan_samples.rel_gain_est)*s_tot

        # mask = experiment_data.processing_mask_map[scan.pix]
        mask = scan.processing_mask_TOD

        # FFT-based N^-1 operation setup
        Ntod = residual_tod.shape[0]
        Nrfft = Ntod // 2 + 1
        freqs = rfftfreq(Ntod, 1.0 / scan.fsamp)
        sigma0 = scan_samples.gain_est*scan_samples.sigma0
        inv_power_spectrum = np.zeros(Nrfft)
        inv_power_spectrum[1:] = 1.0 / (sigma0**2 * (1 + (freqs[1:] / scan_samples.fknee_est)**scan_samples.alpha_est))

        s_tot[~mask] = 0.0  #TODO: Needs to be handled differently.
        residual_tod[~mask] = 0.0
        # s_tot[~mask] = scan_samples.n_corr_est[~mask]/scan_samples.gain_est
        # residual_tod[~mask] = scan_samples.n_corr_est[~mask]
        # Calculate N^-1 * s_tot and N^-1 * residual_tod
        N_inv_s = irfft(rfft(s_tot) * inv_power_spectrum, n=Ntod)
        N_inv_r = irfft(rfft(residual_tod) * inv_power_spectrum, n=Ntod)
        
        # Calculate elements for the linear system
        A_qq = np.dot(s_tot[mask], N_inv_s[mask])
        b_q = np.dot(s_tot[mask], N_inv_r[mask])
        
        A_qq_local.append(A_qq)
        b_q_local.append(b_q)

    A_qq_local = np.array(A_qq_local, dtype=np.float64)
    b_q_local = np.array(b_q_local, dtype=np.float64)

    # Gather all data on the root rank of the band communicator
    scan_counts = det_comm.gather(len(A_qq_local), root=0)
    all_A_qq = det_comm.gather(A_qq_local, root=0)
    all_b_q = det_comm.gather(b_q_local, root=0)
    
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

            # These sigma0 values should match the BP analysis values,
            # sigma0_sq_gain_V2_per_K2 = 3e-4  # V^2/K^2
            # sigma0_sq_gain = 1e-12 * sigma0_sq_gain_V2_per_K2  # V^2/K^2 -> V^2/uK^2

            # However, I found the BP prior too weak, and this gives me more sensible results.
            mean_gain = detector_samples.g0_est + detector_samples.scans[0].rel_gain_est
            sigma0_gain = 1e-4*mean_gain
            sigma0_sq_gain = sigma0_gain**2

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
                if CG_solver.err < 1e-12:
                    break
            logging.info(f"CG solver for gain fluctuations (nu={experiment_data.nu})"
                        f"finished after {i} iterations (residual = {CG_solver.err})")
            delta_g_sample = CG_solver.x
            logger.info(f"delta_g_sample mean = {np.mean(delta_g_sample)}")
            delta_g_sample -= np.mean(delta_g_sample)
            logger.info(f"delta_g: {delta_g_sample}")
            logger.info(f"Band {experiment_data.nu}GHz time-dependent gain: min={np.min(delta_g_sample)*1e9:14.4f} mean={np.mean(delta_g_sample)*1e9:14.4f} std={np.std(delta_g_sample)*1e9:14.4f} max={np.max(delta_g_sample)*1e9:14.4f}")

            if False: #debug stuff
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10,8))
                other_gain = detector_samples.g0_est + detector_samples.scans[0].rel_gain_est
                plt.plot(1e9*(other_gain + delta_g_sample))
                plt.ylim(0.85*1e9*other_gain, 1.15*1e9*other_gain)
                plt.xlabel("PID")
                plt.ylabel("Gain [mV/K]")
                plt.xticks([0, 15000, 30000, 45000])
                plt.savefig(f"{params.output_paths.plots}chain{chain}_iter{iter}_{detector_samples.detname}.png")
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
        det_comm.Scatterv([delta_g_sample, scan_counts, displacements, MPI.DOUBLE], delta_g_local, root=0)
    else:
        delta_g_local = delta_g_sample if delta_g_sample is not None else np.array([])

    # Update local scan objects
    if delta_g_local.size == len(experiment_data.scans):
        for i, scan_samples in enumerate(detector_samples.scans):
            scan_samples.time_dep_rel_gain_est = delta_g_local[i]
    else:
        logger.warning(f"Rank {band_rank} received mismatched number of gain samples."
                       f"Expected {len(experiment_data.scans)}, got {delta_g_local.size}.")

    return detector_samples



def process_tod(TOD_comm: MPI.Comm, band_comm: MPI.Comm, det_comm: MPI.Comm,
                experiment_data: DetectorTOD, detector_samples, compsep_output: NDArray,
                params: Bunch, chain, iter) -> DetectorMap:
    """ Performs a single TOD iteration.

    Input:
        TOD_comm (MPI.Comm): The full TOD communicator between all TOD-processing ranks.
        band_comm (MPI.Comm): The communicator for all the MPI-ranks on our band.
        det_comm (MPI.Comm): The communicator for all the MPI-ranks on our detector.
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
    # 1. Initialize n_corr_est and alpha/fknee values.
    # 2. Sample the gain from the sky-subtracted TOD (Skipped on iter==1 because we don't have a reliable sky-subtracted TOD).
    # 3. Estimate White noise from the sky-subtracted TOD.
    # 4. Sample correlated noise and PS parameters (skipped on iter==1).
    # 5. Mapmaking on TOD - corr_noise_TOD - orb_dipole_TOD.
    # (In other words, on iteration 1 we do just do White noise estimation -> Mapmaking.)

    logger = logging.getLogger(__name__)
    if iter == 1:
        for scan, scan_samples in zip(experiment_data.scans, detector_samples.scans):
            scan_samples.alpha_est = params.noise_alpha
            scan_samples.fknee_est = params.noise_fknee

    if iter >= params.sample_gain_from_iter_num:
        ### ABSOLUTE GAIN CALIBRATION ### 
        t0 = time.time()
        detector_samples, wait_time = sample_absolute_gain(TOD_comm, experiment_data, detector_samples, compsep_output)
        TOD_comm.Barrier()
        tot_time = time.time() - t0
        wait_time = det_comm.reduce(wait_time, op=MPI.SUM, root=0)
        if TOD_comm.Get_rank() == 0:
            logger.info(f"Chain {chain} iter{iter}: Finished absolute gain estimation in {tot_time:.1f}s.")
        if det_comm.Get_rank() == 0:
            wait_time /= det_comm.Get_size()
            logger.info(f"Absolute gain estimation MPI wait overhead for detector {experiment_data.detector_name} ({experiment_data.nu}GHz) = {wait_time:.1f}s.")

        ### RELATIVE GAIN CALIBRATION ### 
        t0 = time.time()
        detector_samples, wait_time = sample_relative_gain(TOD_comm, det_comm, experiment_data, detector_samples, compsep_output)
        TOD_comm.Barrier()
        tot_time = time.time() - t0
        wait_time = det_comm.reduce(wait_time, op=MPI.SUM, root=0)
        if TOD_comm.Get_rank() == 0:
            logger.info(f"Chain {chain} iter{iter}: Finished relative gain estimation in {tot_time:.1f}s.")
        if det_comm.Get_rank() == 0:
            wait_time /= det_comm.Get_size()
            logger.info(f"Relative gain estimation MPI wait overhead for detector {experiment_data.detector_name} ({experiment_data.nu}GHz) = {wait_time:.1f}s.")


        ### TEMPORAL GAIN CALIBRATION ### 
        t0 = time.time()
        detector_samples = sample_temporal_gain_variations(det_comm, experiment_data, detector_samples, compsep_output, chain, iter, params)
        t1 = time.time()
        TOD_comm.Barrier()
        tot_time = time.time() - t0
        wait_time = time.time() - t1
        wait_time = det_comm.reduce(wait_time, op=MPI.SUM, root=0)
        if TOD_comm.Get_rank() == 0:
            logger.info(f"Chain {chain} iter{iter}: Finished temporal gain estimation in {tot_time:.1f}s.")
        if det_comm.Get_rank() == 0:
            wait_time /= det_comm.Get_size()
            logger.info(f"Temporal gain estimation MPI wait overhead for detector {experiment_data.detector_name} ({experiment_data.nu}GHz) = {wait_time:.1f}s.")


        ### Update total gain from sum of all three gain terms. ###
        for scan_samples in detector_samples.scans:
            scan_samples.gain_est = detector_samples.g0_est + scan_samples.rel_gain_est\
                                  + scan_samples.time_dep_rel_gain_est

    ### WHITE NOISE ESTIMATION ###
    t0 = time.time()
    detector_samples = estimate_white_noise(experiment_data, detector_samples, compsep_output, params)
    t1 = time.time()
    TOD_comm.Barrier()
    tot_time = time.time() - t0
    wait_time = time.time() - t1
    wait_time = det_comm.reduce(wait_time, op=MPI.SUM, root=0)
    if TOD_comm.Get_rank() == 0:
        logger.info(f"Chain {chain} iter{iter}: Finished white noise estimation in {tot_time:.1f}s.")
    if det_comm.Get_rank() == 0:
        wait_time /= det_comm.Get_size()
        logger.info(f"White noise estimation MPI wait overhead for detector {experiment_data.detector_name} ({experiment_data.nu}GHz) = {wait_time:.1f}s.")


    if iter >= params.sample_corr_noise_from_iter_num:
        ### CORRELATED NOISE SAMPLING ###
        t0 = time.time()
        detector_samples, mapmaker_corrnoise = sample_noise(det_comm, experiment_data, detector_samples, compsep_output)
        t1 = time.time()
        TOD_comm.Barrier()
        tot_time = time.time() - t0
        wait_time = time.time() - t1
        wait_time = det_comm.reduce(wait_time, op=MPI.SUM, root=0)
        if TOD_comm.Get_rank() == 0:
            logger.info(f"Chain {chain} iter{iter}: Finished noise sampling in {tot_time:.1f}s.")
        if det_comm.Get_rank() == 0:
            wait_time /= det_comm.Get_size()
            logger.info(f"Noise sampling MPI wait overhead for detector {experiment_data.detector_name} ({experiment_data.nu}GHz) = {wait_time:.1f}s.")
    else:
        mapmaker_corrnoise = None

    ### MAPMAKING ###
    t0 = time.time()
    # todproc_output = tod2map(band_comm, experiment_data, compsep_output, detector_samples, params)
    detmap = tod2map(band_comm, experiment_data, compsep_output, detector_samples, params,
                     mapmaker_corrnoise)
    t1 = time.time()
    TOD_comm.Barrier()
    tot_time = time.time() - t0
    wait_time = time.time() - t1
    wait_time = det_comm.reduce(wait_time, op=MPI.SUM, root=0)
    if TOD_comm.Get_rank() == 0:
        logger.info(f"Chain {chain} iter{iter}: Finished mapmaking in {tot_time:.1f}s.")
    if det_comm.Get_rank() == 0:
        wait_time /= det_comm.Get_size()
        logger.info(f"Mapmaking MPI wait overhead for detector {experiment_data.detector_name} ({experiment_data.nu}GHz) = {wait_time:.1f}s.")

    return detmap, detector_samples
