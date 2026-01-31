import numpy as np
from mpi4py import MPI
import logging
from pixell.bunch import Bunch
from scipy.fft import rfft, irfft, rfftfreq
import time
from numpy.typing import NDArray
import math

from commander4.output import log
from commander4.data_models.detector_map import DetectorMap
from commander4.data_models.detector_TOD import DetectorTOD
from commander4.data_models.detector_samples import DetectorSamples
from commander4.data_models.scan_samples import ScanSamples
from commander4.utils.mapmaker import MapmakerIQU, WeightsMapmakerIQU
from commander4.noise_sampling import corr_noise_realization_with_gaps, sample_noise_PS_params
from commander4.utils.map_utils import get_static_sky_TOD, get_s_orb_TOD
from commander4.utils.math_operations import forward_rfft, backward_rfft, calculate_sigma0
from commander4.tod_reader import read_tods_from_file

nthreads=1

def get_empty_compsep_output(staticData: DetectorTOD) -> NDArray[np.float32]:
    "Creates a dummy compsep output for a single band"
    return np.zeros((3, 12*staticData.nside**2), dtype=np.float32)


def tod2map(band_comm: MPI.Comm, experiment_data: DetectorTOD, compsep_output: NDArray,
            detector_samples:DetectorSamples, params: Bunch,
            mapmaker_corrnoise:MapmakerIQU = None) -> DetectorMap:
    # We separate the inverse-variance mapmaking from the other 3 mapmakers.
    # This is purely to reduce the maximum concurrent memory requirement, and is slightly slower
    # as we have to de-compress pix and psi twice.
    mapmaker_invvar = WeightsMapmakerIQU(band_comm, experiment_data.nside)    
    for scan, scan_samples in zip(experiment_data.scans, detector_samples.scans):
        pix = scan.pix
        psi = scan.psi
        inv_var = 1.0/scan_samples.sigma0**2
        mapmaker_invvar.accumulate_to_map(inv_var, pix, psi)
    mapmaker_invvar.gather_map()
    mapmaker_invvar.normalize_map()

    mapmaker = MapmakerIQU(band_comm, experiment_data.nside)
    mapmaker_orbdipole = MapmakerIQU(band_comm, experiment_data.nside)
    mapmaker_skymodel = MapmakerIQU(band_comm, experiment_data.nside)
    for scan, scan_samples in zip(experiment_data.scans, detector_samples.scans):
        pix = scan.pix
        psi = scan.psi
        sky_orb_dipole = get_s_orb_TOD(scan, experiment_data, pix)
        sky_model = get_static_sky_TOD(compsep_output, pix, psi)
        inv_var = 1.0/scan_samples.sigma0**2
        mapmaker.accumulate_to_map(scan.tod/scan_samples.gain_est, inv_var, pix, psi)
        mapmaker_orbdipole.accumulate_to_map(sky_orb_dipole, inv_var, pix, psi)
        mapmaker_skymodel.accumulate_to_map(sky_model, inv_var, pix, psi)

    mapmaker.gather_map()
    mapmaker_orbdipole.gather_map()
    mapmaker_skymodel.gather_map()
    map_rms = mapmaker_invvar.final_rms_map
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
        map_signal -= map_orbdipole
        if mapmaker_corrnoise is not None:
            map_signal -= map_corrnoise
        detmap = DetectorMap(map_signal, map_rms, experiment_data.nu,
                             experiment_data.fwhm, experiment_data.nside)
        detmap.g0 = detector_samples.g0_est
        detmap.gain = detector_samples.scans[0].rel_gain_est + detector_samples.g0_est
        detmap.map_skymodel = map_skymodel
        detmap.map_orbdipole = map_orbdipole
        if mapmaker_corrnoise is not None:
            detmap.map_corrnoise = map_corrnoise
    else:
        detmap = None
    
    # band_comm.Barrier()
    # t_finalize = time.perf_counter() - t0

    # t_huffman = band_comm.reduce(t_huffman, op=MPI.SUM, root=0)
    # t_sky = band_comm.reduce(t_sky, op=MPI.SUM, root=0)
    # t_acc = band_comm.reduce(t_acc, op=MPI.SUM, root=0)
    # t_gather = band_comm.reduce(t_gather, op=MPI.SUM, root=0)
    # if band_comm.Get_rank() == 0:
    #     logger = logging.getLogger(__name__)
    #     logger.info(f"### Mapmaker {experiment_data.nu:.1f} : t_huffman :      {t_huffman/band_comm.Get_size():.4f} s.")
    #     logger.info(f"### Mapmaker {experiment_data.nu:.1f} : t_sky :          {t_sky/band_comm.Get_size():.4f} s.")
    #     logger.info(f"### Mapmaker {experiment_data.nu:.1f} : t_acc :          {t_acc/band_comm.Get_size():.4f} s.")
    #     logger.info(f"### Mapmaker {experiment_data.nu:.1f} : t_gather :       {t_gather/band_comm.Get_size():.4f} s.")
    #     logger.info(f"### Mapmaker {experiment_data.nu:.1f} : t_finalization : {t_finalize:.4f} s.")

    return detmap


def init_tod_processing(mpi_info: Bunch, params: Bunch) -> tuple[bool, MPI.Comm, MPI.Comm, str,
                                                                 dict[str,int], DetectorTOD,
                                                                 DetectorSamples]:
    """To be run once before starting TOD processing.

    Determines whether the process is TOD master, creates the band communicator
    and determines whether the process is the band master. Also reads the
    experiment data.

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        params (Bunch): The parameters from the input parameter file.

    Output:
        mpi_info (Bunch): The data structure containing all MPI relevant data,
            now also with a 'tod' section as well as the dictionary of band
            master mappings.
        my_band_identifier (str): Unique string identifier for the experiment+band this process is responsible for.
        experiment_data (DetectorTOD): THe TOD data for the band of this process.
    """

    logger = logging.getLogger(__name__)

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
    current_detector_id = 0  # A unique number identifying every detector of every band.
    for exp_name in params.experiments:
        experiment = params.experiments[exp_name]
        if not experiment.enabled:
            continue
        for iband, band_name in enumerate(experiment.bands):
            band = experiment.bands[band_name]
            if not band.enabled:
                continue
            if mpi_info.band.name != band_name:
                current_detector_id += len(band.detectors)
                continue
            my_band_name = band_name
            my_band = band
            my_band_id = iband
            for idet, det_name in enumerate(band.detectors):
                if mpi_info.det.name != det_name:
                    current_detector_id += 1  # Update detector counter.
                    continue
                detector = band.detectors[det_name]
                # What is my rank number among the ranks processing this detector?
                my_experiment_name = exp_name
                my_experiment = experiment
                my_det = detector
                my_detector_name = det_name
                # Setting our unique detector id. Note that this is a global, not per band.
                my_detector_id = current_detector_id
                tot_num_scans = experiment.num_scans
                scans = np.arange(tot_num_scans)
                my_scans = np.array_split(scans, mpi_info.det.size)[mpi_info.det.rank]
                my_scans_start = my_scans[0]
                my_scans_stop = my_scans[-1]
                current_detector_id += 1  # Update detector counter.
    mpi_info.tod.comm.Barrier()


    logger.info(f"TOD-rank {mpi_info.tod.rank:4} (on machine {mpi_info.processor_name}), dedicated to "
                f"detector {my_detector_id:4}, with local rank {mpi_info.det.rank:4} (local communicator "
                f"size: {mpi_info.det.size:4}).")
    time.sleep(mpi_info.tod.rank*1e-3)  # Small sleep to get prints in nice order.
    # MPIcolor_band = MPIrank_tod%tot_num_bands  # Spread the MPI tasks over the different bands.
    band_comm = mpi_info.band.comm
    MPIsize_band, MPIrank_band = band_comm.Get_size(), band_comm.Get_rank()  # Get my local rank, and the total size of, the band-communicator I'm on.
    logger.info(f"TOD-rank {mpi_info.tod.rank:4} (on machine {mpi_info.processor_name}), dedicated to band {my_band_id:4}, with local rank {mpi_info.band.rank:4} (local communicator size: {mpi_info.band.size:4}).")
    
    det_comm = band_comm.Split(my_detector_id, key=mpi_info.tod.rank)  # Create communicators for each different band.
    MPIsize_det, MPIrank_det = det_comm.Get_size(), band_comm.Get_rank()  # Get my local rank, and the total size of, the band-communicator I'm on.
    logger.info(f"TOD-rank {mpi_info.tod.rank:4} (on machine {mpi_info.processor_name}), dedicated to detector {my_detector_id:4}, with local rank {mpi_info.det.rank:4} (local communicator size: {mpi_info.det.size:4}).")

    # Creating "tod_band_masters", an array which maps the band index to the rank of the master of that band.
    my_band_identifier = f"{my_experiment_name}$$${my_band_name}"
    data_world = (my_band_identifier, mpi_info.world.rank) if mpi_info.band.is_master else None
    data_tod = (my_band_identifier, mpi_info.tod.rank) if mpi_info.band.is_master else None
    all_data_world = mpi_info.tod.comm.allgather(data_world)
    all_data_tod = mpi_info.tod.comm.allgather(data_tod)
    world_band_masters_dict = {item[0]: item[1] for item in all_data_world if item is not None}
    tod_band_masters_dict = {item[0]: item[1] for item in all_data_tod if item is not None}
    logger.info(f"world_band_masters_dict: {world_band_masters_dict}")
    logger.info(f"tod_band_masters_dict: {tod_band_masters_dict}")
    logger.info(f"TOD: Rank {mpi_info.tod.rank:4} assigned scans {my_scans_start:6} - {my_scans_stop:6} on "
                f"band {my_band_id:4}, det{my_detector_id:4}.")
    
    mpi_info['world']['tod_band_masters'] = world_band_masters_dict
    mpi_info['tod']['tod_band_masters'] = tod_band_masters_dict
    t0 = time.time()

    experiment_data = read_tods_from_file(my_experiment, my_band, my_det, params,
                                               my_detector_id, my_scans_start, my_scans_stop)

    mpi_info.tod.comm.Barrier()
    if mpi_info.tod.is_master:
        logger.info(f"TOD: Finished reading all files in {time.time()-t0:.1f}s.")

    num_scans = len(experiment_data.scans)
    scansample_list = []
    for iscan in range(num_scans):
        scansample_list.append(ScanSamples())
        scansample_list[-1].time_dep_rel_gain_est = 0.0
        scansample_list[-1].rel_gain_est = my_det.rel_gain_est - params.initial_g0
        scansample_list[-1].gain_est = params.initial_g0 + my_det.rel_gain_est
    det_samples = DetectorSamples(scansample_list)
    det_samples.detector_id = my_detector_id
    det_samples.g0_est = params.initial_g0
    det_samples.detname = my_det.name

    return mpi_info, my_band_identifier, experiment_data, det_samples


def estimate_white_noise(experiment_data: DetectorTOD, detector_samples: DetectorSamples, det_compsep_map: NDArray, params: Bunch) -> DetectorTOD:
    """Estimate the white noise level in the TOD data, add it to the scans, and return the updated experiment data.
    Input:
        experiment_data (DetectorTOD): The experiment TOD object.
        params (Bunch): The parameters from the input parameter file.
    Output:
        experiment_data (DetectorTOD): The experiment TOD with the estimated white noise level added to each scan.
    """
    for scan, scan_samples in zip(experiment_data.scans, detector_samples.scans):
        pix = scan.pix
        psi = scan.psi
        # Should maybe n_corr be subtracted here as well?
        sky_subtracted_tod = scan.tod.copy()
        sky_subtracted_tod -= scan_samples.gain_est*get_static_sky_TOD(det_compsep_map, pix, psi)
        sky_subtracted_tod -= scan_samples.gain_est*get_s_orb_TOD(scan, experiment_data, pix)
        mask = scan.processing_mask_TOD
        sigma0 = calculate_sigma0(sky_subtracted_tod, mask)
        scan_samples.sigma0 = float(sigma0/scan_samples.gain_est)
    return detector_samples



def sample_noise(band_comm: MPI.Comm, experiment_data: DetectorTOD,
                 detector_samples: DetectorSamples, det_compsep_map: NDArray, chain, iteration) -> DetectorTOD:
    logger = logging.getLogger(__name__)
    num_failed_convergence = 0
    worst_residual = 0.0
    alphas = []
    fknees = []
    mapmaker = MapmakerIQU(band_comm, experiment_data.nside)
    for scan, scansamples in zip(experiment_data.scans, detector_samples.scans):
        f_samp = scan.fsamp
        # raw_tod = scan.tod
        pix = scan.pix
        psi = scan.psi

        s_tot = get_s_orb_TOD(scan, experiment_data, pix)

        s_tot += get_static_sky_TOD(det_compsep_map, pix, psi)

        sky_subtracted_TOD = scan.tod.copy()
        sky_subtracted_TOD -= scansamples.gain_est*s_tot
        Ntod = sky_subtracted_TOD.shape[0]
        Nfft = Ntod//2 + 1
        freq = rfftfreq(Ntod, d = 1/f_samp)
        fknee = scansamples.fknee_est
        alpha = scansamples.alpha_est
        mask = scan.processing_mask_TOD
        sigma0 = calculate_sigma0(sky_subtracted_TOD, mask)
        C_1f_inv = np.zeros(Nfft)
        C_1f_inv[1:] = 1.0 / (sigma0**2*(freq[1:]/fknee)**alpha)
        # C_1f_inv[0] = C_1f_inv[-1]  # Test: try and constrain DC mode somewhat.
        err_tol = 1e-8
        n_corr_est, residual = corr_noise_realization_with_gaps(sky_subtracted_TOD,
                                                                mask, sigma0, C_1f_inv,
                                                                err_tol=err_tol)
        inv_var = 1.0/scansamples.sigma0**2
        mapmaker.accumulate_to_map((n_corr_est/scansamples.gain_est).astype(np.float32), inv_var, pix, psi)
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

        if band_comm.Get_rank() == 91 and chain == 1:
            if int(scan.scanID)%21 == 0:
                np.save(f"/mn/stornext/d23/cmbco/jonas/c4_testing/misc_data_output2/n_corr_{scan.scanID}_iter{iteration}.npy", n_corr_est)
                np.save(f"/mn/stornext/d23/cmbco/jonas/c4_testing/misc_data_output2/mask_{scan.scanID}_iter{iteration}.npy", mask)
                np.save(f"/mn/stornext/d23/cmbco/jonas/c4_testing/misc_data_output2/n_corr_params_{scan.scanID}_iter{iteration}.npy", np.array([fknee, alpha]))

    t0 = time.time()
    band_comm.Barrier()
    wait_time = time.time() - t0
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
        logger.info(f"{experiment_data.nu}GHz: fknees {np.min(fknees):.4f} {np.percentile(fknees, 1):.4f} {np.mean(fknees):.4f} {np.percentile(fknees, 99):.4f} {np.max(fknees):.4f}")
        logger.info(f"{experiment_data.nu}GHz: alphas {np.min(alphas):.4f} {np.percentile(alphas, 1):.4f} {np.mean(alphas):.4f} {np.percentile(alphas, 99):.4f} {np.max(alphas):.4f}")
    # t_finalize = time.perf_counter() - tx

    # t_Huffman = band_comm.reduce(t_Huffman, op=MPI.SUM, root=0)
    # t_orb = band_comm.reduce(t_orb, op=MPI.SUM, root=0)
    # t_sky = band_comm.reduce(t_sky, op=MPI.SUM, root=0)
    # t_proj = band_comm.reduce(t_proj, op=MPI.SUM, root=0)
    # t_cg = band_comm.reduce(t_cg, op=MPI.SUM, root=0)
    # t_PS = band_comm.reduce(t_PS, op=MPI.SUM, root=0)
    # t_mapmaker = band_comm.reduce(t_mapmaker, op=MPI.SUM, root=0)
    # if band_comm.Get_rank() == 0:
    #     logger.info(f"### Noise-samp : t_huffman :      {t_Huffman/band_comm.Get_size():.4f} s.")
    #     logger.info(f"### Noise-samp : t_orb :          {t_orb/band_comm.Get_size():.4f} s.")
    #     logger.info(f"### Noise-samp : t_sky :          {t_sky/band_comm.Get_size():.4f} s.")
    #     logger.info(f"### Noise-samp : t_proj :          {t_proj/band_comm.Get_size():.4f} s.")
    #     logger.info(f"### Noise-samp : t_cg :           {t_cg/band_comm.Get_size():.4f} s.")
    #     logger.info(f"### Noise-samp : t_PS :           {t_PS/band_comm.Get_size():.4f} s.")
    #     logger.info(f"### Noise-samp : t_mapmaker :     {t_mapmaker/band_comm.Get_size():.4f} s.")
    #     logger.info(f"### Noise-samp : t_finalization : {t_finalize:.4f} s.")

    return detector_samples, mapmaker, wait_time



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
        f_samp = scan.fsamp
        down_factor = int(f_samp)
        indices_edges = np.arange(0, scan.ntod, down_factor)
        indices_centers = (indices_edges[1:] + indices_edges[:-1])//2
        ntod_down = indices_centers.size

        assert((ntod_down+1)*down_factor >= scan.tod.shape[0])

        pix = scan.pix  # Only decompressing pix once for efficiency.
        psi = scan.psi
        pix = pix[indices_centers]
        psi = psi[indices_centers]

        s_orb = get_s_orb_TOD(scan, experiment_data, pix)
        sky_model_TOD = get_static_sky_TOD(det_compsep_map, pix, psi)

        residual_tod = scan.tod[:ntod_down*down_factor].reshape((ntod_down, down_factor))
        residual_tod = np.mean(residual_tod, axis=-1)
        residual_tod -= scan_samples.gain_est*sky_model_TOD  # Subtracting sky signals.
        residual_tod -= scan_samples.gain_est*s_orb 
        mask = scan.processing_mask_TOD[indices_centers]
        sigma0 = calculate_sigma0(residual_tod, mask)
        residual_tod += detector_samples.g0_est*s_orb  # Now we can add back in the orbital dipole.

        Ntod = residual_tod.shape[0]
        Nrfft = Ntod//2+1
        freqs = rfftfreq(Ntod, 1.0)
        # freqs = rfftfreq(Ntod, 1.0/scan.fsamp)
        inv_power_spectrum = np.zeros(Nrfft)
        inv_power_spectrum[1:] = 1.0/(sigma0**2*(1 + (freqs[1:]/scan_samples.fknee_est)**scan_samples.alpha_est))

        ### Solving Equation 16 from BP7 ###
        mask = scan.processing_mask_TOD[indices_centers]
        # In the masked regions, inpaint the orbital dipole times the absolute gain.
        residual_tod[~mask] = detector_samples.g0_est*s_orb[~mask] + np.random.normal(0, sigma0, s_orb[~mask].shape)
        
        s_fft = forward_rfft(s_orb)
        d_fft = forward_rfft(residual_tod)
        N_inv_s_fft = s_fft * inv_power_spectrum
        N_inv_d_fft = d_fft * inv_power_spectrum
        N_inv_s = backward_rfft(N_inv_s_fft, Ntod)
        N_inv_d = backward_rfft(N_inv_d_fft, Ntod)
        
        # We now exclude the time-samples hitting the masked area. We don't want to do this before now, because it would mess up the FFT stuff.

        # mask = experiment_data.processing_mask_map[scan.pix]
        # sum_s_T_N_inv_d += np.dot(s_orb[mask], N_inv_d[mask])  # Add to the numerator and denominator.
        # sum_s_T_N_inv_s += np.dot(s_orb[mask], N_inv_s[mask])
        sum_s_T_N_inv_d += np.dot(s_orb, N_inv_d)
        sum_s_T_N_inv_s += np.dot(s_orb, N_inv_s)

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

    # As of Numpy 2.0 it's good practice to explicitly cast to Python scalar types, as this would
    # otherwise have been a np.float64 type, potentially causing unexpected casting behavior later.
    detector_samples.g0_est = float(g_sampled)
    # t_finalization += time.perf_counter() - t0
    # t_huffman = TOD_comm.reduce(t_huffman, op=MPI.SUM, root=0)
    # t_orb = TOD_comm.reduce(t_orb, op=MPI.SUM, root=0)
    # t_sky = TOD_comm.reduce(t_sky, op=MPI.SUM, root=0)
    # t_res = TOD_comm.reduce(t_res, op=MPI.SUM, root=0)
    # t_inpaint = TOD_comm.reduce(t_inpaint, op=MPI.SUM, root=0)
    # t_ffts = TOD_comm.reduce(t_ffts, op=MPI.SUM, root=0)
    # t_dot = TOD_comm.reduce(t_dot, op=MPI.SUM, root=0)
    # if TOD_comm.Get_rank() == 0:
    #     logger.info(f"### Absgain : t_huffman : {t_huffman/TOD_comm.Get_size():.4f} s.")
    #     logger.info(f"### Absgain : t_orb : {t_orb/TOD_comm.Get_size():.4f} s.")
    #     logger.info(f"### Absgain : t_sky : {t_sky/TOD_comm.Get_size():.4f} s.")
    #     logger.info(f"### Absgain : t_res : {t_res/TOD_comm.Get_size():.4f} s.")
    #     logger.info(f"### Absgain : t_inpaint : {t_inpaint/TOD_comm.Get_size():.4f} s.")
    #     logger.info(f"### Absgain : t_ffts : {t_ffts/TOD_comm.Get_size():.4f} s.")
    #     logger.info(f"### Absgain : t_dot : {t_dot/TOD_comm.Get_size():.4f} s.")
    #     logger.info(f"### Absgain : t_finalization : {t_finalization:.4f} s.")

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
        f_samp = scan.fsamp
        down_factor = int(f_samp)
        indices_edges = np.arange(0, scan.ntod, down_factor)
        indices_centers = (indices_edges[1:] + indices_edges[:-1])//2
        ntod_down = indices_centers.size

        # Define the residual for this sampling step, as per Eq. (17)
        pix = scan.pix
        psi = scan.psi
        pix = pix[indices_centers]
        psi = psi[indices_centers]

        s_tot = get_static_sky_TOD(det_compsep_map, pix, psi)

        s_tot += get_s_orb_TOD(scan, experiment_data, pix)

        # residual_tod = scan.tod.copy()
        gain = detector_samples.g0_est + scan_samples.time_dep_rel_gain_est
        residual_tod = scan.tod[:ntod_down*down_factor].reshape((ntod_down, down_factor))
        residual_tod = np.mean(residual_tod, axis=-1)
        residual_tod -= gain*s_tot
        mask = scan.processing_mask_TOD[indices_centers]
        sigma0 = calculate_sigma0(residual_tod, mask)

        # Setup FFT-based calculation for N^-1 operations
        Ntod = residual_tod.shape[0]
        Nrfft = Ntod // 2 + 1
        # sigma0 = scan_samples.gain_est*scan_samples.sigma0  # scan.sigma0 is in tempearture units.
        # freqs = rfftfreq(Ntod, 1.0 / scan.fsamp)
        freqs = rfftfreq(Ntod, 1.0)
        inv_power_spectrum = np.zeros(Nrfft)
        inv_power_spectrum[1:] = 1.0 / (sigma0**2 * (1 + (freqs[1:] / scan_samples.fknee_est)**scan_samples.alpha_est))

        s_fft = forward_rfft(s_tot)
        N_inv_s_fft = s_fft * inv_power_spectrum
        N_inv_s = backward_rfft(N_inv_s_fft, Ntod)
        
        # mask = experiment_data.processing_mask_map[scan.pix]
        # Inpaint on the masked regions the sky signal times only the detector-residual gain.
        residual_tod[~mask] = scan_samples.rel_gain_est*s_tot[~mask] + np.random.normal(0, sigma0, s_tot[~mask].shape)
        s_T_N_inv_s_scan = np.dot(s_tot, N_inv_s)
        r_T_N_inv_s_scan = np.dot(residual_tod, N_inv_s)
        # s_T_N_inv_s_scan = np.dot(s_tot[mask], N_inv_s[mask])
        # r_T_N_inv_s_scan = np.dot(residual_tod[mask], N_inv_s[mask])

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
                scan_samples.rel_gain_est = float(my_delta_g)
            if det_comm.Get_rank() == 0:
                logging.info(f"Relative gain for detector {experiment_data.detector_name} ({experiment_data.nu}GHz): {detector_samples.g0_est*1e9+my_delta_g*1e9:8.3f} ({detector_samples.g0_est*1e9:8.3f} + {my_delta_g*1e9:8.3f})")
        except ValueError:
            logger.error(f"Rank {global_rank} with detector {my_did} not found in solved gain list.")
    else:
        logger.warning(f"No valid relative gain solution was broadcast. Not updating gains on rank {global_rank}.")

    # t_huffman = TOD_comm.reduce(t_huffman, op=MPI.SUM, root=0)
    # t_orb = TOD_comm.reduce(t_orb, op=MPI.SUM, root=0)
    # t_sky = TOD_comm.reduce(t_sky, op=MPI.SUM, root=0)
    # t_res = TOD_comm.reduce(t_res, op=MPI.SUM, root=0)
    # t_inpaint = TOD_comm.reduce(t_inpaint, op=MPI.SUM, root=0)
    # t_ffts = TOD_comm.reduce(t_ffts, op=MPI.SUM, root=0)
    # if TOD_comm.Get_rank() == 0:
    #     logger.info(f"### relgain : t_huffman : {t_huffman/TOD_comm.Get_size():.4f} s.")
    #     logger.info(f"### relgain : t_orb : {t_orb/TOD_comm.Get_size():.4f} s.")
    #     logger.info(f"### relgain : t_sky : {t_sky/TOD_comm.Get_size():.4f} s.")
    #     logger.info(f"### relgain : t_res : {t_res/TOD_comm.Get_size():.4f} s.")
    #     logger.info(f"### relgain : t_inpaint : {t_inpaint/TOD_comm.Get_size():.4f} s.")
    #     logger.info(f"### relgain : t_ffts : {t_ffts/TOD_comm.Get_size():.4f} s.")
    #     logger.info(f"### relgain : t_finalization : {t_finalization:.4f} s.")

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

        f_samp = scan.fsamp
        down_factor = int(f_samp)
        indices_edges = np.arange(0, scan.ntod, down_factor)
        indices_centers = (indices_edges[1:] + indices_edges[:-1])//2
        ntod_down = indices_centers.size

        # Per Eq. (26), the residual is d - (g0 + Delta_g)*s
        pix = scan.pix
        psi = scan.psi
        pix = pix[indices_centers]
        psi = psi[indices_centers]

        s_tot = get_static_sky_TOD(det_compsep_map, pix, psi)
        s_tot += get_s_orb_TOD(scan, experiment_data, pix)


        residual_tod = scan.tod[:ntod_down*down_factor].reshape((ntod_down, down_factor))
        residual_tod = np.mean(residual_tod, axis=-1)
        gain = detector_samples.g0_est + scan_samples.rel_gain_est
        residual_tod -= gain*s_tot

        mask = scan.processing_mask_TOD[indices_centers]
        sigma0 = calculate_sigma0(residual_tod, mask)

        # FFT-based N^-1 operation setup
        Ntod = residual_tod.shape[0]
        Nrfft = Ntod // 2 + 1
        # freqs = rfftfreq(Ntod, 1.0 / scan.fsamp)
        freqs = rfftfreq(Ntod, 1.0)
        sigma0 = scan_samples.gain_est*scan_samples.sigma0
        inv_power_spectrum = np.zeros(Nrfft)
        inv_power_spectrum[1:] = 1.0 / (sigma0**2 * (1 + (freqs[1:] / scan_samples.fknee_est)**scan_samples.alpha_est))

        # s_tot[~mask] = 0.0  #TODO: Needs to be handled differently.
        # residual_tod[~mask] = 0.0
        # s_tot[~mask] = scan_samples.n_corr_est[~mask]/scan_samples.gain_est
        # residual_tod[~mask] = scan_samples.n_corr_est[~mask]

        # In the masked regions, inpaint the total sky model times only the temporal gain estimate.
        residual_tod[~mask] = scan_samples.time_dep_rel_gain_est*s_tot[~mask] + np.random.normal(0, sigma0, s_tot[~mask].shape)

        # Calculate N^-1 * s_tot and N^-1 * residual_tod
        N_inv_s = backward_rfft(forward_rfft(s_tot) * inv_power_spectrum, Ntod)
        N_inv_r = backward_rfft(forward_rfft(residual_tod) * inv_power_spectrum, Ntod)
        
        # Calculate elements for the linear system
        A_qq = np.dot(s_tot, N_inv_s)
        b_q = np.dot(s_tot, N_inv_r) # v2 = removed the masks from this step.
        
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
                g_inv_v = backward_rfft(forward_rfft(v) * prior_ps_inv, n_scans_total).real
                diag_v = A_diag * v
                return g_inv_v + diag_v

            # Construct RHS of the sampling equation (Eq. 30)
            eta1 = np.random.randn(n_scans_total)
            fluctuation1 = np.sqrt(np.maximum(A_diag, 0)) * eta1

            eta2 = np.random.randn(n_scans_total)
            fluctuation2 = backward_rfft(forward_rfft(eta2) * prior_ps_inv_sqrt, n_scans_total).real

            # logger.info(f"DATA TERM (A_diag) MEAN: {np.mean(A_diag)}")
            # logger.info(f"PRIOR TERM (G^-1) MEAN: {np.mean(prior_ps_inv)}")
            # logger.info(f"RHS = {b} + {fluctuation1} + {fluctuation2}")
            # logger.info(f"RHS (means) = {np.nanmean(b)} + {np.nanmean(fluctuation1)} + {np.nanmean(fluctuation2)}")
            RHS = b + fluctuation1 + fluctuation2

            ### Simpler sanity check solution  ##
            epsilon = 1e-12
            # The mean value is simply b / A
            g_mean = b / (A_diag + epsilon)
            # The standard deviation is 1 / sqrt(A)
            g_std = 1.0 / np.sqrt(np.maximum(A_diag, 0) + epsilon)
            # logger.info(f"Sanity check solution A, b = {A_diag} {b}")
            # logger.info(f"Sanity check solution: {g_mean} {g_std}")
            # logger.info(f"Sanity check solution(means): {np.mean(g_mean)} {np.mean(g_std)}")

            from pixell import utils
            CG_solver = utils.CG(matvec, RHS, x0=g_mean)
            for i in range(200):
                CG_solver.step()
                if CG_solver.err < 1e-10:
                    break
            # logging.info(f"CG solver for gain fluctuations (nu={experiment_data.nu})"
            #             f"finished after {i} iterations (residual = {CG_solver.err})")
            delta_g_sample = CG_solver.x
            logger.info(f"delta_g_sample mean = {np.mean(delta_g_sample)}")
            delta_g_sample -= np.mean(delta_g_sample)
            logger.info(f"delta_g: {delta_g_sample}")
            logger.info(f"Band {experiment_data.nu}GHz time-dependent gain: min={np.min(delta_g_sample)*1e9:14.4f} mean={np.mean(delta_g_sample)*1e9:14.4f} std={np.std(delta_g_sample)*1e9:14.4f} max={np.max(delta_g_sample)*1e9:14.4f}")

            if True: #debug stuff
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10,8))
                other_gain = detector_samples.g0_est + detector_samples.scans[0].rel_gain_est
                plt.plot(1e9*(other_gain + delta_g_sample))
                # plt.ylim(0.85*1e9*other_gain, 1.15*1e9*other_gain)
                plt.ylim(0, np.max(1e9*(other_gain + delta_g_sample)))
                plt.xlabel("PID")
                plt.ylabel("Gain [mV/K]")
                plt.xticks([0, 15000, 30000, 45000])
                plt.savefig(f"{params.output_paths.plots}chain{chain}_iter{iter}_{detector_samples.detname}.png")
                plt.close()
                if chain == 1:
                    np.save(f"gain_temp_iter{iter}_{detector_samples.detname}.npy", 1e9*(other_gain + delta_g_sample))
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
            scan_samples.time_dep_rel_gain_est = delta_g_local[i].astype(np.float32)
    else:
        logger.warning(f"Rank {band_rank} received mismatched number of gain samples."
                       f"Expected {len(experiment_data.scans)}, got {delta_g_local.size}.")

    # t_finalization = time.perf_counter() - t0
    # t_huffman = det_comm.reduce(t_huffman, op=MPI.SUM, root=0)
    # t_orb = det_comm.reduce(t_orb, op=MPI.SUM, root=0)
    # t_sky = det_comm.reduce(t_sky, op=MPI.SUM, root=0)
    # t_res = det_comm.reduce(t_res, op=MPI.SUM, root=0)
    # t_inpaint = det_comm.reduce(t_inpaint, op=MPI.SUM, root=0)
    # t_ffts = det_comm.reduce(t_ffts, op=MPI.SUM, root=0)
    # t_dot = det_comm.reduce(t_dot, op=MPI.SUM, root=0)
    # if det_comm.Get_rank() == 0:
    #     logger.info(f"### tempgain {experiment_data.nu:.1f} : t_huffman : {t_huffman/det_comm.Get_size():.4f} s.")
    #     logger.info(f"### tempgain {experiment_data.nu:.1f} : t_orb : {t_orb/det_comm.Get_size():.4f} s.")
    #     logger.info(f"### tempgain {experiment_data.nu:.1f} : t_sky : {t_sky/det_comm.Get_size():.4f} s.")
    #     logger.info(f"### tempgain {experiment_data.nu:.1f} : t_res : {t_res/det_comm.Get_size():.4f} s.")
    #     logger.info(f"### tempgain {experiment_data.nu:.1f} : t_inpaint : {t_inpaint/det_comm.Get_size():.4f} s.")
    #     logger.info(f"### tempgain {experiment_data.nu:.1f} : t_ffts : {t_ffts/det_comm.Get_size():.4f} s.")
    #     logger.info(f"### tempgain {experiment_data.nu:.1f} : t_dot : {t_dot/det_comm.Get_size():.4f} s.")
    #     logger.info(f"### tempgain {experiment_data.nu:.1f} : t_finalization : {t_finalization:.4f} s.")

    return detector_samples



def process_tod(mpi_info: Bunch, experiment_data: DetectorTOD,
                detector_samples, compsep_output: NDArray,
                params: Bunch, chain, iter) -> DetectorMap:
    """ Performs a single TOD iteration.

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
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
    
    timing_dict = {}
    waittime_dict = {}

    det_comm = mpi_info.det.comm
    band_comm = mpi_info.band.comm
    TOD_comm = mpi_info.tod.comm
    ### WHITE NOISE ESTIMATION ###
    t0 = time.time()
    detector_samples = estimate_white_noise(experiment_data, detector_samples, compsep_output, params)
    timing_dict["wn-est-1"] = time.time() - t0
    if mpi_info.tod.is_master:
        logger.info(f"Chain {chain} iter{iter} {experiment_data.nu}GHz: Finished white noise estimation in {timing_dict['wn-est-1']:.1f}s.")

    ### ABSOLUTE GAIN CALIBRATION ### 
    if params.sample_abs_gain and iter >= params.sample_abs_gain_from_iter_num:
        t0 = time.time()
        detector_samples, wait_time = sample_absolute_gain(TOD_comm, experiment_data, detector_samples, compsep_output)
        timing_dict["abs-gain"] = time.time() - t0
        waittime_dict["abs-gain"] = wait_time
        if mpi_info.band.is_master:
            logger.info(f"Chain {chain} iter{iter} {experiment_data.nu}GHz: Finished absolute gain estimation in {timing_dict['abs-gain']:.1f}s.")

    ### RELATIVE GAIN CALIBRATION ### 
    if params.sample_rel_gain and iter >= params.sample_rel_gain_from_iter_num:
        t0 = time.time()
        detector_samples, wait_time = sample_relative_gain(TOD_comm, det_comm, experiment_data, detector_samples, compsep_output)
        timing_dict["rel-gain"] = time.time() - t0
        waittime_dict["rel-gain"] = wait_time
        if mpi_info.band.is_master:
            logger.info(f"Chain {chain} iter{iter} {experiment_data.nu}GHz: Finished relative gain estimation in {timing_dict['rel-gain']:.1f}s.")


    ### TEMPORAL GAIN CALIBRATION ### 
    if params.sample_temporal_gain and iter >= params.sample_temporal_gain_from_iter_num:
        t0 = time.time()
        detector_samples = sample_temporal_gain_variations(det_comm, experiment_data, detector_samples, compsep_output, chain, iter, params)
        timing_dict["temp-gain"] = time.time() - t0
        if mpi_info.band.is_master:
            logger.info(f"Chain {chain} iter{iter} {experiment_data.nu}GHz: Finished temporal gain estimation in {timing_dict['temp-gain']:.1f}s.")

        ### Update total gain from sum of all three gain terms. ###
        for scan_samples in detector_samples.scans:
            scan_samples.gain_est = detector_samples.g0_est + scan_samples.rel_gain_est\
                                  + scan_samples.time_dep_rel_gain_est

    ### WHITE NOISE ESTIMATION ###
    t0 = time.time()
    detector_samples = estimate_white_noise(experiment_data, detector_samples, compsep_output, params)
    timing_dict["wn-est-2"] = time.time() - t0
    if band_comm.Get_rank() == 0:
        logger.info(f"Chain {chain} iter{iter} {experiment_data.nu}GHz: Finished white noise estimation in {timing_dict['wn-est-2']:.1f}s.")

    if iter >= params.sample_corr_noise_from_iter_num:
        ### CORRELATED NOISE SAMPLING ###
        t0 = time.time()
        detector_samples, mapmaker_corrnoise, wait_time = sample_noise(band_comm, experiment_data, detector_samples, compsep_output, chain, iter)
        timing_dict["corr-noise"] = time.time() - t0
        waittime_dict["corr-noise"] = wait_time
        if mpi_info.band.is_master:
            logger.info(f"Chain {chain} iter{iter} {experiment_data.nu}GHz: Finished correlated noise sampling in {timing_dict['corr-noise']:.1f}s.")
    else:
        mapmaker_corrnoise = None

    ### MAPMAKING ###
    t0 = time.time()
    detmap = tod2map(band_comm, experiment_data, compsep_output, detector_samples, params,
                     mapmaker_corrnoise)
    timing_dict["mapmaker"] = time.time() - t0
    if band_comm.Get_rank() == 0:
        logger.info(f"Chain {chain} iter{iter} {experiment_data.nu}GHz: Finished mapmaking in {timing_dict['mapmaker']:.1f}s.")

    t0 = time.time()
    TOD_comm.Barrier()
    waittime_dict["end-barrier"] = time.time() - t0

    for key in timing_dict:
        timing_dict[key] = band_comm.reduce(timing_dict[key], op=MPI.SUM, root=0)
    for key in waittime_dict:
        waittime_dict[key] = band_comm.reduce(waittime_dict[key], op=MPI.SUM, root=0)
    
    if mpi_info.band.is_master:
        for key in timing_dict:
            timing_dict[key] /= band_comm.Get_size()
            logger.info(f"Average time spent for {experiment_data.nu}GHz on {key} = {timing_dict[key]:.1f}s.")

        for key in waittime_dict:
            waittime_dict[key] /= band_comm.Get_size()
            logger.info(f"Average wait overhead for {experiment_data.nu}GHz on {key} = {waittime_dict[key]:.1f}s.")

    return detmap, detector_samples
