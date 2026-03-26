import numpy as np
import pixell
from pixell import utils
from mpi4py import MPI
import logging
from scipy.fft import rfftfreq
import time
from numpy.typing import NDArray

import healpy as hp
import matplotlib.pyplot as plt
from pixell.bunch import Bunch
from astropy.io import fits
import pysm3.units as pysm3_u

from commander4.data_models.detector_map import DetectorMap
from commander4.data_models.detector_group_TOD import DetGroupTOD
from commander4.data_models.TOD_samples import TODSamples
from commander4.utils.mapmaker import MapmakerIQU, WeightsMapmakerIQU, WeightsMapmaker, Mapmaker
from commander4.utils.CG_mapmaker import CGMapmakerI, CGMapmakerIQU
from commander4.solvers.preconditioners import InvNPreconditionerI, InvNPreconditionerIQU
from commander4.noise_sampling import corr_noise_realization_with_gaps, sample_noise_PS_params, fill_all_masked
from commander4.utils.map_utils import get_static_sky_TOD, get_s_orb_TOD
from commander4.utils.math_operations import forward_rfft, backward_rfft, calculate_sigma0
from commander4.tod_reader import read_tods_from_file
from commander4.output.write_chains_files import write_tod_chain_to_file, write_map_chain_to_file
from commander4.logging.performance_logger import benchmark, bench_summary, start_bench,\
                                            stop_bench, log_memory, increment_count, bench_reset


def get_initial_sky(experiment_data: DetGroupTOD) -> NDArray[np.float32]:
    """ Returns a sky realization from a set of components. The set of components are listed in
        the provided DetGroupTOD object, originally specified in the parameter file. 
    """
    initial_sky = np.zeros((3, 12*experiment_data.nside**2), dtype=np.float32)
    for skyfile in experiment_data.sky_init_files:
        with fits.open(skyfile) as hdul:
            fields = ["TEMPERATURE", "Q_POLARISATION", "U_POLARISATION"]
            for i, field in enumerate(fields):
                data = hdul[1].data[field].flatten()
                nside = hp.npix2nside(data.size)
                if nside != experiment_data.nside:
                    data = hp.ud_grade(data, experiment_data.nside)
                initial_sky[i] += data

    # Convert from uK_CMB to uK_RJ
    initial_sky *= (1*pysm3_u.uK_CMB).to(pysm3_u.uK_RJ,
                    equivalencies=pysm3_u.cmb_equivalencies(experiment_data.nu*pysm3_u.GHz)).value
    return initial_sky

def called_on_non_master(arr):
    logger = logging.getLogger(__name__)
    logger.debug("Dummy precond has been called")
    return np.copy(arr)

def tod2map_CG(band_comm: MPI.Comm, experiment_data: DetGroupTOD, compsep_output: NDArray,
            detector_samples: TODSamples, params: Bunch, chain: int, iter: int,
            do_ncorr_sampling: bool) -> dict[str, DetectorMap]:
    """ Commander4 CG mapmaking. All ranks on the provided MPI communicator collaborates on creating
        the band maps (sky signal, inverse variance, possibly also aux maps like orbital dipole).
    Args:
        band_comm (Comm): The communicator consisting of all MPI ranks which holds TOD data that
                          should go into the same map.
        experiment_data (DetGroupTOD): TOD data class to be made into maps.
        compsep_output (NDArray): The sky model at our band. Not used, but written to chain file.
        detector_samples (TODSamples): Sampled TOD parameters, such as gain.
        params (Bunch): Parameter file as 'Param' object.
        chain (int): Current chain number.
        iter (int): Current Gibbs iteration.
        do_ncorr_sampling (bool): Perform correlated noise sampling or not.
    Output:
        dict[str, DetectorMap]: Dictionary containing the solved detector maps, keyed by
            polarization component ('I', 'QU').

    """
    logger = logging.getLogger(__name__)
    ismaster = band_comm.Get_rank() == 0
    ### CG MAPMAKER ###
    # We separate the inverse-variance mapmaking from the other 3 mapmakers.
    # This is purely to reduce the maximum concurrent memory requirement, and is slightly slower
    # as we have to de-compress pix and psi twice.
    pols = experiment_data.pols
    if pols == "IQU":
        mapmaker_invvar = WeightsMapmakerIQU(band_comm, experiment_data.nside)
        for scan, scan_samples in zip(experiment_data.scans, detector_samples.scans):
            pix = scan.pix
            psi = scan.psi
            inv_var = 1.0/scan_samples.sigma0**2
            mapmaker_invvar.accumulate_to_map(inv_var, pix, psi)
        mapmaker_invvar.gather_map()
        mapmaker_invvar.normalize_map()
        if ismaster:
            precond = InvNPreconditionerIQU(mapmaker_invvar.final_rms_map**2)
        else:
            precond = called_on_non_master
        cg_mapmaker = CGMapmakerIQU(experiment_data, detector_samples, band_comm,
                    preconditioner=precond, nthreads=params.general.nthreads_tod, 
                    CG_maxiter=params.general.CG_mapmaker.maxiter)
    elif pols == "I":
        mapmaker_invvar = WeightsMapmaker(band_comm, experiment_data.nside)
        for scan, scan_samples in zip(experiment_data.scans, detector_samples.scans):
            pix = scan.pix
            inv_var = 1.0/scan_samples.sigma0**2
            mapmaker_invvar.accumulate_to_map(inv_var, pix)
        mapmaker_invvar.gather_map()
        if ismaster:
            precond = InvNPreconditionerI(utils.without_nan(1./mapmaker_invvar.final_map))
        else:
            precond = called_on_non_master
        cg_mapmaker = CGMapmakerI(experiment_data, detector_samples, band_comm,
                    preconditioner=precond, nthreads=params.general.nthreads_tod, 
                    CG_maxiter=params.general.CG_mapmaker.maxiter)
    else:
        raise ValueError(f"specified polarizations {pols} is notsupported yet.")

    BinMapmaker = MapmakerIQU if pols == "IQU" else Mapmaker #general bin mapmaker class object.
    # mapmaker = BinMapmaker(band_comm, experiment_data.nside)
    mapmaker_orbdipole = BinMapmaker(band_comm, experiment_data.nside)

    if do_ncorr_sampling:
        mapmaker_ncorr = BinMapmaker(band_comm, experiment_data.nside)
        fknees = []
        alphas = []
        num_failed_convergences_ncorr = 0
        worst_residual_ncorr = 0
    
    ### MAIN SCAN LOOP ###
    for scan, scan_samples in zip(experiment_data.scans, detector_samples.scans):
        d_sky = scan.tod.copy()
        pix = scan.pix
        psi = None if pols=="I" else scan.psi
        inv_var = 1.0/scan_samples.sigma0**2

        ### ORBITAL DIPOLE ###
        sky_orb_dipole = get_s_orb_TOD(scan, experiment_data, pix)
        d_sky -= scan_samples.gain_est*sky_orb_dipole

        ### CORRELATED NOISE SAMPLING ###
        if do_ncorr_sampling:
            s_tot = get_static_sky_TOD(compsep_output, pix, psi=psi)
            s_tot += sky_orb_dipole
            sky_subtracted_TOD = scan.tod.copy()
            sky_subtracted_TOD -= scan_samples.gain_est*s_tot
            Ntod = sky_subtracted_TOD.shape[0]
            Nfft = Ntod + 1  # mirrored FFT: nfft=2*Ntod, n=nfft/2+1=Ntod+1
            freq = rfftfreq(2 * Ntod, d = 1/scan.fsamp)
            fknee = scan_samples.fknee_est
            alpha = scan_samples.alpha_est
            mask = scan.processing_mask_TOD
            sigma0_ncorr = calculate_sigma0(sky_subtracted_TOD, mask)
            C_1f_inv = np.zeros(Nfft)
            C_1f_inv[1:] = 1.0 / (sigma0_ncorr**2*(freq[1:]/fknee)**alpha)
            # fill_all_masked(sky_subtracted_TOD, mask, sigma0_ncorr)
            err_tol = 1e-6
            n_corr_est, residual = corr_noise_realization_with_gaps(sky_subtracted_TOD,
                                                                    mask, sigma0_ncorr, C_1f_inv,
                                                                    err_tol=err_tol)
            mapmaker_ncorr.accumulate_to_map((n_corr_est/scan_samples.gain_est).astype(np.float32),
                                             inv_var, pix, psi)
            if residual > err_tol:
                num_failed_convergences_ncorr += 1
                worst_residual_ncorr = max(worst_residual_ncorr, residual)

            ### CORRELATED NOISE POWER SPECTRUM PARAMETERS SAMPLING ###
            fknee, alpha = sample_noise_PS_params(n_corr_est, sigma0_ncorr, scan.fsamp, alpha,
                                                  freq_max=2.0, n_grid=150, n_burnin=4)
            scan_samples.fknee_est = fknee
            scan_samples.alpha_est = alpha
            alphas.append(alpha)
            fknees.append(fknee)

            d_sky -= n_corr_est

        # mapmaker.accumulate_to_map(d_sky/scan_samples.gain_est, inv_var, pix, psi=psi)
        cg_mapmaker.accum_to_RHS(
                    scan_tod=scan, 
                    scan_samp=scan_samples, 
                    pix=pix,
                    psi=psi,
                    scan_tod_arr=d_sky/scan_samples.gain_est
                    )

    ### PRINT NOISE SAMPLING STATS ###
    if do_ncorr_sampling:
        num_failed_convergences_ncorr = band_comm.reduce(num_failed_convergences_ncorr, op=MPI.SUM)
        worst_residual_ncorr = band_comm.reduce(worst_residual_ncorr, op=MPI.MAX)
        if band_comm.Get_rank() == 0:
            if num_failed_convergences_ncorr > 0:
                logger.info(f"Band {experiment_data.nu}GHz failed noise CG for "\
                            f"{num_failed_convergences_ncorr} scans. "\
                            f"Worst residual = {worst_residual_ncorr:.3e}.")

        alphas = band_comm.gather(alphas, root=0)
        fknees = band_comm.gather(fknees, root=0)
        if band_comm.Get_rank() == 0:
            alphas = np.concatenate(alphas)
            fknees = np.concatenate(fknees)
            logger.info(f"{experiment_data.nu}GHz: fknees {np.min(fknees):.4f} "\
            f"{np.percentile(fknees, 1):.4f} {np.mean(fknees):.4f} {np.percentile(fknees, 99):.4f}"\
            f" {np.max(fknees):.4f}")
            logger.info(f"{experiment_data.nu}GHz: alphas {np.min(alphas):.4f} "\
            f"{np.percentile(alphas, 1):.4f} {np.mean(alphas):.4f} {np.percentile(alphas, 99):.4f}"\
            f" {np.max(alphas):.4f}")
    

    ### GATHER AND NORMALIZE MAPS ###

    if pols == "IQU":
        map_rms = mapmaker_invvar.final_rms_map
        map_cov = mapmaker_invvar.final_cov_map
    else:
        map_cov = mapmaker_invvar.final_map
        map_rms = 1./np.sqrt(map_cov)

    mapmaker_orbdipole.gather_map()
    mapmaker_orbdipole.normalize_map(map_cov)
    map_orbdipole = mapmaker_orbdipole.final_map
    cg_mapmaker.finalize_RHS()
    cg_mapmaker.solve()
    map_signal = cg_mapmaker.solved_map
    
    ########### temporary debug plots
    if ismaster:
        if pols == "IQU":
            plt.figure(figsize=(8.5*3, 5.4))
            npol = 3
            for i in range(npol):
                limup   = np.nanpercentile(cg_mapmaker.RHS_map[i,:], 99)
                limdown = np.nanpercentile(cg_mapmaker.RHS_map[i,:], 1)
                hp.mollview(cg_mapmaker.RHS_map[i,:], cmap='RdBu_r', title='RHS',
                            sub=(1,npol,i+1), min=limdown, max=limup)
            plt.savefig(f"/mn/stornext/u3/leoab/cmdr4_plots/RHS.png")
            plt.close()

            plt.figure(figsize=(8.5*3, 5.4))
            for i in range(npol):
                limup   = np.nanpercentile(precond(cg_mapmaker.RHS_map)[i,:], 99)
                limdown = np.nanpercentile(precond(cg_mapmaker.RHS_map)[i,:], 1)
                hp.mollview(precond(cg_mapmaker.RHS_map)[i,:], cmap='RdBu_r', title='M RHS',
                            sub=(1,npol,i+1), min=limdown, max=limup)
            plt.savefig("/mn/stornext/u3/leoab/cmdr4_plots/M_RHS.png")
            plt.close()
        else:
            plt.figure()
            limup   = np.nanpercentile(cg_mapmaker.RHS_map[0,:], 99)
            limdown = np.nanpercentile(cg_mapmaker.RHS_map[0,:], 1)
            hp.mollview(cg_mapmaker.RHS_map[0,:], cmap='RdBu_r', title='RHS',
                            min=limdown, max=limup)
            plt.savefig(f"/mn/stornext/u3/leoab/cmdr4_plots/RHS.png")
            plt.close()

            plt.figure()
            limup   = np.nanpercentile(precond(cg_mapmaker.RHS_map)[0,:], 99)
            limdown = np.nanpercentile(precond(cg_mapmaker.RHS_map)[0,:], 1)
            hp.mollview(precond(cg_mapmaker.RHS_map)[0,:], cmap='RdBu_r', title='M RHS',
                            min=limdown, max=limup)
            plt.savefig(f"/mn/stornext/u3/leoab/cmdr4_plots/M_RHS.png")
            plt.close()

    #####################

    if do_ncorr_sampling:
        mapmaker_ncorr.gather_map()
        mapmaker_ncorr.normalize_map(map_cov)
        map_corrnoise = mapmaker_ncorr.final_map

    ### FINAL CLEANUP ON MASTER RANK ###
    detmap_dict_out = {}
    if band_comm.Get_rank() == 0:
        #Here we split here between I and QU
        if "I" in pols:
            detmap_I = DetectorMap(map_signal[0,:], map_rms[0,:], experiment_data.nu,
                                experiment_data.fwhm, experiment_data.nside)
            detmap_I.g0 = detector_samples.g0_est
            detmap_I.gain = detector_samples.scans[0].rel_gain_est + detector_samples.g0_est
            detmap_dict_out.update({"I": detmap_I})
        if "QU" in pols:
            detmap_QU = DetectorMap(map_signal[1:3,:], map_rms[1:3,:], experiment_data.nu,
                                experiment_data.fwhm, experiment_data.nside)
            detmap_QU.g0 = detector_samples.g0_est
            detmap_QU.gain = detector_samples.scans[0].rel_gain_est + detector_samples.g0_est
            detmap_dict_out.update({"QU": detmap_QU})

        maps_to_file = {}
        maps_to_file["map_observed_sky"] = map_signal
        maps_to_file["map_rms"] = map_rms
        if params.general.write_orb_dipole_maps_to_chain:
            maps_to_file["map_orbdipole"] = map_orbdipole
        if params.general.write_corr_noise_maps_to_chain and do_ncorr_sampling:
            maps_to_file["map_corrnoise"] = map_corrnoise
        if params.general.write_sky_model_maps_to_chain:
            maps_to_file["map_skymodel"] = compsep_output

        write_map_chain_to_file(params, chain, iter, experiment_data.experiment_name,
                                experiment_data.band_name, maps_to_file)

    return detmap_dict_out #empty on non-master ranks


def tod2map_bin(band_comm: MPI.Comm, experiment_data: DetGroupTOD, compsep_output: NDArray,
            tod_samples: TODSamples, params: Bunch, chain: int, iter: int,
            do_ncorr_sampling: bool) -> dict[str, DetectorMap]:
    """ Commander4 bin mapmaking. All ranks on the provided MPI communicator collaborates on creating
        the band maps (sky signal, inverse variance, possibly also aux maps like orbital dipole).
    Args:
        band_comm (Comm): The communicator consisting of all MPI ranks which holds TOD data that
                          should go into the same map.
        experiment_data (DetGroupTOD): TOD data class to be made into maps.
        compsep_output (NDArray): The sky model at our band. Not used, but written to chain file.
        tod_samples (TODSamples): Sampled TOD parameters, such as gain.
        params (Bunch): Parameter file as 'Param' object.
        chain (int): Current chain number.
        iter (int): Current Gibbs iteration.
        do_ncorr_sampling (bool): Perform correlated noise sampling or not.
    Output:
        dict[str, DetectorMap]: Dictionary containing the solved detector maps, keyed by
            polarization component ('I', 'QU').

    """
    logger = logging.getLogger(__name__)
    ### INVERSE VARIANCE MAPMAKER ###
    # We separate the inverse-variance mapmaking from the other 3 mapmakers.
    # This is purely to reduce the maximum concurrent memory requirement, and is slightly slower
    # as we have to de-compress pix and psi twice.
    start_bench("binned-mapmaker")
    nscans = len(experiment_data.scans)
    ndet = experiment_data.ndet
    pols = experiment_data.pols
    mapmaker_invvar = WeightsMapmakerIQU(band_comm, experiment_data.nside)    
    for iscan, scan in enumerate(experiment_data.scans):
        for idet, det in enumerate(scan.detectors):
            pix = det.pix
            psi = det.psi
            inv_var = 1.0/tod_samples.sigma0_est[idet,iscan]**2
            mapmaker_invvar.accumulate_to_map(inv_var, pix, psi)
    mapmaker_invvar.gather_map()
    mapmaker_invvar.normalize_map()

    mapmaker = MapmakerIQU(band_comm, experiment_data.nside)
    mapmaker_orbdipole = MapmakerIQU(band_comm, experiment_data.nside)
    
    if do_ncorr_sampling:
        mapmaker_ncorr = MapmakerIQU(band_comm, experiment_data.nside)
        fknees = []
        alphas = []
        residuals = []
        niters = []
        num_failed_convergences_ncorr = 0
        num_too_high_var_ncorr = 0
        worst_residual_ncorr = 0
    stop_bench("binned-mapmaker")

    ### MAIN SCAN LOOP ###
    for iscan, scan in enumerate(experiment_data.scans):
        for idet, det in enumerate(scan.detectors):
            start_bench("binned-mapmaker")
            d_sky = det.tod.copy()
            pix = det.pix
            psi = det.psi
            inv_var = 1.0/tod_samples.sigma0_est[idet,iscan]**2
            gain = tod_samples.gain_est[idet,iscan]

            ### ORBITAL DIPOLE ###
            sky_orb_dipole = get_s_orb_TOD(det, experiment_data, pix)
            d_sky -= gain*sky_orb_dipole

            stop_bench("binned-mapmaker", increment_count=False)
            ### CORRELATED NOISE SAMPLING ###
            if do_ncorr_sampling:
                start_bench("ncorr-sampling")
                s_tot = get_static_sky_TOD(compsep_output, pix, psi)
                s_tot += sky_orb_dipole
                sky_subtracted_TOD = det.tod.copy()
                sky_subtracted_TOD -= gain*s_tot
                Ntod = sky_subtracted_TOD.shape[0]
                Nfft = Ntod + 1  # mirrored FFT: nfft=2*Ntod, n=nfft/2+1=Ntod+1
                freq = rfftfreq(2 * Ntod, d = 1/det.fsamp)
                fknee = tod_samples.fknee_est[idet,iscan]
                alpha = tod_samples.alpha_est[idet,iscan]
                mask = det.processing_mask_TOD
                sigma0_ncorr = calculate_sigma0(sky_subtracted_TOD, mask)
                C_1f_inv = np.zeros(Nfft)
                C_1f_inv[1:] = 1.0 / (sigma0_ncorr**2*(freq[1:]/fknee)**alpha)
                err_tol = 1e-6
                # Inpaint masked regions with linear slope + white noise.
                # In the CG solver this is only used to define the starting guess,
                # but if the CG fails it is also used to generate the fallback solution.
                fill_all_masked(sky_subtracted_TOD, mask, sigma0_ncorr)
                n_corr_est, residual, niter = corr_noise_realization_with_gaps(sky_subtracted_TOD,
                                                                    mask, sigma0_ncorr, C_1f_inv,
                                                                    err_tol=err_tol)

                # if band_comm.Get_rank() == 0 and idet == 0 and chain == 1:
                #     if iscan == 300 or iscan == 600 or iscan == 900:
                #         np.save(f"corrdata/mirrorfft_ncorr_signal_{experiment_data.band_name}_{iscan}_{iter}.npy", sky_subtracted_TOD)
                #         np.save(f"corrdata/mirrorfft_ncorr_ncorr_{experiment_data.band_name}_{iscan}_{iter}.npy", n_corr_est)
                #         np.save(f"corrdata/mirrorfft_ncorr_mask_{experiment_data.band_name}_{iscan}_{iter}.npy", mask)
                #         np.save(f"corrdata/mirrorfft_ncorr_C_1f_inv_{experiment_data.band_name}_{iscan}_{iter}.npy", C_1f_inv)
                resid = (sky_subtracted_TOD - n_corr_est) * mask
                var_resid = np.dot(resid, resid)
                var_data = np.dot(sky_subtracted_TOD * mask, sky_subtracted_TOD * mask)
                # If either of the two tests failed, use fallback for n_corr.
                if var_resid > var_data or residual > err_tol:
                    # Direcly solve constrained realization system without a mask.
                    n_corr_est, residual, niter = corr_noise_realization_with_gaps(sky_subtracted_TOD,
                                             np.ones_like(mask, dtype=bool), sigma0_ncorr, C_1f_inv)
                    # if band_comm.Get_rank() == 0 and idet == 0 and chain == 1:
                    #     if iscan == 300 or iscan == 600 or iscan == 900:
                    #         np.save(f"corrdata/mirrorfft_corrected_ncorr_signal_{experiment_data.band_name}_{iscan}_{iter}.npy", sky_subtracted_TOD)
                    #         np.save(f"corrdata/mirrorfft_corrected_ncorr_ncorr_{experiment_data.band_name}_{iscan}_{iter}.npy", n_corr_est)
                mapmaker_ncorr.accumulate_to_map((n_corr_est/gain).astype(np.float32),
                                                  inv_var, pix, psi)
                if residual > err_tol:
                    num_failed_convergences_ncorr += 1
                if var_resid > var_data:
                    num_too_high_var_ncorr += 1
                worst_residual_ncorr = max(worst_residual_ncorr, residual)

                ### CORRELATED NOISE POWER SPECTRUM PARAMETERS SAMPLING ###
                fknee, alpha = sample_noise_PS_params(n_corr_est, sigma0_ncorr, det.fsamp, alpha,
                                                      freq_max=2.0, n_grid=150, n_burnin=4)
                tod_samples.fknee_est[idet,iscan] = fknee
                tod_samples.alpha_est[idet,iscan] = alpha
                alphas.append(alpha)
                fknees.append(fknee)
                residuals.append(residual)
                niters.append(niter)

                d_sky -= n_corr_est
                stop_bench("ncorr-sampling", increment_count=False)
                if iscan == len(experiment_data.scans) - 1:
                    log_memory("ncorr-sampling")

            start_bench("binned-mapmaker")
            mapmaker.accumulate_to_map(d_sky/gain, inv_var, pix, psi)
            mapmaker_orbdipole.accumulate_to_map(sky_orb_dipole, inv_var, pix, psi)
            stop_bench("binned-mapmaker", increment_count=False)

    ### PRINT NOISE SAMPLING STATS ###
    if do_ncorr_sampling:
        num_failed_convergences_ncorr = band_comm.reduce(num_failed_convergences_ncorr, op=MPI.SUM)
        num_too_high_var_ncorr = band_comm.reduce(num_too_high_var_ncorr, op=MPI.SUM)
        worst_residual_ncorr = band_comm.reduce(worst_residual_ncorr, op=MPI.MAX)
        nscans_global = band_comm.reduce(nscans*ndet, op=MPI.SUM)
        if band_comm.Get_rank() == 0:
            logger.debug(f"Worst corr-noise sampling residual (band {experiment_data.nu}GHz) = "\
                         f"{worst_residual_ncorr:.2e}.")
            if num_failed_convergences_ncorr > 0:
                logger.warning(f"Band {experiment_data.nu}GHz failed noise CG for "\
                               f"{num_failed_convergences_ncorr} out of {nscans_global} scans. "\
                               f"Worst residual = {worst_residual_ncorr:.3e}.")
            if num_too_high_var_ncorr > 0:
                logger.warning(f"Band {experiment_data.nu}GHz failed variance sanity check for "\
                               f"{num_too_high_var_ncorr} out of {nscans_global} scans. ")

        alphas = band_comm.gather(alphas, root=0)
        fknees = band_comm.gather(fknees, root=0)
        residuals = band_comm.gather(residuals, root=0)
        niters = band_comm.gather(niters, root=0)
        if band_comm.Get_rank() == 0:
            alphas = np.concatenate(alphas)
            fknees = np.concatenate(fknees)
            residuals = np.concatenate(residuals)
            residuals = residuals[residuals != 0]
            residuals = np.array([0]) if len(residuals) == 0 else residuals
            niters = np.concatenate(niters)
            logger.info(f"{experiment_data.nu}GHz: fknees {np.min(fknees):.4f} "\
            f"{np.percentile(fknees, 1):.4f} {np.mean(fknees):.4f} {np.percentile(fknees, 99):.4f}"\
            f" {np.max(fknees):.4f}")
            logger.info(f"{experiment_data.nu}GHz: alphas {np.min(alphas):.4f} "\
            f"{np.percentile(alphas, 1):.4f} {np.mean(alphas):.4f} {np.percentile(alphas, 99):.4f}"\
            f" {np.max(alphas):.4f}")
            logger.info(f"{experiment_data.nu}GHz: residuals {np.min(residuals):.2e} "\
            f"{np.percentile(residuals, 1):.2e} {np.mean(residuals):.2e} {np.percentile(residuals, 99):.2e}"\
            f" {np.max(residuals):.2e}")
            logger.info(f"{experiment_data.nu}GHz: iterations {np.min(niters):.4f} "\
            f"{np.percentile(niters, 1):.4f} {np.mean(niters):.4f} {np.percentile(niters, 99):.4f}"\
            f" {np.max(niters):.4f}")


    start_bench("binned-mapmaker")
    ### GATHER AND NORMALIZE MAPS ###
    mapmaker.gather_map()
    mapmaker_orbdipole.gather_map()
    map_rms = mapmaker_invvar.final_rms_map
    map_cov = mapmaker_invvar.final_cov_map
    mapmaker.normalize_map(map_cov)
    map_signal = mapmaker.final_map
    mapmaker_orbdipole.normalize_map(map_cov)
    map_orbdipole = mapmaker_orbdipole.final_map
    if do_ncorr_sampling:
        mapmaker_ncorr.gather_map()
        mapmaker_ncorr.normalize_map(map_cov)
        map_corrnoise = mapmaker_ncorr.final_map
    stop_bench("binned-mapmaker", increment_count=False)
    log_memory("binned-mapmaker")

    ### FINAL CLEANUP ON MASTER RANK ###
    detmap_dict_out = {}
    if band_comm.Get_rank() == 0:
        #Here we split here between I and QU
        if "I" in pols:
            detmap_I = DetectorMap(map_signal[0,:], map_rms[0,:], experiment_data.nu,
                                experiment_data.fwhm, experiment_data.nside)
            detmap_I.g0 = tod_samples.g0_est
            
            detmap_dict_out.update({"I": detmap_I})
        if "QU" in pols:
            detmap_QU = DetectorMap(map_signal[1:3,:], map_rms[1:3,:], experiment_data.nu,
                                experiment_data.fwhm, experiment_data.nside)
            detmap_QU.g0 = tod_samples.g0_est
            detmap_dict_out.update({"QU": detmap_QU})

        maps_to_file = {}
        maps_to_file["map_observed_sky"] = map_signal
        maps_to_file["map_rms"] = map_rms
        if params.general.write_orb_dipole_maps_to_chain:
            maps_to_file["map_orbdipole"] = map_orbdipole
        if params.general.write_corr_noise_maps_to_chain and do_ncorr_sampling:
            maps_to_file["map_corrnoise"] = map_corrnoise
        if params.general.write_sky_model_maps_to_chain:
            maps_to_file["map_skymodel"] = compsep_output

        start_bench("filewrite-datamaps")
        write_map_chain_to_file(params, chain, iter, experiment_data.experiment_name,
                                experiment_data.band_name, maps_to_file)
        stop_bench("filewrite-datamaps")

    return detmap_dict_out #empty on non-master ranks


def init_tod_processing(mpi_info: Bunch, params: Bunch) -> tuple[Bunch, str, DetGroupTOD,
                                                                 TODSamples]:
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
        todproc_my_band_id (str): Unique string identifier for the experiment+band this process is
          responsible for, regardless of polarization.
        experiment_data (DetGroupTOD): The TOD data for the band of this process.
    """

    logger = logging.getLogger(__name__)

    # We now loop over all bands in all experiments, and allocate them to the first ranks of the
    # TOD MPI communicator. These ranks will then become the "band masters" for those bands,
    # handling all communication with CompSep.
    # All the non-master ranks will have None values, and receive info from master further down.
    nranks_on_my_band = mpi_info.band.size
    ndets_on_my_band = None
    det_names = []
    my_experiment_name = None
    my_band_name = None
    my_experiment = None
    my_band = None
    my_band_id = None
    my_band_pol = None #string identifying the polarization type, e.g. "IQU", "I", "QU"
    my_detector_id = None
    my_scans_start = None
    my_scans_stop = None
    all_det_gains = []
    current_detector_id = 0  # A unique number identifying every detector of every band.
    for exp_name in params.experiments:
        experiment = params.experiments[exp_name]
        if not experiment.enabled:
            continue
        for iband, band_name in enumerate(experiment.bands):
            band = experiment.bands[band_name]
            if not band.enabled:
                continue
            # Checking if our rank is allocated to this experiment + band.
            if mpi_info.experiment.name == exp_name and mpi_info.band.name == band_name:
                my_band_name = band_name
                my_band = band
                my_band_pol = band.polarization
                my_band_id = iband
                # What is my rank number among the ranks processing this detector?
                my_experiment_name = exp_name
                my_experiment = experiment
                # Setting our unique detector id. Note that this is a global, not per band.
                tot_num_scans = experiment.num_scans
                scans = np.arange(tot_num_scans)
                my_scans = np.array_split(scans, mpi_info.band.size)[mpi_info.band.rank]
                my_scans_start = my_scans[0]
                my_scans_stop = my_scans[-1]
                ndets_on_my_band = len(band.detectors)
                det_names = [det for det in band.detectors]
                for idet, det_name in enumerate(band.detectors):
                    detector = band.detectors[det_name]
                    all_det_gains.append(detector.gain_est)
    all_det_gains = np.array(all_det_gains)
    mpi_info.tod.comm.Barrier()


    logger.debug(f"TOD-rank {mpi_info.tod.rank:4} (on machine {mpi_info.processor_name}), "\
                #  f"dedicated to detector {my_detector_id:4}, with local rank {mpi_info.det.rank:4}"\
                 f" (local communicator size: {mpi_info.det.size:4}).")
    time.sleep(mpi_info.tod.rank*1e-5)  # Small sleep to get prints in nice order.
    # MPIcolor_band = MPIrank_tod%tot_num_bands  # Spread the MPI tasks over the different bands.
    band_comm = mpi_info.band.comm
    logger.debug(f"TOD-rank {mpi_info.tod.rank:4} (on machine {mpi_info.processor_name}), "\
                 f"dedicated to band {my_band_id:4}, with local rank {mpi_info.band.rank:4} "\
                 f"(local communicator size: {mpi_info.band.size:4}).")
    
    # Create communicators for each different band.
    # det_comm = band_comm.Split(my_detector_id, key=mpi_info.tod.rank)
    logger.debug(f"TOD-rank {mpi_info.tod.rank:4} (on machine {mpi_info.processor_name}), "\
                #  f"dedicated to detector {my_detector_id:4}, with local rank {mpi_info.det.rank:4}"\
                 f" (local communicator size: {mpi_info.det.size:4}).")

    t0 = time.time()
    with benchmark("fileread-tod"):
        experiment_data = read_tods_from_file(band_comm, my_experiment, my_band, det_names, params,
                                              my_scans_start, my_scans_stop)
    #FIXME: Make this not a hacky fix.
    if "init_sky_from" in my_band:
        experiment_data.sky_init_files = my_band.init_sky_from
    else:
        experiment_data.sky_init_files = []
    mpi_info.tod.comm.Barrier()
    if mpi_info.tod.is_master:
        logger.info(f"TOD: Finished reading all files in {time.time()-t0:.1f}s.")

    # The actual number of scans may be less than my_scans_stop - my_scans_start, because
    # bad PIDs are filtered out during TOD reading.
    my_num_scans = len(experiment_data.scans)

    # Find the detector-averaged gain for our experiment, which serves as the initial g0 estimate.
    initial_g0_est = float(np.mean(all_det_gains))
    tod_samples = TODSamples()
    tod_samples.experiment_name = my_experiment._name
    tod_samples.band_name = my_band._name
    tod_samples.g0_est = initial_g0_est
    tod_samples.rel_gain_est = np.zeros(ndets_on_my_band, dtype=np.float32)
    tod_samples.rel_gain_est[:] = all_det_gains - initial_g0_est
    tod_samples.time_dep_rel_gain_est = np.zeros((ndets_on_my_band, my_num_scans), dtype=np.float32)
    tod_samples.gain_est = tod_samples.g0_est + tod_samples.rel_gain_est[:,None] + tod_samples.time_dep_rel_gain_est
    tod_samples.sigma0_est = np.zeros((ndets_on_my_band, my_num_scans), dtype=np.float32)
    tod_samples.alpha_est = np.zeros((ndets_on_my_band, my_num_scans), dtype=np.float32) + params.general.noise_alpha
    tod_samples.fknee_est = np.zeros((ndets_on_my_band, my_num_scans), dtype=np.float32) + params.general.noise_fknee

    if mpi_info.band.is_master:
        logger.debug(f"Initial absolute gain estimate for {my_band._name}: {initial_g0_est:.3e}.")
        logger.debug(f"Initial rel gain estimates for {my_band._name}: {tod_samples.rel_gain_est}.")

    # Creating "tod_band_masters", an array which maps the band index to the rank of the master
    # of that band.
    todproc_my_band_id = f"{my_experiment_name}$$${my_band_name}"
    data_world = (todproc_my_band_id, mpi_info.world.rank) if mpi_info.band.is_master else None
    data_tod = (todproc_my_band_id, mpi_info.tod.rank) if mpi_info.band.is_master else None
    # pols_tod_bands = (todproc_my_band_id, my_band_pol) if mpi_info.band.is_master else None
    all_data_world = mpi_info.tod.comm.allgather(data_world)
    all_data_tod = mpi_info.tod.comm.allgather(data_tod)
    # all_pol_data = mpi_info.tod.comm.allgather(pols_tod_bands)

    world_band_masters_dict = {}
    if "I" in my_band_pol:
        world_band_masters_dict.update({item[0]+'_I':  # First I:
                                        item[1] for item in all_data_world if item is not None})
    if "QU" in my_band_pol:
        world_band_masters_dict.update({item[0]+'_QU':  # Then QU:
                                        item[1] for item in all_data_world if item is not None})
    tod_band_masters_dict = {item[0]: item[1] for item in all_data_tod if item is not None}
    # tod_band_pol_dict = {item[0]: item[1] for item in all_pol_data if item is not None}
    # logger.info(f"world_band_masters_dict: {world_band_masters_dict}")
    # logger.info(f"tod_band_masters_dict: {tod_band_masters_dict}")
    # logger.info(f"tod_band_pol_dict: {tod_band_pol_dict}")
    # logger.info(f"TOD: Rank {mpi_info.tod.rank:4} assigned scans {my_scans_start:6} - "\
    #             f"{my_scans_stop:6} on band {my_band_id:4}.")
    mpi_info['world']['tod_band_masters'] = world_band_masters_dict
    mpi_info['tod']['tod_band_masters'] = tod_band_masters_dict
    # mpi_info['world']['tod_band_pols'] = tod_band_pol_dict

    return mpi_info, todproc_my_band_id, experiment_data, tod_samples


def estimate_white_noise(experiment_data: DetGroupTOD, tod_samples: TODSamples,
                         det_compsep_map: NDArray, params: Bunch) -> TODSamples:
    """ Estimate the white noise level in the TOD data and return the updated TOD samples.
    Input:
        experiment_data (DetGroupTOD): The experiment TOD object.
        tod_samples (TODSamples): Current sampled TOD parameters.
        det_compsep_map (NDArray): The component-separation sky map for the detector.
        params (Bunch): The parameters from the input parameter file.
    Output:
        tod_samples (TODSamples): Updated TOD samples with sigma0 estimates.
    """
    for iscan, scan in enumerate(experiment_data.scans):
        for idetector, detector in enumerate(scan.detectors):
            pix = detector.pix
            psi = detector.psi if "QU" in experiment_data.pols else None
            # FIXME: Should maybe n_corr be subtracted here as well?
            gain = tod_samples.gain_est[idetector,iscan]
            sky_subtracted_tod = detector.tod.copy()
            sky_subtracted_tod -= gain*get_static_sky_TOD(det_compsep_map, pix, psi=psi)
            sky_subtracted_tod -= gain*get_s_orb_TOD(detector, experiment_data, pix)
            mask = detector.processing_mask_TOD
            sigma0 = calculate_sigma0(sky_subtracted_tod, mask)
            tod_samples.sigma0_est[idetector,iscan] = sigma0/gain
        if iscan == len(experiment_data.scans) - 1:
            log_memory("sigma0-est")
    return tod_samples



def sample_absolute_gain(band_comm: MPI.Comm, experiment_data: DetGroupTOD, tod_samples: TODSamples,
                         det_compsep_map: NDArray):
    """ Draw a realization of the absolute gain term, g0, which is constant across all
        detectors and all scans within a band. For frequencies < 380.0 GHz this is done using
        only the orbital dipole, and above it uses the full sky.
    Args:
        band_comm (MPI.Comm): The band-level MPI communicator.
        experiment_data (DetGroupTOD): The object holding all the scan data.
        tod_samples (TODSamples): Current sampled TOD parameters (updated in-place with g0).
        det_compsep_map (NDArray): The component-separation sky map for the detector.
    Returns:
        tod_samples (TODSamples): Updated TOD samples with the new g0 estimate.
        wait_time (float): Time spent waiting at the MPI barrier.
    """
    logger = logging.getLogger(__name__)

    sum_s_T_N_inv_d = 0  # Accumulators for the numerator and denominator of eqn 16.
    sum_s_T_N_inv_s = 0

    # Calibrate on the full sky at high frequencies, as the orbital dipole is too faint.
    calibrate_on_full_sky = experiment_data.nu > 380.0

    for iscan, scan in enumerate(experiment_data.scans):
        for idet, det in enumerate(scan.detectors):
            f_samp = det.fsamp
            down_factor = int(f_samp)
            indices_edges = np.arange(0, det.ntod, down_factor)
            indices_centers = (indices_edges[1:] + indices_edges[:-1])//2
            ntod_down = indices_centers.size

            assert((ntod_down+1)*down_factor >= det.tod.shape[0])

            pix = det.pix  # Only decompressing pix once for efficiency.
            psi = det.psi
            pix = pix[indices_centers]
            psi = psi[indices_centers]

            s_orb = get_s_orb_TOD(det, experiment_data, pix)
            sky_model_TOD = get_static_sky_TOD(det_compsep_map, pix, psi=psi)

            if calibrate_on_full_sky:
                # Calibrate on the full sky model (static sky + orbital dipole),
                # analogous to sample_relative_gain / sample_temporal_gain_variations.
                s_cal = sky_model_TOD + s_orb
                gain = tod_samples.rel_gain_est[idet] + tod_samples.time_dep_rel_gain_est[idet,iscan]
                residual_tod = det.tod[:ntod_down*down_factor].reshape((ntod_down, down_factor))
                residual_tod = np.mean(residual_tod, axis=-1)
                residual_tod -= gain*s_cal
            else:
                # Default: calibrate on the orbital dipole only.
                s_cal = s_orb
                residual_tod = det.tod[:ntod_down*down_factor].reshape((ntod_down, down_factor))
                residual_tod = np.mean(residual_tod, axis=-1)
                residual_tod -= tod_samples.gain_est[idet,iscan]*sky_model_TOD  # Subtracting sky signals.
                residual_tod -= tod_samples.gain_est[idet,iscan]*s_orb
                residual_tod += tod_samples.g0_est*s_orb  # Now we can add back in the orbital dipole.

            mask = det.processing_mask_TOD[indices_centers]
            sigma0 = calculate_sigma0(residual_tod, mask)

            Ntod = residual_tod.shape[0]
            Nrfft = Ntod//2+1
            freqs = rfftfreq(Ntod, 1.0)
            inv_power_spectrum = np.zeros(Nrfft)
            inv_power_spectrum[1:] = 1.0/(sigma0**2*(1 + (freqs[1:]/tod_samples.fknee_est[idet,iscan])\
                                                    **tod_samples.alpha_est[idet,iscan]))

            ### Solving Equation 16 from BP7 ###
            mask = det.processing_mask_TOD[indices_centers]
            # In the masked regions, inpaint the calibration signal times the absolute gain.
            # TODO: Shouldn't this be using the full gain, and not just g0?
            residual_tod[~mask] = tod_samples.g0_est*s_cal[~mask]\
                                + np.random.normal(0, sigma0, s_cal[~mask].shape)
            
            s_fft = forward_rfft(s_cal)
            d_fft = forward_rfft(residual_tod)
            N_inv_s_fft = s_fft * inv_power_spectrum
            N_inv_d_fft = d_fft * inv_power_spectrum
            N_inv_s = backward_rfft(N_inv_s_fft, Ntod)
            N_inv_d = backward_rfft(N_inv_d_fft, Ntod)
            
            # We now exclude the time-samples hitting the masked area.
            # We don't want to do this before now, because it would mess up the FFT stuff.

            # mask = experiment_data.processing_mask_map[scan.pix]
            # Add to the numerator and denominator.
            # sum_s_T_N_inv_d += np.dot(s_cal[mask], N_inv_d[mask])
            # sum_s_T_N_inv_s += np.dot(s_cal[mask], N_inv_s[mask])
            sum_s_T_N_inv_d += np.dot(s_cal, N_inv_d)
            sum_s_T_N_inv_s += np.dot(s_cal, N_inv_s)

    # The g0 term is fully global, so we reduce across both all scans and all bands:
    sum_s_T_N_inv_d = band_comm.reduce(sum_s_T_N_inv_d, op=MPI.SUM, root=0)
    sum_s_T_N_inv_s = band_comm.reduce(sum_s_T_N_inv_s, op=MPI.SUM, root=0)
    g_sampled = 0.0
    # Rank 0 draws a sample of g0 from eq (16) from BP6, and bcasts it to the other ranks.
    if band_comm.Get_rank() == 0:
        eta = np.random.randn()
        g_mean = sum_s_T_N_inv_d / sum_s_T_N_inv_s
        g_std = 1.0 / np.sqrt(sum_s_T_N_inv_s)

        g_sampled = g_mean + eta * g_std
        logger.info(f"Band {experiment_data.band_name} g0: {tod_samples.g0_est:.4e} "\
                    f"-> {g_sampled:.4e} (+/- {g_std:.4e})")

    t0 = time.time()
    band_comm.Barrier()
    wait_time = time.time() - t0
    g_sampled = band_comm.bcast(g_sampled, root=0)
    log_memory("abs-gain")

    # As of Numpy 2.0 it's good practice to explicitly cast to Python scalar types, as this would
    # otherwise have been a np.float64 type, potentially causing unexpected casting behavior later.
    tod_samples.g0_est = float(g_sampled)

    return tod_samples, wait_time


def sample_relative_gain(band_comm: MPI.Comm, experiment_data: DetGroupTOD,
                         tod_samples: TODSamples, det_compsep_map: NDArray):
    """ Samples the detector-dependent relative gain (Delta g_i). This function implements the
        logic from Sec. 3.4 of BP7.
    Args:
        band_comm (MPI.Comm): The band-level MPI communicator.
        experiment_data (DetGroupTOD): The object holding scan data for the band.
        tod_samples (TODSamples): Current sampled TOD parameters.
        det_compsep_map (NDArray): The component-separation sky map for the detector.
    Returns:
        tod_samples (TODSamples): Updated TOD samples with relative gain estimates.
    """
    logger = logging.getLogger(__name__)
    global_rank = band_comm.Get_rank()
    # band_rank = det_comm.Get_rank()
    ndet = experiment_data.ndet

    #### 1. Local Calculation (on each rank) ###
    # Each rank calculates the sum of terms for its local subset of scans.
    # local_s_T_N_inv_s = 0.0
    local_s_T_N_inv_s = np.zeros(ndet, dtype=np.float32)

    # local_r_T_N_inv_s = 0.0
    local_r_T_N_inv_s = np.zeros(ndet, dtype=np.float32)


    for iscan, scan in enumerate(experiment_data.scans):
        for idet, det in enumerate(scan.detectors):
            f_samp = det.fsamp
            down_factor = int(f_samp)
            indices_edges = np.arange(0, det.ntod, down_factor)
            indices_centers = (indices_edges[1:] + indices_edges[:-1])//2
            ntod_down = indices_centers.size

            # Define the residual for this sampling step, as per Eq. (17)
            pix = det.pix
            psi = det.psi
            pix = pix[indices_centers]
            psi = psi[indices_centers]

            s_tot = get_static_sky_TOD(det_compsep_map, pix, psi)

            s_tot += get_s_orb_TOD(det, experiment_data, pix)

            gain = tod_samples.g0_est + tod_samples.time_dep_rel_gain_est[idet,iscan]
            residual_tod = det.tod[:ntod_down*down_factor].reshape((ntod_down, down_factor))
            residual_tod = np.mean(residual_tod, axis=-1)
            residual_tod -= gain*s_tot
            mask = det.processing_mask_TOD[indices_centers]
            sigma0 = calculate_sigma0(residual_tod, mask)

            # Setup FFT-based calculation for N^-1 operations
            Ntod = residual_tod.shape[0]
            Nrfft = Ntod // 2 + 1
            freqs = rfftfreq(Ntod, 1.0)
            inv_power_spectrum = np.zeros(Nrfft)
            inv_power_spectrum[1:] = 1.0 / (sigma0**2 * (1 + (freqs[1:] / tod_samples.fknee_est[idet,iscan])\
                                                        **tod_samples.alpha_est[idet,iscan]))

            s_fft = forward_rfft(s_tot)
            N_inv_s_fft = s_fft * inv_power_spectrum
            N_inv_s = backward_rfft(N_inv_s_fft, Ntod)
            
            # Inpaint on the masked regions the sky signal times only the detector-residual gain.
            residual_tod[~mask] = tod_samples.rel_gain_est[idet]*s_tot[~mask]\
                                + np.random.normal(0, sigma0, s_tot[~mask].shape)
            s_T_N_inv_s_scan = np.dot(s_tot, N_inv_s)
            r_T_N_inv_s_scan = np.dot(residual_tod, N_inv_s)
            # s_T_N_inv_s_scan = np.dot(s_tot[mask], N_inv_s[mask])
            # r_T_N_inv_s_scan = np.dot(residual_tod[mask], N_inv_s[mask])

            # Add the contribution from this scan to the local sum
            local_s_T_N_inv_s[idet] += s_T_N_inv_s_scan
            local_r_T_N_inv_s[idet] += r_T_N_inv_s_scan

    ### 2. Intra-Detector Reduction ###
    # Sum the local values across all ranks that share the same detector using det_comm.
    # After this, every rank in the det_comm will have the total sum for their detector.
    band_comm.Allreduce(MPI.IN_PLACE, local_s_T_N_inv_s, op=MPI.SUM)
    band_comm.Allreduce(MPI.IN_PLACE, local_r_T_N_inv_s, op=MPI.SUM)

    ### 3. Solve Global System ###
    delta_g_samples = np.zeros(ndet, dtype=np.float32)
    if band_comm.Get_rank() == 0:
        A = np.zeros((ndet + 1, ndet + 1))
        b = np.zeros(ndet + 1)
        diagonal = np.array(local_s_T_N_inv_s)
        A[:ndet, :ndet] = np.diag(diagonal)
        A[:ndet, ndet] = 0.5
        A[ndet, :ndet] = 1.0
        eta = np.random.randn(ndet)
        fluctuation_term = np.sqrt(diagonal) * eta
        
        b[:ndet] = np.array(local_r_T_N_inv_s) + fluctuation_term
    
        try:
            solution = np.linalg.solve(A, b)
            delta_g_samples[:] = solution[:ndet]
            logger.info(f"Solved global relative gains for {ndet} detectors.")
        except np.linalg.LinAlgError:
            logger.error("Failed to solve global linear system for relative gain: Not updating")
    band_comm.Bcast(delta_g_samples, root=0)
    log_memory("rel-gain")
    
    wait_time = 0
    tod_samples.rel_gain_est[:] = delta_g_samples
    if band_comm.Get_rank() == 0:
        logger.info(f"Rel gain for band {experiment_data.band_name}: {tod_samples.g0_est:.3e} "\
                    f"+/- {np.std(delta_g_samples):.3e}")

    return tod_samples, wait_time



def sample_temporal_gain_variations(band_comm: MPI.Comm, experiment_data: DetGroupTOD,
                                    tod_samples: TODSamples, det_compsep_map: NDArray,
                                    chain: int, iter: int, params: Bunch):
    """ Samples the time-dependent relative gain variations (delta g_qi). This function implements
        the logic from Sec. 3.5 of the BP7 paper, using a Wiener filter to smooth the gain solution
        over time (PIDs). It solves a global system for all scans of a given detector, which are
        distributed across the ranks of the band_comm.

    Args:
        band_comm (MPI.Comm): The communicator for ranks sharing the same band.
        experiment_data (DetGroupTOD): The object holding scan data.
        tod_samples (TODSamples): The sampled TOD parameters.
        det_compsep_map (NDArray): The sky model at our band.
        chain (int): Current chain number.
        iter (int): Current Gibbs iteration.
        params (Bunch): Parameters from the parameter file.
    """
    logger = logging.getLogger(__name__)
    band_rank = band_comm.Get_rank()
    band_size = band_comm.Get_size()
    ndet = experiment_data.ndet
    nscans_local = len(experiment_data.scans)

    # Local calculations on each rank
    A_qq_local = np.zeros((ndet, nscans_local), dtype=np.float64)
    b_q_local = np.zeros((ndet, nscans_local), dtype=np.float64)

    for iscan, scan in enumerate(experiment_data.scans):
        for idet, det in enumerate(scan.detectors):
            # I'm still not sure what way of dealing with the masked samples are best:
            # 1. Replace masked values with 0s before FFT.
            # 2. Replace masked values with n_corr realizations before FFT.
            # 3. Remove masked values by reducing TOD size before FFTs.
            # (simply passing the full data through the FFTs seems like a bad idea because of
            # ringing from the large residual in the galactic plane).

            f_samp = det.fsamp
            down_factor = int(f_samp)
            indices_edges = np.arange(0, det.ntod, down_factor)
            indices_centers = (indices_edges[1:] + indices_edges[:-1])//2
            ntod_down = indices_centers.size

            # Per Eq. (26), the residual is d - (g0 + Delta_g)*s
            pix = det.pix
            psi = det.psi
            pix = pix[indices_centers]
            psi = psi[indices_centers]

            s_tot = get_static_sky_TOD(det_compsep_map, pix, psi)
            s_tot += get_s_orb_TOD(det, experiment_data, pix)

            gain = tod_samples.g0_est + tod_samples.rel_gain_est[idet]
            residual_tod = det.tod[:ntod_down*down_factor].reshape((ntod_down, down_factor))
            residual_tod = np.mean(residual_tod, axis=-1)
            residual_tod -= gain*s_tot

            mask = det.processing_mask_TOD[indices_centers]
            sigma0 = calculate_sigma0(residual_tod, mask)

            # FFT-based N^-1 operation setup
            Ntod = residual_tod.shape[0]
            Nrfft = Ntod // 2 + 1
            freqs = rfftfreq(Ntod, 1.0)
            inv_power_spectrum = np.zeros(Nrfft)
            inv_power_spectrum[1:] = 1.0 / (sigma0**2 * (1 + (freqs[1:] / tod_samples.fknee_est[idet,iscan])\
                                                         **tod_samples.alpha_est[idet,iscan]))

            # In the masked regions, inpaint the total sky model times only the temporal gain estimate.
            residual_tod[~mask] = tod_samples.time_dep_rel_gain_est[idet,iscan]*s_tot[~mask]\
                                + np.random.normal(0, sigma0, s_tot[~mask].shape)

            # Calculate N^-1 * s_tot and N^-1 * residual_tod
            N_inv_s = backward_rfft(forward_rfft(s_tot) * inv_power_spectrum, Ntod)
            N_inv_r = backward_rfft(forward_rfft(residual_tod) * inv_power_spectrum, Ntod)

            # Calculate elements for the linear system
            A_qq = np.dot(s_tot, N_inv_s)
            b_q = np.dot(s_tot, N_inv_r)

            A_qq_local[idet, iscan] = A_qq
            b_q_local[idet, iscan] = b_q

    # Gather scan counts on all ranks (needed for gather/scatter with varying roots)
    scan_counts = np.array(band_comm.allgather(nscans_local), dtype=int)
    displacements = np.insert(np.cumsum(scan_counts), 0, 0)[:-1]

    # Distribute detector solves across ranks in round-robin fashion.
    # Each detector's equation system is gathered to, solved on, and scattered from
    # the rank given by solving_rank = idet % band_size.
    for idet in range(ndet):
        solving_rank = idet % band_size

        all_A_qq = band_comm.gather(A_qq_local[idet], root=solving_rank)
        all_b_q = band_comm.gather(b_q_local[idet], root=solving_rank)

        delta_g_sample = None
        if band_rank == solving_rank:
            # Concatenate gathered arrays into single flat arrays
            A_diag = np.concatenate(all_A_qq)
            b = np.concatenate(all_b_q)

            n_scans_total = len(A_diag)
            if n_scans_total > 1:
                # Define Prior (Wiener Filter) based on Eq. (31)
                alpha_gain = -2.5
                fknee_gain = 1.0  # Hour (which equals 1 scan)

                # However, I found the BP prior too weak, and this gives me more sensible results.
                mean_gain = tod_samples.g0_est + tod_samples.rel_gain_est[idet]
                sigma0_gain = 1e-4*mean_gain
                sigma0_sq_gain = sigma0_gain**2

                gain_freqs = rfftfreq(n_scans_total, d=1.0)
                prior_ps = np.zeros_like(gain_freqs)
                prior_ps[1:] = sigma0_sq_gain * (np.abs(gain_freqs[1:]) / fknee_gain)**alpha_gain

                prior_ps_inv = np.zeros_like(gain_freqs)
                prior_ps_inv[prior_ps > 0] = 1.0 / prior_ps[prior_ps > 0]
                prior_ps_inv_sqrt = np.sqrt(prior_ps_inv)

                # Define Linear Operator for Conjugate Gradient Solver
                def matvec(v, A_diag=A_diag, prior_ps_inv=prior_ps_inv,
                           n_scans_total=n_scans_total):
                    g_inv_v = backward_rfft(forward_rfft(v) * prior_ps_inv, n_scans_total).real
                    diag_v = A_diag * v
                    return g_inv_v + diag_v

                # Construct RHS of the sampling equation (Eq. 30)
                eta1 = np.random.randn(n_scans_total)
                fluctuation1 = np.sqrt(np.maximum(A_diag, 0)) * eta1

                eta2 = np.random.randn(n_scans_total)
                fluctuation2 = backward_rfft(forward_rfft(eta2) * prior_ps_inv_sqrt, n_scans_total).real

                RHS = b + fluctuation1 + fluctuation2

                ### Simpler sanity check solution  ##
                epsilon = 1e-12
                g_mean = b / (A_diag + epsilon)
                g_std = 1.0 / np.sqrt(np.maximum(A_diag, 0) + epsilon)

                CG_solver = pixell.utils.CG(matvec, RHS, x0=g_mean)
                for i in range(200):
                    CG_solver.step()
                    if CG_solver.err < 1e-10:
                        break

                delta_g_sample = CG_solver.x
                delta_g_sample -= np.mean(delta_g_sample)
                # logger.info(f"Band {experiment_data.nu}GHz det {idet} time-dependent gain: "\
                #             f"min={np.min(delta_g_sample)*1e9:14.4f} "\
                #             f"mean={np.mean(delta_g_sample)*1e9:14.4f} "\
                #             f"std={np.std(delta_g_sample)*1e9:14.4f} "\
                #             f"max={np.max(delta_g_sample)*1e9:14.4f}")

                if False: #debug stuff
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10,8))
                    other_gain = tod_samples.g0_est + tod_samples.rel_gain_est[idet]
                    plt.plot(1e9*(other_gain + delta_g_sample))
                    plt.ylim(0, np.max(1e9*(other_gain + delta_g_sample)))
                    plt.xlabel("PID")
                    plt.ylabel("Gain [mV/K]")
                    plt.savefig(f"{params.general.output_paths.plots}chain{chain}_iter{iter}_"
                                f"det{idet}_{experiment_data.band_name}.png")
                    plt.close()
            else:
                delta_g_sample = np.zeros(n_scans_total)

        # Scatter the results back to all ranks from the solving rank
        if band_size > 1:
            delta_g_local = np.empty(nscans_local, dtype=np.float64)
            if band_rank == solving_rank:
                sendbuf = [delta_g_sample, scan_counts, displacements, MPI.DOUBLE]
            else:
                sendbuf = None
            band_comm.Scatterv(sendbuf, delta_g_local, root=solving_rank)
        else:
            delta_g_local = delta_g_sample if delta_g_sample is not None else np.array([])
        log_memory("temporal-gain")

        # Update tod_samples for this detector
        if delta_g_local.size == nscans_local:
            tod_samples.time_dep_rel_gain_est[idet, :] = delta_g_local.astype(np.float32)
        else:
            logger.warning(f"Rank {band_rank} received mismatched number of gain samples "\
                           f"for det {idet}. Expected {nscans_local}, got {delta_g_local.size}.")

    return tod_samples


def process_tod(mpi_info: Bunch, experiment_data: DetGroupTOD,
                tod_samples: TODSamples, compsep_output: NDArray,
                params: Bunch, chain: int, iter: int) -> tuple[dict[str, DetectorMap], TODSamples]:
    """ Performs a single TOD iteration.

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        experiment_data (DetGroupTOD): The input experiment TOD for the band
            belonging to the current process.
        tod_samples (TODSamples): Sampled TOD parameters (gain, noise, etc.).
        compsep_output (NDArray): The current best estimate of the sky model
            as seen by the band belonging to the current process.
        params (Bunch): The parameters from the input parameter file.
        chain (int): ID of the current chain.
        iter (int): Iteration within the Gibbs chain.

    Output:
        dict[str, DetectorMap]: Correlated-noise-subtracted TOD data projected into map
            space for the band belonging to the current process.
        tod_samples (TODSamples): Updated sampled TOD parameters.
    """
    # Steps:
    # 1. Initialize n_corr_est and alpha/fknee values.
    # 2. Sample the gain from the sky-subtracted TOD (Skipped on iter==1 because we don't have
    #   a reliable sky-subtracted TOD).
    # 3. Estimate White noise from the sky-subtracted TOD.
    # 4. Sample correlated noise and PS parameters (skipped on iter==1).
    # 5. Mapmaking on TOD - corr_noise_TOD - orb_dipole_TOD.
    # (In other words, on iteration 1 we do just do White noise estimation -> Mapmaking.)

    logger = logging.getLogger(__name__)

    timing_dict = {}
    waittime_dict = {}

    det_comm = mpi_info.det.comm
    band_comm = mpi_info.band.comm
    TOD_comm = mpi_info.tod.comm
    ### WHITE NOISE ESTIMATION ###
    t0 = time.time()
    with benchmark("sigma0-est"):
        tod_samples = estimate_white_noise(experiment_data, tod_samples, compsep_output, params)
    timing_dict["wn-est-1"] = time.time() - t0
    if mpi_info.tod.is_master:
        logger.info(f"Chain {chain} iter{iter} {experiment_data.nu}GHz: Finished white noise "\
                    f"estimation in {timing_dict['wn-est-1']:.1f}s.")

    ### ABSOLUTE GAIN CALIBRATION ### 
    if params.general.sample_abs_gain and iter >= params.general.sample_abs_gain_from_iter_num:
        t0 = time.time()
        with benchmark("abs-gain"):
            tod_samples, wait_time = sample_absolute_gain(band_comm, experiment_data, tod_samples,
                                                          compsep_output)
        timing_dict["abs-gain"] = time.time() - t0
        waittime_dict["abs-gain"] = wait_time
        if mpi_info.band.is_master:
            logger.info(f"Chain {chain} iter{iter} {experiment_data.nu}GHz: Finished absolute "\
                        f"gain estimation in {timing_dict['abs-gain']:.1f}s.")

    ### RELATIVE GAIN CALIBRATION ### 
    if params.general.sample_rel_gain and iter >= params.general.sample_rel_gain_from_iter_num:
        t0 = time.time()
        with benchmark("rel-gain"):
            tod_samples, wait_time = sample_relative_gain(band_comm, experiment_data, tod_samples,
                                                          compsep_output)
        timing_dict["rel-gain"] = time.time() - t0
        waittime_dict["rel-gain"] = wait_time
        if mpi_info.band.is_master:
            logger.info(f"Chain {chain} iter{iter} {experiment_data.nu}GHz: Finished relative "\
                        f"gain estimation in {timing_dict['rel-gain']:.1f}s.")


    ### TEMPORAL GAIN CALIBRATION ### 
    if params.general.sample_temporal_gain\
    and iter >= params.general.sample_temporal_gain_from_iter_num:
        t0 = time.time()
        with benchmark("temporal-gain"):
            tod_samples = sample_temporal_gain_variations(band_comm, experiment_data,
                                                tod_samples, compsep_output, chain, iter, params)
        timing_dict["temp-gain"] = time.time() - t0
        if mpi_info.band.is_master:
            logger.info(f"Chain {chain} iter{iter} {experiment_data.nu}GHz: Finished temporal "\
                        f"gain estimation in {timing_dict['temp-gain']:.1f}s.")

    ### Update total gain from sum of all three gain terms. ###
    tod_samples.gain_est = tod_samples.g0_est + tod_samples.rel_gain_est[:,None] + tod_samples.time_dep_rel_gain_est

    ### WHITE NOISE ESTIMATION ###
    t0 = time.time()
    with benchmark("sigma0-est"):
        tod_samples = estimate_white_noise(experiment_data, tod_samples, compsep_output, params)
    timing_dict["wn-est-2"] = time.time() - t0
    if band_comm.Get_rank() == 0:
        logger.info(f"Chain {chain} iter{iter} {experiment_data.nu}GHz: Finished white noise "\
                    f"estimation in {timing_dict['wn-est-2']:.1f}s.")

    ### MAPMAKING ###
    do_ncorr_sampling = params.general.sample_corr_noise and iter >=\
                        params.general.sample_corr_noise_from_iter_num
    t0 = time.time()

    if "mapmaker" in params.experiments[experiment_data.experiment_name].bands[experiment_data.band_name]:
        mapmaker_str = params.experiments[experiment_data.experiment_name].bands[experiment_data.band_name].mapmaker
    elif "mapmaker" in params.experiments[experiment_data.experiment_name]:
        mapmaker_str = params.experiments[experiment_data.experiment_name].mapmaker
    else:
        raise ValueError(f"Unspecified mapmaker for experiment {experiment_data.experiment_name}," \
                        f" band {experiment_data.band_name}.")

    if mapmaker_str == "CG":
        detmap_dict = tod2map_CG(band_comm, experiment_data, compsep_output, tod_samples, params, chain,
                     iter, do_ncorr_sampling)
    elif mapmaker_str == "bin":
        detmap_dict = tod2map_bin(band_comm, experiment_data, compsep_output, tod_samples, params, 
                            chain, iter, do_ncorr_sampling)
    else:
        raise ValueError(f'Mapmaker must be either "CG" or "bin", but {mapmaker_str} was given for'\
                         f' experiment {experiment_data.experiment_name}, band {experiment_data.band_name}')
    timing_dict["mapmaker"] = time.time() - t0
    if band_comm.Get_rank() == 0:
        logger.info(f"Chain {chain} iter{iter} {experiment_data.nu}GHz: Finished mapmaking in "\
                    f"{timing_dict['mapmaker']:.1f}s.")

    ### WRITE CHAIN TO FILE ###
    with benchmark("filewrite-tod"):
        write_tod_chain_to_file(band_comm, tod_samples, params, chain, iter)

    t0 = time.time()
    with benchmark("end-barrier"):
        TOD_comm.Barrier()
    waittime_dict["end-barrier"] = time.time() - t0

    bench_summary(TOD_comm, label="All bands")
    bench_summary(band_comm, label=f"Band {experiment_data.band_name}")
    bench_reset()

    for key in timing_dict:
        timing_dict[key] = band_comm.reduce(timing_dict[key], op=MPI.SUM, root=0)
    for key in waittime_dict:
        waittime_dict[key] = band_comm.reduce(waittime_dict[key], op=MPI.SUM, root=0)
    
    if mpi_info.band.is_master:
        for key in timing_dict:
            timing_dict[key] /= band_comm.Get_size()
            logger.info(f"Average time spent for {experiment_data.nu}GHz on {key} = "\
                        f"{timing_dict[key]:.1f}s.")

        for key in waittime_dict:
            waittime_dict[key] /= band_comm.Get_size()
            logger.info(f"Average wait overhead for {experiment_data.nu}GHz on {key} = "\
                        f"{waittime_dict[key]:.1f}s.")

    return detmap_dict, tod_samples