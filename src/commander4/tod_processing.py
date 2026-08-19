import numpy as np
import pixell
from pixell import utils
from mpi4py import MPI
import logging
from scipy.fft import rfftfreq
import time
from dataclasses import dataclass, field, fields
from typing import ClassVar, Self
from numpy.typing import NDArray
from contextlib import contextmanager

from pixell.bunch import Bunch

from commander4.param_schema import resolve_param
from commander4.output.log import logassert
from commander4.data_models.detector_map import DetectorMap
from commander4.data_models.detector_group_TOD import DetGroupTOD
from commander4.data_models.TOD_samples import TODSamples
from commander4.data_selection import masked_chisq_z, log_dataselect_summary
from commander4.data_models.jump_corrections import JumpCorrection
from commander4.data_models.tod_view import TODView
from commander4.utils.mapmaker import MapmakerIQU, WeightsMapmakerIQU, WeightsMapmaker, Mapmaker
from commander4.utils.CG_mapmaker import CGMapmakerI, CGMapmakerIQU
from commander4.solvers.preconditioners import InvNPreconditionerI, InvNPreconditionerIQU
from commander4.noise_sampling.sample_ncorr import sample_correlated_noise, log_corr_noise_stats,\
    SIGMA0_METHODS, GAIN_GAP_FILL_METHODS
from commander4.noise_sampling.noise_sampling import fill_all_masked
from commander4.utils.math_operations import forward_rfft, backward_rfft
from commander4.utils.execution_ids import get_execution_band_ids
from commander4.noise_sampling.sigma0 import calc_sigma0_robust, calc_sigma0_binned_psd
from commander4.tod_reader import read_tods_from_file
from commander4.output.write_chains_files import write_map_chain_to_file
from commander4.logging.performance_logger import benchmark, bench_summary, start_bench,\
                                            stop_bench, log_memory, increment_count, bench_reset

logger = logging.getLogger(__name__)


def called_on_non_master(arr):
    logger.debug("Dummy precond has been called")
    return np.copy(arr)


def _binned_tod_power_spectrum(tod: NDArray, fsamp: float, nbin: int) -> tuple[NDArray, NDArray]:
    """ Log-binned periodogram of a TOD, for low-resolution diagnostics written to the chain.

        Computes the one-sided periodogram ``|rfft(tod)|^2 / Ntod`` on the natural frequency grid,
        then averages it into exponentially spaced bins (``pixell.utils.expbin`` with ``nmin=1``).
        expbin returns at most ``nbin`` bins (fewer for short TODs), so the binned frequencies and
        power are returned padded to length ``nbin`` with NaN, giving a fixed width for the
        per-detector-scan chain arrays.
    Args:
        tod (NDArray): Time-ordered data (any units; e.g. the raw TOD or an n_corr realization).
        fsamp (float): Sampling rate (Hz).
        nbin (int): Fixed output length (the maximum number of bins).
    Returns:
        (freqs, power): Each a length-``nbin`` array, NaN-padded beyond the actual bin count.
    """
    ntod = len(tod)
    freqs = rfftfreq(ntod, 1.0 / fsamp)
    power = (1.0 / ntod) * np.abs(forward_rfft(tod)) ** 2
    bins = pixell.utils.expbin(freqs.size, nbin=nbin, nmin=1)
    nb = bins.shape[0]
    freqs_binned = np.full(nbin, np.nan, dtype=np.float64)
    power_binned = np.full(nbin, np.nan, dtype=np.float64)
    freqs_binned[:nb] = pixell.utils.bin_data(bins, freqs)
    power_binned[:nb] = pixell.utils.bin_data(bins, power)
    return freqs_binned, power_binned


def _record_tod_diagnostics(tod_samples: TODSamples, iscan: int, idet: int, view: TODView,
                            n_corr: NDArray | None) -> None:
    """ Record per-detector-scan TOD diagnostics into the chain arrays.

        Stores the low-resolution log-binned power spectra (sharing one binned frequency axis) of
        four detector-unit TOD views:
          * ``raw``:      the raw detector TOD.
          * ``ncorrsub``: the TOD with only the correlated noise subtracted (sky signal, orbital
                          dipole, and white noise retained); equals ``raw`` when no n_corr drawn.
          * ``residual``: the noise residual, with the sky model, orbital dipole, and correlated
                          noise all subtracted.
          * ``ncorr``:    the correlated-noise realization itself, stored only when one was drawn.
        ``ncorrsub`` and ``residual`` use the jump-corrected stream (matching mapmaking and n_corr
        sampling). When the off-by-default DEBUG full-``n_corr`` collection is enabled, also stores
        the entire ``n_corr`` TOD for this detector-scan.
    """
    nbin = tod_samples.TOD_PS_NBIN
    freqs_binned, raw_binned = _binned_tod_power_spectrum(view.tod, view.fsamp, nbin)
    tod_samples.tod_ps_freqs[iscan, idet] = freqs_binned
    tod_samples.tod_ps_raw[iscan, idet] = raw_binned

    # Sky+orbital-dipole-subtracted residual, and the TOD with only the correlated noise removed.
    # Both are fresh writable copies, so n_corr (when present) is subtracted in place from each.
    residual_tod = view.get_tod(subtract=(("sky", TODView._ALL_GAIN_TERMS),
                                          ("orbital_dipole", TODView._ALL_GAIN_TERMS)))
    ncorrsub_tod = view.get_tod()
    if n_corr is not None:
        residual_tod -= n_corr
        ncorrsub_tod -= n_corr
    _, residual_binned = _binned_tod_power_spectrum(residual_tod, view.fsamp, nbin)
    _, ncorrsub_binned = _binned_tod_power_spectrum(ncorrsub_tod, view.fsamp, nbin)
    tod_samples.tod_ps_residual[iscan, idet] = residual_binned
    tod_samples.tod_ps_ncorrsub[iscan, idet] = ncorrsub_binned

    if n_corr is not None:
        _, ncorr_binned = _binned_tod_power_spectrum(n_corr, view.fsamp, nbin)
        tod_samples.tod_ps_ncorr[iscan, idet] = ncorr_binned
        if tod_samples.ncorr_tods is not None:
            tod_samples.ncorr_tods[iscan][idet] = n_corr.astype(np.float32, copy=False)

    # White-noise-residual chi-squared z-score (uses this iteration's sigma0, sampled just above);
    # stored to the chain and consumed by the data-selection (accept) cuts.
    tod_samples.chisq_z[iscan, idet] = masked_chisq_z(
        residual_tod, view.get_mask(proc_mask=False), view.sigma0)


def tod2map_CG(band_comm: MPI.Comm, experiment_data: DetGroupTOD, compsep_output: NDArray,
               tod_samples: TODSamples, iteration: int,
               mapmaking: "MapmakingConfig", correlated_noise: "CorrelatedNoiseConfig",
               data_selection: "DataSelectionConfig",
               ) -> tuple[dict[str, DetectorMap], dict[str, NDArray]]:
    """ Commander4 CG mapmaking. All ranks on the provided MPI communicator collaborates on creating
        the band maps (sky signal, inverse variance, possibly also aux maps like orbital dipole).
    Args:
        band_comm (Comm): The communicator consisting of all MPI ranks which holds TOD data that
                          should go into the same map.
        experiment_data (DetGroupTOD): TOD data class to be made into maps.
        compsep_output (NDArray): The sky model at our band. Not used, but written to chain file.
        tod_samples (TODSamples): Sampled TOD parameters, such as gain.
        iteration: Current Gibbs iteration.
        mapmaking: Validated mapmaking settings.
        correlated_noise: Validated correlated-noise settings.
        data_selection: Validated detector-scan selection settings.
    Output:
        Detector maps for component separation and maps selected for chain output.

    """
    ismaster = band_comm.Get_rank() == 0
    corr_noise_active = correlated_noise.is_active(iteration)
    selection_active = data_selection.cuts_are_active(iteration, correlated_noise)
    ### CG MAPMAKER ###
    # Single fused scan loop (mirrors Commander3's process_TOD): each detector-scan samples
    # correlated noise / sigma0 *first*, then every sigma0-dependent quantity -- inverse-variance
    # weights (preconditioner + rms/cov), orbital-dipole and corr-noise maps, and the CG RHS -- is
    # accumulated with that freshly-sampled sigma0. Previously the inverse-variance map was built in a
    # separate up-front pass on the previous iteration's sigma0, which (since the LHS operator and
    # preconditioner read the live, updated sigma0) left the CG RHS inconsistent with its own A.
    pols = experiment_data.pols
    scan_view = TODView(experiment_data, tod_samples, compsep_output=compsep_output)
    # Optional per-experiment sparse map storage: each rank holds only its locally-observed pixels
    # rather than a full sky map. The band master still ends up with full-sky maps.
    domain = experiment_data.get_pixel_domain(scan_view, band_comm, mapmaking.sparse_maps)
    # The inverse-variance map (preconditioner + rms/cov) is accumulated inside the fused loop below,
    # so it -- and thus cg_mapmaker.M -- can only be finalized afterwards. cg_mapmaker is constructed
    # here with a placeholder preconditioner; M is unused until solve() and accum_to_RHS never reads
    # it, so it is reassigned to the real Jacobi preconditioner after the loop.
    if pols == "IQU":
        mapmaker_invvar = WeightsMapmakerIQU(band_comm, experiment_data.nside, pixel_domain=domain)
        cg_mapmaker = CGMapmakerIQU(experiment_data, tod_samples, band_comm,
                    preconditioner=called_on_non_master, nthreads=mapmaking.num_threads,
                    CG_maxiter=mapmaking.cg.max_iter, pixel_domain=domain)
    elif pols == "I":
        mapmaker_invvar = WeightsMapmaker(band_comm, experiment_data.nside, pixel_domain=domain)
        cg_mapmaker = CGMapmakerI(experiment_data, tod_samples, band_comm,
                    preconditioner=called_on_non_master, nthreads=mapmaking.num_threads,
                    CG_maxiter=mapmaking.cg.max_iter, pixel_domain=domain)
    else:
        raise ValueError(f"specified polarizations {pols} is notsupported yet.")

    BinMapmaker = MapmakerIQU if pols == "IQU" else Mapmaker #general bin mapmaker class object.
    # mapmaker = BinMapmaker(band_comm, experiment_data.nside)
    mapmaker_orbdipole = BinMapmaker(band_comm, experiment_data.nside, pixel_domain=domain)

    if corr_noise_active:
        mapmaker_ncorr = BinMapmaker(band_comm, experiment_data.nside, pixel_domain=domain)
        sampled_params = []
        residuals = []
        niters = []
        num_failed_convergences_ncorr = 0
        num_too_high_var_ncorr = 0
        worst_residual_ncorr = 0

    ### MAIN SCAN LOOP ###
    for view in scan_view.iter_focused(accepted_only=True):
        # Full-length pointing (no good_data_mask compaction): the CG operator gap-fills flagged
        # samples rather than removing them, so every sample carries weight (gain/sigma0)^2 and the
        # inverse-variance / preconditioner must count them all to match the A operator.
        pix, psi = view.pix, view.psi
        good_data_mask = view.get_mask(proc_mask=False)
        gain = view.get_gain()
        response = view.det_response if pols == "IQU" else None

        ### DATA-SELECTION VETO 1 (too little unflagged data).
        good_frac = good_data_mask.mean()
        tod_samples.good_fraction[view.iscan, view.idet] = good_frac
        if selection_active and good_frac < data_selection.min_good_fraction:
            tod_samples.accept[view.iscan, view.idet] = False
            continue

        ### CORRELATED NOISE / SIGMA0 SAMPLING (first, so the weights below use the new sigma0) ###
        n_corr_est = None
        if corr_noise_active:
            sky_subtracted_TOD = view.get_tod(
                subtract=(("sky", TODView._ALL_GAIN_TERMS),
                          ("orbital_dipole", TODView._ALL_GAIN_TERMS)),
            )
            res = sample_correlated_noise(
                sky_subtracted_TOD, view.get_mask(proc_mask_type="ncorr"),
                np.array(view.noise_params, copy=True),
                experiment_data.noise_model, view.fsamp, cg_err_tol=correlated_noise.cg.err_tol,
                cg_max_iter=correlated_noise.cg.max_iter,
                sample_params=correlated_noise.sample_psd_params,
                sample_sigma0=correlated_noise.sample_sigma0,
                sigma0_method=correlated_noise.sigma0_method,
                nomono=correlated_noise.nomono,
                onlymono=correlated_noise.onlymono,
                sigma0_dec=correlated_noise.sigma0_decimation,
                psd_fit_nu_min=correlated_noise.psd_fit_nu_min,
                psd_fit_nu_max=correlated_noise.psd_fit_nu_max,
                psd_bin=correlated_noise.psd_bin)
            n_corr_est = res.n_corr
            tod_samples.noise_params[view.iscan, view.idet, :] = res.noise_params
            if correlated_noise.sample_psd_params:
                sampled_params.append(np.array(res.noise_params, copy=True))
            if not res.converged:
                num_failed_convergences_ncorr += 1
            if res.high_var:
                num_too_high_var_ncorr += 1
            worst_residual_ncorr = max(worst_residual_ncorr, res.residual)
            residuals.append(res.residual)
            niters.append(res.niter)
        elif correlated_noise.sample_sigma0:
            # No correlated noise this iteration: estimate sigma0 here, at the same point in the
            # chain (after gain) as the n_corr-coupled estimate, instead of a separate pre-gain pass.
            tod_samples.noise_params[view.iscan, view.idet, 0] = _estimate_standalone_sigma0(
                view, correlated_noise.sigma0_method)

        # Diagnostics (incl. chisq_z) before any accumulation, so a veto below leaves this
        # detector-scan out of every map product (the CG operator passes re-read `accept`).
        _record_tod_diagnostics(tod_samples, view.iscan, view.idet, view, n_corr_est)

        ### DATA-SELECTION VETO 2 (catastrophic chi^2), applied in-loop so this iteration's maps
        ### already exclude the scan.
        if selection_active:
            z = tod_samples.chisq_z[view.iscan, view.idet]
            if not (np.isfinite(z) and abs(z) <= data_selection.chisq_abs_threshold):
                tod_samples.accept[view.iscan, view.idet] = False
                continue

        # sigma0 now reflects this iteration's estimate; every weight below is consistent with it.
        sigma0 = view.sigma0
        inv_var = (gain/sigma0)**2

        ### INVERSE-VARIANCE WEIGHTS (preconditioner + rms/cov) ###
        if pols == "IQU":
            mapmaker_invvar.accumulate_to_map(inv_var, pix, psi, response=response)
        else:
            mapmaker_invvar.accumulate_to_map(inv_var, pix)

        ### ORBITAL DIPOLE ###
        sky_orb_dipole = view.get_orbital_dipole_tod()
        d_sky = view.get_tod(subtract=(("orbital_dipole", TODView._ALL_GAIN_TERMS),))
        if pols == "IQU":
            mapmaker_orbdipole.accumulate_to_map(sky_orb_dipole, inv_var, pix, psi,
                                                 response=response)
        else:
            mapmaker_orbdipole.accumulate_to_map(sky_orb_dipole, inv_var, pix, psi)

        ### CORRELATED-NOISE MAP ###
        if corr_noise_active:
            if pols == "IQU":
                mapmaker_ncorr.accumulate_to_map(
                    (n_corr_est/gain).astype(np.float32, copy=False),
                    inv_var, pix, psi, response=response)
            else:
                mapmaker_ncorr.accumulate_to_map(
                    (n_corr_est/gain).astype(np.float32, copy=False),
                    inv_var, pix, psi)
            d_sky -= n_corr_est

        # Gap-fill flagged samples instead of compacting them away. The CG operator applies a
        # Fourier transform (apply_T), which requires a continuous, full-length TOD: removing masked
        # samples corrupts the FFT, and a single non-finite sample (or an empty compacted scan)
        # otherwise poisons/crashes the whole solve. fill_all_masked (linear interpolation + white
        # noise) is the same gap-filling used in correlated-noise sampling; the filled samples are
        # noisy realizations carrying weight 1/sigma0^2, consistent with the full-length A operator.
        fill_all_masked(d_sky, good_data_mask, sigma0)

        cg_mapmaker.accum_to_RHS(
                    scan_tod=view.detector,
                    sigma0=sigma0,
                    pix=pix,
                    psi=psi,
                    scan_tod_arr=d_sky/gain
                    )

    ### PRINT NOISE SAMPLING STATS ###
    if corr_noise_active:
        log_corr_noise_stats(band_comm, experiment_data.nu, experiment_data.noise_model,
                             sampled_params, residuals, niters, num_failed_convergences_ncorr,
                             num_too_high_var_ncorr, worst_residual_ncorr,
                             sum(len(s.detectors) for s in experiment_data.scans))


    ### FINALIZE INVERSE-VARIANCE MAP, BUILD PRECONDITIONER, GATHER/NORMALIZE ###
    # The inverse-variance map is now complete (accumulated with this iteration's sigma0); finalize
    # it and assign cg_mapmaker.M before solving, so the preconditioner matches the RHS and the LHS
    # operator (which reads the live sigma0 too).
    mapmaker_invvar.gather_map()
    if pols == "IQU":
        mapmaker_invvar.normalize_map()
        if ismaster:
            # Jacobi preconditioner M = 1/diag(A), where A is the accumulated inverse-noise matrix
            # (final_cov_map holds its 6 unique elements; [0,3,5] are A_II, A_QQ, A_UU). The previous
            # choice -- diag(A^-1) via rms**2 -- blows up at near-singular pixels (poor per-pixel
            # polarization-angle coverage, where the 3x3 inverse is inflated by a vanishing
            # determinant), wrecking the conditioning and making PCG diverge. 1/diag(A) stays bounded
            # by the actual per-component inverse variance; without_nan zeros unobserved pixels.
            A_diag = mapmaker_invvar.final_cov_map[(0, 3, 5), :]
            cg_mapmaker.M = InvNPreconditionerIQU(utils.without_nan(1.0 / A_diag))
        map_rms = mapmaker_invvar.final_rms_map
        map_cov = mapmaker_invvar.final_cov_map
    else:
        if ismaster:
            cg_mapmaker.M = InvNPreconditionerI(utils.without_nan(1./mapmaker_invvar.final_map))
        map_cov = mapmaker_invvar.final_map
        map_rms = 1./np.sqrt(map_cov)

    mapmaker_orbdipole.gather_map()
    mapmaker_orbdipole.normalize_map(map_cov)
    map_orbdipole = mapmaker_orbdipole.final_map
    cg_mapmaker.finalize_RHS()
    cg_mapmaker.solve()
    map_signal = cg_mapmaker.solved_map

    if corr_noise_active:
        mapmaker_ncorr.gather_map()
        mapmaker_ncorr.normalize_map(map_cov)
        map_corrnoise = mapmaker_ncorr.final_map

    ### FINAL CLEANUP ON MASTER RANK ###
    detmap_dict_out = {}
    maps_to_file = {}
    if band_comm.Get_rank() == 0:
        #Here we split here between I and QU
        # Smooth maps to the common analysis resolution after mapmaking; 0 leaves bands at their
        # native beam.
        common_res_fwhm = mapmaking.common_res_fwhm
        if "I" in pols:
            detmap_I = DetectorMap(map_signal[0,:], map_rms[0,:], experiment_data.nu,
                                experiment_data.fwhm, experiment_data.nside)
            detmap_I.g0 = tod_samples.abs_gain
            if common_res_fwhm:
                detmap_I.smooth_to_resolution(common_res_fwhm)
            detmap_dict_out.update({"I": detmap_I})
        if "QU" in pols:
            detmap_QU = DetectorMap(map_signal[1:3,:], map_rms[1:3,:], experiment_data.nu,
                                experiment_data.fwhm, experiment_data.nside)
            detmap_QU.g0 = tod_samples.abs_gain
            if common_res_fwhm:
                detmap_QU.smooth_to_resolution(common_res_fwhm)
            detmap_dict_out.update({"QU": detmap_QU})

        maps_to_file["map_observed_sky"] = map_signal
        maps_to_file["map_rms"] = map_rms
        if mapmaking.include_orbital_dipole_maps:
            maps_to_file["map_orbdipole"] = map_orbdipole
        if mapmaking.include_corr_noise_maps and corr_noise_active:
            maps_to_file["map_corrnoise"] = map_corrnoise
        if mapmaking.include_sky_model_maps:
            maps_to_file["map_skymodel"] = compsep_output

    return detmap_dict_out, maps_to_file


def tod2map_bin(band_comm: MPI.Comm, experiment_data: DetGroupTOD, compsep_output: NDArray,
                tod_samples: TODSamples, iteration: int,
                mapmaking: "MapmakingConfig", correlated_noise: "CorrelatedNoiseConfig",
                data_selection: "DataSelectionConfig",
                ) -> tuple[dict[str, DetectorMap], dict[str, NDArray]]:
    """ Commander4 bin mapmaking. All ranks on the provided MPI communicator collaborates on creating
        the band maps (sky signal, inverse variance, possibly also aux maps like orbital dipole).
    Args:
        band_comm (Comm): The communicator consisting of all MPI ranks which holds TOD data that
                          should go into the same map.
        experiment_data (DetGroupTOD): TOD data class to be made into maps.
        compsep_output (NDArray): The sky model at our band. Not used, but written to chain file.
        tod_samples (TODSamples): Sampled TOD parameters, such as gain.
        iteration: Current Gibbs iteration.
        mapmaking: Validated mapmaking settings.
        correlated_noise: Validated correlated-noise settings.
        data_selection: Validated detector-scan selection settings.
    Output:
        Detector maps for component separation and maps selected for chain output.

    """
    start_bench("binned-mapmaker")
    corr_noise_active = correlated_noise.is_active(iteration)
    selection_active = data_selection.cuts_are_active(iteration, correlated_noise)
    pols = experiment_data.pols
    scan_view = TODView(experiment_data, tod_samples, compsep_output=compsep_output)
    # Optional per-experiment sparse map storage: each rank holds only its locally-observed pixels
    # rather than a full sky map. The band master still ends up with full-sky maps.
    domain = experiment_data.get_pixel_domain(scan_view, band_comm, mapmaking.sparse_maps)

    # Set up various mapmakers.
    mapmaker_invvar = WeightsMapmakerIQU(band_comm, experiment_data.nside, pixel_domain=domain)
    mapmaker = MapmakerIQU(band_comm, experiment_data.nside, pixel_domain=domain)
    mapmaker_orbdipole = MapmakerIQU(band_comm, experiment_data.nside, pixel_domain=domain)
    if corr_noise_active:
        mapmaker_ncorr = MapmakerIQU(band_comm, experiment_data.nside, pixel_domain=domain)
        sampled_params = []
        residuals = []
        niters = []
        num_failed_convergences_ncorr = 0
        num_too_high_var_ncorr = 0
        worst_residual_ncorr = 0
    stop_bench("binned-mapmaker")

    ### MAIN SCAN LOOP ###
    for view in scan_view.iter_focused(accepted_only=True):
        start_bench("binned-mapmaker")
        good_data_mask = view.get_mask(proc_mask=False)
        pix, psi = view.pix, view.psi
        pix_masked = pix[good_data_mask]
        psi_masked = psi[good_data_mask]
        response = view.det_response
        gain = view.get_gain()
        stop_bench("binned-mapmaker", increment_count=False)

        ### DATA-SELECTION VETO 1 (too little unflagged data).
        good_frac = good_data_mask.mean()
        tod_samples.good_fraction[view.iscan, view.idet] = good_frac
        if selection_active and good_frac < data_selection.min_good_fraction:
            tod_samples.accept[view.iscan, view.idet] = False
            continue

        ### CORRELATED NOISE / SIGMA0 SAMPLING (first, so the weights below use the new sigma0) ###
        n_corr_est = None
        if corr_noise_active:
            start_bench("ncorr-sampling")
            sky_subtracted_TOD = view.get_tod(
                subtract=(("sky", TODView._ALL_GAIN_TERMS),
                          ("orbital_dipole", TODView._ALL_GAIN_TERMS)),
            )
            res = sample_correlated_noise(
                sky_subtracted_TOD, view.get_mask(proc_mask_type="ncorr"),
                np.array(view.noise_params, copy=True),
                experiment_data.noise_model, view.fsamp, cg_err_tol=correlated_noise.cg.err_tol,
                cg_max_iter=correlated_noise.cg.max_iter,
                sample_params=correlated_noise.sample_psd_params,
                sample_sigma0=correlated_noise.sample_sigma0,
                sigma0_method=correlated_noise.sigma0_method,
                nomono=correlated_noise.nomono,
                onlymono=correlated_noise.onlymono,
                sigma0_dec=correlated_noise.sigma0_decimation,
                psd_fit_nu_min=correlated_noise.psd_fit_nu_min,
                psd_fit_nu_max=correlated_noise.psd_fit_nu_max,
                psd_bin=correlated_noise.psd_bin)
            n_corr_est = res.n_corr
            tod_samples.noise_params[view.iscan, view.idet, :] = res.noise_params
            if correlated_noise.sample_psd_params:
                sampled_params.append(np.array(res.noise_params, copy=True))
            if not res.converged:
                num_failed_convergences_ncorr += 1
            if res.high_var:
                num_too_high_var_ncorr += 1
            worst_residual_ncorr = max(worst_residual_ncorr, res.residual)
            residuals.append(res.residual)
            niters.append(res.niter)
            stop_bench("ncorr-sampling")
        elif correlated_noise.sample_sigma0:
            # No correlated noise this iteration: estimate sigma0 here, at the same point in the
            # chain (after gain) as the n_corr-coupled estimate, instead of a separate pre-gain pass.
            tod_samples.noise_params[view.iscan, view.idet, 0] = _estimate_standalone_sigma0(
                view, correlated_noise.sigma0_method)

        _record_tod_diagnostics(tod_samples, view.iscan, view.idet, view, n_corr_est)

        ### DATA-SELECTION VETO 2 (catastrophic chi^2)
        if selection_active:
            z = tod_samples.chisq_z[view.iscan, view.idet]
            if not (np.isfinite(z) and abs(z) <= data_selection.chisq_abs_threshold):
                tod_samples.accept[view.iscan, view.idet] = False
                continue

        start_bench("binned-mapmaker")
        # Retrieve the new sigma0 for this det-scan, sampled above.
        sigma0 = view.sigma0
        # sigma0 is in detector-units, transform into uK_RJ by dividing it by the gain.
        inv_var = (gain/sigma0)**2
        mapmaker_invvar.accumulate_to_map(inv_var, pix_masked, psi_masked, response=response)

        ### ORBITAL DIPOLE ###
        sky_orb_dipole = view.get_orbital_dipole_tod()
        d_sky = view.get_tod(subtract=(("orbital_dipole", TODView._ALL_GAIN_TERMS),))

        # If we're doing ncorr, accumulate to map and subtract from sky TOD.
        if corr_noise_active:
            mapmaker_ncorr.accumulate_to_map(
                (n_corr_est[good_data_mask]/gain).astype(np.float32, copy=False),
                inv_var, pix_masked, psi_masked, response=response)
            d_sky -= n_corr_est

        d_sky_masked = d_sky[good_data_mask]
        mapmaker.accumulate_to_map(d_sky_masked/gain, inv_var, pix_masked, psi_masked, response=response)
        mapmaker_orbdipole.accumulate_to_map(sky_orb_dipole[good_data_mask], inv_var,
                                             pix_masked, psi_masked, response=response)
        stop_bench("binned-mapmaker", increment_count=False)
    if corr_noise_active:
        log_memory("ncorr-sampling")

    ### PRINT NOISE SAMPLING STATS ###
    if corr_noise_active:
        log_corr_noise_stats(band_comm, experiment_data.nu, experiment_data.noise_model,
                             sampled_params, residuals, niters, num_failed_convergences_ncorr,
                             num_too_high_var_ncorr, worst_residual_ncorr,
                             sum(len(s.detectors) for s in experiment_data.scans))


    start_bench("binned-mapmaker")
    ### GATHER AND NORMALIZE MAPS ###
    # Finalize the inverse-variance map (now accumulated with this iteration's sigma0) before reading
    # its rms/cov, which normalize the signal, orbital-dipole, and corr-noise maps below.
    mapmaker_invvar.gather_map()
    mapmaker_invvar.normalize_map()
    mapmaker.gather_map()
    mapmaker_orbdipole.gather_map()
    map_rms = mapmaker_invvar.final_rms_map
    map_cov = mapmaker_invvar.final_cov_map
    mapmaker.normalize_map(map_cov)
    map_signal = mapmaker.final_map
    mapmaker_orbdipole.normalize_map(map_cov)
    map_orbdipole = mapmaker_orbdipole.final_map
    if corr_noise_active:
        mapmaker_ncorr.gather_map()
        mapmaker_ncorr.normalize_map(map_cov)
        map_corrnoise = mapmaker_ncorr.final_map
    stop_bench("binned-mapmaker", increment_count=False)
    log_memory("binned-mapmaker")

    ### FINAL CLEANUP ON MASTER RANK ###
    detmap_dict_out = {}
    maps_to_file = {}
    if band_comm.Get_rank() == 0:
        #Here we split here between I and QU
        # Smooth maps to the common analysis resolution after mapmaking; 0 leaves bands at their
        # native beam.
        common_res_fwhm = mapmaking.common_res_fwhm
        if "I" in pols:
            detmap_I = DetectorMap(map_signal[0,:], map_rms[0,:], experiment_data.nu,
                                experiment_data.fwhm, experiment_data.nside)
            detmap_I.g0 = tod_samples.abs_gain
            if common_res_fwhm:
                detmap_I.smooth_to_resolution(common_res_fwhm)
            detmap_dict_out.update({"I": detmap_I})
        if "QU" in pols:
            detmap_QU = DetectorMap(map_signal[1:3,:], map_rms[1:3,:], experiment_data.nu,
                                experiment_data.fwhm, experiment_data.nside)
            detmap_QU.g0 = tod_samples.abs_gain
            if common_res_fwhm:
                detmap_QU.smooth_to_resolution(common_res_fwhm)
            detmap_dict_out.update({"QU": detmap_QU})

        maps_to_file["map_observed_sky"] = map_signal
        maps_to_file["map_rms"] = map_rms
        if mapmaking.include_orbital_dipole_maps:
            maps_to_file["map_orbdipole"] = map_orbdipole
        if mapmaking.include_corr_noise_maps and corr_noise_active:
            maps_to_file["map_corrnoise"] = map_corrnoise
        if mapmaking.include_sky_model_maps:
            maps_to_file["map_skymodel"] = compsep_output

    return detmap_dict_out, maps_to_file


def init_tod_processing(mpi_info: Bunch, params: Bunch) -> tuple[Bunch, str, DetGroupTOD,
                                                                 TODSamples, TODSamples]:
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

    # We now loop over all bands in all experiments, and allocate them to the first ranks of the
    # TOD MPI communicator. These ranks will then become the "band masters" for those bands,
    # handling all communication with CompSep.
    # All the non-master ranks will have None values, and receive info from master further down.
    det_names = []
    my_band_name = None
    my_experiment = None
    my_band = None
    my_band_id = None
    my_band_pol = None #string identifying the polarization type, e.g. "IQU", "I", "QU"
    my_scans_start = None
    my_scans_stop = None
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
                my_experiment = experiment
                # Setting our unique detector id. Note that this is a global, not per band.
                tot_num_scans = resolve_param(params, "num_scans",
                                              (f"experiments.{exp_name}.bands.{band_name}",
                                               f"experiments.{exp_name}"))
                scans = np.arange(tot_num_scans)
                my_scans = np.array_split(scans, mpi_info.band.size)[mpi_info.band.rank]
                my_scans_start = my_scans[0]
                my_scans_stop = my_scans[-1]
                det_names = [det for det in band.detectors]
    mpi_info.tod.comm.Barrier()

    time.sleep(mpi_info.tod.rank*1e-5)  # Small sleep to get prints in nice order.
    # MPIcolor_band = MPIrank_tod%tot_num_bands  # Spread the MPI tasks over the different bands.
    band_comm = mpi_info.band.comm
    logger.debug(f"TOD-rank {mpi_info.tod.rank:4} (on machine {mpi_info.processor_name}), "\
                 f"dedicated to band {my_band_id:4}, with local rank {mpi_info.band.rank:4} "\
                 f"(local communicator size: {mpi_info.band.size:4}).")

    t0 = time.time()
    with benchmark("fileread-tod"):
        experiment_data = read_tods_from_file(band_comm, my_experiment, my_band, det_names, params,
                                              my_scans_start, my_scans_stop)
    mpi_info.tod.comm.Barrier()
    if mpi_info.tod.is_master:
        logger.info(f"TOD: Finished reading all files in {time.time()-t0:.1f}s.")

    tod_samples_chain1 = TODSamples(experiment_data, params, my_band, band_comm, 1)
    tod_samples_chain2 = TODSamples(experiment_data, params, my_band, band_comm, 2)

    # Build the band's map-distribution PixelDomain once now: the pointing is static, so it is
    # reused across Gibbs iterations by both the mapmakers and the sky-model distribution (which
    # needs it to give each rank only its local pixels). In full mode this is a cheap no-op.
    sparse_maps = resolve_param(params, "sparse_maps",
                                (f"experiments.{experiment_data.experiment_name}",),
                                default=MapmakingConfig.sparse_maps)
    experiment_data.get_pixel_domain(TODView(experiment_data, tod_samples_chain1), band_comm,
                                     sparse_maps)

    # Creating "tod_band_masters", an array which maps the band index to the rank of the master
    # of that band.
    todproc_my_band_id = my_band_name
    data_world = (
        todproc_my_band_id,
        mpi_info.world.rank,
        my_band_pol,
    ) if mpi_info.band.is_master else None
    data_tod = (todproc_my_band_id, mpi_info.tod.rank) if mpi_info.band.is_master else None
    all_data_world = mpi_info.tod.comm.allgather(data_world)
    all_data_tod = mpi_info.tod.comm.allgather(data_tod)

    world_band_masters_dict = {
        execution_band_id: item[1]
        for item in all_data_world if item is not None
        for execution_band_id in get_execution_band_ids(item[0], item[2])
    }
    tod_band_masters_dict = {item[0]: item[1] for item in all_data_tod if item is not None}
    mpi_info['world']['tod_band_masters'] = world_band_masters_dict
    mpi_info['tod']['tod_band_masters'] = tod_band_masters_dict

    return mpi_info, todproc_my_band_id, experiment_data, tod_samples_chain1, tod_samples_chain2


def _estimate_standalone_sigma0(view: TODView, sigma0_method: str) -> float:
    """ White-noise sigma0 for one detector-scan when correlated noise is *not* being sampled.

    Estimated from the sky- and orbital-dipole-subtracted residual (which still contains the 1/f
    component; both estimators target the white floor). This mirrors the sigma0 estimate that
    ``sample_correlated_noise`` performs when n_corr is sampled, so sigma0 is always (re)estimated at
    the same point in the chain -- inside the mapmaker scan loop, after gain -- matching Commander3.

    Args:
        view: The focused TODView for one detector-scan.
        sigma0_method: ``'pairwise'`` (first-difference) or ``'binned_psd'`` (bottom of binned PSD).
    Returns:
        The estimated white-noise level (float).
    """
    residual = view.get_tod(subtract=(("sky", TODView._ALL_GAIN_TERMS),
                                      ("orbital_dipole", TODView._ALL_GAIN_TERMS)))
    mask = view.get_mask(proc_mask_type="ncorr")
    if sigma0_method == "binned_psd":
        sigma0 = calc_sigma0_binned_psd(residual, mask, view.fsamp)
    else:
        sigma0 = calc_sigma0_robust(residual, mask)
    logassert(sigma0 != 0, "sigma0 is 0, which should never happen.", logger)
    logassert(sigma0 != np.inf, "sigma0 is inf, which should never happen.", logger)
    return sigma0


def sample_jump_detection(band_comm: MPI.Comm, experiment_data: DetGroupTOD,
                          tod_samples: TODSamples,
                          config: "JumpDetectionConfig") -> TODSamples:
    """Detect jump discontinuities from the flag stream and store additive post-jump offsets.

    A jump is identified by a contiguous region with a non-zero
    ``flag & experiments.[experiment_name].jump_bitmask``. For each region, the offset is
    estimated from the last ``window`` valid samples before the jump and the first ``window``
    valid samples after it, where validity is defined by ``full_mask``. The correction is then
    applied to all later samples when a TOD is requested through ``TODView.get_tod()``.
    """
    scan_view = TODView(experiment_data, tod_samples)
    num_applied_local = 0
    num_skipped_local = 0
    offsets_local = []
    jump_counts_local = []

    for view in scan_view.iter_focused():
        # Jump detection needs the flag stream both to locate jumps (via jump_bitmask) and to
        # define valid pre/post-jump samples; skip detector-scans without it.
        if getattr(view.detector, "_flag_encoded", None) is None:
            tod_samples.jumps.set(view.iscan, view.idet, None)
            jump_counts_local.append(0)
            continue
        jump, num_skipped = JumpCorrection.detect(
            view.tod,
            view.flag,
            view.get_mask(proc_mask_type="jump"),
            config.window,
            jump_bitmask=config.jump_bitmask,
        )
        tod_samples.jumps.set(view.iscan, view.idet, jump)
        jump_counts_local.append(jump.size)
        num_skipped_local += num_skipped
        if not jump.is_empty():
            offsets_local.extend(jump.offsets.astype(np.float64, copy=False))
            num_applied_local += jump.size

    num_applied = band_comm.reduce(num_applied_local, op=MPI.SUM, root=0)
    num_skipped = band_comm.reduce(num_skipped_local, op=MPI.SUM, root=0)
    gathered_offsets = band_comm.gather(np.asarray(offsets_local, dtype=np.float64), root=0)
    gathered_jump_counts = band_comm.gather(np.asarray(jump_counts_local, dtype=np.int32), root=0)

    if band_comm.Get_rank() == 0:
        all_jump_counts = np.concatenate(gathered_jump_counts) if gathered_jump_counts else np.empty(0)
        if all_jump_counts.size > 0:
            logger.debug(
                f"Band {experiment_data.band_name} jump counts per detector-scan: "
                f"min={np.min(all_jump_counts)}, avg={np.mean(all_jump_counts):.2f}, "
                f"max={np.max(all_jump_counts)} over {all_jump_counts.size} samples."
            )
        if num_applied > 0:
            all_offsets = np.concatenate([arr for arr in gathered_offsets if arr.size > 0])
            logger.info(f"Band {experiment_data.band_name} jump detection: applied {num_applied} "
                        f"offsets, skipped {num_skipped}, median |offset| = "
                        f"{np.median(np.abs(all_offsets)):.3e}.")
        elif num_skipped > 0:
            logger.info(f"Band {experiment_data.band_name} jump detection skipped {num_skipped} "
                        f"flagged regions because there were not enough valid samples around them.")

    log_memory("jump-detect")
    return tod_samples


# Which signal a gain term is calibrated against -- i.e. what the calibration residual is reduced
# to. `TODView.get_calib_tod` owns the actual signal bookkeeping; the three choices are:
#   orbital_dipole -- the CMB dipole induced by the observer's motion, alone. It is known a priori
#                     from the spacecraft velocity and the CMB monopole temperature, so it does not
#                     depend on the sky model. That makes it the only *absolute* calibrator,
#                     and the natural default for the absolute gain (the average across detectors).
#   sky            -- the entire modelled signal, static sky and orbital dipole. The default for
#                     the relative and temporal terms: those only have to track gain *differences*
#                     between detectors or scans, so they can use every bit of signal available.
#   sky_no_dipole  -- the static sky model from component separation, with the dipole left out.
# Each term's default and any per-band override are resolved by ``GainConfig.from_params``.
_VALID_CALIB_TARGETS = ("orbital_dipole", "sky", "sky_no_dipole")

# Above this freq the orbital dipole is a poor calibrator, getting faint compared to foregrounds.
_ORBITAL_DIPOLE_MAX_FREQ_GHZ = 400.0


def _solve_relative_gain_system(s_weights: NDArray, r_weights: NDArray, prev_rel_gain: NDArray,
                                rng=None) -> NDArray:
    """ Draw relative-gain deviations Delta g_i from the BP7 Sec. 3.4 constrained Gaussian.

        Solves the bordered linear system enforcing ``sum(Delta g_i) = 0`` over the *active*
        detectors only -- those with nonzero calibration weight ``s_weights``. A detector rejected
        on every scan (or with a vanishing calibrator) has ``s_weights == 0``; it would contribute
        an all-zero row/column and, if two or more are present, make the matrix singular. Such
        detectors are excluded from the solve (shrinking the system to the active set) and held at
        their current relative gain.

    Args:
        s_weights (NDArray): Per-detector ``sum_scans s^T N^-1 s`` (calibration weight), shape (ndet,).
        r_weights (NDArray): Per-detector ``sum_scans r^T N^-1 s`` (residual projection), shape (ndet,).
        prev_rel_gain (NDArray): Current relative gains, kept for excluded detectors, shape (ndet,).
        rng: Optional NumPy random generator for the fluctuation term (defaults to ``np.random``).

    Returns:
        NDArray: Full-length (ndet,) float32 relative-gain vector with the active entries resampled.

    Raises:
        np.linalg.LinAlgError: If the reduced system is singular (left for the caller to handle).
    """
    rng = np.random if rng is None else rng
    out = np.array(prev_rel_gain, dtype=np.float32)
    idx = np.flatnonzero(np.asarray(s_weights) > 0.0)
    n = idx.size
    if n == 0:
        return out
    d = np.asarray(s_weights)[idx].astype(np.float64)
    r = np.asarray(r_weights)[idx].astype(np.float64)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = np.diag(d)
    A[:n, n] = 0.5     # Lagrange-multiplier column enforcing the zero-sum constraint.
    A[n, :n] = 1.0     # Constraint row: sum of active Delta g_i = 0.
    b = np.zeros(n + 1)
    b[:n] = r + np.sqrt(d) * rng.standard_normal(n)
    solution = np.linalg.solve(A, b)   # Raises LinAlgError if singular.
    out[idx] = solution[:n].astype(np.float32)
    return out


def sample_absolute_gain(band_comm: MPI.Comm, experiment_data: DetGroupTOD,
                         tod_samples: TODSamples, det_compsep_map: NDArray,
                         config: "GainConfig") -> TODSamples:
    """ Draw a realization of the absolute gain term, g0, which is constant across all
        detectors and all scans within a band, using the calibrator selected by ``config``.
    Args:
        band_comm (MPI.Comm): The band-level MPI communicator.
        experiment_data (DetGroupTOD): The object holding all the scan data.
        tod_samples (TODSamples): Current sampled TOD parameters (updated in-place with g0).
        det_compsep_map (NDArray): The component-separation sky map for the detector.
        config: Validated absolute-gain settings.
    Returns:
        tod_samples (TODSamples): Updated TOD samples with the new g0 estimate.
    """
    sum_s_T_N_inv_d = 0  # Accumulators for the numerator and denominator of eqn 16.
    sum_s_T_N_inv_s = 0

    scan_view = TODView(experiment_data, tod_samples, compsep_output=det_compsep_map,
                        downsample_factor=config.downsample_factor)

    # Skip detector-scans flagged as bad (accepted_only); they carry no gain info.
    for view in scan_view.iter_focused(accepted_only=True):
        calib = view.get_calib_tod("abs", config.calibrate_against,
                                   gap_fill_method=config.gap_fill_method,
                                   proc_mask_type="gain")
        s_cal = calib.s_cal
        residual_tod = calib.tod

        # Calibration TODs are block-averaged, so their true rate is fsamp/downsample_factor;
        # apply_N_inv needs it to place the 1/f noise weight at the correct frequencies.
        gain_samprate = view.fsamp / view.downsample_factor
        N_inv_s = experiment_data.apply_N_inv(s_cal, view.noise_params, samprate=gain_samprate)
        N_inv_d = experiment_data.apply_N_inv(residual_tod, view.noise_params, samprate=gain_samprate)

        # Add to the numerator and denominator.
        sum_s_T_N_inv_d += np.dot(s_cal, N_inv_d)
        sum_s_T_N_inv_s += np.dot(s_cal, N_inv_s)

    # The g0 term is fully global, so we reduce across both all scans and all bands:
    sum_s_T_N_inv_d = band_comm.reduce(sum_s_T_N_inv_d, op=MPI.SUM, root=0)
    sum_s_T_N_inv_s = band_comm.reduce(sum_s_T_N_inv_s, op=MPI.SUM, root=0)
    # Default to the current value so a skipped or ill-posed solve leaves the gain unchanged.
    g_sampled = tod_samples.abs_gain
    # Rank 0 draws a sample of g0 from eq (16) from BP6, and bcasts it to the other ranks.
    if band_comm.Get_rank() == 0:
        if not np.isfinite(sum_s_T_N_inv_s) or sum_s_T_N_inv_s <= 0.0:
            logger.error(f"Band {experiment_data.band_name} absolute gain has no calibration "
                         f"weight (all detector-scans rejected or zero calibrator): not updating.")
        else:
            eta = np.random.randn()
            g_mean = sum_s_T_N_inv_d / sum_s_T_N_inv_s
            g_std = 1.0 / np.sqrt(sum_s_T_N_inv_s)
            g_sampled = g_mean + eta * g_std
            logger.info(f"Band {experiment_data.band_name} g0: {tod_samples.abs_gain:.4e} "\
                        f"-> {g_sampled:.4e} (+/- {g_std:.4e})")

    with benchmark("abs-gain-barrier"):   # reported across ranks by bench_summary
        band_comm.Barrier()
    g_sampled = band_comm.bcast(g_sampled, root=0)
    log_memory("abs-gain")

    # As of Numpy 2.0 it's good practice to explicitly cast to Python scalar types, as this would
    # otherwise have been a np.float64 type, potentially causing unexpected casting behavior later.
    tod_samples.abs_gain = float(g_sampled)

    return tod_samples


def sample_relative_gain(band_comm: MPI.Comm, experiment_data: DetGroupTOD,
                         tod_samples: TODSamples, det_compsep_map: NDArray,
                         config: "GainConfig") -> TODSamples:
    """ Samples the detector-dependent relative gain (Delta g_i). This function implements the
        logic from Sec. 3.4 of BP7.
    Args:
        band_comm (MPI.Comm): The band-level MPI communicator.
        experiment_data (DetGroupTOD): The object holding scan data for the band.
        tod_samples (TODSamples): Current sampled TOD parameters.
        det_compsep_map (NDArray): The component-separation sky map for the detector.
        config: Validated relative-gain settings.
    Returns:
        tod_samples (TODSamples): Updated TOD samples with relative gain estimates.
    """
    ndet = experiment_data.ndet

    #### 1. Local Calculation (on each rank) ###
    # Each rank calculates the sum of terms for its local subset of scans.
    # local_s_T_N_inv_s = 0.0
    local_s_T_N_inv_s = np.zeros(ndet, dtype=np.float32)

    # local_r_T_N_inv_s = 0.0
    local_r_T_N_inv_s = np.zeros(ndet, dtype=np.float32)
    scan_view = TODView(experiment_data, tod_samples, compsep_output=det_compsep_map,
                        downsample_factor=config.downsample_factor)

    # Skip detector-scans flagged as bad (accepted_only); they carry no gain info.
    for view in scan_view.iter_focused(accepted_only=True):
        calib = view.get_calib_tod("rel", config.calibrate_against,
                                   gap_fill_method=config.gap_fill_method,
                                   proc_mask_type="gain")
        s_cal = calib.s_cal
        residual_tod = calib.tod
        # Calibration TODs are block-averaged, so their true rate is fsamp/downsample_factor;
        # apply_N_inv needs it to place the 1/f noise weight at the correct frequencies.
        gain_samprate = view.fsamp / view.downsample_factor
        N_inv_s = experiment_data.apply_N_inv(s_cal, view.noise_params, samprate=gain_samprate)

        s_T_N_inv_s_scan = np.dot(s_cal, N_inv_s)
        r_T_N_inv_s_scan = np.dot(residual_tod, N_inv_s)

        # Add the contribution from this scan to the local sum (full-band detector column).
        local_s_T_N_inv_s[view.idet] += s_T_N_inv_s_scan
        local_r_T_N_inv_s[view.idet] += r_T_N_inv_s_scan

    ### 2. Intra-Detector Reduction ###
    # Sum the local values across all ranks that share the same detector using det_comm.
    # After this, every rank in the det_comm will have the total sum for their detector.
    band_comm.Allreduce(MPI.IN_PLACE, local_s_T_N_inv_s, op=MPI.SUM)
    band_comm.Allreduce(MPI.IN_PLACE, local_r_T_N_inv_s, op=MPI.SUM)

    ### 3. Solve Global System ###
    # Solve the constrained system (sum of Delta g_i = 0) over the active detectors only; detectors
    # rejected on every scan or with a vanishing calibrator carry zero weight, are held at their
    # current value, and are excluded so the bordered matrix stays non-singular.
    delta_g_samples = np.array(tod_samples.rel_gain, dtype=np.float32)  # default: leave unchanged
    if band_comm.Get_rank() == 0:
        n_active = int(np.count_nonzero(local_s_T_N_inv_s > 0.0))
        n_excluded = ndet - n_active
        if n_active == 0:
            logger.error(f"Band {experiment_data.band_name}: no detectors with calibration weight "
                         f"for relative gain; not updating.")
        else:
            try:
                delta_g_samples = _solve_relative_gain_system(local_s_T_N_inv_s,
                                                local_r_T_N_inv_s, tod_samples.rel_gain)
                msg = f"Solved relative gains for {n_active} active detectors"
                if n_excluded:
                    msg += f" ({n_excluded} excluded: rejected on all scans or zero calibrator)"
                logger.info(msg + ".")
            except np.linalg.LinAlgError:
                logger.error("Failed to solve linear system for relative gain: Not updating.")
    # Broadcast and apply on every rank, so all band ranks hold the identical relative-gain vector.
    prev_rel_gain = np.array(tod_samples.rel_gain)
    band_comm.Bcast(delta_g_samples, root=0)
    tod_samples.rel_gain[:] = delta_g_samples
    log_memory("rel-gain")

    if band_comm.Get_rank() == 0:
        logger.info(f"Rel gain for band {experiment_data.band_name}: min = "\
                    f"{np.min(delta_g_samples):.3e} max = {np.max(delta_g_samples):.3e}")
        logger.debug(f"Rel gains for band {experiment_data.band_name}: {delta_g_samples}\n"\
                     f"Average change = {np.mean(np.abs(prev_rel_gain - delta_g_samples))}")

    return tod_samples


def sample_temporal_gain_variations(band_comm: MPI.Comm, experiment_data: DetGroupTOD,
                                    tod_samples: TODSamples, det_compsep_map: NDArray,
                                    config: "GainConfig") -> TODSamples:
    """ Samples the time-dependent relative gain variations (delta g_qi). This function implements
        the logic from Sec. 3.5 of the BP7 paper, using a Wiener filter to smooth the gain solution
        over time (PIDs). It solves a global system for all scans of a given detector, which are
        distributed across the ranks of the band_comm.

    Args:
        band_comm (MPI.Comm): The communicator for ranks sharing the same band.
        experiment_data (DetGroupTOD): The object holding scan data.
        tod_samples (TODSamples): The sampled TOD parameters.
        det_compsep_map (NDArray): The sky model at our band.
        config: Validated temporal-gain settings.
    Returns:
        tod_samples (TODSamples): Updated TOD samples with per-scan gain variations.
    """
    band_rank = band_comm.Get_rank()
    band_size = band_comm.Get_size()
    ndet = experiment_data.ndet
    nscans_local = len(experiment_data.scans)

    # Local calculations on each rank
    A_qq_local = np.zeros((ndet, nscans_local), dtype=np.float64)
    b_q_local = np.zeros((ndet, nscans_local), dtype=np.float64)
    scan_view = TODView(experiment_data, tod_samples, compsep_output=det_compsep_map,
                        downsample_factor=config.downsample_factor)

    # I'm still not sure what way of dealing with the masked samples are best:
    # 1. Replace masked values with 0s before FFT.
    # 2. Replace masked values with n_corr realizations before FFT.
    # 3. Remove masked values by reducing TOD size before FFTs.
    # (simply passing the full data through the FFTs seems like a bad idea because of
    # ringing from the large residual in the galactic plane).
    # Rejected detector-scans (accepted_only) contribute zero weight (A_qq = b_q = 0); the Wiener
    # prior then fills their temporal gain from neighbors.
    for view in scan_view.iter_focused(accepted_only=True):
        calib = view.get_calib_tod("temp", config.calibrate_against,
                                   gap_fill_method=config.gap_fill_method,
                                   proc_mask_type="gain")
        s_cal = calib.s_cal
        residual_tod = calib.tod

        # Calibration TODs are block-averaged, so their true rate is fsamp/downsample_factor;
        # apply_N_inv needs it to place the 1/f noise weight at the correct frequencies.
        gain_samprate = view.fsamp / view.downsample_factor
        N_inv_s = experiment_data.apply_N_inv(s_cal, view.noise_params, samprate=gain_samprate)
        N_inv_r = experiment_data.apply_N_inv(residual_tod, view.noise_params, samprate=gain_samprate)

        # Calculate elements for the linear system
        A_qq = np.dot(s_cal, N_inv_s)
        b_q = np.dot(s_cal, N_inv_r)

        A_qq_local[view.idet, view.iscan] = A_qq
        b_q_local[view.idet, view.iscan] = b_q

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
                mean_gain = tod_samples.abs_gain + tod_samples.rel_gain[idet]
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
            tod_samples.temporal_gain[:,idet] = delta_g_local.astype(np.float32, copy=False)
        else:
            logger.warning(f"Rank {band_rank} received mismatched number of gain samples "\
                           f"for det {idet}. Expected {nscans_local}, got {delta_g_local.size}.")

    return tod_samples


# Each config class owns the parameter names, defaults, and validation for one TOD operation.
# ``process_tod`` still states the physical execution order explicitly. Correlated noise and data
# selection stay inside the mapmaker scan loops because their position relative to sigma0,
# diagnostics, vetoes, and map accumulation is part of the algorithm.
@dataclass(frozen=True)
class StepConfig:
    """Common parameter construction and iteration gate for a TOD step."""

    enabled: bool = False
    from_iter: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be true or false.")
        if not isinstance(self.from_iter, int) or isinstance(self.from_iter, bool):
            raise ValueError("from_iter must be an integer.")
        if self.from_iter < 1:
            raise ValueError("from_iter must be at least 1.")

    def is_active(self, iteration: int) -> bool:
        """Whether this step runs in the given Gibbs iteration."""
        return self.enabled and iteration >= self.from_iter

    @classmethod
    def _from_block(cls, block_name: str, block: Bunch | dict, **resolved_values) -> Self:
        """Construct this step config and reject fields it does not own."""
        try:
            values = dict(block)
        except (TypeError, ValueError) as error:
            raise ValueError(f"'{block_name}' must be a parameter block.") from error
        constructor_fields = {item.name for item in fields(cls) if item.init}
        parameter_fields = constructor_fields - set(resolved_values)
        unknown = sorted(set(values) - parameter_fields)
        if unknown:
            raise ValueError(f"Unknown key(s) {unknown} in '{block_name}'. That block accepts "
                             f"{sorted(parameter_fields)}.")
        try:
            return cls(**values, **resolved_values)
        except TypeError as error:
            raise ValueError(f"Invalid '{block_name}' configuration: {error}") from error


@dataclass(frozen=True)
class JumpDetectionConfig(StepConfig):
    """Validated jump-detection parameters and experiment-specific flag bitmask."""

    PARAMETER_NAME: ClassVar[str] = "jump_detection"

    window: int = 10
    jump_bitmask: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.window, int) or isinstance(self.window, bool) or self.window < 1:
            raise ValueError("jump_detection.window must be an integer of at least 1.")
        if self.enabled and self.jump_bitmask is None:
            raise ValueError("Jump detection is enabled, but the experiment has no jump_bitmask.")
        if self.jump_bitmask is not None and not isinstance(self.jump_bitmask, int):
            raise ValueError("The experiment jump_bitmask must be an integer.")

    @classmethod
    def from_params(cls, params: Bunch, experiment_data: DetGroupTOD) -> "JumpDetectionConfig":
        """Build jump settings from their step block and the experiment flag bitmask."""
        experiment = params.experiments[experiment_data.experiment_name]
        jump_bitmask = experiment.jump_bitmask if "jump_bitmask" in experiment else None
        block = (params.tod_processing[cls.PARAMETER_NAME]
                 if cls.PARAMETER_NAME in params.tod_processing else Bunch())
        return cls._from_block(f"tod_processing.{cls.PARAMETER_NAME}", block,
                               jump_bitmask=jump_bitmask)


@dataclass(frozen=True)
class GainConfig(StepConfig):
    """Validated settings needed to execute one gain-sampling step."""

    calibrate_against: str = "sky"
    gap_fill_method: str = "wn"
    downsample_time: float = 1.0
    sampling_rate: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.calibrate_against not in _VALID_CALIB_TARGETS:
            raise ValueError(f"calibrate_against must be one of {_VALID_CALIB_TARGETS}, got "
                             f"{self.calibrate_against!r}.")
        if self.gap_fill_method not in GAIN_GAP_FILL_METHODS:
            raise ValueError(f"gap_fill_method must be one of {GAIN_GAP_FILL_METHODS}, got "
                             f"{self.gap_fill_method!r}.")
        if not np.isfinite(self.downsample_time) or self.downsample_time < 0:
            raise ValueError("downsample_time must be a finite, non-negative number.")
        if not np.isfinite(self.sampling_rate) or self.sampling_rate <= 0:
            raise ValueError("The experiment sampling rate must be positive and finite.")

    @property
    def downsample_factor(self) -> int:
        """Number of native samples averaged into one gain-calibration sample."""
        return max(1, round(self.downsample_time * self.sampling_rate))

    @classmethod
    def from_params(cls, params: Bunch, experiment_data: DetGroupTOD, step_name: str,
                    default_calibrator: str, iteration: int, is_master: bool) -> "GainConfig":
        """Build one self-contained gain config, including a per-band calibrator override."""
        block = dict(params.tod_processing[step_name]
                     if step_name in params.tod_processing else Bunch())
        configured_calibrator = block.get("calibrate_against", default_calibrator)
        exp_name = experiment_data.experiment_name
        band_name = experiment_data.band_name
        block["calibrate_against"] = resolve_param(
            params, "calibrate_against",
            (f"experiments.{exp_name}.bands.{band_name}.{step_name}",),
            default=configured_calibrator, raise_on_missing_scope=False,
            legal_values=_VALID_CALIB_TARGETS,
        )
        config = cls._from_block(
            f"tod_processing.{step_name}", block,
            sampling_rate=float(experiment_data.fsamp),
        )
        if (config.is_active(iteration) and is_master
                and config.calibrate_against == "orbital_dipole"
                and experiment_data.nu > _ORBITAL_DIPOLE_MAX_FREQ_GHZ):
            logger.warning(f"{step_name} for band {band_name} ({experiment_data.nu} GHz) is "
                           f"calibrated against the orbital dipole, but above "
                           f"{_ORBITAL_DIPOLE_MAX_FREQ_GHZ:.0f} GHz the dipole is faint compared "
                           f"with the foregrounds and makes a poor calibrator; consider 'sky', "
                           f"or an externally determined gain for this band.")
        return config


@dataclass(frozen=True)
class CGConfig:
    """Conjugate-gradient controls shared by the two CG uses in TOD processing."""

    max_iter: int = 0
    err_tol: float = 1.0e-4

    def __post_init__(self) -> None:
        if not isinstance(self.max_iter, int) or isinstance(self.max_iter, bool):
            raise ValueError("max_iter must be an integer.")
        if self.max_iter < 0:
            raise ValueError("max_iter cannot be negative.")
        if not np.isfinite(self.err_tol) or self.err_tol < 0:
            raise ValueError("err_tol must be a finite, non-negative number.")

    @classmethod
    def from_block(cls, block_name: str, block: Bunch | dict,
                   require_all: bool = False) -> "CGConfig":
        """Build CG controls and optionally require both fields to be stated."""
        try:
            values = dict(block)
        except (TypeError, ValueError) as error:
            raise ValueError(f"'{block_name}' must be a parameter block.") from error
        if require_all:
            missing = sorted({"max_iter", "err_tol"} - set(values))
            if missing:
                raise ValueError(f"Missing required key(s) {missing} in '{block_name}'.")
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"Invalid '{block_name}' configuration: {error}") from error


@dataclass(frozen=True)
class CorrelatedNoiseConfig(StepConfig):
    """Validated correlated-noise and sigma0 sampling settings."""

    PARAMETER_NAME: ClassVar[str] = "corr_noise"

    sample_psd_params: bool = False
    sample_sigma0: bool = True
    sigma0_method: str = "pairwise"
    sigma0_decimation: int = 1
    nomono: bool = False
    onlymono: bool = False
    psd_fit_nu_min: float = 0.0
    psd_fit_nu_max: float = float("inf")
    psd_bin: bool = False
    cg: CGConfig = field(default_factory=CGConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sample_psd_params and not self.enabled:
            raise ValueError("corr_noise.sample_psd_params requires enabled=True.")
        if self.sigma0_method not in SIGMA0_METHODS:
            raise ValueError(f"corr_noise.sigma0_method must be one of {SIGMA0_METHODS}, got "
                             f"{self.sigma0_method!r}.")
        if (not isinstance(self.sigma0_decimation, int)
                or isinstance(self.sigma0_decimation, bool) or self.sigma0_decimation < 1):
            raise ValueError("corr_noise.sigma0_decimation must be an integer of at least 1.")

    @classmethod
    def from_params(cls, params: Bunch, is_master: bool) -> "CorrelatedNoiseConfig":
        """Build correlated-noise settings, including its nested CG block."""
        block = dict(params.tod_processing[cls.PARAMETER_NAME]
                     if cls.PARAMETER_NAME in params.tod_processing else Bunch())
        cg = CGConfig.from_block(f"tod_processing.{cls.PARAMETER_NAME}.cg",
                                 block.pop("cg", Bunch()))
        config = cls._from_block(f"tod_processing.{cls.PARAMETER_NAME}", block, cg=cg)
        if config.nomono and config.onlymono and is_master:
            logger.error("tod_processing.corr_noise.nomono and onlymono are both True, which is "
                         "contradictory; onlymono takes precedence.")
        return config


@dataclass(frozen=True)
class DataSelectionConfig(StepConfig):
    """Validated detector-scan selection thresholds and iteration range."""

    PARAMETER_NAME: ClassVar[str] = "data_selection"

    until_iter: int | None = None
    chisq_abs_threshold: float = 1.0e4
    min_good_fraction: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.until_iter is not None:
            if not isinstance(self.until_iter, int) or isinstance(self.until_iter, bool):
                raise ValueError("data_selection.until_iter must be an integer or null.")
            if self.until_iter < self.from_iter:
                raise ValueError("data_selection.until_iter cannot be before from_iter.")
        if not np.isfinite(self.chisq_abs_threshold) or self.chisq_abs_threshold <= 0:
            raise ValueError("data_selection.chisq_abs_threshold must be positive and finite.")
        if not 0.0 <= self.min_good_fraction <= 1.0:
            raise ValueError("data_selection.min_good_fraction must be between 0 and 1.")

    @classmethod
    def from_params(cls, params: Bunch) -> "DataSelectionConfig":
        """Build detector-scan selection settings from its parameter block."""
        block = (params.tod_processing[cls.PARAMETER_NAME]
                 if cls.PARAMETER_NAME in params.tod_processing else Bunch())
        return cls._from_block(f"tod_processing.{cls.PARAMETER_NAME}", block)

    def is_available(self, iteration: int,
                     correlated_noise: CorrelatedNoiseConfig) -> bool:
        """Whether diagnostics can be reported after waiting for configured n_corr sampling."""
        return self.enabled and (correlated_noise.is_active(iteration)
                                 or not correlated_noise.enabled)

    def cuts_are_active(self, iteration: int,
                        correlated_noise: CorrelatedNoiseConfig) -> bool:
        """Whether this iteration applies detector-scan vetoes."""
        before_end = self.until_iter is None or iteration <= self.until_iter
        return (self.is_available(iteration, correlated_noise)
                and super().is_active(iteration) and before_end)


@dataclass(frozen=True)
class MapmakingConfig:
    """Validated mapmaking resources, algorithm controls, and map-output selection."""

    mapmaker: str
    num_threads: int
    include_orbital_dipole_maps: bool
    include_corr_noise_maps: bool
    include_sky_model_maps: bool
    sparse_maps: bool = False
    common_res_fwhm: float = 0.0
    cg: CGConfig = field(default_factory=CGConfig)

    def __post_init__(self) -> None:
        if self.mapmaker not in ("CG", "bin"):
            raise ValueError(f"mapmaker must be 'CG' or 'bin', got {self.mapmaker!r}.")
        if not isinstance(self.num_threads, int) or self.num_threads < 1:
            raise ValueError("resources.tod.num_threads must be an integer of at least 1.")
        if self.common_res_fwhm < 0:
            raise ValueError("compsep.common_res_fwhm cannot be negative.")

    @classmethod
    def from_params(cls, params: Bunch,
                    experiment_data: DetGroupTOD) -> "MapmakingConfig":
        """Build mapmaking settings using band, experiment, and global precedence."""
        exp_name = experiment_data.experiment_name
        band_name = experiment_data.band_name
        mapmaker = resolve_param(
            params, "mapmaker",
            (f"experiments.{exp_name}.bands.{band_name}", f"experiments.{exp_name}",
             "tod_processing"),
            legal_values=("CG", "bin"),
        )
        tod = params.tod_processing
        if "cg_mapmaker" in tod:
            cg = CGConfig.from_block("tod_processing.cg_mapmaker", tod.cg_mapmaker,
                                     require_all=mapmaker == "CG")
        elif mapmaker == "CG":
            raise ValueError("tod_processing.cg_mapmaker is required for the CG mapmaker.")
        include = params.output.chains.include
        resolved = {
            "mapmaker": mapmaker,
            "sparse_maps": bool(resolve_param(params, "sparse_maps", (f"experiments.{exp_name}",),
                                              default=cls.sparse_maps)),
            "common_res_fwhm": float(resolve_param(params, "common_res_fwhm", ("compsep",),
                                                   default=cls.common_res_fwhm)),
            "num_threads": params.resources.tod.num_threads,
            "include_orbital_dipole_maps": bool(include.orbital_dipole_maps),
            "include_corr_noise_maps": bool(include.corr_noise_maps),
            "include_sky_model_maps": bool(include.sky_model_maps),
        }
        if "cg_mapmaker" in tod:
            resolved["cg"] = cg
        return cls(**resolved)


def process_tod(mpi_info: Bunch, experiment_data: DetGroupTOD,
                tod_samples: TODSamples, compsep_output: NDArray,
                params: Bunch, chain: int, iter: int) -> tuple[dict[str, DetectorMap], TODSamples]:
    """Run one TOD iteration for one band.

    The function states the scientific order directly. Correlated noise, sigma0, diagnostics, and
    data-selection vetoes run inside the selected mapmaker because they operate on one focused
    detector-scan and must occur at exact positions relative to map accumulation.

    Returns:
        The detector maps for component separation and the updated TOD samples.
    """
    timing_dict = {}
    waittime_dict = {}
    band_comm = mpi_info.band.comm
    tod_comm = mpi_info.tod.comm
    is_master = mpi_info.band.is_master

    # Resolve every config before executing any step, so invalid later settings fail before an
    # earlier sampler changes the chain state. Each class owns its own unpacking and validation.
    mapmaking = MapmakingConfig.from_params(params, experiment_data)
    jump_detection = JumpDetectionConfig.from_params(params, experiment_data)
    absolute_gain = GainConfig.from_params(
        params, experiment_data, "abs_gain", "orbital_dipole", iter, is_master,
    )
    relative_gain = GainConfig.from_params(
        params, experiment_data, "rel_gain", "sky", iter, is_master,
    )
    temporal_gain = GainConfig.from_params(
        params, experiment_data, "temporal_gain", "sky", iter, is_master,
    )
    correlated_noise = CorrelatedNoiseConfig.from_params(params, is_master)
    data_selection = DataSelectionConfig.from_params(params)

    @contextmanager
    def timed_step(label: str):
        """Benchmark + wall-time + master log line around one TOD sampling step."""
        t0 = time.time()
        with benchmark(label):
            yield
        timing_dict[label] = time.time() - t0
        if mpi_info.band.is_master:
            logger.info(f"Chain {chain} iter{iter} {experiment_data.nu}GHz: Finished {label} "
                        f"in {timing_dict[label]:.1f}s.")

    # Jump corrections must be stored before any later step reads the TOD.
    if jump_detection.is_active(iter):
        with timed_step("jump-detect"):
            tod_samples = sample_jump_detection(band_comm, experiment_data, tod_samples,
                                                jump_detection)

    # Gain uses the previous iteration's sigma0. The new sigma0 is estimated later, inside the
    # mapmaker scan loop, matching Commander3's gain -> n_corr -> bin_TOD order.
    if absolute_gain.is_active(iter):
        with timed_step("abs-gain"):
            tod_samples = sample_absolute_gain(band_comm, experiment_data, tod_samples,
                                               compsep_output, absolute_gain)

    if relative_gain.is_active(iter):
        with timed_step("rel-gain"):
            tod_samples = sample_relative_gain(band_comm, experiment_data, tod_samples,
                                               compsep_output, relative_gain)

    if temporal_gain.is_active(iter):
        with timed_step("temp-gain"):
            tod_samples = sample_temporal_gain_variations(band_comm, experiment_data, tod_samples,
                                                          compsep_output, temporal_gain)

    t0 = time.time()
    # A finite diagnostic means "evaluated this iteration". Previously rejected or absent scans
    # remain NaN and are not counted again by the data-selection summary.
    tod_samples.chisq_z[:] = np.nan
    tod_samples.good_fraction[:] = np.nan

    if mapmaking.mapmaker == "CG":
        detmap_dict, maps_to_file = tod2map_CG(
            band_comm, experiment_data, compsep_output, tod_samples, iter, mapmaking,
            correlated_noise, data_selection,
        )
    else:
        detmap_dict, maps_to_file = tod2map_bin(
            band_comm, experiment_data, compsep_output, tod_samples, iter, mapmaking,
            correlated_noise, data_selection,
        )

    # Chain writers retain the full parameter tree because it is serialized into file metadata.
    # The numerical mapmakers themselves only receive the values in their specific configs.
    if is_master:
        start_bench("filewrite-datamaps")
        write_map_chain_to_file(params, chain, iter, experiment_data.experiment_name,
                                experiment_data.band_name, maps_to_file,
                                tod_samples.band_unit_factor, tod_samples.band_unit)
        stop_bench("filewrite-datamaps")
    timing_dict["mapmaker"] = time.time() - t0
    if band_comm.Get_rank() == 0:
        logger.info(f"Chain {chain} iter{iter} {experiment_data.nu}GHz: Finished mapmaking in "
                    f"{timing_dict['mapmaker']:.1f}s.")

    # Report during data-selection warm-up as well as active-cut iterations.
    if data_selection.is_available(iter, correlated_noise):
        log_dataselect_summary(
            band_comm, tod_samples, data_selection,
            active=data_selection.cuts_are_active(iter, correlated_noise),
        )

    with benchmark("filewrite-tod"):
        tod_samples.write_chain_to_file(iter)

    t0 = time.time()
    with benchmark("end-barrier"):
        tod_comm.Barrier()
    waittime_dict["end-barrier"] = time.time() - t0

    bench_summary(tod_comm, label="All bands")
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
