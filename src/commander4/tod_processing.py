import numpy as np
import pixell
from pixell import utils
from mpi4py import MPI
import logging
from scipy.fft import rfftfreq
import time
from numpy.typing import NDArray

from pixell.bunch import Bunch

from commander4.output.log import logassert
from commander4.data_models.detector_map import DetectorMap
from commander4.data_models.detector_group_TOD import DetGroupTOD
from commander4.data_models.TOD_samples import TODSamples
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


def _read_sparse_maps_flag(params: Bunch, experiment_data: DetGroupTOD) -> bool:
    """Whether sparse (per-rank local-pixel) map storage is enabled for this experiment.

    ``sparse_maps`` is an experiment-level option: when set, each rank's map buffers *and* its slice
    of the sky model hold only the pixels its scans observe, rather than a full sky map (the band
    master still assembles full-sky maps). Defaults to the historical full-sky-per-rank layout.
    """
    exp_cfg = params.experiments[experiment_data.experiment_name]
    return bool(exp_cfg["sparse_maps"]) if "sparse_maps" in exp_cfg else False


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


def tod2map_CG(band_comm: MPI.Comm, experiment_data: DetGroupTOD, compsep_output: NDArray,
            tod_samples: TODSamples, params: Bunch, chain: int, iter: int,
            ncorr_cfg: Bunch) -> dict[str, DetectorMap]:
    """ Commander4 CG mapmaking. All ranks on the provided MPI communicator collaborates on creating
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
        ncorr_cfg (Bunch): Correlated-noise sampling config (do_ncorr, do_param, cg_err_tol,
            cg_max_iter).
    Output:
        dict[str, DetectorMap]: Dictionary containing the solved detector maps, keyed by
            polarization component ('I', 'QU').

    """
    ismaster = band_comm.Get_rank() == 0
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
    sparse_maps = _read_sparse_maps_flag(params, experiment_data)
    domain = experiment_data.get_pixel_domain(scan_view, band_comm, sparse_maps)
    # The inverse-variance map (preconditioner + rms/cov) is accumulated inside the fused loop below,
    # so it -- and thus cg_mapmaker.M -- can only be finalized afterwards. cg_mapmaker is constructed
    # here with a placeholder preconditioner; M is unused until solve() and accum_to_RHS never reads
    # it, so it is reassigned to the real Jacobi preconditioner after the loop.
    if pols == "IQU":
        mapmaker_invvar = WeightsMapmakerIQU(band_comm, experiment_data.nside, pixel_domain=domain)
        cg_mapmaker = CGMapmakerIQU(experiment_data, tod_samples, band_comm,
                    preconditioner=called_on_non_master, nthreads=params.general.nthreads_tod,
                    CG_maxiter=params.general.CG_mapmaker.maxiter, pixel_domain=domain)
    elif pols == "I":
        mapmaker_invvar = WeightsMapmaker(band_comm, experiment_data.nside, pixel_domain=domain)
        cg_mapmaker = CGMapmakerI(experiment_data, tod_samples, band_comm,
                    preconditioner=called_on_non_master, nthreads=params.general.nthreads_tod,
                    CG_maxiter=params.general.CG_mapmaker.maxiter, pixel_domain=domain)
    else:
        raise ValueError(f"specified polarizations {pols} is notsupported yet.")

    BinMapmaker = MapmakerIQU if pols == "IQU" else Mapmaker #general bin mapmaker class object.
    # mapmaker = BinMapmaker(band_comm, experiment_data.nside)
    mapmaker_orbdipole = BinMapmaker(band_comm, experiment_data.nside, pixel_domain=domain)

    if ncorr_cfg.do_ncorr:
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

        ### CORRELATED NOISE / SIGMA0 SAMPLING (first, so the weights below use the new sigma0) ###
        n_corr_est = None
        if ncorr_cfg.do_ncorr:
            sky_subtracted_TOD = view.get_tod(
                subtract=(("sky", TODView._ALL_GAIN_TERMS),
                          ("orbital_dipole", TODView._ALL_GAIN_TERMS)),
            )
            res = sample_correlated_noise(
                sky_subtracted_TOD, view.get_mask(proc_mask_type="ncorr"),
                np.array(view.noise_params, copy=True),
                experiment_data.noise_model, view.fsamp, cg_err_tol=ncorr_cfg.cg_err_tol,
                cg_max_iter=ncorr_cfg.cg_max_iter, sample_params=ncorr_cfg.do_param,
                sample_sigma0=ncorr_cfg.sample_sigma0, sigma0_method=ncorr_cfg.sigma0_method,
                nomono=ncorr_cfg.nomono,
                onlymono=ncorr_cfg.onlymono,
                sigma0_dec=ncorr_cfg.sigma0_dec, psd_fit_nu_min=ncorr_cfg.psd_fit_nu_min,
                psd_fit_nu_max=ncorr_cfg.psd_fit_nu_max, psd_bin=ncorr_cfg.psd_bin)
            n_corr_est = res.n_corr
            tod_samples.noise_params[view.iscan, view.idet, :] = res.noise_params
            if ncorr_cfg.do_param:
                sampled_params.append(np.array(res.noise_params, copy=True))
            if not res.converged:
                num_failed_convergences_ncorr += 1
            if res.high_var:
                num_too_high_var_ncorr += 1
            worst_residual_ncorr = max(worst_residual_ncorr, res.residual)
            residuals.append(res.residual)
            niters.append(res.niter)
        elif ncorr_cfg.sample_sigma0:
            # No correlated noise this iteration: estimate sigma0 here, at the same point in the
            # chain (after gain) as the n_corr-coupled estimate, instead of a separate pre-gain pass.
            tod_samples.noise_params[view.iscan, view.idet, 0] = _estimate_standalone_sigma0(
                view, ncorr_cfg.sigma0_method)

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
        if ncorr_cfg.do_ncorr:
            if pols == "IQU":
                mapmaker_ncorr.accumulate_to_map(
                    (n_corr_est/gain).astype(np.float32, copy=False),
                    inv_var, pix, psi, response=response)
            else:
                mapmaker_ncorr.accumulate_to_map(
                    (n_corr_est/gain).astype(np.float32, copy=False),
                    inv_var, pix, psi)
            d_sky -= n_corr_est

        _record_tod_diagnostics(tod_samples, view.iscan, view.idet, view, n_corr_est)

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
    if ncorr_cfg.do_ncorr:
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

    if ncorr_cfg.do_ncorr:
        mapmaker_ncorr.gather_map()
        mapmaker_ncorr.normalize_map(map_cov)
        map_corrnoise = mapmaker_ncorr.final_map

    ### FINAL CLEANUP ON MASTER RANK ###
    detmap_dict_out = {}
    if band_comm.Get_rank() == 0:
        #Here we split here between I and QU
        # Smooth maps to the common analysis resolution after mapmaking (single switch:
        # general.common_res_fwhm; a missing or falsy value leaves bands at their native beam).
        common_res_fwhm = (float(params.general.common_res_fwhm)
                           if "common_res_fwhm" in params.general else 0.0)
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

        maps_to_file = {}
        maps_to_file["map_observed_sky"] = map_signal
        maps_to_file["map_rms"] = map_rms
        if params.general.write_orb_dipole_maps_to_chain:
            maps_to_file["map_orbdipole"] = map_orbdipole
        if params.general.write_corr_noise_maps_to_chain and ncorr_cfg.do_ncorr:
            maps_to_file["map_corrnoise"] = map_corrnoise
        if params.general.write_sky_model_maps_to_chain:
            maps_to_file["map_skymodel"] = compsep_output

        write_map_chain_to_file(params, chain, iter, experiment_data.experiment_name,
                                experiment_data.band_name, maps_to_file,
                                tod_samples.band_unit_factor, tod_samples.band_unit)

    return detmap_dict_out #empty on non-master ranks


def tod2map_bin(band_comm: MPI.Comm, experiment_data: DetGroupTOD, compsep_output: NDArray,
            tod_samples: TODSamples, params: Bunch, chain: int, iter: int,
            ncorr_cfg: Bunch) -> dict[str, DetectorMap]:
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
        ncorr_cfg (Bunch): Correlated-noise sampling config (do_ncorr, do_param, cg_err_tol,
            cg_max_iter).
    Output:
        dict[str, DetectorMap]: Dictionary containing the solved detector maps, keyed by
            polarization component ('I', 'QU').

    """
    start_bench("binned-mapmaker")
    pols = experiment_data.pols
    scan_view = TODView(experiment_data, tod_samples, compsep_output=compsep_output)
    # Optional per-experiment sparse map storage: each rank holds only its locally-observed pixels
    # rather than a full sky map. The band master still ends up with full-sky maps.
    sparse_maps = _read_sparse_maps_flag(params, experiment_data)
    domain = experiment_data.get_pixel_domain(scan_view, band_comm, sparse_maps)

    # Set up various mapmakers.
    mapmaker_invvar = WeightsMapmakerIQU(band_comm, experiment_data.nside, pixel_domain=domain)
    mapmaker = MapmakerIQU(band_comm, experiment_data.nside, pixel_domain=domain)
    mapmaker_orbdipole = MapmakerIQU(band_comm, experiment_data.nside, pixel_domain=domain)
    if ncorr_cfg.do_ncorr:
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

        ### CORRELATED NOISE / SIGMA0 SAMPLING (first, so the weights below use the new sigma0) ###
        n_corr_est = None
        if ncorr_cfg.do_ncorr:
            start_bench("ncorr-sampling")
            sky_subtracted_TOD = view.get_tod(
                subtract=(("sky", TODView._ALL_GAIN_TERMS),
                          ("orbital_dipole", TODView._ALL_GAIN_TERMS)),
            )
            res = sample_correlated_noise(
                sky_subtracted_TOD, view.get_mask(proc_mask_type="ncorr"),
                np.array(view.noise_params, copy=True),
                experiment_data.noise_model, view.fsamp, cg_err_tol=ncorr_cfg.cg_err_tol,
                cg_max_iter=ncorr_cfg.cg_max_iter, sample_params=ncorr_cfg.do_param,
                sample_sigma0=ncorr_cfg.sample_sigma0, sigma0_method=ncorr_cfg.sigma0_method,
                nomono=ncorr_cfg.nomono,
                onlymono=ncorr_cfg.onlymono,
                sigma0_dec=ncorr_cfg.sigma0_dec, psd_fit_nu_min=ncorr_cfg.psd_fit_nu_min,
                psd_fit_nu_max=ncorr_cfg.psd_fit_nu_max, psd_bin=ncorr_cfg.psd_bin)
            n_corr_est = res.n_corr
            tod_samples.noise_params[view.iscan, view.idet, :] = res.noise_params
            if ncorr_cfg.do_param:
                sampled_params.append(np.array(res.noise_params, copy=True))
            if not res.converged:
                num_failed_convergences_ncorr += 1
            if res.high_var:
                num_too_high_var_ncorr += 1
            worst_residual_ncorr = max(worst_residual_ncorr, res.residual)
            residuals.append(res.residual)
            niters.append(res.niter)
            stop_bench("ncorr-sampling")
        elif ncorr_cfg.sample_sigma0:
            # No correlated noise this iteration: estimate sigma0 here, at the same point in the
            # chain (after gain) as the n_corr-coupled estimate, instead of a separate pre-gain pass.
            tod_samples.noise_params[view.iscan, view.idet, 0] = _estimate_standalone_sigma0(
                view, ncorr_cfg.sigma0_method)

        _record_tod_diagnostics(tod_samples, view.iscan, view.idet, view, n_corr_est)

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
        if ncorr_cfg.do_ncorr:
            mapmaker_ncorr.accumulate_to_map(
                (n_corr_est[good_data_mask]/gain).astype(np.float32, copy=False),
                inv_var, pix_masked, psi_masked, response=response)
            d_sky -= n_corr_est

        d_sky_masked = d_sky[good_data_mask]
        mapmaker.accumulate_to_map(d_sky_masked/gain, inv_var, pix_masked, psi_masked, response=response)
        mapmaker_orbdipole.accumulate_to_map(sky_orb_dipole[good_data_mask], inv_var,
                                             pix_masked, psi_masked, response=response)
        stop_bench("binned-mapmaker", increment_count=False)
    if ncorr_cfg.do_ncorr:
        log_memory("ncorr-sampling")

    ### PRINT NOISE SAMPLING STATS ###
    if ncorr_cfg.do_ncorr:
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
    if ncorr_cfg.do_ncorr:
        mapmaker_ncorr.gather_map()
        mapmaker_ncorr.normalize_map(map_cov)
        map_corrnoise = mapmaker_ncorr.final_map
    stop_bench("binned-mapmaker", increment_count=False)
    log_memory("binned-mapmaker")

    ### FINAL CLEANUP ON MASTER RANK ###
    detmap_dict_out = {}
    if band_comm.Get_rank() == 0:
        #Here we split here between I and QU
        # Smooth maps to the common analysis resolution after mapmaking (single switch:
        # general.common_res_fwhm; a missing or falsy value leaves bands at their native beam).
        common_res_fwhm = (float(params.general.common_res_fwhm)
                           if "common_res_fwhm" in params.general else 0.0)
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

        maps_to_file = {}
        maps_to_file["map_observed_sky"] = map_signal
        maps_to_file["map_rms"] = map_rms
        if params.general.write_orb_dipole_maps_to_chain:
            maps_to_file["map_orbdipole"] = map_orbdipole
        if params.general.write_corr_noise_maps_to_chain and ncorr_cfg.do_ncorr:
            maps_to_file["map_corrnoise"] = map_corrnoise
        if params.general.write_sky_model_maps_to_chain:
            maps_to_file["map_skymodel"] = compsep_output

        start_bench("filewrite-datamaps")
        write_map_chain_to_file(params, chain, iter, experiment_data.experiment_name,
                                experiment_data.band_name, maps_to_file,
                                tod_samples.band_unit_factor, tod_samples.band_unit)
        stop_bench("filewrite-datamaps")

    return detmap_dict_out #empty on non-master ranks


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
                # A per-band ``num_scans`` (bands can hold different numbers of scans) takes
                # precedence over the experiment-level value, which is the shared default.
                tot_num_scans = band.num_scans if "num_scans" in band else experiment.num_scans
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
    sparse_maps = _read_sparse_maps_flag(params, experiment_data)
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
                          tod_samples: TODSamples, params: Bunch) -> TODSamples:
    """Detect jump discontinuities from the flag stream and store additive post-jump offsets.

    A jump is identified by a contiguous region with a non-zero
    ``flag & experiments.[experiment_name].jump_bitmask``. For each region, the offset is
    estimated from the last ``N`` valid samples before the jump and the first ``N`` valid samples
    after it, where validity is defined by ``full_mask``. The correction is then applied to all
    later samples when a TOD is requested through ``TODView.get_tod()``.
    """
    n_window = int(getattr(params.general, "jump_detection_window", 10))
    if n_window < 1:
        raise ValueError("jump_detection_window must be >= 1.")
    experiment_params = params.experiments[experiment_data.experiment_name]
    if "jump_bitmask" not in experiment_params or experiment_params.jump_bitmask is None:
        raise ValueError(
            "Jump sampling is enabled, but "
            f"experiments.{experiment_data.experiment_name}.jump_bitmask is not specified."
        )
    jump_bitmask = int(experiment_params.jump_bitmask)

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
            n_window,
            jump_bitmask=jump_bitmask,
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



# Valid calibration targets for gain sampling, and the per-term defaults used when a parameter file
# leaves `calibrate_against` unspecified (preserving the historical low-frequency behavior).
_VALID_CALIB_TARGETS = ("orbital_dipole", "full_sky", "sky")
_DEFAULT_CALIB_TARGETS = {"abs_gain": "orbital_dipole",
                          "rel_gain": "full_sky",
                          "temporal_gain": "full_sky"}


def _resolve_calib_target(params: Bunch, experiment_data: DetGroupTOD, gain_block: str) -> str:
    """ Resolve which signal a gain term is calibrated against for the current band.

        A per-band ``calibrate_against`` (under ``experiments.<exp>.bands.<band>.<gain_block>``)
        overrides ``general.<gain_block>.calibrate_against``, which in turn falls back to the
        term's default in ``_DEFAULT_CALIB_TARGETS``.
    """
    general_block = params.general[gain_block]
    target = (general_block["calibrate_against"] if "calibrate_against" in general_block
              else _DEFAULT_CALIB_TARGETS[gain_block])
    band = params.experiments[experiment_data.experiment_name].bands[experiment_data.band_name]
    if gain_block in band and "calibrate_against" in band[gain_block]:
        target = band[gain_block]["calibrate_against"]
    if target not in _VALID_CALIB_TARGETS:
        raise ValueError(f"{gain_block}.calibrate_against='{target}' is invalid; must be one of "
                         f"{_VALID_CALIB_TARGETS}.")
    return target


def _resolve_gain_downsample_factor(params: Bunch, experiment_data: DetGroupTOD) -> int:
    """ Downsampling factor (in samples) used when building the gain-calibration TODs.

        Derived from ``general.gain_calib_downsample_time`` (a duration in seconds, shared by all
        gain terms) and the band's sampling rate. A duration of 0 disables downsampling.
    """
    return max(1, int(round(params.general.gain_calib_downsample_time * experiment_data.fsamp)))


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


def sample_absolute_gain(band_comm: MPI.Comm, experiment_data: DetGroupTOD, tod_samples: TODSamples,
                         det_compsep_map: NDArray, calibrate_against: str, downsample_factor: int,
                         gap_fill_method: str = "wn"):
    """ Draw a realization of the absolute gain term, g0, which is constant across all
        detectors and all scans within a band, calibrated against ``calibrate_against``.
    Args:
        band_comm (MPI.Comm): The band-level MPI communicator.
        experiment_data (DetGroupTOD): The object holding all the scan data.
        tod_samples (TODSamples): Current sampled TOD parameters (updated in-place with g0).
        det_compsep_map (NDArray): The component-separation sky map for the detector.
        calibrate_against (str): Calibrator signal, one of "orbital_dipole" | "full_sky" | "sky".
        downsample_factor (int): Block-averaging factor for the calibration TODs.
        gap_fill_method (str): Masked-sample fill for the calibration residual, one of "wn" |
            "fallback" | "full_cg" (see TODView.get_calib_tod).
    Returns:
        tod_samples (TODSamples): Updated TOD samples with the new g0 estimate.
        wait_time (float): Time spent waiting at the MPI barrier.
    """

    sum_s_T_N_inv_d = 0  # Accumulators for the numerator and denominator of eqn 16.
    sum_s_T_N_inv_s = 0

    scan_view = TODView(experiment_data, tod_samples, compsep_output=det_compsep_map,
                        downsample_factor=downsample_factor)

    # Skip detector-scans flagged as bad (accepted_only); they carry no gain info.
    for view in scan_view.iter_focused(accepted_only=True):
        calib = view.get_calib_tod("abs", calibrate_against, gap_fill_method=gap_fill_method,
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

    t0 = time.time()
    band_comm.Barrier()
    wait_time = time.time() - t0
    g_sampled = band_comm.bcast(g_sampled, root=0)
    log_memory("abs-gain")

    # As of Numpy 2.0 it's good practice to explicitly cast to Python scalar types, as this would
    # otherwise have been a np.float64 type, potentially causing unexpected casting behavior later.
    tod_samples.abs_gain = float(g_sampled)

    return tod_samples, wait_time


def sample_relative_gain(band_comm: MPI.Comm, experiment_data: DetGroupTOD,
                         tod_samples: TODSamples, det_compsep_map: NDArray, calibrate_against: str,
                         downsample_factor: int, gap_fill_method: str = "wn"):
    """ Samples the detector-dependent relative gain (Delta g_i). This function implements the
        logic from Sec. 3.4 of BP7.
    Args:
        band_comm (MPI.Comm): The band-level MPI communicator.
        experiment_data (DetGroupTOD): The object holding scan data for the band.
        tod_samples (TODSamples): Current sampled TOD parameters.
        det_compsep_map (NDArray): The component-separation sky map for the detector.
        calibrate_against (str): Calibrator signal, one of "orbital_dipole" | "full_sky" | "sky".
        downsample_factor (int): Block-averaging factor for the calibration TODs.
        gap_fill_method (str): Masked-sample fill for the calibration residual, one of "wn" |
            "fallback" | "full_cg" (see TODView.get_calib_tod).
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
                        downsample_factor=downsample_factor)

    # Skip detector-scans flagged as bad (accepted_only); they carry no gain info.
    for view in scan_view.iter_focused(accepted_only=True):
        calib = view.get_calib_tod("rel", calibrate_against, gap_fill_method=gap_fill_method,
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

    wait_time = 0
    return tod_samples, wait_time



def sample_temporal_gain_variations(band_comm: MPI.Comm, experiment_data: DetGroupTOD,
                                    tod_samples: TODSamples, det_compsep_map: NDArray,
                                    chain: int, iter: int, params: Bunch, calibrate_against: str,
                                    downsample_factor: int, gap_fill_method: str = "wn"):
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
        calibrate_against (str): Calibrator signal, one of "orbital_dipole" | "full_sky" | "sky".
        downsample_factor (int): Block-averaging factor for the calibration TODs.
        gap_fill_method (str): Masked-sample fill for the calibration residual, one of "wn" |
            "fallback" | "full_cg" (see TODView.get_calib_tod).
    """
    band_rank = band_comm.Get_rank()
    band_size = band_comm.Get_size()
    ndet = experiment_data.ndet
    nscans_local = len(experiment_data.scans)

    # Local calculations on each rank
    A_qq_local = np.zeros((ndet, nscans_local), dtype=np.float64)
    b_q_local = np.zeros((ndet, nscans_local), dtype=np.float64)
    scan_view = TODView(experiment_data, tod_samples, compsep_output=det_compsep_map,
                        downsample_factor=downsample_factor)

    # I'm still not sure what way of dealing with the masked samples are best:
    # 1. Replace masked values with 0s before FFT.
    # 2. Replace masked values with n_corr realizations before FFT.
    # 3. Remove masked values by reducing TOD size before FFTs.
    # (simply passing the full data through the FFTs seems like a bad idea because of
    # ringing from the large residual in the galactic plane).
    # Rejected detector-scans (accepted_only) contribute zero weight (A_qq = b_q = 0); the Wiener
    # prior then fills their temporal gain from neighbors.
    for view in scan_view.iter_focused(accepted_only=True):
        calib = view.get_calib_tod("temp", calibrate_against, gap_fill_method=gap_fill_method,
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

                if False: #debug stuff
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10,8))
                    other_gain = tod_samples.abs_gain + tod_samples.rel_gain[idet]
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
            tod_samples.temporal_gain[:,idet] = delta_g_local.astype(np.float32, copy=False)
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
    # 1. Detect and store jump corrections from the flag stream.
    # 2. Estimate white noise from the jump-corrected, sky-subtracted TOD.
    # 3. Sample the gain from the jump-corrected, sky-subtracted TOD.
    # 4. Sample correlated noise and PS parameters.
    # 5. Mapmaking on TOD - corr_noise_TOD - orb_dipole_TOD.

    timing_dict = {}
    waittime_dict = {}

    det_comm = mpi_info.det.comm
    band_comm = mpi_info.band.comm
    TOD_comm = mpi_info.tod.comm
    ### JUMP DETECTION ###
    if getattr(params.general, "sample_jump_detection", False) and iter >= int(
        getattr(params.general, "sample_jump_detection_from_iter_num", 1)
    ):
        t0 = time.time()
        with benchmark("jump-detect"):
            tod_samples = sample_jump_detection(band_comm, experiment_data, tod_samples, params)
        timing_dict["jump-detect"] = time.time() - t0
        if mpi_info.band.is_master:
            logger.info(f"Chain {chain} iter{iter} {experiment_data.nu}GHz: Finished jump "
                        f"detection in {timing_dict['jump-detect']:.1f}s.")

    ### CORRELATED-NOISE SAMPLING CONFIG ###
    # All correlated-noise settings live in the nested ``general.corr_noise`` block.
    cn = params.general.corr_noise
    sample_corr_noise = cn.sample_corr_noise
    from_iter = cn.sample_corr_noise_from_iter_num
    sample_noise_params = cn.sample_noise_params
    # Parameter sampling consumes a correlated-noise realization, so it cannot run without it.
    if sample_noise_params and not sample_corr_noise:
        raise ValueError("general.corr_noise.sample_noise_params requires sample_corr_noise=True "
                         "(parameter sampling needs a correlated-noise realization).")
    do_ncorr = sample_corr_noise and iter >= from_iter
    # Per-scan monopole handling mirrors the Fortran flags; both set at once is contradictory.
    nomono = getattr(cn, "nomono", False)
    onlymono = getattr(cn, "onlymono", False)
    if nomono and onlymono and mpi_info.band.is_master:
        logger.error("general.corr_noise.nomono and onlymono are both True, which is contradictory; "
                     "onlymono takes precedence.")
    sigma0_method = getattr(cn, "sigma0_method", "pairwise")
    if sigma0_method not in SIGMA0_METHODS:
        raise ValueError(f"general.corr_noise.sigma0_method must be one of {SIGMA0_METHODS}, got "
                         f"{sigma0_method!r}.")
    ncorr_cfg = Bunch(
        do_ncorr=do_ncorr,
        do_param=do_ncorr and sample_noise_params,
        sample_sigma0=getattr(cn, "sample_sigma0", True),
        sigma0_method=sigma0_method,
        cg_err_tol=cn.CG_err_tol,
        cg_max_iter=cn.CG_max_iter,
        nomono=nomono,
        onlymono=onlymono,
        sigma0_dec=getattr(cn, "sigma0_decimation", 1),
        psd_fit_nu_min=getattr(cn, "psd_fit_nu_min", 0.0),
        psd_fit_nu_max=getattr(cn, "psd_fit_nu_max", float("inf")),
        psd_bin=getattr(cn, "psd_bin", False),
    )

    # Gap-fill method for the non-CG sampling steps (gain calibration). The correlated-noise step's
    # own gap handling is fixed by CG_max_iter (masked CG, or stationary fallback when 0).
    gain_gap_fill = getattr(params.general, "gap_fill_method", "wn")
    if gain_gap_fill not in GAIN_GAP_FILL_METHODS:
        raise ValueError(f"general.gap_fill_method must be one of {GAIN_GAP_FILL_METHODS}, got "
                         f"{gain_gap_fill!r}.")

    # NOTE: sigma0 is estimated inside the mapmaker scan loop (after gain), co-located with the
    # n_corr-coupled estimate -- see the do_ncorr if/elif in tod2map_CG/tod2map_bin. This matches
    # Commander3, where gain always runs on the previous iteration's sigma0.

    ### ABSOLUTE GAIN CALIBRATION ###
    if params.general.abs_gain.sample and iter >= params.general.abs_gain.sample_from_iter_num:
        calib_target = _resolve_calib_target(params, experiment_data, "abs_gain")
        downsample_factor = _resolve_gain_downsample_factor(params, experiment_data)
        t0 = time.time()
        with benchmark("abs-gain"):
            tod_samples, wait_time = sample_absolute_gain(band_comm, experiment_data, tod_samples,
                                                          compsep_output, calib_target,
                                                          downsample_factor, gain_gap_fill)
        timing_dict["abs-gain"] = time.time() - t0
        waittime_dict["abs-gain"] = wait_time
        if mpi_info.band.is_master:
            logger.info(f"Chain {chain} iter{iter} {experiment_data.nu}GHz: Finished absolute "\
                        f"gain estimation in {timing_dict['abs-gain']:.1f}s.")

    ### RELATIVE GAIN CALIBRATION ###
    if params.general.rel_gain.sample and iter >= params.general.rel_gain.sample_from_iter_num:
        calib_target = _resolve_calib_target(params, experiment_data, "rel_gain")
        downsample_factor = _resolve_gain_downsample_factor(params, experiment_data)
        t0 = time.time()
        with benchmark("rel-gain"):
            tod_samples, wait_time = sample_relative_gain(band_comm, experiment_data, tod_samples,
                                                          compsep_output, calib_target,
                                                          downsample_factor, gain_gap_fill)
        timing_dict["rel-gain"] = time.time() - t0
        waittime_dict["rel-gain"] = wait_time
        if mpi_info.band.is_master:
            logger.info(f"Chain {chain} iter{iter} {experiment_data.nu}GHz: Finished relative "\
                        f"gain estimation in {timing_dict['rel-gain']:.1f}s.")


    ### TEMPORAL GAIN CALIBRATION ###
    if params.general.temporal_gain.sample\
    and iter >= params.general.temporal_gain.sample_from_iter_num:
        calib_target = _resolve_calib_target(params, experiment_data, "temporal_gain")
        downsample_factor = _resolve_gain_downsample_factor(params, experiment_data)
        t0 = time.time()
        with benchmark("temporal-gain"):
            tod_samples = sample_temporal_gain_variations(band_comm, experiment_data, tod_samples,
                                    compsep_output, chain, iter, params, calib_target,
                                    downsample_factor, gain_gap_fill)
        timing_dict["temp-gain"] = time.time() - t0
        if mpi_info.band.is_master:
            logger.info(f"Chain {chain} iter{iter} {experiment_data.nu}GHz: Finished temporal "\
                        f"gain estimation in {timing_dict['temp-gain']:.1f}s.")

    ### MAPMAKING ###
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
                     iter, ncorr_cfg)
    elif mapmaker_str == "bin":
        detmap_dict = tod2map_bin(band_comm, experiment_data, compsep_output, tod_samples, params,
                            chain, iter, ncorr_cfg)
    else:
        raise ValueError(f'Mapmaker must be either "CG" or "bin", but {mapmaker_str} was given for'\
                         f' experiment {experiment_data.experiment_name}, band {experiment_data.band_name}')
    timing_dict["mapmaker"] = time.time() - t0
    if band_comm.Get_rank() == 0:
        logger.info(f"Chain {chain} iter{iter} {experiment_data.nu}GHz: Finished mapmaking in "\
                    f"{timing_dict['mapmaker']:.1f}s.")

    ### WRITE CHAIN TO FILE ###
    with benchmark("filewrite-tod"):
        tod_samples.write_chain_to_file(iter)

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