"""`tod2map`: the per-band scan loop that turns the TOD into maps, for both mapmakers.

Each detector-scan first goes through the per-scan steps of the Gibbs chain (data selection,
correlated noise and sigma0, far sidelobes, diagnostics), in Commander3's order, and is then added
to the maps. The binned mapmaker (`BinnedMapmaker`, `mapmaking/binned.py`) solves every map per
pixel after the loop. The CG mapmaker (`CGMapmaker`, `mapmaking/cg.py`) also uses the binned maps,
for its weights and the aux maps, but solves the sky map iteratively after the loop instead, from
the right-hand side it adds up during the loop. Both use only the good samples: the binned maps
leave the flagged ones out, and the CG gives them zero weight.
"""
import time

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.detector_map import DetectorMap
from commander4.data_models.tod_samples import TODSamples
from commander4.diagnostics.performance import benchmark, log_memory, start_bench, stop_bench
from commander4.tod.config import MapmakingConfig, CorrelatedNoiseConfig, DataSelectionConfig
from commander4.tod.data_selection import data_selection_status
from commander4.tod.mapmaking.binned import BinnedMapmaker
from commander4.tod.mapmaking.cg import CGMapmaker
from commander4.tod.mapmaking.output import finalize_band_maps
from commander4.tod.noise.sample_ncorr import sample_correlated_noise, log_corr_noise_stats
from commander4.tod.noise.sigma0 import _estimate_standalone_sigma0
from commander4.tod.scan_diagnostics import _record_tod_diagnostics
from commander4.tod.sidelobe_deconvolve import FarBeamProjector
from commander4.tod.view import TODView


def tod2map(band_comm: MPI.Comm, experiment_data: DetectorGroupTOD, compsep_output: NDArray,
            tod_samples: TODSamples, iteration: int, mapmaking_cfg: MapmakingConfig,
            corr_noise_cfg: CorrelatedNoiseConfig, data_selection_cfg: DataSelectionConfig,
            far_beam_model: FarBeamProjector | None = None,
            ) -> tuple[dict[str, DetectorMap], dict[str, NDArray]]:
    """Make the band maps from the TOD with the binned or the CG mapmaker (collective).

    All ranks of `band_comm` hold TOD of the same band and build its maps together.

    Args:
        band_comm: The ranks holding TOD of this band.
        experiment_data: The band's TOD.
        compsep_output: The sky model at this band, at this rank's local pixels.
        tod_samples: Sampled TOD parameters, such as gain; the per-scan samples drawn here (n_corr,
            sigma0, data selection, diagnostics) are written into it.
        iteration: Current Gibbs iteration.
        mapmaking_cfg: Mapmaking settings; `mapmaking_cfg.mapmaker` picks "bin" or "CG".
        corr_noise_cfg: Correlated-noise settings.
        data_selection_cfg: Detector-scan selection settings.
        far_beam_model: The far-sidelobe model to subtract, or None.

    Returns:
        The detector maps for component separation and the maps for the chain file (both empty on
        all ranks but the master).
    """
    start_bench("setup")
    use_cg = mapmaking_cfg.mapmaker == "CG"
    corr_noise_active = corr_noise_cfg.enabled and iteration >= corr_noise_cfg.from_iter
    _, selection_active = data_selection_status(iteration, data_selection_cfg, corr_noise_cfg)
    sidelobe_active = far_beam_model is not None
    pols = experiment_data.pols
    scan_view = TODView(experiment_data, tod_samples, compsep_output=compsep_output)

    # The maps to bin, named as in the chain file. The binned mapmaker solves the sky map
    # ("signal") per pixel like the aux maps; the CG solves it after the loop instead. Each aux map
    # costs a map and a little time per sample, so one is made only when the chain will hold it.
    map_names = [] if use_cg else ["signal"]
    if mapmaking_cfg.include_orbital_dipole_maps:
        map_names.append("orbdipole")
    if corr_noise_active and mapmaking_cfg.include_corr_noise_maps:
        map_names.append("corrnoise")
    if sidelobe_active and mapmaking_cfg.include_sidelobe_maps:
        map_names.append("sidelobe")
    if mapmaking_cfg.include_residual_maps:
        map_names.append("res")
    binned = BinnedMapmaker(experiment_data.pixel_domain, pols, map_names,
                            count_hits=mapmaking_cfg.include_hit_maps)
    cg = CGMapmaker(experiment_data, tod_samples, band_comm, mapmaking_cfg) if use_cg else None
    if corr_noise_active:
        sampled_params = []
        residuals = []
        niters = []
        num_failed_convergences_ncorr = 0
        num_too_high_var_ncorr = 0
        worst_residual_ncorr = 0
    stop_bench("setup")

    ### MAIN SCAN LOOP ###
    # Each scan's wall time in this loop is recorded as its measured cost, for spreading the scans
    # over the ranks in a later run. A detector-scan's time runs from the start of its pass to the
    # start of the next one, so the passes that a veto ends early are counted too.
    tod_samples.scan_runtime[:] = 0.0
    last_iscan, last_start = None, 0.0
    for view in scan_view.iter_focused(accepted_only=True):
        now = time.perf_counter()
        if last_iscan is not None:
            tod_samples.scan_runtime[last_iscan] += now - last_start
        last_iscan, last_start = view.iscan, now

        start_bench("pix-psi")
        good_data_mask = view.get_mask(proc_mask=False)
        pix, psi = view.pix, view.psi
        response_I_P = view.response_I_P
        gain = view.get_gain()
        stop_bench("pix-psi")

        ### DATA-SELECTION VETO 1 (too little unflagged data).
        start_bench("data-select-1")
        good_frac = good_data_mask.mean()
        tod_samples.good_fraction[view.iscan, view.idet] = good_frac
        if selection_active and good_frac < data_selection_cfg.min_good_fraction:
            tod_samples.accept[view.iscan, view.idet] = False
            stop_bench("data-select-1")
            continue
        stop_bench("data-select-1")

        ### CORRELATED NOISE / SIGMA0 SAMPLING (first, so the weights below use the new sigma0) ###
        n_corr_est = None
        if corr_noise_active:
            start_bench("ncorr")
            sky_subtracted_TOD = view.get_tod(
                subtract=(("sky", TODView._ALL_GAIN_TERMS),
                          ("orbital_dipole", TODView._ALL_GAIN_TERMS)),
            )
            res = sample_correlated_noise(
                sky_subtracted_TOD, view.get_mask(proc_mask_type="ncorr"),
                np.array(view.noise_params, copy=True),
                experiment_data.noise_model, view.fsamp, cg_err_tol=corr_noise_cfg.cg.err_tol,
                cg_max_iter=corr_noise_cfg.cg.max_iter,
                sample_params=corr_noise_cfg.sample_psd_params,
                sample_sigma0=corr_noise_cfg.sample_sigma0,
                sigma0_method=corr_noise_cfg.sigma0_method,
                nomono=corr_noise_cfg.nomono,
                onlymono=corr_noise_cfg.onlymono,
                sigma0_dec=corr_noise_cfg.sigma0_decimation,
                psd_bin=corr_noise_cfg.psd_bin,
                use_dct=corr_noise_cfg.use_dct)
            n_corr_est = res.n_corr
            tod_samples.noise_params[view.iscan, view.idet, :] = res.noise_params
            tod_samples.ncorr_cg_residual[view.iscan, view.idet] = res.residual
            tod_samples.ncorr_cg_niter[view.iscan, view.idet] = res.niter
            tod_samples.ncorr_converged[view.iscan, view.idet] = res.converged and not res.high_var
            if corr_noise_cfg.sample_psd_params:
                sampled_params.append(np.array(res.noise_params, copy=True))
            if not res.converged:
                num_failed_convergences_ncorr += 1
            if res.high_var:
                num_too_high_var_ncorr += 1
            worst_residual_ncorr = max(worst_residual_ncorr, res.residual)
            residuals.append(res.residual)
            niters.append(res.niter)
            stop_bench("ncorr")
        elif corr_noise_cfg.sample_sigma0:
            # No correlated noise this iteration: estimate sigma0 here, at the same point in the
            # chain (after gain) as the n_corr-coupled estimate, not in a separate pre-gain pass.
            with benchmark("sigma0-samp"):
                tod_samples.noise_params[view.iscan, view.idet, 0] = _estimate_standalone_sigma0(
                    view, corr_noise_cfg.sigma0_method)

        sl_tod = None
        if sidelobe_active:
            with benchmark("far-beam-proj"):
                sl_tod = far_beam_model.get_projection(pix, psi, view.idet)
        with benchmark("tod-diagnostics"):
            residual_tod = _record_tod_diagnostics(
                tod_samples, view.iscan, view.idet, view, n_corr_est,
                sidelobe_tod=gain * sl_tod if sl_tod is not None else None)

        ### DATA-SELECTION VETO 2 (catastrophic chi^2), applied in-loop so this iteration's maps
        ### already exclude the scan (the CG operator passes re-read `accept`).
        start_bench("data-select-2")
        if selection_active:
            z = tod_samples.chisq_z[view.iscan, view.idet]
            if not (np.isfinite(z) and abs(z) <= data_selection_cfg.chisq_abs_threshold):
                tod_samples.accept[view.iscan, view.idet] = False
                stop_bench("data-select-2")
                continue
        stop_bench("data-select-2")

        # sigma0 (just sampled above) is in detector units; dividing it by the gain gives the
        # noise of a sample in uK_RJ.
        inv_var = (gain/view.sigma0)**2

        ### MAP TODS, in uK_RJ ###
        # TODO: Known bug with a bolometer transfer function T (`tf_tau_sec`). The data hold
        # T(sky + orbital dipole + sidelobes), but the models subtracted here and in the other
        # Gibbs steps are not passed through T, so the residuals and the CG right-hand side keep
        # (1 - T) times them. To be resolved when/if we migrate to the C3-style design (deconvolve
        # the TOD, filter the models with the same regularization kernel).
        with benchmark("misc"):
            d_sky = view.get_tod(subtract=(("orbital_dipole", TODView._ALL_GAIN_TERMS),))
        if corr_noise_active:
            d_sky -= n_corr_est
        # The far-sidelobe projection comes back in uK_RJ, like the sky and orbital-dipole model
        # TODs, so the map takes it as it is while the detector-unit TOD has it removed at the
        # full gain.
        if sidelobe_active:
            d_sky -= gain * sl_tod
        # The binned maps take only the good samples.
        good = good_data_mask
        tods = {}
        if "signal" in map_names:
            tods["signal"] = d_sky[good]/gain
        if "orbdipole" in map_names:
            # The dipole TOD is cached on the view, so `get_tod` above already paid for it.
            tods["orbdipole"] = view.get_orbital_dipole_tod()[good]
        if "corrnoise" in map_names:
            tods["corrnoise"] = (n_corr_est[good]/gain).astype(np.float32, copy=False)
        if "sidelobe" in map_names:
            tods["sidelobe"] = sl_tod[good]
        if "res" in map_names:
            # `residual_tod` is the detector-unit noise residual `_record_tod_diagnostics` already
            # built (sky model, orbital dipole, n_corr and sidelobes all subtracted).
            tods["res"] = residual_tod[good]/gain

        ### MAP BINNING ###
        with benchmark("map-binning"):
            binned.accumulate(inv_var, view.pix_local[good], psi[good], tods, response_I_P)
            if mapmaking_cfg.include_hit_maps:
                binned.count_hits(view.pix_local[good])
        if use_cg:
            # The CG takes the full-length TOD, since its transfer function needs the whole time
            # axis, and gives the flagged samples zero weight itself.
            with benchmark("cg-rhs"):
                cg.accum_to_RHS(view, d_sky/gain)
    if last_iscan is not None:
        tod_samples.scan_runtime[last_iscan] += time.perf_counter() - last_start

    with benchmark("MPI-sync"):
        band_comm.Barrier()

    ### PRINT NOISE SAMPLING STATS ###
    if corr_noise_active:
        with benchmark("log-corr-noise-stats"):
            log_corr_noise_stats(band_comm, experiment_data,
                                 sampled_params, residuals, niters, num_failed_convergences_ncorr,
                                 num_too_high_var_ncorr, worst_residual_ncorr,
                                 sum(len(s.detectors) for s in experiment_data.scans),
                                 tod_samples.chain, iteration)

    ### GATHER AND SOLVE MAPS ###
    with benchmark("map-gather"):
        binned.finalize()
    if use_cg:
        with benchmark("cg-solve"):
            cg.solve(binned.map_cov)
    log_memory("mapmaker")

    ### FINAL CLEANUP ON MASTER RANK ###
    detmap_dict_out = {}
    maps_to_file = {}
    with benchmark("finalize"):
        if band_comm.Get_rank() == 0:
            # After this, binned.maps holds only the aux maps.
            map_signal = cg.solved_map if use_cg else binned.maps.pop("signal")
            detmap_dict_out, maps_to_file = finalize_band_maps(
                map_signal, binned.map_inv_var, pols, experiment_data, mapmaking_cfg,
                tod_samples, binned.maps, binned.map_nhit, binned.map_cov)

    return detmap_dict_out, maps_to_file
