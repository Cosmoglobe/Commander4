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
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.detector_map import DetectorMap
from commander4.data_models.tod_samples import TODSamples
from commander4.diagnostics.performance import benchmark, log_memory, start_bench, stop_bench
from commander4.parameters.schema import resolve_param
from commander4.tod.mapmaking.binned import BinnedMapmaker
from commander4.tod.mapmaking.cg import CGMapmaker
from commander4.tod.mapmaking.output import finalize_band_maps
from commander4.tod.view import TODView


def tod2map(experiment_data: DetectorGroupTOD, tod_samples: TODSamples, compsep_output: NDArray,
            iteration: int, params: Bunch) -> tuple[dict[str, DetectorMap], dict[str, NDArray]]:
    """Make the band maps from the TOD with the binned or the CG mapmaker (collective).

    All ranks of the band's communicator hold TOD of the same band and build its maps together.

    Args:
        experiment_data: The band's TOD.
        tod_samples: The chain state: the band's communicator and its sampling steps. The steps
            for the per-scan samples drawn here (noise, data selection, diagnostics) write them
            into it, and the far-sidelobe model must be built when it is active.
        compsep_output: The sky model at this band, at this rank's local pixels.
        iteration: Current Gibbs iteration.
        params: The full parameter file, for the mapmaker and the maps the chain gets.

    Returns:
        The detector maps for component separation and the maps for the chain file (both empty on
        all ranks but the master).
    """
    start_bench("setup")
    band_comm = tod_samples.band_comm
    noise, selection = tod_samples.noise, tod_samples.selection
    far_sidelobes = tod_samples.far_sidelobes
    exp, band = experiment_data.experiment_name, experiment_data.band_name
    # The band's own mapmaker, else the experiment's, else the common one.
    mapmaker = resolve_param(params, "mapmaker", (f"experiments.{exp}.bands.{band}",
                                                  f"experiments.{exp}", "tod_processing"),
                             legal_values=("CG", "bin"))
    use_cg = mapmaker == "CG"
    include = params.output.chains.include
    corr_noise_active = noise.cfg.enabled and iteration >= noise.cfg.from_iter
    _, selection_active = selection.status(iteration, noise.cfg)
    sidelobe_active = far_sidelobes.cfg.enabled and iteration >= far_sidelobes.cfg.from_iter
    pols = experiment_data.pols
    scan_view = TODView(experiment_data, tod_samples, compsep_output=compsep_output)

    # The maps to bin, named as in the chain file. The binned mapmaker solves the sky map
    # ("signal") per pixel like the aux maps; the CG solves it after the loop instead. Each aux map
    # costs a map and a little time per sample, so one is made only when the chain will hold it.
    map_names = [] if use_cg else ["signal"]
    if include.orbital_dipole_maps:
        map_names.append("orbdipole")
    if corr_noise_active and include.corr_noise_maps:
        map_names.append("corrnoise")
    if sidelobe_active and getattr(include, "sidelobe_maps", False):
        map_names.append("sidelobe")
    if getattr(include, "residual_maps", False):
        map_names.append("res")
    count_hits = getattr(include, "hit_maps", False)
    binned = BinnedMapmaker(experiment_data.pixel_domain, pols, map_names, count_hits=count_hits)
    cg = CGMapmaker(experiment_data, tod_samples, params) if use_cg else None
    stop_bench("setup")

    ### MAIN SCAN LOOP ###
    # Each scan's wall time in this loop is recorded as its measured cost, for spreading the scans
    # over the ranks in a later run. A detector-scan's time runs from the start of its pass to the
    # start of the next one, so the passes that a veto ends early are counted too.
    diagnostics = tod_samples.diagnostics
    last_iscan, last_start = None, 0.0
    for view in scan_view.iter_focused(accepted_only=True):
        now = time.perf_counter()
        if last_iscan is not None:
            diagnostics.scan_runtime[last_iscan] += now - last_start
        last_iscan, last_start = view.iscan, now

        start_bench("pix-psi")
        good_data_mask = view.get_mask(proc_mask=False)
        pix, psi = view.pix, view.psi
        response_I_P = view.response_I_P
        gain = view.get_gain()
        stop_bench("pix-psi")

        ### DATA-SELECTION VETO 1 (too little unflagged data).
        start_bench("data-select-1")
        if selection.veto_low_good_fraction(view, good_data_mask, selection_active):
            stop_bench("data-select-1")
            continue
        stop_bench("data-select-1")

        ### CORRELATED NOISE / SIGMA0 SAMPLING (first, so the weights below use the new sigma0) ###
        n_corr_est = None
        if corr_noise_active:
            with benchmark("ncorr"):
                n_corr_est = noise.sample_scan(view)
        elif noise.cfg.sample_sigma0:
            # No correlated noise this iteration: estimate sigma0 alone.
            with benchmark("sigma0-samp"):
                noise.sample_sigma0(view)

        sl_tod = None
        if sidelobe_active:
            with benchmark("far-beam-proj"):
                sl_tod = far_sidelobes.get_projection(pix, psi, view.idet)

        ### THE SKY TOD AND THE NOISE RESIDUAL, in detector units ###
        # TODO: Known bug with a bolometer transfer function T (`tf_tau_sec`). The data hold
        # T(sky + orbital dipole + sidelobes), but the models subtracted here and in the other
        # Gibbs steps are not passed through T, so the residuals and the CG right-hand side keep
        # (1 - T) times them. To be resolved when/if we migrate to the C3-style design (deconvolve
        # the TOD, filter the models with the same regularization kernel).
        # `d_sky` is the data with every model but the sky removed, which the maps are made from.
        # Removing the sky model as well leaves the noise residual, which data selection judges and
        # the residual map holds (Commander3's `tod_<freq>_res`). The far-sidelobe projection, like
        # the sky model, is in uK_RJ, so it is removed at the full gain.
        with benchmark("misc"):
            d_sky = view.get_tod(keep=("sky",))
            if corr_noise_active:
                d_sky -= n_corr_est
            if sidelobe_active:
                d_sky -= gain * sl_tod
            residual_tod = d_sky - gain * view.get_sky_tod()
        with benchmark("tod-diagnostics"):
            diagnostics.record_scan(view, residual_tod, n_corr_est)

        ### DATA-SELECTION VETO 2 (catastrophic chi^2), applied in-loop so this iteration's maps
        ### already exclude the scan (the CG operator passes re-read `accept`).
        start_bench("data-select-2")
        if selection.veto_high_chisq(view, residual_tod, selection_active):
            stop_bench("data-select-2")
            continue
        stop_bench("data-select-2")

        # sigma0 (just sampled above) is in detector units; dividing it by the gain gives the
        # noise of a sample in uK_RJ.
        inv_var = (gain/view.sigma0)**2

        ### MAP TODS, in uK_RJ ###
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
            tods["res"] = residual_tod[good]/gain

        ### MAP BINNING ###
        with benchmark("map-binning"):
            binned.accumulate(inv_var, view.pix_local[good], psi[good], tods, response_I_P)
            if count_hits:
                binned.count_hits(view.pix_local[good])
        if use_cg:
            # The CG takes the full-length TOD, since its transfer function needs the whole time
            # axis, and gives the flagged samples zero weight itself.
            with benchmark("cg-rhs"):
                cg.accum_to_RHS(view, d_sky/gain)
    if last_iscan is not None:
        diagnostics.scan_runtime[last_iscan] += time.perf_counter() - last_start

    with benchmark("MPI-sync"):
        band_comm.Barrier()

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
                map_signal, binned.map_inv_var, experiment_data, tod_samples, params, binned.maps,
                binned.map_nhit, binned.map_cov)

    return detmap_dict_out, maps_to_file
