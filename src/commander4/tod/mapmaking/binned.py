"""The binned mapmaker: accumulate P^T N^-1 d and P^T N^-1 P per pixel, then invert per pixel.

The simpler of the two mapmakers (see `mapmaking/cg.py` for the CG one). `BinnedMapmaker` holds
all binned maps of a band: the inverse-variance weights that both mapmakers need, the signal map
and the auxiliary maps. It adds each detector-scan to all of them in one pass. `tod2map_bin` drives
the whole per-band scan loop.

The maps are indexed with `pix_local`: local pixel indices into the rank's map buffers
(`TODView.pix_local`), converted once per detector-scan.
"""
import numpy as np
from mpi4py import MPI
import logging
from numpy.typing import NDArray

from commander4.backend import mapmaker as cpp_mapmaker
from commander4.diagnostics.performance import log_memory, start_bench, stop_bench
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.detector_map import DetectorMap
from commander4.data_models.tod_samples import TODSamples
from commander4.tod.noise.sample_ncorr import sample_correlated_noise, log_corr_noise_stats
from commander4.tod.noise.sigma0 import _estimate_standalone_sigma0
from commander4.tod.scan_diagnostics import _record_tod_diagnostics
from commander4.tod.view import TODView
from commander4.data_models.pixel_domain import PixelDomain
from commander4.tod.config import MapmakingConfig, CorrelatedNoiseConfig, DataSelectionConfig
from commander4.tod.mapmaking.output import finalize_band_maps
from commander4.tod.data_selection import data_selection_status
from commander4.tod.sidelobe_deconvolve import FarBeamProjector
from commander4.diagnostics.performance import benchmark

logger = logging.getLogger(__name__)

class BinnedMapmaker:
    """All binned maps of one band: the weights P^T N^-1 P and any number of TOD maps P^T N^-1 d.

    Every map of a band is binned with the same per-sample weight, so one `accumulate` call per
    detector-scan adds the weights and all the TOD maps together, reading the pointing once. The
    buffers hold this rank's local pixels (`PixelDomain`); `finalize` sums them onto the band
    master and solves each pixel there. Afterwards the master holds (other ranks hold None):

    - `map_cov`: the summed weights, float64. For an IQU or QU band these are the 6 unique elements
      (II, IQ, IU, QQ, QU, UU) of each pixel's 3x3 matrix, (6, npix); for an I-only band (1, npix).
    - `map_inv_var`: the inverse variance 1/diag(A^-1), (ncomp, npix), float32.
    - `maps`: the solved maps by name, (ncomp, npix), float32.
    - `map_nhit`: the number of samples per pixel, (npix,), if `count_hits` was set.

    ncomp is 1 for an I-only band and 3 (I, Q, U) otherwise. A pixel that cannot be solved, because
    it is unobserved or seen at too few polarization angles, gets 0 in `map_inv_var` and `maps`.
    """
    def __init__(self, domain: PixelDomain, pols: str, names: list[str], count_hits: bool = False):
        self.domain = domain
        self.names = list(names)
        self.ncomp = 1 if pols == "I" else 3
        self._weights = np.zeros((1 if pols == "I" else 6, domain.n_local))
        self._maps = np.zeros((len(self.names), self.ncomp, domain.n_local))
        self._hits = np.zeros(domain.n_local) if count_hits else None
        self.map_cov = self.map_inv_var = self.maps = self.map_nhit = None

    def accumulate(self, weight: float, pix_local: NDArray, psi: NDArray,
                   tods: dict[str, NDArray], response_I_P: tuple[float, float] = (1.0, 1.0)):
        """Add one detector-scan: `weight` per sample to the weights, and `weight*tods[name]` to
        each map. Every TOD has one value per entry of `pix_local` (`TODView.pix_local`)."""
        pix_local = np.asarray(pix_local, dtype=np.int64)
        # One row per map, in the order of `names`.
        tod_rows = (np.stack([tods[name] for name in self.names]) if self.names
                    else np.zeros((0, pix_local.size), dtype=np.float32))
        resp_I, resp_P = response_I_P
        cpp_mapmaker.binned_map_accumulator(self._weights, self._maps, tod_rows, float(weight),
                                            pix_local, np.asarray(psi, dtype=np.float64),
                                            response_I=resp_I, response_P=resp_P)

    def count_hits(self, pix_local: NDArray):
        """Add one hit per sample to the hit map."""
        cpp_mapmaker.hit_accumulator(self._hits, np.asarray(pix_local, dtype=np.int64))

    def finalize(self):
        """Sum the buffers of all ranks onto the master and solve each pixel there (collective).

        The maps are summed one at a time, so besides the weights the master holds only one
        unsolved full-sky map at a time. The local buffers are freed.
        """
        is_master = self.domain.comm.Get_rank() == 0
        map_cov = self.domain.reduce_to_full(self._weights)
        if is_master:
            self.map_cov = map_cov
            self.maps = {}
            if self.ncomp == 1:
                # For intensity alone a pixel's normal matrix is one number: the inverse variance.
                self.map_inv_var = map_cov.astype(np.float32)
            else:
                inv_var = np.zeros((3, self.domain.npix))
                cpp_mapmaker.map_inv_var_IQU(inv_var, map_cov)
                self.map_inv_var = inv_var.astype(np.float32)
        for i, name in enumerate(self.names):
            rhs = self.domain.reduce_to_full(self._maps[i])
            if is_master:
                solved = np.zeros((self.ncomp, self.domain.npix))
                if self.ncomp == 1:
                    np.divide(rhs, map_cov, out=solved, where=map_cov != 0)
                else:
                    cpp_mapmaker.map_solve_IQU(solved, rhs, map_cov)
                self.maps[name] = solved.astype(np.float32)
        if self._hits is not None:
            map_nhit = self.domain.reduce_to_full(self._hits)
            if is_master:
                self.map_nhit = np.round(map_nhit).astype(np.int64)
        self._weights = self._maps = self._hits = None


def tod2map_bin(band_comm: MPI.Comm, experiment_data: DetectorGroupTOD, compsep_output: NDArray,
                tod_samples: TODSamples, iteration: int,
                mapmaking_cfg: MapmakingConfig, corr_noise_cfg: CorrelatedNoiseConfig,
                data_selection_cfg: DataSelectionConfig,
                far_beam_model: FarBeamProjector|None = None,
                ) -> tuple[dict[str, DetectorMap], dict[str, NDArray]]:
    """ Commander4 bin mapmaking. All ranks on the provided MPI communicator collaborates on creating
        the band maps (sky signal, inverse variance, possibly also aux maps like orbital dipole).
    Args:
        band_comm (Comm): The communicator consisting of all MPI ranks which holds TOD data that
                          should go into the same map.
        experiment_data (DetectorGroupTOD): TOD data class to be made into maps.
        compsep_output (NDArray): The sky model at our band, at this rank's local pixels.
        tod_samples (TODSamples): Sampled TOD parameters, such as gain.
        iteration: Current Gibbs iteration.
        mapmaking_cfg: Validated mapmaking settings.
        corr_noise_cfg: Validated correlated-noise settings.
        data_selection_cfg: Validated detector-scan selection settings.
    Output:
        Detector maps for component separation and maps selected for chain output.

    """
    start_bench("setup")
    corr_noise_active = corr_noise_cfg.enabled and iteration >= corr_noise_cfg.from_iter
    _, selection_active = data_selection_status(iteration, data_selection_cfg, corr_noise_cfg)
    sidelobe_active = far_beam_model is not None
    pols = experiment_data.pols
    scan_view = TODView(experiment_data, tod_samples, compsep_output=compsep_output)
    # Which pixels each rank's map buffers hold (all of them, or only the locally observed ones
    # with sparse maps). The band master always ends up with full-sky maps.
    domain = experiment_data.pixel_domain

    # The maps to make, named as in the chain file. Each aux map costs a full-sky map and a little
    # time per sample, so one is made only when the chain is going to hold it.
    map_names = ["signal"]
    if mapmaking_cfg.include_orbital_dipole_maps:
        map_names.append("orbdipole")
    if corr_noise_active and mapmaking_cfg.include_corr_noise_maps:
        map_names.append("corrnoise")
    if sidelobe_active and mapmaking_cfg.include_sidelobe_maps:
        map_names.append("sidelobe")
    if mapmaking_cfg.include_residual_maps:
        map_names.append("res")
    binned = BinnedMapmaker(domain, pols, map_names, count_hits=mapmaking_cfg.include_hit_maps)
    if corr_noise_active:
        sampled_params = []
        residuals = []
        niters = []
        num_failed_convergences_ncorr = 0
        num_too_high_var_ncorr = 0
        worst_residual_ncorr = 0
    stop_bench("setup")

    ### MAIN SCAN LOOP ###
    for view in scan_view.iter_focused(accepted_only=True):
        start_bench("pix-psi")
        good_data_mask = view.get_mask(proc_mask=False)
        pix, psi = view.pix, view.psi
        # The maps take local pixel indices, converted once for this det-scan; `pix` itself stays
        # the global index, which the far-sidelobe projection needs.
        pix_local_masked = view.pix_local[good_data_mask]
        psi_masked = psi[good_data_mask]
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
            # chain (after gain) as the n_corr-coupled estimate, instead of a separate pre-gain pass.
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

        ### DATA-SELECTION VETO 2 (catastrophic chi^2)
        start_bench("data-select-2")
        if selection_active:
            z = tod_samples.chisq_z[view.iscan, view.idet]
            if not (np.isfinite(z) and abs(z) <= data_selection_cfg.chisq_abs_threshold):
                tod_samples.accept[view.iscan, view.idet] = False
                stop_bench("data-select-2")
                continue
        stop_bench("data-select-2")

        ### MAP TODS ###
        # Every map gets the TOD of this det-scan in uK_RJ at the unflagged samples, and all of
        # them are binned together below.
        with benchmark("misc"):
            d_sky = view.get_tod(subtract=(("orbital_dipole", TODView._ALL_GAIN_TERMS),))
        tods = {}
        if "corrnoise" in map_names:
            tods["corrnoise"] = (n_corr_est[good_data_mask]/gain).astype(np.float32, copy=False)
        if corr_noise_active:
            d_sky -= n_corr_est
        # The far-sidelobe projection comes back in uK_RJ, like the sky and orbital-dipole model
        # TODs, so the map takes it as it is while the detector-unit TOD has it removed at the
        # full gain.
        if sidelobe_active:
            if "sidelobe" in map_names:
                tods["sidelobe"] = sl_tod[good_data_mask]
            d_sky -= gain * sl_tod
        tods["signal"] = d_sky[good_data_mask]/gain
        if "orbdipole" in map_names:
            # The dipole TOD is cached on the view, so `get_tod` above already paid for it.
            tods["orbdipole"] = view.get_orbital_dipole_tod()[good_data_mask]
        if "res" in map_names:
            # `residual_tod` is the detector-unit noise residual `_record_tod_diagnostics` already
            # built (sky model, orbital dipole and n_corr all subtracted).
            tods["res"] = residual_tod[good_data_mask]/gain

        ### MAP BINNING ###
        with benchmark("map-binning"):
            # sigma0 (just sampled above) is in detector units; dividing it by the gain gives the
            # weight of a sample in uK_RJ.
            inv_var = (gain/view.sigma0)**2
            binned.accumulate(inv_var, pix_local_masked, psi_masked, tods, response_I_P)
            if mapmaking_cfg.include_hit_maps:
                binned.count_hits(pix_local_masked)

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
    log_memory("mapmaker")

    ### FINAL CLEANUP ON MASTER RANK ###
    detmap_dict_out = {}
    maps_to_file = {}
    with benchmark("finalize"):
        if band_comm.Get_rank() == 0:
            aux_maps = {name: binned.maps[name] for name in map_names if name != "signal"}
            detmap_dict_out, maps_to_file = finalize_band_maps(
                binned.maps["signal"], binned.map_inv_var, pols, experiment_data, mapmaking_cfg,
                tod_samples, aux_maps, binned.map_nhit, binned.map_cov)

    return detmap_dict_out, maps_to_file
