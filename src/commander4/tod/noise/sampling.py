"""`Noise`: the noise parameters of every detector-scan, and the steps that sample them.

Both steps run once per detector-scan inside the mapmaking scan loop (`tod2map`), after the gain:
either a correlated-noise realization together with sigma0 and the noise-model parameters, or,
when correlated noise is not sampled, sigma0 alone. The numerics live in `sample_ncorr.py` and
`sigma0.py` next to this file.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.parameters.schema import read_settings
from commander4.tod.noise.sample_ncorr import sample_correlated_noise, SIGMA0_METHODS
from commander4.tod.noise.sigma0 import _estimate_standalone_sigma0
from commander4.tod.step import ChainArrays, SavedBandSample, TODStep
from commander4.tod.view import TODView

if TYPE_CHECKING:
    from commander4.data_models.tod_samples import TODSamples

logger = logging.getLogger(__name__)


class Noise(TODStep):
    """The noise parameters of one band's detector-scans on one rank, and the steps that sample
    them.

    Sampled:
        noise_params: (nscans, ndet, npar) Noise parameters; the first is the white-noise level
            sigma0, in detector units.

    Attributes:
        cfg: The `tod_processing.corr_noise` settings. `enabled` and `from_iter` decide when the
            correlated noise is sampled; `sample_sigma0` also applies without it. The rest are
            passed on to `sample_correlated_noise`, and `cg` holds its CG controls (max_iter=0
            selects the stationary fallback). `psd_fit_nu_min`/`psd_fit_nu_max` are applied to the
            noise model by `NoisePSD.apply_param_file`.
        noise_model: The band's `NoisePSD`, which defines the parameters.
        ncorr_cg_residual, ncorr_cg_niter, ncorr_converged, ncorr_high_var: (nscans, ndet)
            Diagnostics of this iteration's correlated-noise draws, written to the chain: the final
            relative CG residual, the iteration count, whether the CG converged, and whether the
            variance check failed (either sends the draw to the stationary fallback). Reset by
            `start_iteration` (NaN, or -1 for the counts and flags), so a reset entry means "no
            n_corr drawn this iteration". `log_summary` reports the band-wide counts, so these
            arrays are what say *which* scans failed.
    """

    def __init__(self, experiment_data: DetectorGroupTOD, params: Bunch, my_band: Bunch,
                 band_comm: MPI.Comm, chain: int):
        """Read the settings, and start from the noise parameters in the parameter file, the data
        files, or the model.

        `my_band.initial_noise_params` comes first, then the data files' `init_scalars`, then the
        noise model's defaults.
        """
        self.cfg = read_settings(params, "corr_noise", ("tod_processing",),
                                 {"enabled": False, "from_iter": 1, "sample_psd_params": False,
                                  "sample_sigma0": True, "sigma0_method": "pairwise",
                                  "sigma0_decimation": 1, "nomono": False, "onlymono": False,
                                  "psd_bin": False, "use_dct": False, "cg": None,
                                  "psd_fit_nu_min": None, "psd_fit_nu_max": None})
        self.cfg.cg = read_settings(params, "cg", ("tod_processing.corr_noise",),
                                    {"max_iter": 0, "err_tol": 1.0e-4})
        # Both would otherwise be ignored without a word: the PSD parameters are only sampled
        # with n_corr, and the sigma0 estimate without n_corr takes any other method as pairwise.
        if self.cfg.sample_psd_params and not self.cfg.enabled:
            raise ValueError("corr_noise.sample_psd_params requires enabled=True.")
        if self.cfg.sigma0_method not in SIGMA0_METHODS:
            raise ValueError(f"corr_noise.sigma0_method must be one of {SIGMA0_METHODS}.")

        nscans, ndet = experiment_data.nscans, experiment_data.ndet
        self.noise_model = experiment_data.noise_model
        self.noise_params = np.full((nscans, ndet, self.noise_model.npar), np.nan)
        self.ncorr_cg_residual = np.full((nscans, ndet), np.nan)
        self.ncorr_cg_niter = np.full((nscans, ndet), -1, dtype=np.int32)
        self.ncorr_converged = np.full((nscans, ndet), -1, dtype=np.int8)
        self.ncorr_high_var = np.full((nscans, ndet), -1, dtype=np.int8)

        # Find the first detector in the first scan. Used for checking if default values
        # are present. Default to None in case there are no scans.
        rep_det = next((det for _, det in experiment_data.iter_detector_scans()), None)

        if rep_det is not None:
            if "initial_noise_params" in my_band:
                # Option 1: They are specified in the parameter file.
                self.noise_params[:] = np.array(my_band.initial_noise_params)
            elif rep_det.init_scalars is not None:
                # Option 2: There were entries in the read-in files. Index by the detector's
                # full-band column; absent detector-scans stay NaN.
                for iscan, det in experiment_data.iter_detector_scans():
                    self.noise_params[iscan, det.det_idx_fullband] = det.init_scalars[1:]
            else:
                # Option 3: Fall back to the noise model's default parameters (ensuring a finite
                # sigma0, which the model leaves as NaN to be estimated from the data). Warned
                # once per band.
                if band_comm.Get_rank() == 0 and chain == 1:
                    logger.warning(f"Band {experiment_data.band_name}: initial noise parameters "
                                   "were not found; using the noise model defaults.")
                default_params = np.array(self.noise_model.params, dtype=np.float64)
                if not np.isfinite(default_params[0]):
                    default_params[0] = 1.0
                self.noise_params[:] = default_params


    def chain_arrays(self) -> ChainArrays:
        """The noise parameters (restart state) and the CG diagnostics, all per scan."""
        return ChainArrays(per_scan={
            "noise_params": self.noise_params,
            "ncorr_cg_residual": self.ncorr_cg_residual,
            "ncorr_cg_niter": self.ncorr_cg_niter,
            "ncorr_converged": self.ncorr_converged,
            "ncorr_high_var": self.ncorr_high_var,
        })


    def read_chain(self, saved: SavedBandSample) -> None:
        """Continue from the noise parameters in a band chain file."""
        self.noise_params = saved.per_scan("noise_params").astype(self.noise_params.dtype)


    def start_iteration(self) -> None:
        """Forget the CG diagnostics of the previous iteration's draws."""
        self.ncorr_cg_residual[:] = np.nan
        self.ncorr_cg_niter[:] = -1
        self.ncorr_converged[:] = -1
        self.ncorr_high_var[:] = -1


    def sample_scan(self, view: TODView, rng: np.random.Generator) -> NDArray:
        """Draw the correlated noise of the focused detector-scan, and its noise parameters.

        Records the new noise parameters and the CG diagnostics of the draw, and returns the
        realization ``n_corr``, in detector units. Every random draw comes from ``rng``.
        """
        cfg = self.cfg
        res = sample_correlated_noise(
            view.get_tod(), view.get_mask(proc_mask_type="ncorr"),
            np.array(view.noise_params, copy=True),
            self.noise_model, view.fsamp, rng, cg_err_tol=cfg.cg.err_tol,
            cg_max_iter=cfg.cg.max_iter,
            sample_params=cfg.sample_psd_params, sample_sigma0=cfg.sample_sigma0,
            sigma0_method=cfg.sigma0_method, nomono=cfg.nomono, onlymono=cfg.onlymono,
            sigma0_dec=cfg.sigma0_decimation, psd_bin=cfg.psd_bin, use_dct=cfg.use_dct)
        self.noise_params[view.iscan, view.idet, :] = res.noise_params
        self.ncorr_cg_residual[view.iscan, view.idet] = res.residual
        self.ncorr_cg_niter[view.iscan, view.idet] = res.niter
        self.ncorr_converged[view.iscan, view.idet] = res.converged
        self.ncorr_high_var[view.iscan, view.idet] = res.high_var
        return res.n_corr


    def sample_sigma0(self, view: TODView) -> None:
        """Estimate sigma0 alone for the focused detector-scan, when correlated noise is not
        sampled. It runs at the same point in the chain (after gain) as the n_corr-coupled
        estimate, not in a separate pre-gain pass."""
        self.noise_params[view.iscan, view.idet, 0] = _estimate_standalone_sigma0(
            view, self.cfg.sigma0_method)


    def log_summary(self, tod_samples: TODSamples, iteration: int) -> None:
        """Log one per-band summary of this iteration's correlated-noise draws (collective).

        Reads the CG diagnostics `sample_scan` recorded this iteration and, when the noise-model
        parameters are sampled, their distributions, per parameter name (skipping sigma0), so the
        summary adapts to any NoisePSD model.
        """
        band_comm, experiment_data = tod_samples.band_comm, tod_samples.experiment_data
        drawn = self.ncorr_cg_niter >= 0
        residuals = band_comm.gather(self.ncorr_cg_residual[drawn])
        niters = band_comm.gather(self.ncorr_cg_niter[drawn])
        converged = band_comm.gather(self.ncorr_converged[drawn])
        high_var = band_comm.gather(self.ncorr_high_var[drawn])
        sampled_params = band_comm.gather(self.noise_params[drawn])
        if band_comm.Get_rank() != 0:
            return
        residuals, niters = np.concatenate(residuals), np.concatenate(niters)
        converged, high_var = np.concatenate(converged), np.concatenate(high_var)
        sampled_params = np.concatenate(sampled_params)  # (n_total, npar)
        n_total = niters.size
        if n_total == 0:
            return

        # Bands from different experiments can share a centre frequency, and one experiment can
        # hold several detector groups, so the log context names the experiment and band.
        context = (f"Chain {tod_samples.chain} iter{iteration} {experiment_data.experiment_name} "
                   f"{experiment_data.band_name}")

        # CG and variance sanity check failure rates, as warnings if more than 3% of scans fail.
        n_failed_conv = int(np.count_nonzero(converged == 0))
        n_high_var = int(np.count_nonzero(high_var == 1))
        log_failures = logger.warning if n_failed_conv/n_total > 0.03 else logger.verbose
        log_failures(f"{context}: noise CG failed for {100*n_failed_conv/n_total:.1f}% of scans "
                     f"({n_failed_conv}/{n_total}, worst residual = {np.max(residuals):.3e}).")
        log_high_var = logger.warning if n_high_var/n_total > 0.03 else logger.verbose
        log_high_var(f"{context}: variance sanity check failed for {100*n_high_var/n_total:.1f}% "
                     f"of scans ({n_high_var}/{n_total}).")

        # The worst residual and the failure counts are already in the two lines above, so this
        # line only adds the typical CG behaviour. A residual of 0 means no masked CG ran.
        cg_residuals = residuals[residuals != 0]
        median_residual = np.median(cg_residuals) if cg_residuals.size else 0.0
        logger.info(f"{context}: correlated noise for {n_total} detector-scans, median residual "
                    f"{median_residual:.2e}, median iterations {np.median(niters):.1f} "
                    f"(max {np.max(niters):.0f}).")
        if not self.cfg.sample_psd_params:
            return

        # Per-parameter distributions (sigma0 at index 0 is reported elsewhere), plus how many
        # detector-scans ended up sitting on a hard prior bound. A parameter whose true value lies
        # outside `P_uni` cannot be recovered: every scan rails against the nearest edge and then
        # reports a tight scatter around it, which reads as a converged answer in the chain. A
        # genuinely low-fknee detector will also sit near the bound, so this reports rather than
        # warns; it is the large railed fraction, not the railing itself, that says the bound is
        # wrong.
        noise_model = self.noise_model
        for j in range(1, sampled_params.shape[1]):
            name = noise_model.param_names[j] if j < len(noise_model.param_names) else f"p{j}"
            values = sampled_params[:, j]
            logger.verbose(f"{context}: {name} min {np.nanmin(values):.4f}, 1% "
                           f"{np.nanpercentile(values, 1):.4f}, mean {np.nanmean(values):.4f}, "
                           f"99% {np.nanpercentile(values, 99):.4f}, max {np.nanmax(values):.4f}")
            lo, hi = float(noise_model.P_uni[j, 0]), float(noise_model.P_uni[j, 1])
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
                continue
            # Within 1% of the bounds' own span counts as "on" the bound, so a grid endpoint drawn
            # through the inversion sampler's interpolation still registers.
            tol = 0.01*(hi - lo)
            n_lo = int(np.count_nonzero(values <= lo + tol))
            n_hi = int(np.count_nonzero(values >= hi - tol))
            if n_lo or n_hi:
                logger.verbose(
                    f"{context}: {name} on its prior bounds for {n_lo} ({n_lo/n_total:.1%}) "
                    f"of {n_total} detector-scans at the lower bound {lo:.4g}, and "
                    f"{n_hi} ({n_hi/n_total:.1%}) at the upper bound {hi:.4g}.")
