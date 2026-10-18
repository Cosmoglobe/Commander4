"""`Noise`: the noise parameters of every detector-scan, and the steps that sample them.

Both steps run once per detector-scan inside the mapmaking scan loop (`tod2map`), after the gain:
either a correlated-noise realization together with sigma0 and the noise-model parameters, or,
when correlated noise is not sampled, sigma0 alone. The numerics live in `sample_ncorr.py` and
`sigma0.py` next to this file.
"""
from __future__ import annotations

import logging

import numpy as np
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.parameters.schema import read_settings
from commander4.tod.noise.sample_ncorr import sample_correlated_noise, SIGMA0_METHODS
from commander4.tod.noise.sigma0 import _estimate_standalone_sigma0
from commander4.tod.step import ChainArrays, SavedBandSample, TODStep
from commander4.tod.view import TODView

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
        ncorr_cg_residual, ncorr_cg_niter, ncorr_converged: (nscans, ndet) Diagnostics of the
            correlated-noise CG, written to the chain: the final relative CG residual, the
            iteration count, and whether the draw was accepted (converged and not variance-capped;
            -1 = no n_corr drawn yet). The log reports only the band-wide counts of these, so this
            is what says *which* scans failed.
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
        })


    def read_chain(self, saved: SavedBandSample) -> None:
        """Continue from the noise parameters in a band chain file."""
        self.noise_params = saved.per_scan("noise_params").astype(self.noise_params.dtype)


    def sample_scan(self, view: TODView) -> Bunch:
        """Draw the correlated noise of the focused detector-scan, and its noise parameters.

        Returns:
            The result of `sample_correlated_noise`: the realization ``n_corr`` (detector units),
            the new ``noise_params``, and the CG ``residual``, ``niter``, ``converged`` and
            ``high_var``, which the scan loop also sums up for its log.
        """
        cfg = self.cfg
        res = sample_correlated_noise(
            view.get_tod(), view.get_mask(proc_mask_type="ncorr"),
            np.array(view.noise_params, copy=True),
            self.noise_model, view.fsamp, cg_err_tol=cfg.cg.err_tol, cg_max_iter=cfg.cg.max_iter,
            sample_params=cfg.sample_psd_params, sample_sigma0=cfg.sample_sigma0,
            sigma0_method=cfg.sigma0_method, nomono=cfg.nomono, onlymono=cfg.onlymono,
            sigma0_dec=cfg.sigma0_decimation, psd_bin=cfg.psd_bin, use_dct=cfg.use_dct)
        self.noise_params[view.iscan, view.idet, :] = res.noise_params
        self.ncorr_cg_residual[view.iscan, view.idet] = res.residual
        self.ncorr_cg_niter[view.iscan, view.idet] = res.niter
        self.ncorr_converged[view.iscan, view.idet] = res.converged and not res.high_var
        return res


    def sample_sigma0(self, view: TODView) -> None:
        """Estimate sigma0 alone for the focused detector-scan, when correlated noise is not
        sampled. It runs at the same point in the chain (after gain) as the n_corr-coupled
        estimate, not in a separate pre-gain pass."""
        self.noise_params[view.iscan, view.idet, 0] = _estimate_standalone_sigma0(
            view, self.cfg.sigma0_method)
