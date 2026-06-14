import logging
from typing import Optional
import numpy as np
import pixell
from scipy.fft import rfftfreq
from numpy.typing import NDArray
from commander4.utils.math_operations import forward_rfft

logger = logging.getLogger(__name__)


def _inversion_sampler_1d(lnL: NDArray, grid_points: NDArray) -> float:
    """ Performs 1D inversion sampling on a grid. This involves calculating the cumulative
        log-likelihood, normalizing this to be contained in [0,1], drawing a random number in [0,1],
        and matching that to a (interpolated) grid point.
    Args:
        lnL (np.ndarray): Array of log-likelihood values at each grid point.
        grid_points (np.ndarray): The corresponding parameter values for each grid point.
    Returns:
        sample (float): A single random sample drawn from the provided distribution.
    """
    lnL -= np.max(lnL)
    L = np.exp(lnL)  # Calculate the linear likelihood.
    cdf = np.cumsum(L)  # Cumulative likelihood.
    cdf /= cdf[-1]  # Constrain it to [0,1].
    u = np.random.uniform(0, 1)
    sample = np.interp(u, cdf, grid_points)  # Find the x-value that matches the y-value we drew.
    return sample

# ===================================================================
#  Base class
# ===================================================================
class NoisePSD:
    """Base class for noise PSD models.

    Each subclass declares a ``param_names`` tuple listing parameter
    names in order (e.g. ``('sigma0', 'fknee', 'alpha')``).  Values
    are stored in a ``params`` array (float64), accessed by index
    (e.g. ``params[0]`` for sigma0).

    Attributes:
        param_names: Ordered tuple of parameter names (for repr / config parsing).
        P_uni: Uniform (hard) prior bounds ``[lo, hi]`` per parameter, shape (npar, 2).
        P_active: Informative prior ``[mean, rms]`` per parameter, shape (npar, 2).
        P_lognorm: If True the informative prior is log-normal, else Gaussian, shape (npar,).
        nu_fit: Frequency range ``[f_min, f_max]`` (Hz) for fitting, shape (npar, 2).
        is_white: True for purely white models (enables an FFT-free fast path in ``apply_N_inv``).
    """

    param_names: tuple[str, ...] = ()
    is_white: bool = False  # Override to True in purely white-noise subclasses.
    _jit_model_id: int = -1  # Numba model ID; -1 = Python fallback

    def __init__(self,
                 P_active_mean: NDArray,
                 P_active_rms: NDArray,
                 P_uni: NDArray,
                 nu_fit: NDArray,
                 P_lognorm: NDArray[np.bool_]):
        n = len(P_active_mean)

        # Parameter values as a contiguous array
        self.params = np.array(P_active_mean[:n], dtype=np.float64)

        # Prior arrays (indexed in the same order as param_names / params)
        self.P_uni = np.array(P_uni, dtype=np.float32).reshape(n, 2).copy()
        self.P_active = np.empty((n, 2), dtype=np.float32)
        self.P_active[:, 0] = P_active_mean[:n]
        self.P_active[:, 1] = P_active_rms[:n]
        self.P_lognorm = np.array(P_lognorm, dtype=bool).copy()
        self.nu_fit = np.array(nu_fit, dtype=np.float32).reshape(n, 2).copy()

    @property
    def npar(self) -> int:
        """Number of free parameters."""
        return len(self.params)

    # ---- interface (override in subclasses) --------------------------------
    # All models receive the noise parameters explicitly (rather than reading ``self.params``)
    # because the live per-scan/per-detector values are stored in ``TODSamples.noise_params``,
    # while a single shared model instance is attached to each ``DetGroupTOD``.
    def eval_full(self, freqs: NDArray, noise_params: NDArray) -> NDArray:
        """Evaluate the *full* PSD (white + correlated) at each frequency in *freqs* (Hz)."""
        raise NotImplementedError

    def eval_corr(self, freqs: NDArray, noise_params: NDArray) -> NDArray:
        """Evaluate the *correlated-only* PSD at each frequency in *freqs* (Hz)."""
        raise NotImplementedError

    def compute_inv_corr_spectrum(self, freqs: NDArray, noise_params: NDArray) -> NDArray:
        """Return the inverse correlated-noise power spectrum 1 / P_corr(f) for each frequency.

        This is the quantity added to the inverse white-noise level in the correlated-noise CG
        (see ``corr_noise_realization_with_gaps``), so it carries units of inverse variance.
        Frequencies where ``P_corr <= 0`` (e.g. the zero-frequency mode) get ``inv = 0``.
        """
        P_corr = self.eval_corr(freqs, noise_params)
        out = np.zeros_like(P_corr)
        good = P_corr > 0
        out[good] = 1.0 / P_corr[good]
        return out

    def sample_params(self, residual: NDArray, noise_params: NDArray, fsamp: float, *,
                      nu_min: float = 0.0, nu_max: float = np.inf, bin_psd: bool = False) -> NDArray:
        """Draw a new sample of the model parameters (except ``sigma0`` at index 0).

        The fit is performed against the periodogram of the sky-subtracted *residual* TOD (with
        masked gaps inpainted by the caller using the correlated-noise realization plus white
        noise), fitting the *full* PSD model. This mirrors the Commander3 ``sample_noise_psd``
        routine and keeps the Markov-chain correlation length short relative to fitting the
        drawn ``n_corr`` realization directly.

        Args:
            residual: Sky-subtracted residual TOD (white + correlated noise), gaps inpainted.
            noise_params: Current parameter values; ``noise_params[0]`` (sigma0) is held fixed.
            fsamp: Sampling rate of *residual* (Hz).
            nu_min, nu_max: Frequency range (Hz) over which to fit the PSD parameters.
            bin_psd: If True, fit a log-binned periodogram (each bin Whittle-weighted by its mode
                count); if False, fit every Fourier mode in range (default).
        Returns:
            A new ``noise_params`` array with the updated parameter values.
        """
        raise NotImplementedError

    # ---- prior evaluation -------------------------------------------------
    def log_prior(self, param_idx: int, x: float) -> float:
        """Evaluate the log-prior for parameter *param_idx* at value *x*.

        Returns ``-1e30`` if *x* is outside the uniform bounds ``P_uni``.
        """
        lo, hi = self.P_uni[param_idx]
        if x < lo or x > hi:
            return -1e30
        mu = float(self.P_active[param_idx, 0])
        sigma = float(self.P_active[param_idx, 1])
        if sigma <= 0:
            return 0.0  # flat / non-informative
        if self.P_lognorm[param_idx]:
            return -0.5 * (np.log(x) - np.log(mu)) ** 2 / (sigma * np.log(10)) ** 2 - np.log(x)
        else:
            return -0.5 * (x - mu) ** 2 / sigma ** 2

    # ---- repr -------------------------------------------------------------
    def __repr__(self) -> str:
        cls = type(self).__name__
        pnames = type(self).param_names
        parts = []
        for j, val in enumerate(self.params):
            name = pnames[j] if j < len(pnames) else f"p{j}"
            parts.append(f"{name}={val:.4g}")
        return f"{cls}({', '.join(parts)})"


# ===================================================================
#  2. Standard 1/f  (oof)
# ===================================================================
class NoisePSDOof(NoisePSD):
    """P(f) = sigma0^2 (1 + (f / f_knee)^alpha)."""

    param_names = ('sigma0', 'fknee', 'alpha')

    def __init__(self,
                 P_active_mean = [np.nan, 10.0, -2.7],
                 P_active_rms = [np.nan, np.inf, np.inf],
                 P_uni = [[np.nan, np.nan], [1.0, 100.0], [-4.5, -1.0]],
                 nu_fit = [[np.nan, np.nan], [0, 10.0], [0, 10.0]],
                 **kw):
        P_lognorm = np.array([False, True, False])  # sigma0, fknee, alpha
        super().__init__(P_active_mean=P_active_mean[:3], P_active_rms=P_active_rms[:3],
                         P_uni=P_uni[:3], nu_fit=nu_fit[:3], P_lognorm=P_lognorm, **kw)

    def eval_full(self, freqs: NDArray, noise_params: NDArray) -> NDArray:
        s0, fk, a = noise_params
        vals = np.full(len(freqs), s0 ** 2, dtype=np.float64)
        pos = freqs > 0
        vals[pos] = s0 ** 2 * (1.0 + (freqs[pos] / fk) ** a)
        return vals

    def eval_corr(self, freqs: NDArray, noise_params: NDArray) -> NDArray:
        s0, fk, a = noise_params
        vals = np.zeros(len(freqs), dtype=np.float64)
        pos = freqs > 0
        vals[pos] = s0 ** 2 * (freqs[pos] / fk) ** a
        return vals

    def sample_params(self, residual: NDArray, noise_params: NDArray, fsamp: float, *,
                      nu_min: float = 0.0, nu_max: float = np.inf, bin_psd: bool = False,
                      n_grid: int = 150, n_burnin: int = 4) -> NDArray:
        """ Draw a sample of (fknee, alpha) for the 1/f model by fitting the *full* PSD
            P(f) = sigma0^2 (1 + (f/fknee)^alpha) to the periodogram of the sky-subtracted
            *residual* TOD, with sigma0 (= noise_params[0]) held fixed. Uses a short Gibbs loop
            that alternately inversion-samples fknee and alpha on grids spanning the model's
            uniform priors (``P_uni``). See the base-class docstring for argument meaning.
        Returns:
            A new [sigma0, fknee, alpha] array with updated fknee and alpha.
        """
        sigma0_sq = float(noise_params[0])**2
        alpha_current = float(noise_params[2])
        fknee_current = float(noise_params[1])
        Ntod = len(residual)
        freqs = rfftfreq(Ntod, 1.0/fsamp)
        power = (1.0 / Ntod) * np.abs(forward_rfft(residual))**2

        # Restrict to the requested fit range, always excluding the zero-frequency (DC) mode.
        in_fit = (freqs > 0) & (freqs >= nu_min) & (freqs <= nu_max)
        f = freqs[in_fit]
        p = power[in_fit]
        if bin_psd:
            # Log-bin the periodogram; each bin is Whittle-weighted by its number of modes.
            bins = pixell.utils.expbin(f.size, nbin=100, nmin=1)
            weight = (bins[:, 1] - bins[:, 0]).astype(np.float64)
            f = pixell.utils.bin_data(bins, f)
            p = pixell.utils.bin_data(bins, p)
        else:
            weight = np.ones(f.size, dtype=np.float64)
        log_p = np.log(p)
        w = weight[:, np.newaxis]

        # Grids span the uniform priors (single source of truth for the hard parameter bounds).
        fk_lo, fk_hi = float(self.P_uni[1, 0]), float(self.P_uni[1, 1])
        al_lo, al_hi = float(self.P_uni[2, 0]), float(self.P_uni[2, 1])
        fknee_grid = np.logspace(np.log10(fk_lo), np.log10(fk_hi), n_grid)
        alpha_grid = np.linspace(al_lo, al_hi, n_grid)

        for _ in range(n_burnin + 1):
            # Whittle log-likelihood sum_l w_l (log p_l - log S_l - p_l/S_l), with the full model
            # S = sigma0^2 (1 + (f/fknee)^alpha). 1. Sample fknee given the current alpha.
            S = sigma0_sq * (1.0 + (f[:, np.newaxis] / fknee_grid) ** alpha_current)
            resid = log_p[:, np.newaxis] - np.log(S)
            log_L_fknee = np.sum(w * (resid - np.exp(resid)), axis=0)
            fknee_current = float(_inversion_sampler_1d(log_L_fknee, fknee_grid))
            # 2. Sample alpha given the new fknee.
            S = sigma0_sq * (1.0 + (f[:, np.newaxis] / fknee_current) ** alpha_grid)
            resid = log_p[:, np.newaxis] - np.log(S)
            log_L_alpha = np.sum(w * (resid - np.exp(resid)), axis=0)
            alpha_current = float(_inversion_sampler_1d(log_L_alpha, alpha_grid))

        out = np.array(noise_params, dtype=np.float64, copy=True)
        out[1] = fknee_current
        out[2] = alpha_current
        return out