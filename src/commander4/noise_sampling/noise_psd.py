import logging
from typing import Optional
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

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
        apply_filter: Whether to multiply output by a modulation filter spline.
    """

    param_names: tuple[str, ...] = ()
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
    def eval_full(self, freqs: NDArray) -> NDArray:
        """Evaluate the *full* PSD (white + correlated) at each frequency in *freqs* (Hz)."""
        raise NotImplementedError

    def eval_corr(self, freqs: NDArray) -> NDArray:
        """Evaluate the *correlated-only* PSD at each frequency in *freqs* (Hz)."""
        raise NotImplementedError

    def compute_inv_corr_spectrum(self, freqs: NDArray) -> NDArray:
        """Return sigma0^2 / P_corr(f) for each frequency.

        Used to build ``C_corr_inv`` in the noise-sampling CG.
        Frequencies where ``P_corr <= 0`` get ``inv = 0``.
        """
        P_corr = self.eval_corr(freqs)
        out = np.zeros_like(P_corr)
        good = P_corr > 0
        out[good] = self.params[0] ** 2 / P_corr[good]
        return out

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
                 P_active_mean = [np.nan, 0.1, -1.0],
                 P_active_rms = [np.nan, np.inf, np.inf],
                 P_uni = [[np.nan, np.nan], [0.01, 0.5], [-2.5, -0.25]],
                 nu_fit = [[np.nan, np.nan], [0, 3.0], [0, 3.0]],
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