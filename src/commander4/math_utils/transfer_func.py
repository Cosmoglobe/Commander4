"""Bolometer transfer functions (swappable): the per-detector temporal response.

A real bolometer does not respond instantaneously to the incoming optical power: its thermal +
readout chain acts as a causal low-pass filter, smearing and lagging sky features along the scan
direction. A ``TransferFunction`` models this as a multiplicative filter ``H(f)`` applied to a
detector's clean signal TOD:

    apply(signal, fsamp) -> ndarray            # circular FFT convolution, signal shape (ntod,)

Two models are provided, using the Planck HFI bolometers as the baseline behaviour:

* ``SinglePole``: a single thermal time constant ``tau``, ``H(f) = 1 / (1 + 2*pi*i*f*tau)``. This
  is the common case and the one most sims want (one number per detector); HFI time constants are of
  order ~10 ms.
* ``MultiPole``: a sum of single-pole responses ``H(f) = (sum_k a_k / (1 + 2*pi*i*f*tau_k)) / sum_k
  a_k``, the "LFER"-style model HFI actually uses (a handful of time constants). DC-normalised so
  ``H(0) = 1``.

Both are normalised to unit DC gain (``H(0) = 1``), so the mean/calibration of the signal is
preserved and only its temporal shape changes. The filter is applied on a **mirrored** (reflected)
copy of the scan (``[x, x[::-1]]`` extended to length ``2*ntod``, filtered, first ``ntod`` samples
kept), i.e. exactly C4's ``forward_rfft_mirrored`` / ``backward_rfft_mirrored`` convention
([utils/math_operations.py]). The reflection makes the scan boundary value-continuous, so the causal
kernel's tail does not wrap the scan's *end* onto its *start* (as plain circular convolution would).
This is the operator (``T = R F^-1 H F E``, with reflect-extend ``E`` / restrict ``R``) the
Commander4 CG mapmaker's ``apply_T`` must match to deconvolve exactly this response.

Frequency convention (for wiring the matching deconvolution later): ``H`` here is a function of
*physical* frequency in Hz, evaluated on ``rfftfreq(2*ntod, d=1/fsamp)`` (the mirrored length). The
mapmaker's ``T_omega`` receives the *normalised* grid ``rfftfreq(2*ntod)`` (cycles/sample), so the
equivalent single-pole filter there is ``H(f_norm) = 1 / (1 + 2*pi*i*(f_norm*fsamp)*tau)``, the
same ``H`` with ``tau`` in seconds and ``fsamp`` in Hz. Note the mapmaker's adjoint ``apply_T_adjoint``
must conjugate the filter (``H*``), not flip the frequency array, to be the true transpose of ``T``.

Add a new model by subclassing ``TransferFunction`` (implement ``response``); extend
``make_detector_transfer`` to build it from the parameter file.
"""
import logging
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from pixell.bunch import Bunch
from commander4.math_utils.fft import forward_rfft, backward_rfft
from simgen.config import bget
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


class TransferFunction(ABC):
    """Base class for a detector's temporal response, applied as a filter H(f) along the scan."""

    @abstractmethod
    def response(self, freqs_hz: NDArray[np.floating]) -> NDArray[np.complexfloating]:
        """The complex filter ``H(f)`` sampled at the given physical frequencies [Hz]."""

    def apply(self, 
        signal: NDArray[np.floating],
        fsamp: float,
        prior_signal: NDArray[np.floating] | None = None,
        adjoint = False,
        nthreads = 1
    ) -> NDArray[np.floating]:
        """Filter ``signal`` with this transfer function via a mirrored FFT over the scan.

        The forward operator is ``T = R F^-1 diag(H) F E`` where ``E`` reflect-extends the scan to
        length ``2N`` (``x -> [x, x[::-1]]``), ``H = T_omega`` is the filter evaluated on the ``2N``
        frequency grid, and ``R`` restricts back to the first ``N`` samples. The grid is in physical
        Hz (``rfftfreq(2N, d=1/fsamp)``), so ``T_omega`` sees true frequencies and a time constant
        is in seconds, not samples. Mirroring makes the scan boundary continuous so a causal ``H`` does
        not wrap the scan's end onto its start (matching ``apply_N_inv`` and the simulator that bakes
        ``H`` in).

        The transpose is ``T^T = E^T F^-1 diag(H*) F R^T``: ``R^T`` zero-pads (``x -> [x, 0]``), the
        filter is **conjugated** (``H*``; a frequency flip is *not* the transpose for a non-trivial
        ``H``), and ``E^T`` folds the mirror back (``v -> v[:N] + v[N:][::-1]``). Implementing the two
        directions as this exact adjoint pair keeps the mapmaking operator ``P^T T^T N^-1 T P``
        symmetric, so the CG solve stays well-posed. At ``T_omega = 1`` both reduce to the identity.

        Args:
            signal: One detector's clean signal TOD (ntod,), before noise.
            fsamp: Sampling rate [Hz], setting the physical frequency grid.
            prior_signal: Optional noise-free prior TOD (ntod,) used to determine rising vs.
                falling states for asymmetric TFs. Ignored for linear symmetric TFs.

        Returns:
            The filtered TOD, same length and dtype as ``signal``. ``H(0) = 1`` preserves the DC
            level (and hence the gain calibration) up to the small mirrored-boundary effect.
        """
        n = signal.shape[-1]
        freqs = np.fft.rfftfreq(2 * n, d=1.0 / fsamp)

        if adjoint:
            ext = np.concatenate([signal, np.zeros_like(signal)])           # R^T: zero-pad
            filt = np.conj(self.response(freqs))                            # H*
        else:
            ext = np.concatenate([signal, signal[::-1]])                    # E: reflect-extend
            filt = self.response(freqs)                                     # H
        # Keep the filter in the signal's precision, so a float32 signal stays float32 throughout.
        filt = filt.astype(np.complex64 if signal.dtype == np.float32 else np.complex128)
        out = backward_rfft(forward_rfft(ext, nthreads=nthreads) * filt, 2 * n,
                                    nthreads=nthreads)
        if adjoint:
            return out[:n] + out[n:][::-1]                                  # E^T: fold the mirror back
        return np.ascontiguousarray(out[:n])                                # R: keep the first N samples


class SinglePole(TransferFunction):
    """First-order low-pass with a single thermal time constant: ``H(f) = 1/(1 + 2*pi*i*f*tau)``.

    In the time domain this is convolution with a one-sided (causal) decaying exponential of
    e-folding time ``tau``, giving the characteristic scan-direction lag and smearing of a bolometer.
    """
    def __init__(self, tau_sec: float):
        self.tau = float(tau_sec)

    def response(self, freqs_hz):
        return 1.0 / (1.0 + 2j * np.pi * freqs_hz * self.tau)

    def stepup_response(self, A, t,
                      tau_sec:float|None = None,
                      ):
        """
        Computes the TF response to a flat top-hat signal with intensity A for times `t` and 0 otherwise.

        Parameters can be specified to overwrite the fitted ones saved as members.
        """
        tau = self.tau if tau_sec is None else tau_sec
        return A * (1.0 - np.exp(-t / tau))

    def stepdown_response(self, A, t,
                      tau_sec:float|None = None
                      ):
        """
        Computes the TF response to a signal which has intensity A and gets switched off immediately before ``t[0]``.

        Parameters can be specified to overwrite the fitted ones saved as members.
        """
        tau = self.tau if tau_sec is None else tau_sec
        return A * np.exp(-t / tau)


class MultiPole(TransferFunction):
    """Sum of single-pole responses (HFI "LFER" model): ``H = (sum_k a_k/(1+2*pi*i*f*tau_k))/sum a_k``.

    The amplitudes ``a_k`` are forced to be normalised by their sum so ``H(0) = 1`` (unit DC gain), 
    so that for N poles we have N taus and N-1 free amplitudes, setting amp(N) = 1.0 - sum(amps)
    the several-time-constant behaviour of the real HFI detectors while remaining a simple linear,
    calibration-preserving filter.
    """
    def __init__(self, amps: list[float], taus_sec_s: list[float]):
        self.amps = np.asarray(amps, dtype=np.float64)
        self.taus = np.asarray(taus_sec_s, dtype=np.float64)
        if self.amps.shape[0] != self.taus.shape[0]-1 or self.taus.ndim != 1 or self.taus.size == 0:
            raise ValueError("MultiPole needs matching non-empty 1D amp/tau lists.")
        self.norm = self.amps.sum()
        if self.norm == 0.0:
            raise ValueError("MultiPole pole amplitudes sum to zero; cannot DC-normalise.")

    @property
    def Npoles(self):
        return len(self.taus)

    def response(self, freqs_hz):
        w = 2j * np.pi * freqs_hz
        poles = self.amps[:, None] / (1.0 + w[None, :] * self.taus[:, None])
        return poles.sum(axis=0) / self.norm

    def step_up_response(self, A, t,
            amps:list[float]|NDArray|None = None,
            taus:list[float]|NDArray|None = None,
            ):
        """
        Computes the TF response to a flat top-hat signal with intensity A for times `t` and 0 otherwise.

        Parameters can be specified to overwrite the fitted ones saved as members.
        """
        amps = self.amps if amps is None else amps
        taus = self.taus if taus is None else taus
        assert len(amps) == len(taus) - 1, "For N poles only N-1 amplitudes must be defined."
        res = 1.0
        for i, c in enumerate(amps):
            res += -c * np.exp(-t / taus[i])
        #last pole with fixed c
        c = 1.0 - np.sum(amps)
        res += -c * np.exp(-t / taus[-1])
        return A * res
    
    def step_down_response(self, A, t,
            amps:list[float]|NDArray|None = None,
            taus:list[float]|NDArray|None = None,
            ):
        """
        Computes the TF response to a flat top-hat signal with intensity A for times `t` and 0 otherwise.

        Parameters can be specified to overwrite the fitted ones saved as members.
        """
        #FIXME: check this out: TF fitted should have positive amplitudes? Maybe not necessarily
        amps = self.amps if amps is None else amps
        taus = self.taus if taus is None else taus
        assert len(amps) == len(taus) - 1, "For N poles only N-1 amplitudes must be defined."
        res = 0.0
        for i, c in enumerate(amps):
            res += c * np.exp(-t / taus[i])
        #last pole with fixed c
        c = 1.0 - np.sum(amps)
        res += c * np.exp(-t / taus[-1])
        return A * res

    def fit_to_stepfunction(
            self,
            t_data: np.ndarray, 
            y_data: np.ndarray,
            is_stepdown = False,
            A: float|None = None,
            fit_bounds:None|tuple[list] = None
        ) -> dict:
        """
        Fits a dual-exponential transfer function model to step response data.

        Parameters:
        -----------
        t_data : np.ndarray
            Relative time array starting at 0 [s].
        y_data : np.ndarray
            Signal output intensity array [A.U.].

        Optional:
        -----------
        is_stepdown : bool
            if we are fitting for the raising or dropping tophat signal (default False). 
        A : float|None
            amplitude of the tophat signal, if not given can be estimated from the plateau in the
            data, but only when fitting to a raising signal.
        fit_bounds : None|tuple[list]
            fit boundaries to pass to the optimization routine, as a tuple of two lists, lower and
            upper bounds respectively. Within each list the first N-1 
        Returns: 
        --------
        dict containing:
        - 'A': Top plateau amplitude
        - 'params': fitted (c1, tau1, tau2)
        - 'y_fit': fitted time-domain signal
        """
        t = np.asarray(t_data, dtype=float)
        y = np.asarray(y_data, dtype=float)

        if A is None:
            if is_stepdown:
                raise ValueError("Can not fit stepdown with unknown signal amplitude A")
            else:
                #Calculate top plateau A from the average of the last 3 seconds
                print("WARNING: signal amplitude for step response not specifed, estimating from last 3 sec.")
                t_max = t.max()
                last_3s_mask = t >= (t_max - 3.0)
                A = np.mean(y[last_3s_mask])


        p0 = [c for c in self.amps] + [t for t in self.taus]

        if is_stepdown:
            def _step_response(t_val, *params):
                c_s = params[0:self.N_nodes-1]
                tau_s = params[self.N_nodes-1:]
                return self.stepdown_response(A=A, t=t_val, c_s=c_s, tau_s=tau_s)
        else:
            def _step_response(t_val, *params):
                c_s = params[0:self.N_nodes-1]
                tau_s = params[self.N_nodes-1:]
                return self.step_response(A=A, t=t_val, c_s=c_s, tau_s=tau_s)

        # Fit parameters
        if fit_bounds is not None:
            popt, _ = curve_fit(_step_response, t, y, p0=p0, bounds=fit_bounds)
        else:
            popt, _ = curve_fit(_step_response, t, y, p0=p0)

        self.amps = popt[0:self.N_nodes-1]
        self.taus = popt[self.N_nodes-1:]
        
        return A, _step_response(t, *popt)


class AsymmetricTransferFunction(TransferFunction):
    """Applies different transfer functions depending on whether the signal is rising or falling.

    Uses a Filter-then-Splice strategy: both `tf_up` and `tf_down` are evaluated independently 
    over the full scan using mirrored FFTs, and then blended in the time domain using a smooth 
    rising/falling mask derived from a reference signal.
    """

    def __init__(
        self,
        tf_up: TransferFunction,
        tf_down: TransferFunction,
        mask_method: str,
        epsilon: float,
    ):
        """
        Args:
            tf_up: TransferFunction to apply when the signal is rising (d/dt > 0).
            tf_down: TransferFunction to apply when the signal is falling (d/dt < 0).
            mask_method: how to compute the mask to choose between up and down filter.
                For now only 'bin' and 'sigmoid' allowed.
            epsilon: Hysteresis threshold relative to max(|s_dot|) to prevent rapid 
                switching ("chattering") near peaks/troughs, for 'bin' masking method.
                Factor giving the steepness of the sigmoid function for 'sigmoid' masking method,
                as ``mask = sigmoid(epsilon*s_dot)``.
        """
        if not mask_method in ['bin', 'sigmoid']:
            raise ValueError("Unknown masking method specified.")
        else:
            self.mask_method = mask_method
        self.tf_up = tf_up
        self.tf_down = tf_down
        self.epsilon = float(epsilon)

    def response(self, freqs_hz: NDArray[np.floating]) -> NDArray[np.complexfloating]:
        raise NotImplementedError(
            "AsymmetricTransferFunction is state-dependent and cannot be "
            "represented by a single frequency response H(f)."
        )

    def _bin_mask(
        self, s_dot
    ) -> NDArray[np.floating]:
        """
        Computes a binary mask from the derivative of the signal.
        """
        max_abs_dot = np.max(np.abs(s_dot))
        eps = self.epsilon * max_abs_dot if max_abs_dot > 0 else 0.0
        # Construct state mask with hysteresis holding: if the derivative does not change enogh,
        # depending on the relative threshold defined, the mask will not flip.
        mask = np.empty(len(s_dot), dtype=np.float64)
        curr_state = 1.0  # Default initial state
        for i in range(len(s_dot)):
            if s_dot[i] > eps:
                curr_state = 1.0
            elif s_dot[i] < -eps:
                curr_state = 0.0
            mask[i] = curr_state
        return mask

    def _sigmoid_mask(
        self, s_dot
    ) -> NDArray[np.floating]:
        """
        Computes a sigmoid mask from the derivative of the signal, giving a smooth transition between
        up and down transfer functions.
        """
        return 1/(1 + np.exp(-self.epsilon*s_dot))

    def _compute_rising_mask(
        self, ref_signal: NDArray[np.floating], fsamp: float
    ) -> NDArray[np.floating]:
        """Generates a binary state mask (1.0 = rising, 0.0 = falling) with hysteresis."""

        ref_signal = ref_signal.astype(np.float64)

        # Compute smooth derivative s_dot = d/dt
        s_dot = np.gradient(ref_signal, 1.0 / fsamp)
    
        if self.mask_method == 'bin':
            return self._bin_mask(s_dot)
        elif self.mask_method == 'sigmoid':
            return self._sigmoid_mask(s_dot)
        else:
            raise ValueError("Unrecognized masking method")

    def _apply(
        self,
        signal: NDArray[np.floating],
        fsamp: float,
        prior_signal: NDArray[np.floating] | None = None,
        nthreads = 1
    ) -> NDArray[np.floating]:
        """Applies asymmetric filtering to ``signal``. First filters then masks.
        """
        # Generate time-domain mask M (1 for rising, 0 for falling)
        ref_signal = prior_signal if prior_signal is not None else signal
        mask = self._compute_rising_mask(ref_signal, fsamp)

        # Run full mirrored FFT filtering for both filters
        filtered_up = self.tf_up.apply(signal, fsamp, adjoint=False, nthreads=nthreads)
        filtered_down = self.tf_down.apply(signal, fsamp, adjoint=False, nthreads=nthreads)

        # Apply mask in time domain
        filtered = mask * filtered_up + (1.0 - mask) * filtered_down
        return filtered.astype(signal.dtype, copy=False)

    def _apply_adjoint(
        self,
        signal: NDArray[np.floating],
        fsamp: float,
        prior_signal: NDArray[np.floating] | None = None,
        nthreads = 1
    ) -> NDArray[np.floating]:
        """Applies adjoint of asymmetric filtering to ``signal``. First masks then filters.
        """
        # Generate time-domain mask M (1 for rising, 0 for falling)
        ref_signal = prior_signal if prior_signal is not None else signal
        mask = self._compute_rising_mask(ref_signal, fsamp)

        #Mask input signal first
        masked_up = mask * signal.astype(np.float64)
        masked_down = (1.0 - mask) * signal.astype(np.float64)

        # Apply adjoints of the corresponding TFs
        trans_up = self.tf_up.apply(masked_up, fsamp, adjoint=True, nthreads=nthreads)
        trans_down = self.tf_down.apply(masked_down, fsamp, adjoint=True, nthreads=nthreads)

        # Combine
        res = trans_up + trans_down
        return res.astype(signal.dtype, copy=False)

    def apply(
        self,
        signal: NDArray[np.floating],
        fsamp: float,
        prior_signal: NDArray[np.floating] | None = None,
        adjoint = False,
        nthreads = 1
    ) -> NDArray[np.floating]:
        """Applies asymmetric filtering to ``signal``.

        If ``prior_signal`` is provided (e.g. during CG solving), it is used to determine 
        the rising/falling state mask. If ``prior_signal`` is None (e.g. forward simulations),
        the mask is derived directly from ``signal``. ``prior_signal`` is assumed to be already
        correctly smoothed to the telescope FWHM.

        In the adjoint operator, masking is applied before.
        """
        if adjoint:
            return self._apply_adjoint(
                        signal=signal, 
                        fsamp=fsamp, 
                        prior_signal=prior_signal, 
                        nthreads=nthreads
                        )
        else:
            return self._apply(
                        signal=signal, 
                        fsamp=fsamp, 
                        prior_signal=prior_signal, 
                        nthreads=nthreads
                        )

def _tau_sec(spec: Bunch | dict) -> float | None:
    """Read a time constant from a param block, accepting ``tau_sec`` or ``tau_ms`` (None if absent).

    Uses ``bget`` (subscript) rather than attribute access so it works on both a ``Bunch`` and a
    plain ``dict``, since the ``poles`` list entries arrive as dicts (the loader does not bunch-ify
    list elements).
    """
    if bget(spec, "tau_sec", None) is not None:
        return float(bget(spec, "tau_sec"))
    if bget(spec, "tau_ms", None) is not None:
        return float(bget(spec, "tau_ms")) * 1e-3
    return None


def _build_from_node(node: Bunch) -> TransferFunction | None:
    """Build a transfer function from a spec node holding either ``poles`` or a single ``tau_*``.

    A ``poles`` list (each entry an ``{amp, tau_ms|tau_sec}`` block) gives a ``MultiPole``; a single
    ``tau_ms``/``tau_sec`` gives a ``SinglePole``. A missing or zero time constant means identity, for
    which we return ``None`` (no filtering) so the caller can skip the FFT entirely.

    For an asymmetric TF, both ``up`` and ``down`` correctly defined. the ``mask_method`` and ``epsilon``
    parameters will default to 'bin' and 0.05 respectively, if not specified.
    """
    enabled = bget(node, "enabled", True)
    if not enabled:
        return None
    #asymmetric TF case
    if "up" in node:
        tf_up = _build_from_node(node.up)
        if "down" in node:
            tf_down = _build_from_node(node.down)
            rel_eps = bget(node, "epsilon", 0.05)
            mask_method = bget(node, "mask_method", 'bin')
            return AsymmetricTransferFunction(tf_up=tf_up, 
                                              tf_down=tf_down,
                                              mask_method=mask_method,
                                              epsilon=rel_eps)
        else:
            raise ValueError("If a TF up is defined, a TF down must be defined too.")

    #TF with multi-pole case
    poles = bget(node, "poles", None)
    if poles is not None:
        amps, taus = [], []
        for pole in poles:
            tau = _tau_sec(pole)
            if tau is None:
                raise ValueError("Each transfer-function pole needs a 'tau_ms' or 'tau_sec'.")
            amps.append(float(bget(pole, "amp", 1.0)))
            taus.append(tau)
        return MultiPole(amps, taus)
    
    #Single-pole TF
    tau = _tau_sec(node)
    return SinglePole(tau) if tau else None


def make_detector_transfer(det_spec: Bunch, global_default: Bunch | None) -> TransferFunction | None:
    """Resolve one detector's bolometer transfer function from the parameter file.

    Resolution order (first that applies wins): a per-detector ``tau_ms``/``tau_sec`` shorthand, a
    per-detector ``transfer_function`` block (single ``tau_*`` or a ``poles`` list), then the run-wide
    ``simulation.transfer_function`` default. Any level may set ``enabled: false`` to force identity.
    Returns ``None`` (no filtering) when nothing is configured or the response is identity.
    """
    if _tau_sec(det_spec) is not None:
        node = det_spec
    elif "transfer_function" in det_spec:
        node = det_spec.transfer_function
    elif global_default is not None:
        node = global_default
    else:
        return None
    if not bget(node, "enabled", True):
        return None
    return _build_from_node(node)
