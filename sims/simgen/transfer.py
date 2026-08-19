"""Bolometer transfer functions (swappable): the per-detector temporal response.

A real bolometer does not respond instantaneously to the incoming optical power: its thermal +
readout chain acts as a causal low-pass filter, smearing and lagging sky features along the scan
direction. A ``TransferFunction`` models this as a multiplicative filter ``H(f)`` applied to a
detector's clean signal TOD:

    apply(signal, fsamp) -> ndarray            # circular FFT convolution, signal shape (ntod,)

Two models are provided, using the Planck HFI bolometers as the baseline behaviour:

* ``SinglePole`` -- a single thermal time constant ``tau``: ``H(f) = 1 / (1 + 2*pi*i*f*tau)``. This
  is the common case and the one most sims want (one number per detector); HFI time constants are of
  order ~10 ms.
* ``MultiPole`` -- a sum of single-pole responses ``H(f) = (sum_k a_k / (1 + 2*pi*i*f*tau_k)) / sum_k
  a_k``, the "LFER"-style model HFI actually uses (a handful of time constants). DC-normalised so
  ``H(0) = 1``.

Both are normalised to unit DC gain (``H(0) = 1``), so the mean/calibration of the signal is
preserved -- only its temporal shape changes. The filter is applied on a **mirrored** (reflected)
copy of the scan -- ``[x, x[::-1]]`` extended to length ``2*ntod``, filtered, first ``ntod`` samples
kept -- i.e. exactly C4's ``forward_rfft_mirrored`` / ``backward_rfft_mirrored`` convention
([utils/math_operations.py]). The reflection makes the scan boundary value-continuous, so the causal
kernel's tail does not wrap the scan's *end* onto its *start* (as plain circular convolution would).
This is the operator (``T = R F^-1 H F E``, with reflect-extend ``E`` / restrict ``R``) the
Commander4 CG mapmaker's ``apply_T`` must match to deconvolve exactly this response.

Frequency convention (for wiring the matching deconvolution later): ``H`` here is a function of
*physical* frequency in Hz, evaluated on ``rfftfreq(2*ntod, d=1/fsamp)`` (the mirrored length). The
mapmaker's ``T_omega`` receives the *normalised* grid ``rfftfreq(2*ntod)`` (cycles/sample), so the
equivalent single-pole filter there is ``H(f_norm) = 1 / (1 + 2*pi*i*(f_norm*fsamp)*tau)`` -- the
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

from simgen.config import bget

logger = logging.getLogger(__name__)


class TransferFunction(ABC):
    @abstractmethod
    def response(self, freqs_hz: NDArray[np.floating]) -> NDArray[np.complexfloating]:
        """The complex filter ``H(f)`` sampled at the given physical frequencies [Hz]."""

    def apply(self, signal: NDArray[np.floating], fsamp: float) -> NDArray[np.floating]:
        """Filter ``signal`` with this transfer function via a mirrored FFT over the scan.

        The scan is reflected (``[x, x[::-1]]``, length ``2*ntod``), filtered with ``H`` on the
        ``2*ntod`` frequency grid, and the first ``ntod`` samples are returned -- matching C4's
        ``forward_rfft_mirrored``/``backward_rfft_mirrored`` so the CG mapmaker deconvolves the same
        operator. The reflection keeps the causal kernel from wrapping the scan's end onto its start.

        Args:
            signal: One detector's clean signal TOD (ntod,), before noise.
            fsamp: Sampling rate [Hz], setting the physical frequency grid.

        Returns:
            The filtered TOD, same length and dtype as ``signal``. ``H(0) = 1`` preserves the DC
            level (and hence the gain calibration) up to the small mirrored-boundary effect.
        """
        n = signal.shape[-1]
        ext = np.concatenate([signal.astype(np.float64), signal[::-1].astype(np.float64)])
        freqs = np.fft.rfftfreq(2 * n, d=1.0 / fsamp)
        filtered = np.fft.irfft(np.fft.rfft(ext) * self.response(freqs), n=2 * n)[:n]
        return filtered.astype(signal.dtype, copy=False)


class SinglePole(TransferFunction):
    """First-order low-pass with a single thermal time constant: ``H(f) = 1/(1 + 2*pi*i*f*tau)``.

    In the time domain this is convolution with a one-sided (causal) decaying exponential of
    e-folding time ``tau``, giving the characteristic scan-direction lag and smearing of a bolometer.
    """
    def __init__(self, tau_sec: float):
        self.tau = float(tau_sec)

    def response(self, freqs_hz):
        return 1.0 / (1.0 + 2j * np.pi * freqs_hz * self.tau)


class MultiPole(TransferFunction):
    """Sum of single-pole responses (HFI "LFER" model): ``H = (sum_k a_k/(1+2*pi*i*f*tau_k))/sum a_k``.

    The amplitudes ``a_k`` are normalised by their sum so ``H(0) = 1`` (unit DC gain). This captures
    the several-time-constant behaviour of the real HFI detectors while remaining a simple linear,
    calibration-preserving filter.
    """
    def __init__(self, amps: list[float], taus_sec: list[float]):
        self.amps = np.asarray(amps, dtype=np.float64)
        self.taus = np.asarray(taus_sec, dtype=np.float64)
        if self.amps.shape != self.taus.shape or self.amps.ndim != 1 or self.amps.size == 0:
            raise ValueError("MultiPole needs matching non-empty 1D amp/tau lists.")
        self.norm = self.amps.sum()
        if self.norm == 0.0:
            raise ValueError("MultiPole pole amplitudes sum to zero; cannot DC-normalise.")

    def response(self, freqs_hz):
        w = 2j * np.pi * freqs_hz
        poles = self.amps[:, None] / (1.0 + w[None, :] * self.taus[:, None])
        return poles.sum(axis=0) / self.norm


def _tau_sec(spec: Bunch | dict) -> float | None:
    """Read a time constant from a param block, accepting ``tau_sec`` or ``tau_ms`` (None if absent).

    Uses ``bget`` (subscript) rather than attribute access so it works on both a ``Bunch`` and a
    plain ``dict`` -- the ``poles`` list entries arrive as dicts (the loader does not bunch-ify list
    elements).
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
    """
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
