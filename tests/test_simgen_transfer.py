"""Tests for the simgen bolometer transfer functions (sims/simgen/transfer.py).

Pure-numpy: no ducc0/pysm3/camb, so these run even where the heavier simgen sky/reader path can't.
The ``aux`` directory is added to sys.path so the ``simgen`` package imports as it does at runtime.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sims"))

from simgen.config import as_bunch_recursive
from simgen.transfer import SinglePole, MultiPole, make_detector_transfer


def _rms(x):
    return np.sqrt(np.mean(x**2))


def test_single_pole_response_matches_analytic():
    tau, f = 0.012, np.array([0.0, 1.0, 5.0, 20.0])
    expected = 1.0 / (1.0 + 2j * np.pi * f * tau)
    np.testing.assert_allclose(SinglePole(tau).response(f), expected)
    assert SinglePole(tau).response(np.array([0.0]))[0] == 1.0  # unit DC gain


def test_apply_uses_mirrored_reflection_convention():
    """``apply`` must equal reflect-extend ([x, x[::-1]]) -> filter on rfftfreq(2n) -> first n samples.

    This is C4's forward_rfft_mirrored/backward_rfft_mirrored convention; locking it here keeps the
    baked-in convolution identical to the operator the mapmaker's ``apply_T`` will deconvolve.
    """
    x = np.random.default_rng(2).normal(size=200)
    fsamp, tau = 25.0, 0.04
    freqs = np.fft.rfftfreq(2 * x.size, d=1.0 / fsamp)
    ref = np.fft.irfft(np.fft.rfft(np.concatenate([x, x[::-1]]))
                       / (1.0 + 2j * np.pi * freqs * tau), n=2 * x.size)[:x.size]
    np.testing.assert_allclose(SinglePole(tau).apply(x.astype(np.float32), fsamp), ref,
                               rtol=1e-5, atol=1e-6)


def test_single_pole_preserves_dc_level():
    """H(0)=1: a constant TOD is returned exactly unchanged (calibration/mean preserved)."""
    const = np.full(1024, 4.2, dtype=np.float32)
    np.testing.assert_allclose(SinglePole(0.02).apply(const, fsamp=50.0), const, rtol=1e-6, atol=1e-6)
    # For a structured signal the mean is preserved up to the small mirrored-boundary effect.
    signal = np.random.default_rng(0).normal(size=4096).astype(np.float32) + 5.0
    out = SinglePole(0.02).apply(signal, fsamp=50.0)
    assert out.dtype == signal.dtype
    assert np.isclose(out.mean(), signal.mean(), rtol=1e-3)


def test_single_pole_is_causal_lag():
    """A single pole is a one-sided decaying kernel: an impulse smears to *later* samples."""
    n = 2048
    impulse = np.zeros(n, dtype=np.float64)
    impulse[500] = 1.0
    out = SinglePole(0.03).apply(impulse, fsamp=100.0)
    idx = np.arange(n)
    center_of_mass = (idx * out).sum() / out.sum()
    assert center_of_mass > 500                        # response lags the impulse
    assert out[501:511].sum() > out[490:500].sum()     # mass sits on the trailing (later) side
    assert out.max() < 1.0                             # peak is smeared down
    assert np.isclose(out.sum(), 1.0, atol=1e-3)       # area ~conserved (H(0)=1), boundary-limited


def test_single_pole_lowpass_attenuation():
    """A sinusoid's RMS is scaled by ~|H(f0)|: low freq passes, high freq is strongly suppressed.

    The mirrored reflection leaks a little power across frequencies, so the ratio tracks the analytic
    |H(f0)| only to ~1%, not exactly (a plain circular convolution would be exact).
    """
    n, fsamp, tau = 4096, 100.0, 0.05
    t = np.arange(n) / fsamp
    for k in (2, 400):                                 # a low and a high grid frequency
        f0 = k * fsamp / n
        sig = np.cos(2 * np.pi * f0 * t)
        ratio = _rms(SinglePole(tau).apply(sig, fsamp)) / _rms(sig)
        assert np.isclose(ratio, 1.0 / np.sqrt(1.0 + (2 * np.pi * f0 * tau) ** 2), rtol=1e-2)


def test_multipole_dc_normalized():
    """MultiPole normalizes by sum(amps) so H(0)=1 regardless of the raw amplitudes."""
    tf = MultiPole(amps=[0.39, 0.53, 0.08], taus_sec=[0.0105, 0.050, 0.001])
    assert np.isclose(tf.response(np.array([0.0]))[0].real, 1.0)
    const = np.full(1000, 3.0, dtype=np.float32)       # H(0)=1 -> constant passes through exactly
    np.testing.assert_allclose(tf.apply(const, fsamp=30.0), const, rtol=1e-6, atol=1e-6)


def test_multipole_single_entry_equals_single_pole():
    f = np.array([0.0, 3.0, 17.0])
    np.testing.assert_allclose(MultiPole([1.0], [0.02]).response(f), SinglePole(0.02).response(f))
    np.testing.assert_allclose(MultiPole([7.0], [0.02]).response(f), SinglePole(0.02).response(f))


def test_multipole_rejects_bad_input():
    with pytest.raises(ValueError):
        MultiPole([1.0, 2.0], [0.01])          # mismatched lengths
    with pytest.raises(ValueError):
        MultiPole([1.0, -1.0], [0.01, 0.02])   # amplitudes sum to zero -> cannot DC-normalize


def _det(**kw):
    return as_bunch_recursive(dict(kw))


def test_make_detector_transfer_resolution():
    # Per-detector single-pole shorthand (tau_ms / tau_sec equivalent).
    tf = make_detector_transfer(_det(tau_ms=10.0), global_default=None)
    assert isinstance(tf, SinglePole) and np.isclose(tf.tau, 0.010)
    assert np.isclose(make_detector_transfer(_det(tau_sec=0.01), None).tau, 0.010)

    # Per-detector transfer_function block with a poles list -> MultiPole.
    node = _det(transfer_function={"poles": [{"amp": 0.6, "tau_ms": 10.0},
                                             {"amp": 0.4, "tau_ms": 50.0}]})
    tf = make_detector_transfer(node, None)
    assert isinstance(tf, MultiPole) and np.allclose(tf.taus, [0.010, 0.050])

    # Explicit opt-out and "nothing configured" both give identity (None).
    assert make_detector_transfer(_det(transfer_function={"enabled": False}), None) is None
    assert make_detector_transfer(_det(psi_offset_deg=0.0), None) is None
    assert make_detector_transfer(_det(tau_ms=0.0), None) is None   # zero tau == identity

    # Run-wide default is used only when the detector says nothing; a detector tau overrides it.
    default = _det(enabled=True, tau_ms=8.0)
    assert np.isclose(make_detector_transfer(_det(psi_offset_deg=0.0), default).tau, 0.008)
    assert np.isclose(make_detector_transfer(_det(tau_ms=20.0), default).tau, 0.020)
    # A detector opt-out beats the enabled global default.
    assert make_detector_transfer(_det(transfer_function={"enabled": False}), default) is None


def test_pipeline_applies_transfer_to_clean_signal_before_noise():
    """End-to-end wiring: a scan's per-detector TOD is T(gain*signal), with noise added after T.

    Two shared-pointing detectors (identical psi/gain, one with a single-pole TF, one without) are
    simulated with sigma0=0, so the plain detector's TOD is the clean signal and the filtered
    detector's TOD must equal that signal passed through ``SinglePole.apply``. Uses the light raster
    strategy + a synthetic sky map (only healpy needed).
    """
    pytest.importorskip("healpy")
    from simgen.instrument import Band, Detector
    from simgen.pointing import make_pointing
    from simgen.noise import WhiteNoise
    from simgen import pipeline

    tau, fsamp, nside = 0.02, 19.0, 16
    dets = [Detector(name="d_tf", idx=0, sigma0=0.0, gain=1.0, transfer=SinglePole(tau)),
            Detector(name="d_id", idx=1, sigma0=0.0, gain=1.0, transfer=None)]
    band = Band(name="B", exp_name="E", freq=100.0, fwhm_arcmin=0.0, fsamp=fsamp, eval_nside=nside,
                data_nside=nside, units="uK_RJ", polarization="I", detectors=dets, sigma0=0.0,
                noise=None, crosstalk=None)
    strat = make_pointing(as_bunch_recursive({"strategy": "raster", "n_rows": 8,
                                              "samples_per_row": 32}), fsamp)
    ntod = strat.samples_per_scan(0.0, fsamp)
    skymap = np.random.default_rng(1).normal(size=(1, 12 * nside**2)).astype(np.float32)
    _, _, det_tod, _, _ = pipeline._simulate_scan(band, strat, 0, skymap, ntod, 0, 0,
                                                  WhiteNoise(), [], seed=7, include_orbdip=False)

    clean = det_tod["d_id"]                             # no TF, sigma0=0 -> the clean signal
    np.testing.assert_allclose(det_tod["d_tf"], SinglePole(tau).apply(clean, fsamp),
                               rtol=1e-5, atol=1e-6)
    assert not np.allclose(det_tod["d_tf"], clean)      # TF actually altered the TOD
