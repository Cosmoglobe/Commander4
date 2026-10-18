"""Tests for the low-resolution (log-binned) TOD power spectra written to the chain."""

import numpy as np
import pytest

from commander4.tod.scan_diagnostics import _binned_tod_power_spectrum, TODDiagnostics
from tod_step_fakes import make_step


def test_shape_and_nan_padding():
    # expbin returns fewer than nbin bins for a long TOD, so both outputs are NaN-padded to nbin,
    # and the finite entries form a contiguous leading block identical between freqs and power.
    tod = np.random.default_rng(0).standard_normal(100_000)
    nbin = 100
    f, p = _binned_tod_power_spectrum(tod, fsamp=10.0, nbin=nbin)
    assert f.shape == (nbin,) and p.shape == (nbin,)
    finite = np.isfinite(f)
    assert 0 < finite.sum() < nbin
    np.testing.assert_array_equal(finite, np.isfinite(p))
    nb = int(finite.sum())
    assert finite[:nb].all() and not finite[nb:].any()


def test_freqs_monotonic_and_in_range():
    fsamp = 8.0
    tod = np.random.default_rng(1).standard_normal(50_000)
    f, _ = _binned_tod_power_spectrum(tod, fsamp, 100)
    ff = f[np.isfinite(f)]
    assert np.all(np.diff(ff) > 0)                 # strictly increasing bin centers
    assert ff[0] >= 0.0 and ff[-1] <= fsamp / 2 + 1e-6


def test_sinusoid_power_localized():
    fsamp, ntod, f0 = 100.0, 100_000, 5.0
    t = np.arange(ntod) / fsamp
    f, p = _binned_tod_power_spectrum(np.sin(2 * np.pi * f0 * t), fsamp, 100)
    peak_bin = int(np.nanargmax(p))
    assert abs(f[peak_bin] - f0) < 0.5             # peak power bin sits at the injected frequency


def test_white_noise_roughly_flat():
    # Periodogram |rfft|^2 / N of unit-variance white noise has expectation ~1 per mode; the
    # bin-averaged spectrum (skipping the few noisy low-mode bins) should sit near 1.
    tod = np.random.default_rng(3).standard_normal(200_000)
    _, p = _binned_tod_power_spectrum(tod, fsamp=10.0, nbin=100)
    mid = p[np.isfinite(p)][5:]
    assert 0.5 < np.median(mid) < 2.0


def test_short_tod_uses_fewer_bins_without_error():
    # A short TOD yields well under nbin bins; everything beyond stays NaN.
    f, p = _binned_tod_power_spectrum(np.arange(64.0), fsamp=1.0, nbin=100)
    assert np.isfinite(f).sum() < 40
    assert np.isfinite(p).sum() == np.isfinite(f).sum()


# --------------------------------------------------------------------------------------
# TODDiagnostics.record_scan: which TOD view feeds which recorded spectrum
# --------------------------------------------------------------------------------------
class _DiagStubView:
    """Stand-in for TODView exposing only what `record_scan` reads, focused on detector-scan (0, 0).

    ``get_tod(keep=("sky", "orbital_dipole"))`` returns the raw TOD; ``get_tod()`` returns the
    sky+orbital-dipole-subtracted base. Both are fresh copies so the diagnostics helper can subtract
    n_corr in place, mirroring the real TODView contract.
    """
    iscan, idet = 0, 0

    def __init__(self, raw, sky_orb_subtracted, fsamp=10.0):
        self._raw = raw
        self._sky_orb_subtracted = sky_orb_subtracted
        self.fsamp = fsamp

    @property
    def raw_tod(self):
        return self._raw

    def get_tod(self, *, keep=()):
        base = self._raw if "sky" in keep else self._sky_orb_subtracted
        return np.array(base, copy=True)


def _ps(tod, fsamp=10.0):
    return _binned_tod_power_spectrum(tod, fsamp, TODDiagnostics.TOD_PS_NBIN)[1]


def test_record_diagnostics_routes_each_tod_to_its_spectrum():
    rng = np.random.default_rng(7)
    raw = rng.standard_normal(2000)
    sky_orb_sub = rng.standard_normal(2000)        # stand-in for raw - gain*(sky+orb)
    n_corr = rng.standard_normal(2000)
    ts = make_step(TODDiagnostics, 1, 1)
    ts.record_scan(_DiagStubView(raw, sky_orb_sub), n_corr)

    # Recorded spectra are stored as float32, so compare at single-precision tolerance.
    np.testing.assert_allclose(ts.tod_ps_raw[0, 0], _ps(raw), rtol=1e-5)
    np.testing.assert_allclose(ts.tod_ps_ncorr[0, 0], _ps(n_corr), rtol=1e-5)
    # n_corr removed from raw -> sky + white noise retained.
    np.testing.assert_allclose(ts.tod_ps_ncorrsub[0, 0], _ps(raw - n_corr), rtol=1e-5)
    # sky, orbital dipole, and n_corr all removed -> noise residual.
    np.testing.assert_allclose(ts.tod_ps_residual[0, 0], _ps(sky_orb_sub - n_corr), rtol=1e-5)


def test_record_diagnostics_without_ncorr():
    rng = np.random.default_rng(8)
    raw = rng.standard_normal(2000)
    sky_orb_sub = rng.standard_normal(2000)
    ts = make_step(TODDiagnostics, 1, 1)
    ts.record_scan(_DiagStubView(raw, sky_orb_sub), None)

    # With no n_corr drawn: ncorrsub == raw, residual is just the sky-subtracted TOD, ncorr stays NaN.
    np.testing.assert_allclose(ts.tod_ps_ncorrsub[0, 0], _ps(raw), rtol=1e-5)
    np.testing.assert_allclose(ts.tod_ps_residual[0, 0], _ps(sky_orb_sub), rtol=1e-5)
    assert np.isnan(ts.tod_ps_ncorr[0, 0]).all()


def test_record_diagnostics_without_power_spectra_still_gives_the_residual():
    """With the spectra switched off, the data-selection veto and the residual map still need the
    residual TOD."""
    rng = np.random.default_rng(9)
    raw = rng.standard_normal(2000)
    sky_orb_sub = rng.standard_normal(2000)
    n_corr = rng.standard_normal(2000)
    ts = make_step(TODDiagnostics, 1, 1, tod_power_spectra=False)
    assert ts.tod_ps_freqs is None and ts.tod_ps_raw is None

    residual = ts.record_scan(_DiagStubView(raw, sky_orb_sub), n_corr)

    np.testing.assert_allclose(residual, sky_orb_sub - n_corr)
    assert "tod_ps_raw" not in ts.chain_arrays().per_scan


def test_start_iteration_forgets_the_full_tods_and_runtimes_of_the_last_iteration() -> None:
    """A detector-scan rejected since, or without an n_corr draw, writes nothing stale."""
    samples = make_step(TODDiagnostics, 1, 1, residual_tods=True, ncorr_tods=True)
    samples.record_scan(_DiagStubView(np.ones(64), np.ones(64)), np.ones(64))
    samples.scan_runtime[:] = 2.0
    samples.start_iteration()
    assert samples.residual_tods == [[None]] and samples.ncorr_tods == [[None]]
    assert samples.scan_runtime[0] == 0.0


@pytest.mark.parametrize("with_ncorr", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_saved_residual_is_full_observed_minus_model(with_ncorr: bool, dtype: type) -> None:
    """Saving residuals works without n_corr output and survives in-place CG gap filling."""
    raw = np.arange(128, dtype=dtype)
    sky_orb_sub = raw - 20.0
    n_corr = np.full(128, 2.0, dtype=dtype) if with_ncorr else None
    sidelobe = np.full(128, 3.0, dtype=dtype)
    samples = make_step(TODDiagnostics, 1, 1, residual_tods=True)
    residual = samples.record_scan(_DiagStubView(raw, sky_orb_sub), n_corr, sidelobe_tod=sidelobe)

    expected = sky_orb_sub - sidelobe
    if n_corr is not None:
        expected -= n_corr
    saved = samples.residual_tods[0][0]
    assert samples.ncorr_tods is None
    assert saved.dtype == np.float32
    np.testing.assert_array_equal(saved, expected)
    np.testing.assert_array_equal(residual, expected)
    residual[:] = 0.0
    np.testing.assert_array_equal(saved, expected)
