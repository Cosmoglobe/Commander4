"""Tests for the low-resolution (log-binned) TOD power spectra written to the chain."""

import numpy as np
import pytest

from commander4.tod_processing import _binned_tod_power_spectrum, _record_tod_diagnostics
from commander4.data_models.TOD_samples import TODSamples


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
# Optional DEBUG: full n_corr TOD ragged packing
# --------------------------------------------------------------------------------------
def _bare_tod_samples(nscans, ndet, ncorr_tods):
    ts = TODSamples.__new__(TODSamples)          # bypass __init__ (no MPI / data needed)
    ts.nscans, ts.ndet, ts.ncorr_tods = nscans, ndet, ncorr_tods
    return ts


def test_pack_ncorr_tods_ragged():
    a = np.arange(3, dtype=np.float32)
    b = np.arange(5, dtype=np.float32) + 10.0
    # scan0: det0=a (len 3), det1 missing; scan1: det0 missing, det1=b (len 5).
    ts = _bare_tod_samples(2, 2, [[a, None], [None, b]])
    lengths, flat = ts._pack_ncorr_tods()
    assert lengths.tolist() == [[3, 0], [0, 5]]
    # Flat concatenation is scan-major, detector-minor (a before b).
    np.testing.assert_array_equal(flat, np.concatenate([a, b]))
    assert flat.dtype == np.float32


def test_pack_ncorr_tods_empty():
    ts = _bare_tod_samples(2, 2, [[None, None], [None, None]])
    lengths, flat = ts._pack_ncorr_tods()
    assert lengths.sum() == 0
    assert flat.size == 0 and flat.dtype == np.float32


# --------------------------------------------------------------------------------------
# _record_tod_diagnostics: which TOD view feeds which recorded spectrum
# --------------------------------------------------------------------------------------
class _DiagStubView:
    """Stand-in for TODView exposing only what _record_tod_diagnostics reads.

    ``get_tod()`` (no subtract) returns the jump-corrected raw TOD; ``get_tod(subtract=...)``
    returns the sky+orbital-dipole-subtracted base. Both are fresh copies so the diagnostics
    helper can subtract n_corr in place, mirroring the real TODView contract.
    """
    def __init__(self, raw, sky_orb_subtracted, fsamp=10.0):
        self._raw = raw
        self._sky_orb_subtracted = sky_orb_subtracted
        self.fsamp = fsamp

    @property
    def tod(self):
        return self._raw

    def get_tod(self, *, subtract=None, **kwargs):
        base = self._raw if subtract is None else self._sky_orb_subtracted
        return np.array(base, copy=True)


def _diag_tod_samples():
    ts = TODSamples.__new__(TODSamples)          # bypass __init__ (no MPI / data needed)
    shape = (1, 1, TODSamples.TOD_PS_NBIN)
    for name in ("tod_ps_freqs", "tod_ps_raw", "tod_ps_ncorr", "tod_ps_ncorrsub",
                 "tod_ps_residual"):
        setattr(ts, name, np.full(shape, np.nan, dtype=np.float32))
    ts.ncorr_tods = None
    return ts


def _ps(tod, fsamp=10.0):
    return _binned_tod_power_spectrum(tod, fsamp, TODSamples.TOD_PS_NBIN)[1]


def test_record_diagnostics_routes_each_tod_to_its_spectrum():
    rng = np.random.default_rng(7)
    raw = rng.standard_normal(2000)
    sky_orb_sub = rng.standard_normal(2000)        # stand-in for raw - gain*(sky+orb)
    n_corr = rng.standard_normal(2000)
    ts = _diag_tod_samples()
    _record_tod_diagnostics(ts, 0, 0, _DiagStubView(raw, sky_orb_sub), n_corr)

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
    ts = _diag_tod_samples()
    _record_tod_diagnostics(ts, 0, 0, _DiagStubView(raw, sky_orb_sub), None)

    # With no n_corr drawn: ncorrsub == raw, residual is just the sky-subtracted TOD, ncorr stays NaN.
    np.testing.assert_allclose(ts.tod_ps_ncorrsub[0, 0], _ps(raw), rtol=1e-5)
    np.testing.assert_allclose(ts.tod_ps_residual[0, 0], _ps(sky_orb_sub), rtol=1e-5)
    assert np.isnan(ts.tod_ps_ncorr[0, 0]).all()
