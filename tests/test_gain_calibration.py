"""Tests for the gain-calibration configuration and the unified calibrator builder.

Covers two new pieces introduced when the three gain-sampling procedures became nested
parameter-file blocks with a per-term ``calibrate_against`` target:

* ``_resolve_calib_target`` - resolves a gain term's calibrator, with a per-band override
  taking precedence over the general-block value, which falls back to the term default.
* ``TODView.get_calib_tod`` - builds the calibration residual for one gain term against a
  chosen calibrator signal, replacing the former per-term ``get_*_calib_tod`` methods.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from pixell.bunch import Bunch

from commander4.data_models.tod_view import TODView
from commander4.tod_processing import (_resolve_calib_target, _DEFAULT_CALIB_TARGETS,
                                       _VALID_CALIB_TARGETS, _solve_relative_gain_system,
                                       _resolve_gain_downsample_factor)


# --------------------------------------------------------------------------------------
# _resolve_calib_target
# --------------------------------------------------------------------------------------
def _make_params(general_blocks: dict, band_blocks: dict) -> Bunch:
    """Build a params Bunch with the given general and per-band gain blocks."""
    return Bunch(
        general=Bunch(**{name: Bunch(**vals) for name, vals in general_blocks.items()}),
        experiments=Bunch(EXP=Bunch(bands=Bunch(
            BAND=Bunch(**{name: Bunch(**vals) for name, vals in band_blocks.items()})))),
    )


def _exp_data(band="BAND"):
    return SimpleNamespace(experiment_name="EXP", band_name=band)


def test_defaults_when_calibrate_against_absent():
    # No calibrate_against anywhere -> each term falls back to its documented default.
    params = _make_params({"abs_gain": {}, "rel_gain": {}, "temporal_gain": {}}, {})
    for block, default in _DEFAULT_CALIB_TARGETS.items():
        assert _resolve_calib_target(params, _exp_data(), block) == default
    assert _DEFAULT_CALIB_TARGETS["abs_gain"] == "orbital_dipole"
    assert _DEFAULT_CALIB_TARGETS["rel_gain"] == "full_sky"


def test_general_block_value_used():
    params = _make_params({"abs_gain": {"calibrate_against": "full_sky"}}, {})
    assert _resolve_calib_target(params, _exp_data(), "abs_gain") == "full_sky"


def test_band_override_beats_general_and_default():
    # General says full_sky, band overrides to sky -> band wins.
    params = _make_params({"abs_gain": {"calibrate_against": "full_sky"}},
                          {"abs_gain": {"calibrate_against": "sky"}})
    assert _resolve_calib_target(params, _exp_data(), "abs_gain") == "sky"
    # A band with no override block falls back to the general value.
    params2 = _make_params({"abs_gain": {"calibrate_against": "full_sky"}}, {})
    assert _resolve_calib_target(params2, _exp_data(), "abs_gain") == "full_sky"


def test_invalid_target_raises():
    params = _make_params({"abs_gain": {"calibrate_against": "bogus"}}, {})
    with pytest.raises(ValueError):
        _resolve_calib_target(params, _exp_data(), "abs_gain")


def test_valid_targets_contents():
    assert set(_VALID_CALIB_TARGETS) == {"orbital_dipole", "full_sky", "sky"}


# --------------------------------------------------------------------------------------
# TODView.get_calib_tod
# --------------------------------------------------------------------------------------
class _StubView(TODView):
    """A TODView whose data accessors are stubbed so get_calib_tod can be tested in
    isolation: it records the ``subtract`` spec passed to ``get_tod`` and supplies fixed
    sky / orbital-dipole signals."""

    def __init__(self, s_sky, s_orb):
        super().__init__(None, None)
        self._s_sky = s_sky
        self._s_orb = s_orb
        self.captured_subtract = None

    def _materialize_downsampled(self, factor):
        n = self._s_sky.size
        return Bunch(tod=np.zeros(n), pix=np.zeros(n, dtype=int), psi=np.zeros(n))

    def get_mask(self, which, downsample_factor=1):
        return np.ones(self._s_sky.size, dtype=bool)

    def get_static_sky_tod(self, compsep_output=None, downsample_factor=None):
        return self._s_sky

    def get_orbital_dipole_tod(self, downsample_factor=None):
        return self._s_orb

    def get_tod(self, *, subtract=None, downsample_factor=1, compsep_output=None, **kw):
        self.captured_subtract = subtract
        return np.zeros(self._s_sky.size)


def _make_stub():
    return _StubView(np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 20.0, 30.0, 40.0]))


ALL = ("abs", "rel", "temp")


@pytest.mark.parametrize("target,calib,expected_subtract,scal", [
    # Absolute gain on the orbital dipole: sky removed entirely, dipole keeps the abs term.
    ("abs", "orbital_dipole",
     (("sky", ALL), ("orbital_dipole", ("rel", "temp"))), "orb"),
    # Absolute gain on the full sky (clean target-gain-preserving form): both signals keep abs.
    ("abs", "full_sky",
     (("sky", ("rel", "temp")), ("orbital_dipole", ("rel", "temp"))), "sky+orb"),
    # Relative gain on the full sky: both signals keep the rel term.
    ("rel", "full_sky",
     (("sky", ("abs", "temp")), ("orbital_dipole", ("abs", "temp"))), "sky+orb"),
    # Temporal gain on the full sky: both signals keep the temp term.
    ("temp", "full_sky",
     (("sky", ("abs", "rel")), ("orbital_dipole", ("abs", "rel"))), "sky+orb"),
    # Absolute gain on the static sky only: dipole removed entirely, sky keeps abs.
    ("abs", "sky",
     (("sky", ("rel", "temp")), ("orbital_dipole", ALL)), "sky"),
])
def test_get_calib_tod_builds_residual(target, calib, expected_subtract, scal):
    view = _make_stub()
    out = view.get_calib_tod(target, calib, downsample_factor=1, fill_masked=False)
    assert view.captured_subtract == expected_subtract
    expected_scal = {"orb": view._s_orb, "sky": view._s_sky,
                     "sky+orb": view._s_sky + view._s_orb}[scal]
    np.testing.assert_allclose(out.s_cal, expected_scal)


def test_get_calib_tod_rejects_bad_arguments():
    view = _make_stub()
    with pytest.raises(ValueError):
        view.get_calib_tod("bogus", "full_sky", downsample_factor=1)
    with pytest.raises(ValueError):
        view.get_calib_tod("abs", "bogus", downsample_factor=1)


# --------------------------------------------------------------------------------------
# _solve_relative_gain_system (reduced constrained solve + bad-detector exclusion)
# --------------------------------------------------------------------------------------
_ZERO_RNG = SimpleNamespace(standard_normal=lambda n: np.zeros(n))  # disables the fluctuation term


def test_relgain_recovers_constrained_mean():
    # With the fluctuation term zeroed, the solve must reproduce the analytic constrained mean
    # g_i = (r_i - 0.5*lambda)/d_i with lambda set so the active gains sum to zero.
    s = np.array([2.0, 4.0, 1.0, 3.0])
    r = np.array([1.0, -2.0, 0.5, 0.7])
    out = _solve_relative_gain_system(s, r, np.zeros(4), rng=_ZERO_RNG)
    lam = 2.0 * np.sum(r / s) / np.sum(1.0 / s)
    expected = (r - 0.5 * lam) / s
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
    assert abs(out.sum()) < 1e-5          # zero-sum constraint over all (active) detectors


def test_relgain_excludes_zero_weight_detectors():
    # Two detectors have zero calibration weight -> the *full* bordered system is singular, but the
    # helper solves the reduced active system and holds the excluded detectors at their prev value.
    s = np.array([0.0, 2.0, 0.0, 3.0])
    r = np.array([5.0, 1.0, 9.0, -1.0])
    prev = np.array([0.11, 0.22, 0.33, 0.44], dtype=np.float32)

    # Sanity: the un-reduced 4-detector bordered system really is singular.
    n = 4
    A_full = np.zeros((n + 1, n + 1))
    A_full[:n, :n] = np.diag(s); A_full[:n, n] = 0.5; A_full[n, :n] = 1.0
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(A_full, np.ones(n + 1))

    out = _solve_relative_gain_system(s, r, prev, rng=_ZERO_RNG)
    assert np.all(np.isfinite(out))
    assert out[0] == np.float32(0.11) and out[2] == np.float32(0.33)  # excluded held at prev
    assert abs(out[[1, 3]].sum()) < 1e-5                              # active subset sums to zero


def test_relgain_single_zero_detector_is_held():
    s = np.array([0.0, 2.0, 5.0])
    r = np.array([3.0, 1.0, -1.0])
    prev = np.array([0.9, 0.0, 0.0], dtype=np.float32)
    out = _solve_relative_gain_system(s, r, prev, rng=_ZERO_RNG)
    assert out[0] == np.float32(0.9)
    assert abs(out[[1, 2]].sum()) < 1e-5


def test_relgain_no_active_detectors_returns_prev_unchanged():
    prev = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = _solve_relative_gain_system(np.zeros(3), np.zeros(3), prev)
    np.testing.assert_array_equal(out, prev)


def test_relgain_deterministic_with_seeded_rng():
    s = np.array([2.0, 3.0, 4.0]); r = np.array([0.5, -0.5, 0.1]); prev = np.zeros(3)
    a = _solve_relative_gain_system(s, r, prev, rng=np.random.default_rng(5))
    b = _solve_relative_gain_system(s, r, prev, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------------------
# _resolve_gain_downsample_factor (general.gain_calib_downsample_time, in seconds)
# --------------------------------------------------------------------------------------
def test_gain_downsample_factor_from_time():
    make = lambda t: Bunch(general=Bunch(gain_calib_downsample_time=t))
    exp = SimpleNamespace(fsamp=200.0)
    assert _resolve_gain_downsample_factor(make(1.0), exp) == 200
    assert _resolve_gain_downsample_factor(make(0.25), exp) == 50
    assert _resolve_gain_downsample_factor(make(0.0), exp) == 1     # 0 disables downsampling.
    assert _resolve_gain_downsample_factor(make(0.001), exp) == 1   # Clamped to at least 1.
    assert _resolve_gain_downsample_factor(make(1.0), SimpleNamespace(fsamp=32.51)) == 33


# --------------------------------------------------------------------------------------
# Downsampling: model TODs are block-averaged like the data, not block-center sampled
# --------------------------------------------------------------------------------------
NTOD, FACTOR = 12, 4
# arange(0, ntod, factor) defines the block edges and the midpoint construction keeps the
# ntod//factor - 1 leading complete blocks (the trailing block is dropped).
NBLOCKS = NTOD // FACTOR - 1


def _make_real_view(monkeypatch):
    """A TODView over a minimal fake detector, exercising the real downsampling code paths."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")  # Required by get_s_orb_TOD.
    rng = np.random.default_rng(7)
    pix = rng.integers(0, 12, size=NTOD)        # Valid pixels for the nside=1 experiment below.
    psi = rng.uniform(0.0, np.pi, size=NTOD)
    det = SimpleNamespace(tod=rng.normal(size=NTOD), ntod=NTOD, fsamp=float(FACTOR),
                          det_idx_fullband=0, get_pix_psi=lambda: (pix, psi),
                          orb_dir_vec=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    experiment_data = SimpleNamespace(scans=[SimpleNamespace(detectors=[det])], nside=1, nu=30.0)
    no_jump = SimpleNamespace(is_empty=lambda: True)
    tod_samples = SimpleNamespace(jumps=SimpleNamespace(get=lambda iscan, idet: no_jump),
                                  abs_gain=2.0, rel_gain=np.array([0.5]),
                                  temporal_gain=np.array([[0.25]]),
                                  accept=np.ones((1, 1), dtype=bool))
    skymap = rng.normal(size=(3, 12))
    view = TODView(experiment_data, tod_samples, compsep_output=skymap).focus(0, det)
    s_full = skymap[0, pix] + np.cos(2*psi)*skymap[1, pix] + np.sin(2*psi)*skymap[2, pix]
    return view, det, s_full


def _block_mean(arr):
    return arr[:NBLOCKS*FACTOR].reshape(NBLOCKS, FACTOR).mean(axis=-1)


def test_static_sky_downsampling_is_block_average(monkeypatch):
    view, _, s_full = _make_real_view(monkeypatch)
    out = view.get_static_sky_tod(downsample_factor=FACTOR)
    np.testing.assert_allclose(out, _block_mean(s_full), rtol=2e-5, atol=1e-6)
    # Regression guard: must NOT be the model sampled at the block-center pixels.
    block_centers = np.array([2, 6])
    assert not np.allclose(out, s_full[block_centers])


def test_data_and_model_share_block_definition(monkeypatch):
    view, det, _ = _make_real_view(monkeypatch)
    np.testing.assert_allclose(view.get_tod(downsample_factor=FACTOR), _block_mean(det.tod))


def test_orbital_dipole_downsampling_is_block_average(monkeypatch):
    view, _, _ = _make_real_view(monkeypatch)
    orb_full = view.get_orbital_dipole_tod()
    np.testing.assert_allclose(view.get_orbital_dipole_tod(downsample_factor=FACTOR),
                               _block_mean(orb_full), rtol=2e-5, atol=1e-9)


def test_get_calib_tod_downsampled_end_to_end(monkeypatch):
    # Absolute gain against the static sky: residual = <d> - (g_rel+g_temp)<s_sky> - g_all*<s_orb>,
    # with every term block-averaged with the same kernel.
    view, det, s_full = _make_real_view(monkeypatch)
    orb_full = view.get_orbital_dipole_tod()
    out = view.get_calib_tod("abs", "sky", downsample_factor=FACTOR, fill_masked=False)
    np.testing.assert_allclose(out.s_cal, _block_mean(s_full), rtol=2e-5, atol=1e-6)
    expected = _block_mean(det.tod) - 0.75*_block_mean(s_full) - 2.75*_block_mean(orb_full)
    np.testing.assert_allclose(out.tod, expected, rtol=2e-5, atol=1e-6)
