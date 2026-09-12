"""Tests for the gain-calibration configuration and the unified calibrator builder.

Covers two new pieces introduced when the three gain-sampling procedures became nested
parameter-file blocks with a per-term ``calibrate_against`` target:

* the calibrator each gain term uses, with a per-band override (via `resolve_param`)
  taking precedence over the general-block value, which falls back to the term default.
* ``TODView.get_calib_tod`` - builds the calibration residual for one gain term against a
  chosen calibrator signal, replacing the former per-term ``get_*_calib_tod`` methods.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pixell.bunch import Bunch

from commander4.tod.view import TODView
from commander4.tod.gain import GainConfig, _solve_relative_gain_system


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("white_noise", [False, True])
def test_absolute_gain_reuses_filtered_calibrator(
    monkeypatch: pytest.MonkeyPatch, dtype: type, white_noise: bool,
) -> None:
    """One filter per scan reproduces the two-filter gain mean and fluctuation amplitude."""
    from mpi4py import MPI
    from commander4.data_models.detector_group_tod import DetectorGroupTOD
    from commander4.tod.noise.psd import NoisePSDOof
    import commander4.tod.gain as gain

    model = NoisePSDOof()
    model.is_white = white_noise
    experiment = DetectorGroupTOD([], "EXP", "BAND", 1, 100.0, 0.0, 180.0, 1, "I", model)
    config = GainConfig(sampling_rate=180.0, downsample_time=1.0)
    rng = np.random.default_rng(13)
    views = []
    numerator = denominator = 0.0
    for length in (997, 1024):
        calibrator = rng.normal(size=length).astype(dtype)
        residual = (0.7 * calibrator + rng.normal(size=length) + 3.5).astype(dtype)
        noise_params = np.array([2.0, 0.1, -1.5])
        calib = SimpleNamespace(s_cal=calibrator, tod=residual)
        views.append(SimpleNamespace(fsamp=180.0, downsample_factor=180,
                                     noise_params=noise_params,
                                     get_calib_tod=Mock(return_value=calib)))
        # Reference expression before the optimization, using the real mirrored/white operator.
        filtered_data = experiment.apply_N_inv(residual, noise_params, samprate=1.0)
        filtered_calibrator = experiment.apply_N_inv(calibrator, noise_params, samprate=1.0)
        numerator += np.dot(calibrator, filtered_data)
        denominator += np.dot(calibrator, filtered_calibrator)

    scan_view = SimpleNamespace(iter_focused=Mock(return_value=iter(views)))
    monkeypatch.setattr(gain, "TODView", Mock(return_value=scan_view))
    filter_spy = Mock(wraps=experiment.apply_N_inv)
    monkeypatch.setattr(experiment, "apply_N_inv", filter_spy)
    monkeypatch.setattr(gain.np.random, "randn", lambda: 0.37)
    samples = SimpleNamespace(abs_gain=1.0, chain=1)
    result = gain.sample_absolute_gain(MPI.COMM_SELF, experiment, samples, None, config, 1)

    expected = numerator / denominator + 0.37 / np.sqrt(denominator)
    tolerance = 2e-6 if dtype == np.float32 else 1e-12
    assert result.abs_gain == pytest.approx(expected, rel=tolerance)
    assert filter_spy.call_count == len(views)
    for call, view in zip(filter_spy.call_args_list, views):
        assert call.args[0] is view.get_calib_tod.return_value.s_cal
        assert call.kwargs["samprate"] == 1.0


# --------------------------------------------------------------------------------------
# The calibrator a gain term ends up using
# --------------------------------------------------------------------------------------
def _make_params(global_blocks: dict, band_blocks: dict) -> Bunch:
    """Build a params Bunch with the given tod_processing and per-band gain blocks."""
    return Bunch(
        tod_processing=Bunch(**{name: Bunch(**vals) if isinstance(vals, dict) else vals
                                for name, vals in global_blocks.items()}),
        experiments=Bunch(EXP=Bunch(bands=Bunch(
            BAND=Bunch(**{name: Bunch(**vals) for name, vals in band_blocks.items()})))),
        compsep=Bunch(common_res_fwhm=0.0),
    )


def _exp_data(band="BAND"):
    return SimpleNamespace(experiment_name="EXP", band_name=band, fsamp=200.0, nu=100.0)


def _inputs(band_blocks, passed="sky", gain_block="abs_gain", downsample_time=1.0,
            gap_fill="wn", fsamp=200.0, nu=100.0, is_master=True):
    """Resolve one step's calibrator, gap filling, and downsampling."""
    global_blocks = {
        gain_block: {
            "enabled": True,
            "calibrate_against": passed,
            "gap_fill_method": gap_fill,
            "downsample_time": downsample_time,
        },
    }
    params = _make_params(global_blocks, band_blocks)
    experiment = SimpleNamespace(experiment_name="EXP", band_name="BAND", fsamp=fsamp, nu=nu)
    default = "orbital_dipole" if gain_block == "abs_gain" else "sky"
    config = GainConfig.from_params(
        params, experiment, gain_block, default, iteration=1, is_master=is_master,
    )
    return config.calibrate_against, config.downsample_factor, config.gap_fill_method


def test_the_passed_in_calibrator_is_used_when_the_band_does_not_override():
    assert _inputs({})[0] == "sky"
    # A per-band block for a *different* gain term must not be picked up.
    assert _inputs({"rel_gain": {"calibrate_against": "orbital_dipole"}})[0] == "sky"


def test_band_override_beats_the_passed_in_calibrator():
    assert _inputs({"abs_gain": {"calibrate_against": "sky_no_dipole"}})[0] == "sky_no_dipole"
    assert _inputs({"rel_gain": {"calibrate_against": "orbital_dipole"}},
                   gain_block="rel_gain")[0] == "orbital_dipole"


def test_band_override_can_change_every_gain_option():
    global_blocks = {
        "abs_gain": {
            "enabled": True,
            "from_iter": 2,
            "calibrate_against": "sky",
            "gap_fill_method": "wn",
            "downsample_time": 1.0,
        },
    }
    band_blocks = {
        "abs_gain": {
            "enabled": False,
            "from_iter": 4,
            "gap_fill_method": "fallback",
            "downsample_time": 0.25,
        },
    }
    config = GainConfig.from_params(
        _make_params(global_blocks, band_blocks), _exp_data(), "abs_gain", "orbital_dipole",
        iteration=1, is_master=False,
    )

    assert not config.enabled
    assert config.from_iter == 4
    assert config.calibrate_against == "sky"
    assert config.gap_fill_method == "fallback"
    assert config.downsample_factor == 50


def test_invalid_target_raises():
    with pytest.raises(ValueError):
        _inputs({}, passed="bogus")
    # ... including when it arrives via the per-band override.
    with pytest.raises(ValueError):
        _inputs({"abs_gain": {"calibrate_against": "bogus"}})


def test_an_invalid_gap_fill_method_raises():
    assert _inputs({}, gap_fill="full_cg")[2] == "full_cg"
    with pytest.raises(ValueError):
        _inputs({}, gap_fill="bogus")


def test_orbital_dipole_calibration_warns_at_high_frequency(caplog):
    """The dipole is a blackbody signal: in the sub-mm it is faint next to dust, so calibrating a
    545 GHz channel on it is a configuration mistake worth flagging."""
    with caplog.at_level("WARNING", logger="commander4.tod.processing"):
        assert _inputs({}, passed="orbital_dipole", nu=545.0)[0] == "orbital_dipole"
    assert "orbital dipole" in caplog.text and "545" in caplog.text
    # Not at the frequencies where the dipole is the standard absolute calibrator ...
    caplog.clear()
    with caplog.at_level("WARNING", logger="commander4.tod.processing"):
        _inputs({}, passed="orbital_dipole", nu=100.0)
        _inputs({}, passed="orbital_dipole", nu=353.0)
        _inputs({}, passed="sky", nu=857.0)           # ... nor for the other calibrators,
        _inputs({}, passed="orbital_dipole", nu=545.0, is_master=False)   # ... nor off-master.
    assert caplog.text == ""


def test_the_downsample_factor_is_seconds_times_the_sampling_rate():
    assert _inputs({}, downsample_time=1.0, fsamp=200.0)[1] == 200
    assert _inputs({}, downsample_time=0.25, fsamp=200.0)[1] == 50
    assert _inputs({}, downsample_time=0.0)[1] == 1      # 0 disables downsampling.
    assert _inputs({}, downsample_time=0.001)[1] == 1    # clamped to at least 1.
    assert _inputs({}, downsample_time=1.0, fsamp=32.51)[1] == 33


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

    def get_mask(self, good_data_mask=True, proc_mask=True, proc_mask_type=""):
        return np.ones(self._s_sky.size, dtype=bool)

    def get_static_sky_tod(self, compsep_output=None):
        return self._s_sky

    def get_orbital_dipole_tod(self):
        return self._s_orb

    def get_tod(self, *, subtract=None, compsep_output=None, **kw):
        self.captured_subtract = subtract
        return np.zeros(self._s_sky.size)


def _make_stub():
    return _StubView(np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 20.0, 30.0, 40.0]))


ALL = ("abs", "rel", "temp")


@pytest.mark.parametrize("target,calib,expected_subtract,scal", [
    # Absolute gain on the orbital dipole: sky removed entirely, dipole keeps the abs term.
    ("abs", "orbital_dipole",
     (("sky", ALL), ("orbital_dipole", ("rel", "temp"))), "orb"),
    # Absolute gain on the whole sky (clean target-gain-preserving form): both signals keep abs.
    ("abs", "sky",
     (("sky", ("rel", "temp")), ("orbital_dipole", ("rel", "temp"))), "sky+orb"),
    # Relative gain on the whole sky: both signals keep the rel term.
    ("rel", "sky",
     (("sky", ("abs", "temp")), ("orbital_dipole", ("abs", "temp"))), "sky+orb"),
    # Temporal gain on the whole sky: both signals keep the temp term.
    ("temp", "sky",
     (("sky", ("abs", "rel")), ("orbital_dipole", ("abs", "rel"))), "sky+orb"),
    # Absolute gain on the static sky only: dipole removed entirely, sky keeps abs.
    ("abs", "sky_no_dipole",
     (("sky", ("rel", "temp")), ("orbital_dipole", ALL)), "sky"),
])
def test_get_calib_tod_builds_residual(target, calib, expected_subtract, scal):
    view = _make_stub()
    out = view.get_calib_tod(target, calib, fill_masked=False)
    assert view.captured_subtract == expected_subtract
    expected_scal = {"orb": view._s_orb, "sky": view._s_sky,
                     "sky+orb": view._s_sky + view._s_orb}[scal]
    np.testing.assert_allclose(out.s_cal, expected_scal)


def test_get_calib_tod_rejects_bad_arguments():
    view = _make_stub()
    with pytest.raises(ValueError):
        view.get_calib_tod("bogus", "sky")
    with pytest.raises(ValueError):
        view.get_calib_tod("abs", "bogus")


# --------------------------------------------------------------------------------------
# Masked-sample gap fill in the gain residual (wn | fallback | full_cg)
# --------------------------------------------------------------------------------------
class _FillStubView(TODView):
    """Stub exposing just enough state for ``_fill_masked_calibration_samples`` to run all three
    gap-fill methods over a residual with a real masked region."""

    def __init__(self, n, mask, noise_model, noise_params, fsamp=1.0, gain=2.0):
        super().__init__(None, None)
        self.experiment_data = SimpleNamespace(noise_model=noise_model)
        self._n = n
        self._mask = mask
        self._np = np.asarray(noise_params, float)
        self._fsamp = fsamp
        self._gain = gain
        self._data = np.random.default_rng(0).normal(0.0, noise_params[0], n)

    @property
    def fsamp(self): return self._fsamp
    @property
    def noise_params(self): return self._np
    @property
    def sigma0(self): return float(self._np[0])
    @property
    def corrected_tod(self): return self._data
    def get_gain(self, gain_terms=None): return self._gain
    def get_mask(self, good_data_mask=True, proc_mask=True, proc_mask_type=""): return self._mask
    def get_static_sky_tod(self, compsep_output=None): return np.zeros(self._n)
    def get_orbital_dipole_tod(self): return np.zeros(self._n)


def test_gain_gap_fill_methods_replace_masked_samples():
    """All three methods fill the masked residual with target gain x s_cal plus a noise draw."""
    from commander4.tod.noise.psd import NoisePSDOof
    n = 4096
    mask = np.ones(n, dtype=bool)
    mask[1000:1400] = False
    params = np.array([1.0, 0.3, -1.8])
    s_cal = np.ones(n)
    resid = np.zeros(n)  # valid-sample placeholder; masked entries get overwritten
    gap = ~mask
    for method in ("wn", "fallback", "full_cg"):
        view = _FillStubView(n, mask, NoisePSDOof(), params, fsamp=1.0, gain=2.0)
        np.random.seed(0)
        filled = view._fill_masked_calibration_samples(resid, mask, s_cal, ("abs",), None,
                                                       method=method)
        assert np.array_equal(filled[mask], resid[mask])         # valid samples untouched
        assert np.all(np.isfinite(filled))
        assert not np.allclose(filled[gap], 0.0)                 # gaps were filled
        # Masked fill is centered on target gain x calibrator (= 2.0); the noise is zero-mean.
        assert filled[gap].mean() == pytest.approx(2.0, abs=0.4)
        if method != "wn":
            assert method in view._gap_noise   # the shared realization was cached


def test_gain_gap_fill_realization_is_shared_across_terms():
    """The cached 1/f gap draw is reused for every gain term (one realization per scan)."""
    from commander4.tod.noise.psd import NoisePSDOof
    n = 2048
    mask = np.ones(n, dtype=bool)
    mask[800:1000] = False
    params = np.array([1.0, 0.3, -1.8])
    s_cal = np.ones(n)
    resid = np.zeros(n)
    view = _FillStubView(n, mask, NoisePSDOof(), params, fsamp=1.0, gain=1.0)
    np.random.seed(0)
    a = view._fill_masked_calibration_samples(resid, mask, s_cal, ("abs",), None,
                                              method="fallback")
    b = view._fill_masked_calibration_samples(resid, mask, s_cal, ("rel",), None,
                                              method="fallback")
    # Same gain and cached draw -> identical masked fill across the two terms.
    np.testing.assert_array_equal(a[~mask], b[~mask])
    assert len(view._gap_noise) == 1


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
# GainConfig converts each step's downsample_time into a sampling-rate-specific factor.


# --------------------------------------------------------------------------------------
# Downsampling: model TODs are block-averaged like the data, not block-center sampled
# --------------------------------------------------------------------------------------
NTOD, FACTOR = 12, 4
# Downsampling cuts the full-rate stream into contiguous blocks of `factor` samples and keeps every
# complete block (ntod // factor of them); only a trailing *partial* block is dropped. Here ntod is
# an exact multiple of factor, so all ntod // factor blocks are kept.
NBLOCKS = NTOD // FACTOR


def _make_real_view(monkeypatch, good=None):
    """Factory for a TODView over a minimal fake detector, exercising the real downsampling paths.

    Returns ``make_view(downsample_factor)`` so a test can build views at full *and* downsampled
    resolution over the same detector (the factor is now fixed per view), plus the detector and the
    full-rate static-sky TOD for the expected-value comparisons. ``good`` is an optional full-rate
    flag cut; without one every sample passes.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "1")  # Required by get_s_orb_tod.
    rng = np.random.default_rng(7)
    pix = rng.integers(0, 12, size=NTOD)        # Valid pixels for the nside=1 experiment below.
    psi = rng.uniform(0.0, np.pi, size=NTOD)
    det = SimpleNamespace(tod=rng.normal(size=NTOD), ntod=NTOD, fsamp=float(FACTOR), nside=1,
                          det_idx_fullband=0, get_pix_psi=lambda: (pix, psi),
                          response_I_P=(1.0, 1.0),
                          orbital_velocity_m_per_s=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    experiment_data = SimpleNamespace(scans=[SimpleNamespace(detectors=[det])], nside=1, nu=30.0)
    no_jump = SimpleNamespace(is_empty=lambda: True)
    tod_samples = SimpleNamespace(jumps=SimpleNamespace(get=lambda iscan, idet: no_jump),
                                  abs_gain=2.0, rel_gain=np.array([0.5]),
                                  temporal_gain=np.array([[0.25]]),
                                  accept=np.ones((1, 1), dtype=bool))
    if good is not None:
        det._good_data_mask = good          # presence of this attribute enables the flag cut
        det.good_data_mask = good
    skymap = rng.normal(size=(3, 12))

    def make_view(downsample_factor=1, mask_threshold=0.5):
        return TODView(experiment_data, tod_samples, compsep_output=skymap,
                       downsample_factor=downsample_factor,
                       mask_threshold=mask_threshold).focus(0, det)

    s_full = skymap[0, pix] + np.cos(2*psi)*skymap[1, pix] + np.sin(2*psi)*skymap[2, pix]
    return make_view, det, s_full


def _block_mean(arr):
    return arr[:NBLOCKS*FACTOR].reshape(NBLOCKS, FACTOR).mean(axis=-1)


def test_static_sky_downsampling_is_block_average(monkeypatch):
    make_view, _, s_full = _make_real_view(monkeypatch)
    out = make_view(FACTOR).get_static_sky_tod()
    np.testing.assert_allclose(out, _block_mean(s_full), rtol=2e-5, atol=1e-6)
    # Regression guard: must NOT be the model sampled at the block-center pixels.
    block_centers = np.array([2, 6, 10])
    assert not np.allclose(out, s_full[block_centers])


def test_get_calib_tod_downsampled_end_to_end(monkeypatch):
    # Absolute gain against the static sky: residual = <d> - (g_rel+g_temp)<s_sky> - g_all*<s_orb>,
    # with every term block-averaged with the same kernel.
    make_view, det, s_full = _make_real_view(monkeypatch)
    orb_full = make_view(1).get_orbital_dipole_tod()
    out = make_view(FACTOR).get_calib_tod("abs", "sky_no_dipole", fill_masked=False)
    np.testing.assert_allclose(out.s_cal, _block_mean(s_full), rtol=2e-5, atol=1e-6)
    expected = _block_mean(det.tod) - 0.75*_block_mean(s_full) - 2.75*_block_mean(orb_full)
    np.testing.assert_allclose(out.tod, expected, rtol=2e-5, atol=1e-6)


# --------------------------------------------------------------------------------------
# Downsampling with flagged samples: a block survives on its kept fraction, and the block
# average uses only the samples that passed.
# --------------------------------------------------------------------------------------
# Three blocks of four: fully good, three-quarters good, one-quarter good.
_PARTLY_FLAGGED = np.array([1, 1, 1, 1,  1, 1, 1, 0,  0, 1, 0, 0], dtype=bool)


def test_a_block_survives_on_its_kept_fraction_not_on_being_spotless(monkeypatch):
    make_view, _, _ = _make_real_view(monkeypatch, good=_PARTLY_FLAGGED)
    # 100%, 75% and 25% of samples pass, so the default 0.5 threshold keeps the first two.
    np.testing.assert_array_equal(make_view(FACTOR).get_mask(), [True, True, False])
    # Raising the threshold above 0.75 drops the partly-flagged block as well; demanding every
    # sample (the old behaviour) would have dropped it at any threshold.
    np.testing.assert_array_equal(make_view(FACTOR, mask_threshold=0.8).get_mask(),
                                  [True, False, False])
    # A threshold of 0 still requires *some* good sample, so the block average is never empty.
    np.testing.assert_array_equal(make_view(FACTOR, mask_threshold=0.0).get_mask(),
                                  [True, True, True])


def test_block_average_ignores_flagged_samples_so_a_glitch_cannot_shift_a_kept_block(monkeypatch):
    make_view, det, _ = _make_real_view(monkeypatch, good=_PARTLY_FLAGGED)
    clean = make_view(FACTOR).get_tod()
    # Put a cosmic-ray-sized hit on the one flagged sample of the surviving second block.
    det.tod[7] = 1e6
    glitched = make_view(FACTOR).get_tod()
    np.testing.assert_allclose(glitched, clean, rtol=1e-6)
    # The kept block is the mean of its three good samples, not of all four.
    np.testing.assert_allclose(glitched[1], det.tod[4:7].mean(), rtol=1e-6)


def test_model_is_averaged_over_the_same_samples_as_the_data(monkeypatch):
    # The scan moves during a block, so a model averaged over all samples would not match a data
    # average taken over the good ones alone.
    make_view, _, s_full = _make_real_view(monkeypatch, good=_PARTLY_FLAGGED)
    out = make_view(FACTOR).get_static_sky_tod()
    np.testing.assert_allclose(out[1], s_full[4:7].mean(), rtol=2e-5, atol=1e-6)
    assert not np.isclose(out[1], s_full[4:8].mean(), rtol=2e-5, atol=1e-6)


def test_mask_threshold_must_leave_some_sample_in_a_surviving_block(monkeypatch):
    make_view, _, _ = _make_real_view(monkeypatch)
    with pytest.raises(ValueError, match="mask_threshold"):
        make_view(FACTOR, mask_threshold=1.0)
    with pytest.raises(ValueError, match="mask_threshold"):
        make_view(FACTOR, mask_threshold=-0.1)
    with pytest.raises(ValueError, match="mask_threshold"):
        GainConfig(sampling_rate=180.0, mask_threshold=1.0)


def test_get_tod_subtracts_each_named_model_and_rejects_anything_else(monkeypatch):
    """``subtract`` dispatches on the two canonical signal names, and nothing else is accepted."""
    make_view, det, s_full = _make_real_view(monkeypatch)
    view = make_view(1)
    orb = view.get_orbital_dipole_tod()
    gain = 2.75  # abs 2.0 + rel 0.5 + temp 0.25, the fake view's full gain

    np.testing.assert_allclose(view.get_tod(subtract=(("sky", ALL),)),
                               det.tod - gain*s_full, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(view.get_tod(subtract=(("orbital_dipole", ALL),)),
                               det.tod - gain*orb, rtol=2e-5, atol=1e-6)
    with pytest.raises(ValueError, match="Unknown TOD signal"):
        view.get_tod(subtract=(("orb", ALL),))
