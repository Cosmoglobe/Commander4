"""The gain sampling: its settings, the sums each term is drawn from, the three solves, and the
block-averaged TODs the gain fit reads from `TODView`."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.pixel_domain import PixelDomain
from commander4.data_models.scan_tod import ScanTOD
from commander4.tod.data_selection import DataSelection
from commander4.tod.glitches.sampling import Glitches
from commander4.tod.hfi_demodulation import HFIModulation
from commander4.tod.jumps.sampling import Jumps
from commander4.tod.noise.gap_filling import fill_all_masked
from commander4.tod.noise.psd import NoisePSDOof
from commander4.tod.sidelobe_deconvolve import FarSidelobes
from commander4.tod.view import TODView
from commander4.tod.gain import Gain, _solve_relative_gain_system
from tod_step_fakes import make_step


def _gain_step(global_blocks: dict | None = None, band_blocks: dict | None = None,
               nu: float = 100.0, rank: int = 0, chain: int = 1) -> Gain:
    """A `Gain` for band BAND of experiment EXP, on a band rank without scans.

    `global_blocks` and `band_blocks` are the gain blocks of `tod_processing` and of the band's own
    parameter block (e.g. ``{"abs_gain": {"enabled": True}}``).
    """
    experiment = SimpleNamespace(experiment_name="EXP", band_name="BAND", nu=nu, nscans=0, ndet=1,
                                 band_unit_factor=1.0, iter_detector_scans=lambda: iter(()))
    params = Bunch(tod_processing=Bunch(**(global_blocks or {})),
                   experiments=Bunch(EXP=Bunch(bands=Bunch(BAND=Bunch(**(band_blocks or {}))))))
    return Gain(experiment, params, params.experiments.EXP.bands.BAND,
                SimpleNamespace(Get_rank=lambda: rank, allreduce=lambda value: value), chain)


class _FakeScan:
    """One detector-scan as `Gain._calibration_sums` reads it: the three signal models, the data
    with every signal removed at the full gain (`get_tod`), the gain mask and the noise."""

    def __init__(self, iscan: int, tod: np.ndarray, s_sky: np.ndarray | None = None,
                 s_orb: np.ndarray | None = None, s_sl: np.ndarray | None = None,
                 mask: np.ndarray | None = None, noise_params: tuple = (1.0, 0.1, -1.0),
                 fsamp: float = 1.0, downsample_factor: int = 1):
        zeros = np.zeros_like(tod)
        self.iscan, self.idet = iscan, 0
        self.tod = tod
        self.s_sky = zeros if s_sky is None else s_sky
        self.s_orb = zeros if s_orb is None else s_orb
        self.s_sl = zeros if s_sl is None else s_sl
        self.mask = np.ones(tod.size, dtype=bool) if mask is None else mask
        self.noise_params = np.array(noise_params)
        self.sigma0 = 0.0  # No white noise in the filled blocks, so the sums are exact.
        self.fsamp, self.downsample_factor = fsamp, downsample_factor

    def get_sky_tod(self): return self.s_sky
    def get_orbital_dipole_tod(self): return self.s_orb
    def get_far_sidelobe_tod(self): return self.s_sl
    def get_tod(self): return self.tod.copy()
    def get_mask(self, proc_mask_type): return self.mask


def _band(nscans: int, fsamp: float = 1.0, white_noise: bool = True) -> DetectorGroupTOD:
    """Band EXP/BAND with one detector and `nscans` scans (IDs 1, 2, ...), holding no data."""
    model = NoisePSDOof()
    model.is_white = white_noise
    scans = [ScanTOD([], 0.0, scan_id) for scan_id in range(1, nscans + 1)]
    return DetectorGroupTOD(scans, "EXP", "BAND", 1, 100.0, 0.0, fsamp, 1, "I", model)


def _sums(monkeypatch: pytest.MonkeyPatch, gains: Gain, band: DetectorGroupTOD,
          scans: list[_FakeScan], term: str) -> tuple[np.ndarray, np.ndarray]:
    """`gains._calibration_sums` for `term`, over the detector-scans `scans` of `band`."""
    import commander4.tod.gain as gain

    monkeypatch.setattr(gain, "TODView", lambda *args, **kwargs: SimpleNamespace(
        iter_focused=lambda **kwargs: iter(scans)))
    return gains._calibration_sums(SimpleNamespace(experiment_data=band), None, term,
                                   np.random.default_rng(0))


def test_the_initial_gain_split_uses_the_whole_band():
    """The absolute and relative gains are means over the detector-scans of all ranks, so every
    rank starts from the same ones. The temporal gain keeps the rest."""
    # This rank has one scan with gains 1 (d0) and 3 (d1). The other rank has one scan, with only
    # d0 present, at gain 7. Its per-detector sums and counts arrive through `allreduce`.
    dets = [SimpleNamespace(name=name, det_idx_fullband=idet, init_scalars=[gain])
            for idet, (name, gain) in enumerate((("d0", 1.0), ("d1", 3.0)))]
    experiment = SimpleNamespace(experiment_name="EXP", band_name="BAND", nu=100.0, nscans=1,
                                 ndet=2, band_unit_factor=1.0,
                                 iter_detector_scans=lambda: iter([(0, det) for det in dets]))
    other_rank = iter([np.array([7.0, 0.0]), np.array([1, 0])])
    band_comm = SimpleNamespace(Get_rank=lambda: 0,
                                allreduce=lambda value: value + next(other_rank))
    params = Bunch(tod_processing=Bunch(), experiments=Bunch(EXP=Bunch(bands=Bunch(BAND=Bunch(
        detectors=Bunch(d0=Bunch(), d1=Bunch()))))))

    gain = Gain(experiment, params, params.experiments.EXP.bands.BAND, band_comm, chain=1)

    assert gain.abs_gain == pytest.approx(11/3)
    np.testing.assert_allclose(gain.rel_gain, [4 - 11/3, 3 - 11/3])
    np.testing.assert_allclose(gain.temporal_gain, [[1 - 4, 0.0]], atol=1e-12)


@pytest.mark.parametrize("sampler_name", ["sample_absolute", "sample_relative", "sample_temporal"])
@pytest.mark.parametrize("seconds, fsamp, expected", [
    (1.0, 200.0, 200), (0.25, 200.0, 50), (0.0, 200.0, 1),
    (0.001, 200.0, 1), (1.0, 32.51, 33),
])
def test_gain_samplers_convert_seconds_to_native_samples(
    monkeypatch: pytest.MonkeyPatch, sampler_name: str, seconds: float, fsamp: float, expected: int,
) -> None:
    """Check the actual averaging factor passed to TODView by each gain routine."""
    import commander4.tod.gain as gain

    experiment = SimpleNamespace(fsamp=fsamp, ndet=1, scans=[])
    gains = _gain_step({term: {"downsample_time": seconds}
                        for term in ("abs_gain", "rel_gain", "temporal_gain")})
    view = Mock(side_effect=RuntimeError("stop at TODView"))
    monkeypatch.setattr(gain, "TODView", view)
    with pytest.raises(RuntimeError, match="stop at TODView"):
        samples = SimpleNamespace(band_comm=MPI.COMM_SELF, experiment_data=experiment)
        getattr(gains, sampler_name)(samples, None, 1, np.random.default_rng(0))
    assert view.call_args.kwargs["downsample_factor"] == expected


@pytest.mark.parametrize("nu, target, rank, chain, warns", [
    (545.0, "orbital_dipole", 0, 1, True),
    (353.0, "orbital_dipole", 0, 1, False),
    (857.0, "sky", 0, 1, False),
    (545.0, "orbital_dipole", 1, 1, False),
    (545.0, "orbital_dipole", 0, 2, False),
])
def test_dipole_calibration_warning_only_above_400_ghz_once_per_band(
    caplog: pytest.LogCaptureFixture, nu: float, target: str, rank: int, chain: int, warns: bool,
) -> None:
    """Above 400 GHz the dipole is faint next to dust, so calibrating on it deserves a warning,
    given once per band: by the band master, for chain 1."""
    with caplog.at_level("WARNING", logger="commander4.tod.gain"):
        _gain_step({"abs_gain": {"enabled": True, "calibrate_against": target}}, nu=nu,
                   rank=rank, chain=chain)
    assert ("orbital dipole" in caplog.text) == warns


@pytest.mark.parametrize("step_name", ["abs_gain", "rel_gain", "temporal_gain"])
@pytest.mark.parametrize("enabled", [False, True])
def test_each_sampled_gain_term_warns_about_dipole_calibration(
    caplog: pytest.LogCaptureFixture, step_name: str, enabled: bool,
) -> None:
    """The warning is given when the settings are read, and only for a term that is sampled."""
    with caplog.at_level("WARNING", logger="commander4.tod.gain"):
        _gain_step({step_name: {"enabled": enabled, "calibrate_against": "orbital_dipole"}},
                   nu=545.0)
    assert (f"{step_name} for band BAND" in caplog.text) == enabled


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("white_noise", [False, True])
@pytest.mark.parametrize("white_weights", [False, True])
def test_absolute_gain_sums_come_from_the_filled_whitened_good_blocks(
    monkeypatch: pytest.MonkeyPatch, dtype: type, white_noise: bool, white_weights: bool,
) -> None:
    """s^T N^-1 s and s^T N^-1 r come from the gap-filled, N^-1/2-whitened calibrator and
    residual, summed over the good blocks only, at the block-averaged sample rate (as C3's
    `accumulate_abscal`). With `white_noise_weights`, N is the white-noise level alone."""
    band = _band(2, fsamp=180.0, white_noise=white_noise)
    gains = _gain_step({"abs_gain": {"downsample_time": 1.0,
                                     "white_noise_weights": white_weights}})
    gains.abs_gain = 1.0
    rng = np.random.default_rng(13)
    scans = []
    expected_s_N_inv_s = expected_s_N_inv_r = 0.0
    for iscan, length in enumerate((997, 1024)):
        calibrator = rng.normal(size=length).astype(dtype)
        residual = (0.7 * calibrator + rng.normal(size=length) + 3.5).astype(dtype)
        mask = rng.random(length) > 0.3
        noise_params = (2.0, 0.1, -1.5)
        # The orbital dipole is the calibrator, and `get_tod` has it removed at the gain of 1.
        scans.append(_FakeScan(iscan, residual - calibrator, s_orb=calibrator, mask=mask,
                               noise_params=noise_params, fsamp=180.0, downsample_factor=180))
        s_filled, r_filled = calibrator.copy(), (residual - calibrator) + calibrator
        fill_all_masked(s_filled, mask, 0.0, np.random.default_rng(0))
        fill_all_masked(r_filled, mask, 0.0, np.random.default_rng(0))
        s_white = band.apply_N_inv(s_filled, np.array(noise_params), samprate=1.0, power=0.5,
                                   white_only=white_weights)
        r_white = band.apply_N_inv(r_filled, np.array(noise_params), samprate=1.0, power=0.5,
                                   white_only=white_weights)
        expected_s_N_inv_s += np.sum(s_white[mask]**2)
        expected_s_N_inv_r += np.sum(s_white[mask] * r_white[mask])

    s_N_inv_s, s_N_inv_r = _sums(monkeypatch, gains, band, scans, "abs_gain")

    tolerance = 2e-6 if dtype == np.float32 else 1e-12
    assert s_N_inv_s.sum() == pytest.approx(expected_s_N_inv_s, rel=tolerance)
    assert s_N_inv_r.sum() == pytest.approx(expected_s_N_inv_r, rel=tolerance)


def test_masked_blocks_do_not_pull_the_absolute_gain_toward_its_old_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With most of the calibrator weight inside the mask, the draw lands at the true gain, not
    near the old one. Here a bright crossing that the sky model gets 20% wrong is masked, and the
    gain starts at twice the true value."""
    import commander4.tod.gain as gain

    rng = np.random.default_rng(3)
    n, g_true, g_old, noise_rms = 20000, 1.0, 2.0, 1.0
    phase = np.arange(n) % 60
    crossing = 30*np.exp(-0.5*((phase - 30)/2.0)**2)
    s_cal = 3*np.sin(2*np.pi*phase/60) + crossing
    mask = np.abs(phase - 30) > 6
    data = g_true*(s_cal + 0.2*crossing*~mask) + rng.normal(0.0, noise_rms, n)

    # The sky model is the calibrator, and `get_tod` has it removed at the old gain.
    scan = _FakeScan(0, data - g_old*s_cal, s_sky=s_cal, mask=mask,
                     noise_params=(noise_rms, 0.1, -1.0))
    monkeypatch.setattr(gain, "TODView", lambda *args, **kwargs: SimpleNamespace(
        iter_focused=lambda **kwargs: iter([scan])))
    gains = _gain_step({"abs_gain": {"downsample_time": 0.0, "calibrate_against": "sky"}})
    gains.abs_gain = g_old
    samples = SimpleNamespace(band_comm=MPI.COMM_SELF, experiment_data=_band(1), chain=1)
    gains.sample_absolute(samples, None, 1, np.random.default_rng(0))

    # The draw's own scatter is noise_rms/sqrt(sum of s^2 over the good blocks), about 0.004 here.
    # Before the fix the masked blocks held g_old * s, and the draw came out near g_old.
    assert gains.abs_gain == pytest.approx(g_true, abs=0.05)


def test_temporal_gain_is_solved_in_time_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shuffling a rank's scans gives every scan the same gain, since the solve sorts by scan ID.

    The Wiener prior couples scans that are neighbours in time, so a solve in rank order would
    pair the wrong scans and change the result.
    """
    import commander4.tod.gain as gain

    rng = np.random.default_rng(5)
    nscans = 6
    # One calibrator and residual per scan, listed by the scan's place in time.
    calibs = [SimpleNamespace(s_cal=rng.normal(size=500), tod=rng.normal(size=500))
              for _ in range(nscans)]
    model = NoisePSDOof()
    model.is_white = True

    def solve(time_order: np.ndarray) -> np.ndarray:
        """Temporal gain of each scan, with the rank holding its scans in the given order of their
        places in time. The scan IDs increase with time, but are not the places themselves."""
        scan_ids = 100 + 10*time_order
        scans = [ScanTOD([], 0.0, int(scan_id)) for scan_id in scan_ids]
        experiment = DetectorGroupTOD(scans, "EXP", "BAND", 1, 100.0, 0.0, 10.0, 1, "I", model)
        # The temporal gain starts at 0, so `get_tod` is the residual itself.
        views = [_FakeScan(iscan, calibs[t].tod, s_sky=calibs[t].s_cal, fsamp=10.0)
                 for iscan, t in enumerate(time_order)]
        scan_view = SimpleNamespace(iter_focused=Mock(return_value=iter(views)))
        monkeypatch.setattr(gain, "TODView", Mock(return_value=scan_view))
        gains = _gain_step()
        gains.abs_gain, gains.rel_gain = 1.0, np.zeros(1)
        gains.temporal_gain = np.zeros((nscans, 1), dtype=np.float32)
        samples = SimpleNamespace(band_comm=MPI.COMM_SELF, experiment_data=experiment, chain=1,
                                  scan_ids=scan_ids)
        # The same fluctuation draws, which the solve makes in time order.
        gains.sample_temporal(samples, None, 1, np.random.default_rng(7))
        return gains.temporal_gain[:, 0]

    in_time_order = solve(np.arange(nscans))
    shuffled_order = np.array([3, 0, 5, 1, 4, 2])
    shuffled = solve(shuffled_order)

    np.testing.assert_array_equal(shuffled, in_time_order[shuffled_order])


def test_temporal_draw_without_fluctuations_or_prior_is_the_per_scan_fit_minus_its_mean() -> None:
    from commander4.tod.gain import _draw_temporal_gain

    rng = np.random.default_rng(0)
    weights = rng.uniform(1.0, 2.0, 500)
    per_scan_fit = rng.normal(size=500)
    no_fluctuations = SimpleNamespace(standard_normal=lambda n: np.zeros(n))
    # sigma0 = 1e6 makes the prior's inverse power at most 1e-12.
    draw, cg_err = _draw_temporal_gain(weights, weights*per_scan_fit, 1e6, 1.0, -2.5,
                                       no_fluctuations)
    assert cg_err < 1e-10
    np.testing.assert_allclose(draw, per_scan_fit - per_scan_fit.mean(), atol=1e-10)


# --------------------------------------------------------------------------------------
# The calibrator a gain term ends up using
# --------------------------------------------------------------------------------------
def _inputs(band_blocks: dict, passed: str = "sky", gain_block: str = "abs_gain",
            downsample_time: float = 1.0) -> tuple[str, float]:
    """Resolve one step's calibrator and downsampling."""
    global_blocks = {
        gain_block: {
            "enabled": True,
            "calibrate_against": passed,
            "downsample_time": downsample_time,
        },
    }
    config = _gain_step(global_blocks, band_blocks).cfg[gain_block]
    return config.calibrate_against, config.downsample_time


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
            "downsample_time": 1.0,
            "mask_threshold": 0.5,
        },
    }
    band_blocks = {
        "abs_gain": {
            "enabled": False,
            "from_iter": 4,
            "downsample_time": 0.25,
            "mask_threshold": 0.8,
        },
    }
    config = _gain_step(global_blocks, band_blocks).cfg.abs_gain

    assert not config.enabled
    assert config.from_iter == 4
    assert config.calibrate_against == "sky"
    assert config.downsample_time == 0.25
    assert config.mask_threshold == 0.8


# --------------------------------------------------------------------------------------
# Gain._calibration_sums: the calibrator and the residual of each term
# --------------------------------------------------------------------------------------
# Distinct values for the three gain terms, and three signal models, on six samples.
_GAINS = {"abs_gain": 2.0, "rel_gain": 0.3, "temporal_gain": 0.05}
_S_SKY = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
_S_ORB = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
_S_SL = np.array([0.5, -0.5, 0.25, -0.25, 0.5, -0.5])


def _gains_of(term: str, calibrate_against: str) -> Gain:
    """A `Gain` with the gain terms of `_GAINS`, fitting `term` against `calibrate_against`."""
    gains = _gain_step({term: {"calibrate_against": calibrate_against}})
    gains.abs_gain = _GAINS["abs_gain"]
    gains.rel_gain = np.array([_GAINS["rel_gain"]])
    gains.temporal_gain = np.array([[_GAINS["temporal_gain"]]])
    return gains


@pytest.mark.parametrize("term,calibrate_against,calibrator", [
    ("abs_gain", "orbital_dipole", _S_ORB),
    ("abs_gain", "sky", _S_SKY + _S_ORB + _S_SL),
    ("rel_gain", "sky", _S_SKY + _S_ORB + _S_SL),
    ("temporal_gain", "sky", _S_SKY + _S_ORB + _S_SL),
    ("abs_gain", "sky_no_dipole", _S_SKY + _S_SL),
])
def test_each_term_is_fitted_with_its_own_gain_against_its_calibrator(
    monkeypatch: pytest.MonkeyPatch, term: str, calibrate_against: str, calibrator: np.ndarray,
) -> None:
    """Without noise the residual is exactly g_term * s, so s^T N^-1 r / s^T N^-1 s is the term's
    own gain, and s^T N^-1 s shows the calibrator. A calibrator with the sky model has the
    far-sidelobe pickup as well, as C3's `s_tot`."""
    scan = _FakeScan(0, np.zeros(6), s_sky=_S_SKY, s_orb=_S_ORB, s_sl=_S_SL)
    s_N_inv_s, s_N_inv_r = _sums(monkeypatch, _gains_of(term, calibrate_against), _band(1),
                                 [scan], term)
    # With white noise of sigma0 = 1, N^-1/2 only removes the mean.
    assert s_N_inv_s[0, 0] == pytest.approx(np.sum((calibrator - calibrator.mean())**2))
    assert s_N_inv_r[0, 0]/s_N_inv_s[0, 0] == pytest.approx(_GAINS[term])


def test_an_unknown_calibrator_is_refused_at_startup() -> None:
    with pytest.raises(ValueError, match="calibrate_against"):
        _gain_step({"rel_gain": {"calibrate_against": "dipole"}})


def test_the_masked_blocks_of_residual_and_calibrator_are_filled_alike(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both get the same straight line across a gap, so the residual stays the term's gain times
    the calibrator, and nothing from inside the gap is left."""
    mask = np.array([True, True, False, False, True, True])
    s_sky = _S_SKY.copy()
    s_sky[2:4] = 1e3               # A bright, badly modelled crossing in the masked blocks.
    scan = _FakeScan(0, np.zeros(6), s_sky=s_sky, s_orb=_S_ORB.copy(), s_sl=_S_SL, mask=mask)
    s_N_inv_s, s_N_inv_r = _sums(monkeypatch, _gains_of("rel_gain", "sky_no_dipole"), _band(1),
                                 [scan], "rel_gain")
    assert s_N_inv_r[0, 0]/s_N_inv_s[0, 0] == pytest.approx(_GAINS["rel_gain"])
    # The line runs between the means of the good blocks on each side of the gap.
    s_filled = _S_SKY + _S_SL
    left, right = s_filled[:2].mean(), s_filled[4:].mean()
    s_filled[2:4] = left + (right - left)*np.array([1, 2])/3
    assert s_N_inv_s[0, 0] == pytest.approx(np.sum(mask*(s_filled - s_filled.mean())**2))
    # The dipole calibrator is the view's cached dipole TOD itself, which the fill must not change.
    _sums(monkeypatch, _gains_of("abs_gain", "orbital_dipole"), _band(1), [scan], "abs_gain")
    assert scan.s_orb[2] == _S_ORB[2]


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
    out = _solve_relative_gain_system(np.zeros(3), np.zeros(3), prev, rng=_ZERO_RNG)
    np.testing.assert_array_equal(out, prev)


def test_relgain_deterministic_with_seeded_rng():
    s = np.array([2.0, 3.0, 4.0]); r = np.array([0.5, -0.5, 0.1]); prev = np.zeros(3)
    a = _solve_relative_gain_system(s, r, prev, rng=np.random.default_rng(5))
    b = _solve_relative_gain_system(s, r, prev, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------------------
# Gain routines convert downsample_time [s] to a number of native samples.


# --------------------------------------------------------------------------------------
# Downsampling: model TODs are block-averaged like the data, not block-center sampled
# --------------------------------------------------------------------------------------
NTOD, FACTOR = 12, 4
# Downsampling cuts the full-rate stream into contiguous blocks of `factor` samples and keeps every
# complete block (ntod // factor of them); only a trailing *partial* block is dropped. Here ntod is
# an exact multiple of factor, so all ntod // factor blocks are kept.
NBLOCKS = NTOD // FACTOR


def _make_real_view(monkeypatch, good=None, sidelobe=None):
    """Factory for a TODView over a minimal fake detector, exercising the real downsampling paths.

    Returns ``make_view(downsample_factor)`` so a test can build views at full *and* downsampled
    resolution over the same detector (the factor is now fixed per view), plus the detector and the
    full-rate sky TOD for the expected-value comparisons. ``good`` is an optional full-rate
    flag cut; without one every sample passes. ``sidelobe`` is an optional full-rate far-sidelobe
    pickup, given by a model that is switched on and built; without one there is no model.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "1")  # Required by get_s_orb_tod.
    rng = np.random.default_rng(7)
    pix = rng.integers(0, 12, size=NTOD)        # Valid pixels for the nside=1 experiment below.
    psi = rng.uniform(0.0, np.pi, size=NTOD)
    det = SimpleNamespace(tod=rng.normal(size=NTOD), ntod=NTOD, fsamp=float(FACTOR), nside=1,
                          det_idx_fullband=0,
                          pointing=SimpleNamespace(get_pix_psi=lambda: (pix, psi)),
                          response_I_P=(1.0, 1.0),
                          orbital_velocity_m_per_s=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                          good_data_mask=np.ones(NTOD, dtype=bool) if good is None else good)
    experiment_data = SimpleNamespace(scans=[SimpleNamespace(detectors=[det])], nside=1, nu=30.0,
                                      pixel_domain=PixelDomain(MPI.COMM_SELF, 1, "full"))
    tod_samples = SimpleNamespace(jumps=make_step(Jumps, 1, 1), glitches=make_step(Glitches, 1, 1),
                                  gain=SimpleNamespace(abs_gain=2.0, rel_gain=np.array([0.5]),
                                                       temporal_gain=np.array([[0.25]])),
                                  selection=make_step(DataSelection, 1, 1),
                                  hfi=make_step(HFIModulation, 1, 1),
                                  far_sidelobes=make_step(FarSidelobes, 1, 1),
                                  noise=SimpleNamespace(noise_params=np.ones((1, 1, 3))))
    if sidelobe is not None:
        # The projection is asked for at the full rate, like the sky model.
        tod_samples.far_sidelobes = SimpleNamespace(
            cfg=Bunch(enabled=True), cubes=[None],
            get_projection=lambda p, s, idet: sidelobe if p is pix and s is psi else None)
    skymap = rng.normal(size=(3, 12))

    def make_view(downsample_factor=1, mask_threshold=0.5):
        return TODView(experiment_data, tod_samples, compsep_output=skymap,
                       downsample_factor=downsample_factor,
                       mask_threshold=mask_threshold).focus(0, det)

    s_full = skymap[0, pix] + np.cos(2*psi)*skymap[1, pix] + np.sin(2*psi)*skymap[2, pix]
    return make_view, det, s_full


def _block_mean(arr):
    return arr[:NBLOCKS*FACTOR].reshape(NBLOCKS, FACTOR).mean(axis=-1)


def test_sky_downsampling_is_block_average(monkeypatch):
    make_view, _, s_full = _make_real_view(monkeypatch)
    out = make_view(FACTOR).get_sky_tod()
    np.testing.assert_allclose(out, _block_mean(s_full), rtol=2e-5, atol=1e-6)
    # Regression guard: must NOT be the model sampled at the block-center pixels.
    block_centers = np.array([2, 6, 10])
    assert not np.allclose(out, s_full[block_centers])


def test_downsampled_get_tod_removes_the_block_averaged_signals_at_the_full_gain(monkeypatch):
    # <d> - g_all*(<s_sky> + <s_orb>), with every term block-averaged with the same kernel.
    make_view, det, s_full = _make_real_view(monkeypatch)
    orb_full = make_view(1).get_orbital_dipole_tod()
    expected = _block_mean(det.tod) - 2.75*(_block_mean(s_full) + _block_mean(orb_full))
    np.testing.assert_allclose(make_view(FACTOR).get_tod(), expected, rtol=2e-5, atol=1e-6)


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
    signals = ("sky", "orbital_dipole")
    clean = make_view(FACTOR).get_tod(keep=signals)
    # Put a cosmic-ray-sized hit on the one flagged sample of the surviving second block.
    det.tod[7] = 1e6
    glitched = make_view(FACTOR).get_tod(keep=signals)
    np.testing.assert_allclose(glitched, clean, rtol=1e-6)
    # The kept block is the mean of its three good samples, not of all four.
    np.testing.assert_allclose(glitched[1], det.tod[4:7].mean(), rtol=1e-6)


def test_model_is_averaged_over_the_same_samples_as_the_data(monkeypatch):
    # The scan moves during a block, so a model averaged over all samples would not match a data
    # average taken over the good ones alone.
    make_view, _, s_full = _make_real_view(monkeypatch, good=_PARTLY_FLAGGED)
    out = make_view(FACTOR).get_sky_tod()
    np.testing.assert_allclose(out[1], s_full[4:7].mean(), rtol=2e-5, atol=1e-6)
    assert not np.isclose(out[1], s_full[4:8].mean(), rtol=2e-5, atol=1e-6)


def test_mask_threshold_must_leave_some_sample_in_a_surviving_block(monkeypatch):
    make_view, _, _ = _make_real_view(monkeypatch)
    with pytest.raises(ValueError, match="mask_threshold"):
        make_view(FACTOR, mask_threshold=1.0)
    with pytest.raises(ValueError, match="mask_threshold"):
        make_view(FACTOR, mask_threshold=-0.1)


def test_get_tod_removes_each_signal_unless_kept_and_rejects_anything_else(monkeypatch):
    sl = np.linspace(-1.0, 1.0, NTOD)
    make_view, det, s_full = _make_real_view(monkeypatch, sidelobe=sl)
    view = make_view(1)
    orb = view.get_orbital_dipole_tod()
    gain = 2.75  # abs 2.0 + rel 0.5 + temp 0.25, the fake view's full gain

    np.testing.assert_allclose(view.get_tod(), det.tod - gain*(s_full + orb + sl),
                               rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(view.get_tod(keep=("orbital_dipole",)), det.tod - gain*(s_full + sl),
                               rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(view.get_tod(keep=("sky",)), det.tod - gain*(orb + sl),
                               rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(view.get_tod(keep=("far_sidelobes",)), det.tod - gain*(s_full + orb),
                               rtol=2e-5, atol=1e-6)
    with pytest.raises(ValueError, match="Unknown models"):
        view.get_tod(keep=("orb",))


def test_far_sidelobes_are_block_averaged_and_zero_without_a_model(monkeypatch):
    sl = np.linspace(-1.0, 1.0, NTOD)
    make_view, _, _ = _make_real_view(monkeypatch, sidelobe=sl)
    np.testing.assert_allclose(make_view(FACTOR).get_far_sidelobe_tod(), _block_mean(sl))
    make_view, _, _ = _make_real_view(monkeypatch)
    np.testing.assert_array_equal(make_view(FACTOR).get_far_sidelobe_tod(), np.zeros(NBLOCKS))
