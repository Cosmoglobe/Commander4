"""Planck HFI modulation phase, stochastic parity baselines, and lazy TOD demodulation."""
from types import SimpleNamespace

import numpy as np
import pytest
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.pixel_domain import PixelDomain
from commander4.tod.data_selection import DataSelection
from commander4.tod.glitches.events import GlitchEvents
from commander4.tod.glitches.sampling import Glitches
from commander4.tod.jumps.events import JumpEvents
from commander4.tod.jumps.sampling import Jumps
from commander4.data_models.pointing import PixelPointing
from commander4.data_models.scan_tod import ScanTOD
from commander4.tod.hfi_demodulation import HFIModulation
from commander4.tod.sidelobe_deconvolve import FarSidelobes
from commander4.tod.view import TODView
from tod_step_fakes import make_step

_NSIDE = 1
_NPIX = 12
_NPAIR = 50
_NTOD = 2 * _NPAIR
_GAIN = 2.0
_SIGMA0 = 2.0
_BASELINES = np.array([100.0, 200.0])
# Kept in by the tests of the instrument models, which compare against the demodulated sky.
_SIGNALS = ("sky", "orbital_dipole")


class _ZeroRNG:
    def normal(self) -> float:
        return 0.0


class _SequenceRNG:
    def __init__(self, values: list[float]):
        self.values = iter(values)

    def normal(self) -> float:
        return next(self.values)


def _build_hfi_case(phase: int = -1, pulse: np.ndarray | None = None):
    """Build one modulated detector-scan with a bright middle-RING-pixel crossing.

    An optional glitch `pulse` (detector units) is added and modulated like the sky.
    """
    pix = np.zeros(_NTOD, dtype=np.int64)
    pix[:20] = 6  # nside=1's only pixel strictly inside C3's 0.48--0.52*npix cut.
    psi = np.zeros(_NTOD)
    sky_map = np.zeros((1, _NPIX))
    sky_map[0, 6] = 10.0
    sky_tod = sky_map[0, pix]

    signal = _GAIN * sky_tod if pulse is None else _GAIN * sky_tod + pulse
    raw_tod = np.empty(_NTOD, dtype=np.float32)
    raw_tod[0::2] = phase * signal[0::2] + _BASELINES[0]
    raw_tod[1::2] = -phase * signal[1::2] + _BASELINES[1]

    pointing = PixelPointing(pix, psi, _NSIDE, _NSIDE, _NTOD, _NTOD)
    detector = DetectorTOD(
        name="100-1a", det_idx_fullband=0, tod=raw_tod, pointing=pointing,
        sampling_rate_hz=180.0, orbital_velocity_m_per_s=np.zeros(3, dtype=np.float32),
        default_proc_mask=np.ones(_NPIX, dtype=bool), specific_proc_masks={},
        flag=np.zeros(_NTOD, dtype=np.int64), bad_data_bitmask=1,
    )
    noise_model = SimpleNamespace(npar=1, params=np.array([_SIGMA0]))
    band = DetectorGroupTOD(
        [ScanTOD([detector], 0.0, 1)], "PlanckHFI", "Planck100GHz", _NSIDE, 100.0,
        10.0, 180.0, 1, "I", noise_model, hfi_demodulation=True,
    )
    samples = SimpleNamespace(
        band_comm=MPI.COMM_SELF,
        experiment_data=band,
        hfi=HFIModulation(band, Bunch(), None, MPI.COMM_SELF, 1),
        noise=SimpleNamespace(noise_params=np.array([[[_SIGMA0]]])),
        gain=SimpleNamespace(abs_gain=_GAIN, rel_gain=np.zeros(1), temporal_gain=np.zeros((1, 1))),
        selection=make_step(DataSelection, 1, 1),
        jumps=make_step(Jumps, 1, 1),
        glitches=make_step(Glitches, 1, 1),
        far_sidelobes=make_step(FarSidelobes, 1, 1),
    )
    band.pixel_domain = PixelDomain(MPI.COMM_SELF, _NSIDE, "full")
    return band, samples, sky_map, raw_tod, sky_tod


@pytest.mark.parametrize("phase", [-1, 1])
def test_first_pass_finds_phase_then_next_pass_fits_and_demodulates(
    monkeypatch, phase: int,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    band, samples, sky_map, raw_tod, sky_tod = _build_hfi_case(phase=phase)

    # C3's first pass fits raw parity means and uses them to determine the modulation phase.
    samples.hfi.sample(samples,sky_map, rng=_ZeroRNG())
    assert samples.hfi.initialized
    assert samples.hfi.modulation_phase[0, 0] == phase

    # Every later Gibbs pass conditions the baseline draw on the current gain-scaled sky model.
    samples.hfi.sample(samples,sky_map, rng=_ZeroRNG())
    np.testing.assert_allclose(samples.hfi.baselines[0, 0], _BASELINES)

    view = TODView(band, samples, compsep_output=sky_map).focus(0, band.scans[0].detectors[0])
    np.testing.assert_array_equal(view.raw_tod, raw_tod)
    np.testing.assert_allclose(view.get_tod(keep=_SIGNALS), _GAIN * sky_tod)


def test_baseline_sample_includes_c3_white_noise_fluctuation(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    band, samples, sky_map, _, _ = _build_hfi_case(phase=-1)
    samples.hfi.modulation_phase[0, 0] = -1
    samples.hfi.initialized = True

    samples.hfi.sample(samples,sky_map, rng=_SequenceRNG([1.5, -2.0]))

    expected = _BASELINES + np.array([1.5, -2.0]) * _SIGMA0 / np.sqrt(_NPAIR)
    np.testing.assert_allclose(samples.hfi.baselines[0, 0], expected)


def test_baseline_fit_removes_glitch_pulses_like_the_sky(monkeypatch) -> None:
    """Left in, the pulses' mean would go into the baselines and be removed again after demod."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    pulse = np.zeros(_NTOD)
    pulse[40:60] = 30.0
    band, samples, sky_map, _, sky_tod = _build_hfi_case(phase=-1, pulse=pulse)
    samples.hfi.modulation_phase[0, 0] = -1
    samples.hfi.initialized = True
    # One template event, whose placeholder pulse model is replaced by the known pulse.
    samples.glitches = make_step(Glitches, 1, 1, settings={"glitches": {"enabled": True}})
    samples.glitches.events[0, 0] = GlitchEvents([40], [20], [0], [30.0], [40], [40], [0], [])
    samples.glitches.template_amps = np.zeros((1, 3, 8))
    samples.glitches.template_taus = np.ones((1, 3, 8))
    monkeypatch.setattr(GlitchEvents, "pulse_tod", lambda self, ntod, amps, taus, fsamp, bright,
                        faint: pulse if faint else np.zeros(ntod))

    samples.hfi.sample(samples,sky_map, rng=_ZeroRNG())

    np.testing.assert_allclose(samples.hfi.baselines[0, 0], _BASELINES)
    view = TODView(band, samples, compsep_output=sky_map).focus(0, band.scans[0].detectors[0])
    np.testing.assert_allclose(view.get_tod(keep=_SIGNALS), _GAIN * sky_tod, atol=1e-4)
    # The event is faint, so only keeping the faint glitches leaves its pulse in.
    np.testing.assert_allclose(view.get_tod(keep=_SIGNALS + ("faint_glitches",)),
                               _GAIN * sky_tod + pulse, atol=1e-4)
    np.testing.assert_allclose(view.get_tod(keep=_SIGNALS + ("bright_glitches",)),
                               _GAIN * sky_tod, atol=1e-4)


def _with_jump(samples: SimpleNamespace) -> None:
    """Give `samples` one stored jump at sample 4."""
    samples.jumps = make_step(Jumps, 1, 1, settings={"jump_detection": {"enabled": True}},
                              experiment={"jump_bitmask": 1})
    samples.jumps.events[0, 0] = JumpEvents([4], [3.0])


def test_get_tod_applies_hfi_demodulation_then_jumps_and_leaves_in_what_it_is_told_to_keep(
) -> None:
    band, samples, sky_map, raw_tod, _ = _build_hfi_case(phase=-1)
    samples.hfi.modulation_phase[0, 0] = -1
    samples.hfi.initialized = True
    samples.hfi.baselines[0, 0] = _BASELINES
    _with_jump(samples)

    # The jump is a step of the demodulated TOD, so it is added after the sign flip.
    expected = raw_tod.copy()
    expected[0::2] = -(expected[0::2] - _BASELINES[0])
    expected[1::2] = expected[1::2] - _BASELINES[1]
    expected[4:] += 3.0

    view = TODView(band, samples, compsep_output=sky_map).focus(0, band.scans[0].detectors[0])
    np.testing.assert_array_equal(view.raw_tod, raw_tod)
    np.testing.assert_allclose(view.get_tod(keep=_SIGNALS), expected)

    # Each model named in `keep` stays in the TOD, and every other one is still removed. The sign
    # flip of the demodulation is always applied.
    baselines_kept = raw_tod.copy()
    baselines_kept[0::2] *= -1
    baselines_kept[4:] += 3.0
    np.testing.assert_allclose(view.get_tod(keep=_SIGNALS + ("hfi_baselines",)), baselines_kept)
    jumps_kept = raw_tod.copy()
    jumps_kept[0::2] = -(jumps_kept[0::2] - _BASELINES[0])
    jumps_kept[1::2] = jumps_kept[1::2] - _BASELINES[1]
    np.testing.assert_allclose(view.get_tod(keep=_SIGNALS + ("jumps",)), jumps_kept)
    with pytest.raises(ValueError, match="Unknown models"):
        view.get_tod(keep=("jump",))


def test_residual_removes_hfi_baselines_jumps_and_sky(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    band, samples, sky_map, _, _ = _build_hfi_case(phase=-1)
    samples.hfi.modulation_phase[0, 0] = -1
    samples.hfi.initialized = True
    samples.hfi.baselines[0, 0] = _BASELINES
    n_corr = np.full(_NTOD, 3.0, dtype=np.float32)
    residual = np.full(_NTOD, 0.5, dtype=np.float32)
    detector = band.scans[0].detectors[0]
    # The jump that `_with_jump` corrects: a step of the demodulated TOD.
    jump = np.zeros(_NTOD, dtype=np.float32)
    jump[4:] = -3.0
    # Add the noise and the jump with the raw modulation signs; the residual is in demodulated
    # detector units.
    detector._tod[0::2] -= n_corr[0::2] + residual[0::2] + jump[0::2]
    detector._tod[1::2] += n_corr[1::2] + residual[1::2] + jump[1::2]
    _with_jump(samples)
    view = TODView(band, samples, compsep_output=sky_map).focus(0, detector)

    np.testing.assert_allclose(view.get_tod() - n_corr, residual, atol=1e-4)
