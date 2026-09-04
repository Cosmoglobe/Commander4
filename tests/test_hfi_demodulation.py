"""Planck HFI modulation phase, stochastic parity baselines, and lazy TOD demodulation."""
from types import SimpleNamespace

import numpy as np
import pytest

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.jump_corrections import JumpCorrection
from commander4.data_models.pointing import PixelPointing
from commander4.data_models.scan_tod import ScanTOD
from commander4.tod.hfi_demodulation import sample_hfi_baselines
from commander4.tod.view import TODView

_NSIDE = 1
_NPIX = 12
_NPAIR = 50
_NTOD = 2 * _NPAIR
_GAIN = 2.0
_SIGMA0 = 2.0
_BASELINES = np.array([100.0, 200.0])


class _ZeroRNG:
    def normal(self) -> float:
        return 0.0


class _SequenceRNG:
    def __init__(self, values: list[float]):
        self.values = iter(values)

    def normal(self) -> float:
        return next(self.values)


def _build_hfi_case(phase: int = -1):
    """Build one modulated detector-scan with a bright middle-RING-pixel crossing."""
    pix = np.zeros(_NTOD, dtype=np.int64)
    pix[:20] = 6  # nside=1's only pixel strictly inside C3's 0.48--0.52*npix cut.
    psi = np.zeros(_NTOD)
    sky_map = np.zeros((1, _NPIX))
    sky_map[0, 6] = 10.0
    sky_tod = sky_map[0, pix]

    raw_tod = np.empty(_NTOD, dtype=np.float32)
    raw_tod[0::2] = phase * _GAIN * sky_tod[0::2] + _BASELINES[0]
    raw_tod[1::2] = -phase * _GAIN * sky_tod[1::2] + _BASELINES[1]

    pointing = PixelPointing(pix, psi, np.array([0], dtype=np.int64), None, None,
                             _NSIDE, _NSIDE, _NTOD, _NTOD)
    detector = DetectorTOD(
        name="100-1a", det_idx_fullband=0, tod=raw_tod, pointing=pointing,
        sampling_rate_hz=180.0, orbital_velocity_m_per_s=np.zeros(3, dtype=np.float32),
        huffman_tree=None, huffman_symbols=None, default_proc_mask=np.ones(_NPIX, dtype=bool),
        specific_proc_masks={}, flag_encoded=np.zeros(_NTOD, dtype=np.int64),
        bad_data_bitmask=1, flag_is_compressed=False,
    )
    noise_model = SimpleNamespace(npar=1, params=np.array([_SIGMA0]))
    band = DetectorGroupTOD(
        [ScanTOD([detector], 0.0, 1)], "PlanckHFI", "Planck100GHz", _NSIDE, 100.0,
        10.0, 180.0, 1, "I", noise_model, hfi_demodulation=True,
    )
    no_jump = SimpleNamespace(is_empty=lambda: True)
    samples = SimpleNamespace(
        hfi_demodulation=True,
        modulation_phase=np.ones((1, 1), dtype=np.int8),
        modulation_phase_initialized=False,
        baselines=np.zeros((1, 1, 2)),
        noise_params=np.array([[[_SIGMA0]]]),
        abs_gain=_GAIN,
        rel_gain=np.zeros(1),
        temporal_gain=np.zeros((1, 1)),
        accept=np.ones((1, 1), dtype=bool),
        jumps=SimpleNamespace(get=lambda iscan, idet: no_jump),
    )
    return band, samples, sky_map, raw_tod, sky_tod


@pytest.mark.parametrize("phase", [-1, 1])
def test_first_pass_finds_phase_then_next_pass_fits_and_demodulates(
    monkeypatch, phase: int,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    band, samples, sky_map, raw_tod, sky_tod = _build_hfi_case(phase=phase)

    # C3's first pass fits raw parity means and uses them to determine the modulation phase.
    sample_hfi_baselines(band, samples, sky_map, rng=_ZeroRNG())
    assert samples.modulation_phase_initialized
    assert samples.modulation_phase[0, 0] == phase

    # Every later Gibbs pass conditions the baseline draw on the current gain-scaled sky model.
    sample_hfi_baselines(band, samples, sky_map, rng=_ZeroRNG())
    np.testing.assert_allclose(samples.baselines[0, 0], _BASELINES)

    view = TODView(band, samples, compsep_output=sky_map).focus(0, band.scans[0].detectors[0])
    np.testing.assert_array_equal(view.raw_tod, raw_tod)
    np.testing.assert_allclose(view.get_tod(), _GAIN * sky_tod)


def test_baseline_sample_includes_c3_white_noise_fluctuation(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    band, samples, sky_map, _, _ = _build_hfi_case(phase=-1)
    samples.modulation_phase[0, 0] = -1
    samples.modulation_phase_initialized = True

    sample_hfi_baselines(band, samples, sky_map, rng=_SequenceRNG([1.5, -2.0]))

    expected = _BASELINES + np.array([1.5, -2.0]) * _SIGMA0 / np.sqrt(_NPAIR)
    np.testing.assert_allclose(samples.baselines[0, 0], expected)


def test_corrected_tod_applies_jumps_then_hfi_demodulation() -> None:
    band, samples, sky_map, raw_tod, _ = _build_hfi_case(phase=-1)
    samples.modulation_phase[0, 0] = -1
    samples.modulation_phase_initialized = True
    samples.baselines[0, 0] = _BASELINES
    jump = JumpCorrection(np.array([4]), np.array([3.0], dtype=np.float32))
    samples.jumps = SimpleNamespace(get=lambda iscan, idet: jump)

    expected = raw_tod.copy()
    expected[4:] += 3.0
    expected[0::2] = -(expected[0::2] - _BASELINES[0])
    expected[1::2] = expected[1::2] - _BASELINES[1]

    view = TODView(band, samples, compsep_output=sky_map).focus(0, band.scans[0].detectors[0])
    np.testing.assert_array_equal(view.raw_tod, raw_tod)
    np.testing.assert_allclose(view.corrected_tod, expected)
    np.testing.assert_allclose(view.get_tod(), expected)
