"""Jump detection from the flag stream, the jump offset model, and its use in TODView."""
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.pointing import PixelPointing
from commander4.data_models.scan_tod import ScanTOD
from commander4.tod.config import JumpDetectionConfig
from commander4.tod.glitches.events import empty_glitch_grid
from commander4.tod.jumps.events import JumpEvents, empty_jump_grid
from commander4.tod.jumps.sampling import _detect_jumps, sample_jump_detection
from commander4.tod.view import TODView

_NTOD = 40
_JUMP_BIT = 2


def _step_case() -> tuple[np.ndarray, np.ndarray]:
    """A flat TOD that steps up by 5 at sample 12, with samples 10-11 flagged as the jump."""
    tod = np.zeros(_NTOD, dtype=np.float32)
    tod[12:] += 5.0
    flag = np.zeros(_NTOD, dtype=np.int64)
    flag[10:12] = _JUMP_BIT
    return tod, flag


def test_detect_jumps_measures_the_step_across_the_flagged_region() -> None:
    tod, flag = _step_case()
    flag[30] = 1  # Other flag bits are not jumps.

    jumps, num_skipped = _detect_jumps(tod, flag, np.ones(_NTOD, dtype=bool), 5, _JUMP_BIT)

    assert num_skipped == 0
    np.testing.assert_array_equal(jumps.locations, [12])
    np.testing.assert_allclose(jumps.offsets, [-5.0])


def test_detect_jumps_skips_a_jump_without_enough_valid_samples_before_it() -> None:
    tod, flag = _step_case()
    valid = np.ones(_NTOD, dtype=bool)
    valid[:8] = False  # Only samples 8-9 are valid before the jump, fewer than the window of 5.

    jumps, num_skipped = _detect_jumps(tod, flag, valid, 5, _JUMP_BIT)

    assert num_skipped == 1
    assert jumps.num_events == 0


def test_offset_tod_adds_each_offset_from_its_location_on() -> None:
    jumps = JumpEvents([3, 6], [1.0, 2.0])
    np.testing.assert_array_equal(jumps.offset_tod(8), [0, 0, 0, 1, 1, 1, 3, 3])


def _band_and_samples(tod: np.ndarray, flag: np.ndarray, pix: np.ndarray, gain: float):
    """A one-scan, one-detector intensity band at nside 1, and samples with only a gain."""
    pointing = PixelPointing(pix, np.zeros(_NTOD), np.array([0], dtype=np.int64), None, None,
                             1, 1, _NTOD, _NTOD)
    detector = DetectorTOD(
        name="det", det_idx_fullband=0, tod=tod, pointing=pointing, sampling_rate_hz=10.0,
        orbital_velocity_m_per_s=np.zeros(3, dtype=np.float32), huffman_tree=None,
        huffman_symbols=None, default_proc_mask=np.ones(12, dtype=bool), specific_proc_masks={},
        flag_encoded=flag, bad_data_bitmask=1, flag_is_compressed=False,
    )
    band = DetectorGroupTOD([ScanTOD([detector], 0.0, 1)], "EXP", "BAND", 1, 100.0, 10.0, 10.0, 1,
                            "I", SimpleNamespace(npar=1, params=np.array([1.0])))
    samples = SimpleNamespace(jumps=empty_jump_grid(1, 1), glitches=empty_glitch_grid(1, 1),
                              accept=np.ones((1, 1), dtype=bool), chain=1, abs_gain=gain,
                              rel_gain=np.zeros(1), temporal_gain=np.zeros((1, 1)))
    return band, samples, detector


_CONFIG = JumpDetectionConfig(enabled=True, window=5, jump_bitmask=_JUMP_BIT)


def test_sampled_jumps_are_removed_from_the_corrected_tod(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    tod, flag = _step_case()
    band, samples, detector = _band_and_samples(tod, flag, np.zeros(_NTOD, dtype=np.int64), 1.0)

    sample_jump_detection(MPI.COMM_SELF, band, samples, np.zeros((1, 12)), _CONFIG, iteration=1)

    np.testing.assert_array_equal(samples.jumps[0, 0].locations, [12])
    view = TODView(band, samples).focus(0, detector)
    np.testing.assert_allclose(view.corrected_tod, np.zeros(_NTOD))
    np.testing.assert_array_equal(view.raw_tod, tod)  # The stored data are never modified.


def test_a_sky_change_across_the_jump_is_not_taken_as_part_of_it(monkeypatch) -> None:
    """The scan moves from a pixel with sky 0 to one with sky 10 exactly where the jump is."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    gain = 2.0
    tod, flag = _step_case()
    pix = np.zeros(_NTOD, dtype=np.int64)
    pix[12:] = 1
    sky_map = np.zeros((1, 12))
    sky_map[0, 1] = 10.0
    tod = tod + gain * sky_map[0, pix].astype(np.float32)  # The raw step is 5 + 2*10 = 25.
    band, samples, _ = _band_and_samples(tod, flag, pix, gain)

    sample_jump_detection(MPI.COMM_SELF, band, samples, sky_map, _CONFIG, iteration=1)

    np.testing.assert_allclose(samples.jumps[0, 0].offsets, [-5.0])
