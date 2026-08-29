"""Regression test for overlapping point-source beam discs."""

import numpy as np

from commander4.sky.point_sources import _numba_eval_from_map, _numba_proj2map


def test_overlapping_sources_accumulate_into_the_same_pixel() -> None:
    sky_map = np.zeros(2, dtype=np.float64)
    beam_pixels = np.array([0, 0], dtype=np.int64)
    beam_values = np.array([1.0, 1.0])
    beam_offsets = np.array([0, 1, 2], dtype=np.int64)
    amplitudes = np.array([2.0, 3.0])

    _numba_proj2map(sky_map, beam_pixels, beam_values, beam_offsets, amplitudes)

    np.testing.assert_array_equal(sky_map, [5.0, 0.0])


def test_flat_beam_offsets_preserve_ragged_source_discs() -> None:
    sky_map = np.array([2.0, 4.0, 8.0])
    beam_pixels = np.array([0, 1, 2], dtype=np.int64)
    beam_values = np.array([1.0, 0.25, 0.75])
    beam_offsets = np.array([0, 1, 3], dtype=np.int64)
    amplitudes = np.zeros(2)

    _numba_eval_from_map(sky_map, beam_pixels, beam_values, beam_offsets, amplitudes)

    np.testing.assert_array_equal(amplitudes, [2.0, 7.0])
