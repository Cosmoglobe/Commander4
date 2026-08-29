"""Regression test for overlapping point-source beam discs."""

import numpy as np

from commander4.sky.point_sources import _numba_proj2map


def test_overlapping_sources_accumulate_into_the_same_pixel() -> None:
    sky_map = np.zeros(2, dtype=np.float64)
    pixel_discs = [np.array([0]), np.array([0])]
    beam_values = [np.array([1.0]), np.array([1.0])]
    amplitudes = np.array([2.0, 3.0])

    _numba_proj2map(sky_map, pixel_discs, beam_values, amplitudes)

    np.testing.assert_array_equal(sky_map, [5.0, 0.0])
