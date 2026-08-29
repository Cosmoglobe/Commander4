"""Regression test for overlapping point-source beam discs."""

import numpy as np

from commander4.sky.point_sources import _evaluate_sources_from_map, _project_sources_to_map


def test_overlapping_sources_accumulate_into_the_same_pixel() -> None:
    sky_map = np.zeros(2, dtype=np.float64)
    pixel_discs = [np.array([0]), np.array([0])]
    beam_discs = [np.array([1.0]), np.array([1.0])]
    amplitudes = np.array([2.0, 3.0])

    _project_sources_to_map(sky_map, pixel_discs, beam_discs, amplitudes)

    np.testing.assert_array_equal(sky_map, [5.0, 0.0])


def test_ragged_source_discs_are_evaluated_independently() -> None:
    sky_map = np.array([2.0, 4.0, 8.0])
    pixel_discs = [np.array([0]), np.array([1, 2])]
    beam_discs = [np.array([1.0]), np.array([0.25, 0.75])]
    amplitudes = np.zeros(2)

    _evaluate_sources_from_map(sky_map, pixel_discs, beam_discs, amplitudes)

    np.testing.assert_array_equal(amplitudes, [2.0, 7.0])
