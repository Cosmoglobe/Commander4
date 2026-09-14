"""Tests for selecting fast TOD lengths with ducc0's built-in FFT size model."""

import ducc0.fft
import pytest

from commander4.file_io.experiments.read_utils import find_good_fourier_size


@pytest.mark.parametrize("ntod", [2, 17, 64, 100, 9750, 400_000])
def test_selected_size_is_the_closest_good_real_fft_size_at_most_ntod(ntod: int) -> None:
    selected = find_good_fourier_size(ntod)

    assert selected <= ntod
    assert ducc0.fft.good_size(selected, True) == selected
    for candidate in range(selected + 1, ntod + 1):
        assert ducc0.fft.good_size(candidate, True) != candidate


@pytest.mark.parametrize("ntod", [64, 100, 995_328, 1_296_000])
def test_an_already_fast_length_keeps_every_sample(ntod: int) -> None:
    # Commander3's get_closest_fft_magic_number returns n itself in this case; dropping to the
    # next size down would discard up to ~1% of a scan for nothing.
    assert ducc0.fft.good_size(ntod, True) == ntod          # precondition of this test
    assert find_good_fourier_size(ntod) == ntod


@pytest.mark.parametrize("ntod", [-1, 0, 1])
def test_size_search_rejects_lengths_without_a_positive_fft(ntod: int) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        find_good_fourier_size(ntod)
