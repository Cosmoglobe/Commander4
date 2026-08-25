"""Tests for selecting fast TOD lengths with ducc0's built-in FFT size model."""

import ducc0.fft
import pytest

from commander4.file_io.experiments import read_utils
from commander4.file_io.experiments.read_utils import find_good_fourier_size


@pytest.mark.parametrize("ntod", [2, 17, 64, 100, 9750, 400_000])
def test_selected_size_is_the_closest_smaller_good_real_fft_size(ntod: int) -> None:
    selected = find_good_fourier_size(ntod)

    assert selected < ntod
    assert ducc0.fft.good_size(selected, True) == selected
    for candidate in range(selected + 1, ntod):
        assert ducc0.fft.good_size(candidate, True) != candidate


def test_size_search_walks_downward_and_requests_a_real_fft(monkeypatch) -> None:
    calls = []

    def fake_good_size(candidate: int, real: bool) -> int:
        calls.append((candidate, real))
        next_good_size = {9: 10, 8: 9, 7: 7}
        return next_good_size[candidate]

    monkeypatch.setattr(read_utils.ducc0.fft, "good_size", fake_good_size)

    assert find_good_fourier_size(10) == 7
    assert calls == [(9, True), (8, True), (7, True)]


@pytest.mark.parametrize("ntod", [-1, 0, 1])
def test_size_search_rejects_lengths_without_a_smaller_positive_fft(ntod: int) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        find_good_fourier_size(ntod)
