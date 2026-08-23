"""Tests for the explicit DetectorTOD construction and pointing-owned lengths."""

from types import SimpleNamespace

import numpy as np
import pytest

from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.pointing import PixelPointing
from commander4.tod.sky_projection import get_s_orb_tod


def _pointing(ntod_original: int = 8, ntod: int = 6) -> PixelPointing:
    pixels = np.arange(ntod_original, dtype=np.int64)
    psi = np.linspace(0.0, np.pi, ntod_original, dtype=np.float64)
    return PixelPointing(
        pixels, psi, np.array([0], dtype=np.int64), None, None,
        nside=1, data_nside=1, ntod_original=ntod_original, ntod=ntod,
    )


def _detector(
    pointing: PixelPointing,
    orbital_velocity_m_per_s: np.ndarray | None = None,
) -> DetectorTOD:
    return DetectorTOD(
        name="detector",
        det_idx_fullband=2,
        tod=np.arange(pointing.ntod_original, dtype=np.float32),
        pointing=pointing,
        sampling_rate_hz=32.5,
        orbital_velocity_m_per_s=orbital_velocity_m_per_s,
        huffman_tree=None,
        huffman_symbols=None,
        default_proc_mask=None,
        specific_proc_masks={},
        flag_encoded=np.zeros(pointing.ntod_original, dtype=np.int64),
        bad_data_bitmask=1,
        flag_is_compressed=False,
    )


def test_lengths_are_derived_from_pointing() -> None:
    detector = _detector(_pointing(ntod_original=8, ntod=6))

    assert detector.ntod_original == 8
    assert detector.ntod == 6
    assert detector.tod.shape == (6,)
    assert detector.flag.shape == (6,)


def test_constructor_keeps_only_the_full_band_detector_index() -> None:
    detector = _detector(_pointing())

    assert detector.det_idx_fullband == 2
    assert not hasattr(detector, "det_idx_local")


def test_physical_arguments_have_explicit_units() -> None:
    velocity = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    detector = _detector(_pointing(), velocity)

    assert detector.fsamp == 32.5
    assert detector.orbital_velocity_m_per_s.dtype == np.float32
    np.testing.assert_array_equal(detector.orbital_velocity_m_per_s, velocity)


def test_missing_orbital_velocity_is_returned_as_none() -> None:
    detector = _detector(_pointing())

    assert detector.orbital_velocity_m_per_s is None


def test_missing_orbital_velocity_produces_zero_orbital_dipole() -> None:
    detector = _detector(_pointing())
    experiment = SimpleNamespace(nu=30.0, nside=1)
    pixels = np.arange(detector.ntod, dtype=np.int64)

    orbital_dipole = get_s_orb_tod(detector, experiment, pixels, nthreads=1)

    np.testing.assert_array_equal(orbital_dipole, np.zeros(detector.ntod, dtype=np.float32))


def test_orbital_velocity_must_have_three_entries() -> None:
    with pytest.raises(ValueError, match="vector of size 3"):
        _detector(_pointing(), np.ones(2))


def test_constructor_is_keyword_only() -> None:
    pointing = _pointing()
    with pytest.raises(TypeError):
        DetectorTOD(
            "detector", 0, np.zeros(8, dtype=np.float32), pointing, 32.5, None,
            None, None, None, {},
        )
