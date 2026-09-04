"""Tests for the explicit DetectorTOD construction and pointing-owned lengths."""

from types import SimpleNamespace

import numpy as np
import pytest

from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.pointing import PixelPointing
from commander4.compression import huffman
from commander4.tod.sky_projection import get_s_orb_tod, get_static_sky_tod


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
    det_response: np.ndarray | None = None,
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
        det_response=det_response,
    )


def test_lengths_are_derived_from_pointing() -> None:
    detector = _detector(_pointing(ntod_original=8, ntod=6))

    assert detector.ntod_original == 8
    assert detector.ntod == 6
    assert detector.tod.shape == (6,)
    assert detector.flag.shape == (6,)


def test_compressed_psi_bins_decode_to_bin_centers() -> None:
    bins = np.array([1, 2, 8, 4], dtype=np.int64)
    differences = huffman.preproc_diff(bins)
    tree, symbols, codes, lengths = huffman.build_huffman_tree([differences])
    encoded = huffman.huffman_compress_array(differences, codes, lengths)
    pointing = PixelPointing(
        np.arange(bins.size), encoded, tree, symbols, npsi=8, nside=1, data_nside=1,
        ntod_original=bins.size, ntod=bins.size,
    )

    bin_width = np.float32(2*np.pi/8)
    expected = (bins.astype(np.float32) - np.float32(0.5))*bin_width

    np.testing.assert_array_equal(pointing.get_psi(), expected)


def test_physical_arguments_have_explicit_units() -> None:
    velocity = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    detector = _detector(_pointing(), velocity)

    assert detector.fsamp == 32.5
    assert detector.orbital_velocity_m_per_s.dtype == np.float32
    np.testing.assert_array_equal(detector.orbital_velocity_m_per_s, velocity)


def test_missing_orbital_velocity_produces_zero_orbital_dipole() -> None:
    detector = _detector(_pointing())
    experiment = SimpleNamespace(nu=30.0, nside=1)
    pixels = np.arange(detector.ntod, dtype=np.int64)

    orbital_dipole = get_s_orb_tod(detector, experiment, pixels, nthreads=1)

    np.testing.assert_array_equal(orbital_dipole, np.zeros(detector.ntod, dtype=np.float32))


def test_static_sky_projection_skips_inactive_response_components() -> None:
    pixels = np.array([0, 1, 2], dtype=np.int64)
    psi = np.array([0.1, 0.4, 0.8])
    sky = np.arange(36, dtype=np.float64).reshape(3, 12)
    cos_2psi = np.cos(2 * psi)
    sin_2psi = np.sin(2 * psi)

    intensity_sky = sky.copy()
    intensity_sky[1:3] = np.nan
    intensity = get_static_sky_tod(
        intensity_sky, pixels, psi, response=np.array([1.0, 0.0]),
    )
    np.testing.assert_allclose(intensity, sky[0, pixels])

    polarization_sky = sky.copy()
    polarization_sky[0] = np.nan
    polarization = get_static_sky_tod(
        polarization_sky, pixels, psi, response=np.array([0.0, 1.0]),
    )
    expected_polarization = sky[1, pixels] * cos_2psi + sky[2, pixels] * sin_2psi
    np.testing.assert_allclose(polarization, expected_polarization, rtol=1e-6)

    zero = get_static_sky_tod(
        np.full_like(sky, np.nan), pixels, psi, response=np.array([0.0, 0.0]),
    )
    np.testing.assert_array_equal(zero, 0.0)

    general_response = np.array([0.25, 0.75])
    general = get_static_sky_tod(sky, pixels, psi, response=general_response)
    expected_general = (general_response[0] * sky[0, pixels]
                        + general_response[1] * expected_polarization)
    np.testing.assert_allclose(general, expected_general, rtol=1e-6)


def test_orbital_dipole_applies_intensity_response() -> None:
    pointing = _pointing()
    velocity = np.array([1000.0, 2000.0, 3000.0])
    experiment = SimpleNamespace(nu=30.0, nside=1)
    pixels = np.arange(pointing.ntod, dtype=np.int64)
    standard = get_s_orb_tod(
        _detector(pointing, velocity), experiment, pixels, nthreads=1,
    )
    intensity_only = get_s_orb_tod(
        _detector(pointing, velocity, np.array([1.0, 0.0])), experiment, pixels, nthreads=1,
    )
    polarization_only = get_s_orb_tod(
        _detector(pointing, velocity, np.array([0.0, 1.0])), experiment, pixels, nthreads=1,
    )
    scaled = get_s_orb_tod(
        _detector(pointing, velocity, np.array([0.25, 1.0])), experiment, pixels, nthreads=1,
    )

    np.testing.assert_allclose(intensity_only, standard)
    np.testing.assert_array_equal(polarization_only, 0.0)
    np.testing.assert_allclose(scaled, 0.25 * standard)


def test_inplace_sim_orbital_dipole_applies_intensity_response() -> None:
    from commander4.simulations.inplace_litebird_sim import get_orbital_dipole
    import pysm3.units as units

    pointing = _pointing()
    velocity = np.array([1000.0, 2000.0, 3000.0])
    pixels = np.arange(pointing.ntod, dtype=np.int64)
    standard = get_orbital_dipole(_detector(pointing, velocity), pixels, 30.0, units.uK_RJ)
    polarization_only = get_orbital_dipole(
        _detector(pointing, velocity, np.array([0.0, 1.0])), pixels, 30.0, units.uK_RJ,
    )

    assert np.any(standard != 0.0)
    np.testing.assert_array_equal(polarization_only, 0.0)


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
