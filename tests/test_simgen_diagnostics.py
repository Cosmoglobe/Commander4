"""Tests for lightweight simgen diagnostic map products."""
from types import SimpleNamespace

import h5py
import numpy as np

from simgen.diagnostics import (hit_map, noise_map, noise_map_rhs, normal_matrix_rms,
                                white_noise_normal_matrix, write_band_diagnostics)


def test_normal_matrix_rms_is_an_rms_not_a_variance():
    """N samples of white noise at sigma0 must give sigma0/sqrt(N), not its square."""
    nhit, sigma0 = 16, 3.0
    band = SimpleNamespace(name="B", eval_nside=1, polarization="I",
                           detectors=[SimpleNamespace(name="det", sigma0=sigma0)])
    det_pix = {"det": np.zeros(nhit, dtype=np.int64)}
    det_psi = {"det": np.zeros(nhit)}
    normal = white_noise_normal_matrix(band, det_pix, det_psi)

    rms = normal_matrix_rms(normal, "I")
    np.testing.assert_allclose(rms[0, 0], sigma0 / np.sqrt(nhit))


def test_normal_matrix_rms_uses_the_full_iqu_inverse():
    """Poor polarization-angle coverage must inflate the RMS above the naive 1/sqrt(diagonal)."""
    sigma0 = 1.0
    band = SimpleNamespace(name="B", eval_nside=1, polarization="IQU",
                           detectors=[SimpleNamespace(name="det", sigma0=sigma0)])
    # Three angles, but unevenly spread, so I, Q and U are correlated rather than independent.
    psi = np.deg2rad(np.array([0.0, 30.0, 75.0]))
    normal = white_noise_normal_matrix(band, {"det": np.zeros(3, dtype=np.int64)}, {"det": psi})

    rms = normal_matrix_rms(normal, "IQU")[:, 0]
    matrix = np.array([[normal[0, 0], normal[1, 0], normal[2, 0]],
                       [normal[1, 0], normal[3, 0], normal[4, 0]],
                       [normal[2, 0], normal[4, 0], normal[5, 0]]])
    np.testing.assert_allclose(rms, np.sqrt(np.diagonal(np.linalg.inv(matrix))))
    assert np.all(rms > 1.0 / np.sqrt(np.diagonal(matrix)))


def test_realized_noise_diagnostic_map(tmp_path):
    """The binned one-pixel noise map is the inverse-variance weighted TOD average."""
    band = SimpleNamespace(
        name="B", eval_nside=1, polarization="I", units="uK_RJ", freq=30.0,
        fwhm_arcmin=30.0, detectors=[SimpleNamespace(name="det", sigma0=1.0)],
    )
    det_pix = {"det": np.array([0, 0], dtype=np.int64)}
    det_psi = {"det": np.zeros(2)}
    det_noise = {"det": np.array([3.0, 5.0])}
    normal = white_noise_normal_matrix(band, det_pix, det_psi)
    rhs = noise_map_rhs(band, det_pix, det_psi, det_noise)

    np.testing.assert_allclose(noise_map(normal, rhs, "I")[0, 0], 4.0)

    write_band_diagnostics(str(tmp_path), band, np.zeros((1, 12)), hit_map(band, det_pix), normal,
                           rhs)
    with h5py.File(tmp_path / "B_diagnostics.h5", "r") as infile:
        np.testing.assert_allclose(infile["maps/noise"][0, 0], 4.0)
    assert (tmp_path / "B_noise_I.png").is_file()
