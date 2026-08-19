"""Tests for lightweight simgen diagnostic map products."""
import os
import sys
from types import SimpleNamespace

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "aux"))

from simgen.diagnostics import (hit_map, noise_map, noise_map_rhs, white_noise_normal_matrix,
                                write_band_diagnostics)


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