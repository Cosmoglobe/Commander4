from pathlib import Path

import healpy as hp
import numpy as np
import pytest
from astropy.io import fits

from commander4.data_models.detector_map import DetectorMap
from commander4.file_io.map_reader import read_data_map_from_file
from commander4.parameters.bunch import as_bunch_recursive


def _read_band(path: Path, rms: np.ndarray, eval_nside: int | None = None) -> DetectorMap:
    """Write a unit sky map and `rms` to a FITS table, and read them back as one I-only band."""
    columns = [fits.Column(name="I_STOKES", format="E", unit="uK_RJ", array=np.ones(rms.size)),
               fits.Column(name="I_RMS", format="E", unit="uK_RJ", array=rms)]
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.header["ORDERING"] = "RING"
    hdu.writeto(path, overwrite=True)
    band = {"polarization": "I", "freq": 30.0, "fwhm": 60.0,
            "signal_map": {"path": str(path), "dataset_names": ["I_STOKES", "", ""]},
            "rms_map": {"path": str(path), "dataset_names": ["I_RMS", "", ""]}}
    if eval_nside is not None:
        band["eval_nside"] = eval_nside
    params = as_bunch_recursive({"compsep": {"bands": {"b30": {}}}})
    return read_data_map_from_file(as_bunch_recursive(band, name="b30"), params)


def test_rms_maps_are_read_as_inverse_variance(tmp_path: Path) -> None:
    """RMS becomes 1/RMS^2, an RMS of 0 or inf means unobserved, and downgrading adds weights."""
    rms = np.full(hp.nside2npix(2), 2.0)
    rms[[5, 9]] = [0.0, np.inf]

    detmap = _read_band(tmp_path / "band.fits", rms)
    expected = np.full(rms.size, 0.25)
    expected[[5, 9]] = 0.0
    np.testing.assert_array_equal(detmap.map_inv_var[0], expected)

    downgraded = _read_band(tmp_path / "band.fits", rms, eval_nside=1)
    assert downgraded.map_inv_var.shape == (1, hp.nside2npix(1))
    assert downgraded.map_inv_var.sum() == pytest.approx(expected.sum())
    assert (downgraded.map_inv_var > 0).all()


def test_detector_map_rejects_a_pixel_count_inconsistent_with_nside() -> None:
    with pytest.raises(ValueError, match="pixel count 13 does not match nside 1"):
        DetectorMap(np.zeros(13), np.ones(13), nu=30.0, fwhm=60.0, nside=1)
