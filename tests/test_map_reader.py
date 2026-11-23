from pathlib import Path

import healpy as hp
import numpy as np
import pytest
from astropy.io import fits

from commander4.data_models.detector_map import DetectorMap
from commander4.file_io.map_reader import read_data_map_from_file
from commander4.parameters.bunch import as_bunch_recursive


def _read_band(path: Path, rms: np.ndarray, eval_nside: int | None = None,
               sky: np.ndarray | None = None) -> DetectorMap:
    """Write a sky map (default: ones) and `rms` to a FITS table, and read them back as one I-only
    band."""
    sky = np.ones(rms.size) if sky is None else sky
    columns = [fits.Column(name="I_STOKES", format="D", unit="uK_RJ", array=sky),
               fits.Column(name="I_RMS", format="D", unit="uK_RJ", array=rms)]
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.header["ORDERING"] = "RING"
    hdu.writeto(path, overwrite=True)
    band = {"polarization": "I", "freq": 30.0, "fwhm": 60.0,
            "signal_map": {"path": str(path), "dataset_names": ["I_STOKES", "", ""]},
            "rms_map": {"path": str(path), "dataset_names": ["I_RMS", "", ""]}}
    if eval_nside is not None:
        band["eval_nside"] = eval_nside
    params = as_bunch_recursive({"compsep": {"bands": {"b30": {}}}})
    return read_data_map_from_file(as_bunch_recursive(band, name="b30"), "I", params)


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


def test_downgrading_takes_the_inverse_variance_weighted_mean(tmp_path: Path) -> None:
    """A merged pixel holds the mean of its sub-pixels weighted by 1/RMS^2, the combination whose
    inverse variance is the summed one. An unobserved sub-pixel, NaN signal included, is ignored.
    So is an UNSEEN or NaN signal pixel whose RMS is positive."""
    rng = np.random.default_rng(3)
    sky = rng.normal(0.0, 10.0, hp.nside2npix(2))
    rms = rng.uniform(1.0, 5.0, sky.size)
    sky[7], rms[7] = np.nan, 0.0  # An unobserved pixel with a bad signal value.
    sky[[11, 20]] = [hp.UNSEEN, np.nan]  # Bad signal values with a valid RMS.

    downgraded = _read_band(tmp_path / "band.fits", rms, eval_nside=1, sky=sky)

    # In NEST ordering the four nside-2 sub-pixels of nside-1 pixel p are 4p, ..., 4p + 3.
    weight = np.divide(1.0, rms**2, out=np.zeros_like(rms), where=rms > 0)
    weight[[11, 20]] = 0.0
    sky_nest = hp.reorder(np.where(weight > 0, sky, 0.0), r2n=True).reshape(-1, 4)
    weight_nest = hp.reorder(weight, r2n=True).reshape(-1, 4)
    expected = hp.reorder((sky_nest*weight_nest).sum(axis=1)/weight_nest.sum(axis=1), n2r=True)
    np.testing.assert_allclose(downgraded.map_sky[0], expected, rtol=1e-5)
    np.testing.assert_allclose(downgraded.map_inv_var[0],
                               hp.reorder(weight_nest.sum(axis=1), n2r=True), rtol=1e-5)


def test_detector_map_rejects_a_pixel_count_inconsistent_with_nside() -> None:
    with pytest.raises(ValueError, match="pixel count 13 does not match nside 1"):
        DetectorMap(np.zeros(13), np.ones(13), nu=30.0, fwhm=60.0, nside=1)
