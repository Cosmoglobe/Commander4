import healpy as hp
import numpy as np
import pytest
from pixell.bunch import Bunch

from commander4.data_models.detector_map import DetectorMap
from commander4.file_io.map_reader import _resample_rms_map, read_data_map_from_file


def _file_band(tmp_path, **settings) -> tuple[Bunch, Bunch]:
    """An intensity file band at nside 1, and the parameter file around it."""
    signal_path, rms_path = str(tmp_path / "signal.fits"), str(tmp_path / "rms.fits")
    hp.write_map(signal_path, np.arange(12, dtype=np.float64), column_names=["TEMPERATURE"],
                 column_units="K_CMB", overwrite=True)
    hp.write_map(rms_path, np.full(12, 2.0), column_names=["TEMPERATURE"], overwrite=True)
    band = Bunch(enabled=True, get_from="file", polarization="I", freq=100.0, fwhm=60.0,
                 signal_map=Bunch(path=signal_path, dataset_names=["TEMPERATURE", "", ""]),
                 rms_map=Bunch(type="rms", path=rms_path, dataset_names=["TEMPERATURE", "", ""]),
                 **settings)
    object.__setattr__(band, "_name", "Planck100GHz")
    params = Bunch(compsep=Bunch(bands=Bunch(Planck100GHz=band)))
    return band, params


def test_file_maps_stay_in_their_band_unit(tmp_path) -> None:
    """The maps are not converted: CompSep brings the sky model to each band's unit instead."""
    band, params = _file_band(tmp_path, band_unit="uK_CMB")

    detmap = read_data_map_from_file(band, params)

    assert detmap.unit == "uK_CMB"
    np.testing.assert_allclose(detmap.map_sky[0], np.arange(12))
    np.testing.assert_allclose(detmap.map_rms[0], 2.0)


def test_a_file_band_must_state_its_unit(tmp_path) -> None:
    band, params = _file_band(tmp_path)
    with pytest.raises(ValueError, match="must set band_unit"):
        read_data_map_from_file(band, params)

    band, params = _file_band(tmp_path, units="uK_CMB")
    with pytest.raises(ValueError, match="renamed to 'band_unit'"):
        read_data_map_from_file(band, params)


def test_rms_resampling_preserves_total_inverse_variance() -> None:
    rms = np.full(hp.nside2npix(2), 2.0)

    downgraded = _resample_rms_map(rms, nside_out=1)
    upgraded = _resample_rms_map(downgraded, nside_out=2)

    np.testing.assert_allclose(downgraded, 1.0)
    np.testing.assert_allclose(upgraded, rms)
    assert np.sum(1.0/downgraded**2) == pytest.approx(np.sum(1.0/rms**2))


def test_detector_map_rejects_a_pixel_count_inconsistent_with_nside() -> None:
    with pytest.raises(ValueError, match="pixel count 13 does not match nside 1"):
        DetectorMap(np.zeros(13), np.ones(13), nu=30.0, unit="uK_RJ", fwhm=60.0, nside=1)
