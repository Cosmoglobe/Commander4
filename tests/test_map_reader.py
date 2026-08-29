import healpy as hp
import numpy as np
import pytest

from commander4.data_models.detector_map import DetectorMap
from commander4.file_io.map_reader import _resample_rms_map


def test_rms_resampling_preserves_total_inverse_variance() -> None:
    rms = np.full(hp.nside2npix(2), 2.0)

    downgraded = _resample_rms_map(rms, nside_out=1)
    upgraded = _resample_rms_map(downgraded, nside_out=2)

    np.testing.assert_allclose(downgraded, 1.0)
    np.testing.assert_allclose(upgraded, rms)
    assert np.sum(1.0/downgraded**2) == pytest.approx(np.sum(1.0/rms**2))


def test_detector_map_rejects_a_pixel_count_inconsistent_with_nside() -> None:
    with pytest.raises(ValueError, match="pixel count 13 does not match nside 1"):
        DetectorMap(np.zeros(13), np.ones(13), nu=30.0, fwhm=60.0, nside=1)
