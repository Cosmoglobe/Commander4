"""Smoothing a band to a common resolution (`general.common_res_fwhm`) on partial sky coverage.

A band that covers only part of the sky -- any ground-based TOD band -- has zero inverse-noise
weight outside its footprint, and far enough outside it the *smoothed* weight is zero as well.
Dividing the smoothed signal by that weight used to produce NaN there, and since the per-pixel
solver converts its result to alms, a single NaN pixel turned every component's alms into NaN.
Unobserved pixels have to stay unobserved: zero signal, zero weight.
"""
import healpy as hp
import numpy as np

from commander4.data_models.detector_map import (DetectorMap, smooth_inv_var_map_noiseweighted,
                                                 smooth_signal_map_noiseweighted)

NSIDE = 32
NPIX = hp.nside2npix(NSIDE)


def _patch_mask(radius_deg: float = 20.0) -> np.ndarray:
    """Boolean map: a disc of the given radius around the north pole."""
    vec = hp.pix2vec(NSIDE, np.arange(NPIX))
    return np.array(vec[2]) > np.cos(np.deg2rad(radius_deg))


def _partial_sky_maps(rms_in_patch: float = 10.0):
    """(signal, inv_var) for a band observing only a disc; unobserved pixels have zero weight."""
    np.random.seed(4)
    observed = _patch_mask()
    inv_var = np.zeros(NPIX)
    inv_var[observed] = 1.0/rms_in_patch**2
    signal = np.zeros(NPIX)
    signal[observed] = np.random.normal(0.0, 100.0, observed.sum())
    return signal, inv_var, observed


def _smoothed_weight(inv_var: np.ndarray, fwhm: float) -> np.ndarray:
    """The weight map the smoothing normalizes by; its sign decides which pixels are observed.

    Beam ringing around a sharp coverage edge drives this slightly negative over a large part of
    the sphere -- roughly half of it for the patch used here -- which is what the guards are for.
    """
    return hp.smoothing(inv_var, fwhm=fwhm)


def test_partial_sky_signal_smoothing_stays_finite():
    signal, inv_var, observed = _partial_sky_maps()
    fwhm = np.deg2rad(3.0)

    smoothed = smooth_signal_map_noiseweighted(signal, inv_var, fwhm)

    assert np.isfinite(smoothed).all(), "smoothed signal must not contain NaN/inf"
    # Pixels the smoothing does not reach with positive weight carry no signal at all.
    unreached = _smoothed_weight(inv_var, fwhm) <= 0.0
    assert unreached.sum() > 0.1*NPIX, "expected the guard to be exercised over much of the sphere"
    np.testing.assert_array_equal(smoothed[unreached], 0.0)
    # Inside the patch the smoothing did something and kept the signal scale.
    assert np.std(smoothed[observed]) > 0.0
    assert np.std(smoothed[observed]) < np.std(signal[observed])


def test_partial_sky_inv_var_smoothing_marks_unobserved_pixels():
    """Outside the footprint the inverse variance must be exactly zero, never negative or NaN."""
    _, inv_var, observed = _partial_sky_maps()
    fwhm = np.deg2rad(3.0)

    smoothed_inv_var = smooth_inv_var_map_noiseweighted(inv_var, fwhm)

    assert np.isfinite(smoothed_inv_var).all()
    np.testing.assert_array_equal(smoothed_inv_var[_smoothed_weight(inv_var, fwhm) <= 0.0], 0.0)
    # Smoothing averages down the noise inside the patch, raising the inverse variance above the
    # input 1/10^2, but never to infinity.
    deep = np.array(hp.pix2vec(NSIDE, np.arange(NPIX))[2]) > np.cos(np.deg2rad(10.0))
    assert np.median(smoothed_inv_var[deep]) > 1.0/10.0**2


def test_smoothing_a_full_sky_band_is_unchanged():
    """The guards must be a no-op where every pixel is observed, which is the full-sky case."""
    np.random.seed(11)
    inv_var = np.full(NPIX, 1.0/10.0**2)
    signal = np.random.normal(0.0, 100.0, NPIX)
    fwhm = np.deg2rad(3.0)

    # Reference: the plain expressions, valid because every smoothed weight is positive here.
    weight = hp.smoothing(inv_var, fwhm=fwhm)
    assert (weight > 0).all()
    expected_signal = hp.smoothing(signal*inv_var, fwhm=fwhm)/weight

    np.testing.assert_allclose(smooth_signal_map_noiseweighted(signal, inv_var, fwhm),
                               expected_signal)
    assert (smooth_inv_var_map_noiseweighted(inv_var, fwhm) > 0).all()


def test_detector_map_smooth_to_resolution_on_a_patch():
    """The end-to-end path the parameter `general.common_res_fwhm` drives."""
    signal, inv_var, observed = _partial_sky_maps()
    detmap = DetectorMap(map_sky=signal[None, :], map_inv_var=inv_var[None, :], nu=90.0, fwhm=30.0,
                         nside=NSIDE)

    detmap.smooth_to_resolution(60.0)

    assert detmap.fwhm == 60.0
    assert np.isfinite(detmap.map_sky).all(), "signal map must not go NaN outside the footprint"
    assert np.isfinite(detmap.map_inv_var).all(), "unobserved pixels must have zero weight, not NaN"
    assert (detmap.map_inv_var >= 0).all()
    assert detmap.map_inv_var[0][observed].max() > 0.0


def test_smoothing_to_a_finer_beam_is_refused(caplog):
    """Smoothing only coarsens; a finer target leaves the band alone rather than sharpening it."""
    signal, inv_var, _ = _partial_sky_maps()
    detmap = DetectorMap(map_sky=signal[None, :], map_inv_var=inv_var[None, :], nu=90.0, fwhm=30.0,
                         nside=NSIDE)
    before = detmap.map_sky.copy()

    detmap.smooth_to_resolution(10.0)

    assert detmap.fwhm == 30.0
    np.testing.assert_array_equal(detmap.map_sky, before)
    assert "finer" in caplog.text
