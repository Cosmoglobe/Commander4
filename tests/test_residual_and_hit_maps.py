"""The binned mapmaker's residual and hit maps, Commander3's `tod_<freq>_res` and its companion.

`maps/res` is the noise residual `_record_tod_diagnostics` already builds per detector-scan (data
minus sky model, orbital dipole and correlated noise), binned with the same inverse-variance
weights as the signal map. So a TOD that is exactly `gain * sky` must bin to a residual of zero
while still producing the right signal map -- that is the property these tests pin.

`maps/nhit` counts unflagged samples per pixel, which no other output records: `maps/rms` counts
them weighted by `(gain/sigma0)^2`, so it cannot be inverted back into a sample count in general.
"""
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI

from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.scan_tod import ScanTOD
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.pointing import PixelPointing
import commander4.tod.processing as tod_processing

_BITMASK = 1
_NSIDE = 1
_NPIX = 12*_NSIDE**2
_GAIN = 1.5   # abs_gain below; rel and temporal gain are zero, so this is the whole gain.


def _build_band(pix, psi, tod, flag=None) -> DetectorGroupTOD:
    """One IQU detector-scan with uncompressed pointing and no orbital motion.

    The zero orbital velocity is what makes the expected residual exactly the noise: with no
    spacecraft velocity the orbital-dipole TOD the mapmaker subtracts is identically zero, so the
    only model term left is the sky projection.
    """
    ntod = pix.size
    pointing = PixelPointing(pix.astype(np.int64), psi.astype(np.float64), np.array([0], np.int64),
                             None, None, _NSIDE, _NSIDE, ntod, ntod)
    det = DetectorTOD(
        name="d0", det_idx_fullband=0, tod=tod.astype(np.float32), pointing=pointing,
        sampling_rate_hz=1.0, orbital_velocity_m_per_s=np.zeros(3, dtype=np.float32),
        huffman_tree=None, huffman_symbols=None, default_proc_mask=np.ones(_NPIX, bool),
        specific_proc_masks={},
        flag_encoded=(np.zeros(ntod) if flag is None else flag).astype(np.int64),
        bad_data_bitmask=_BITMASK, flag_is_compressed=False,
    )
    noise_model = SimpleNamespace(npar=1, params=np.array([np.nan]))
    return DetectorGroupTOD([ScanTOD([det], 0.0, 0)], "EXP", "B", nside=_NSIDE, nu=30.0, fwhm=0.0,
                            fsamp=1.0, ndet=1, pols="IQU", noise_model=noise_model)


def _fake_tod_samples(sigma0: float = 2.0) -> SimpleNamespace:
    """Minimal stand-in exposing exactly the fields tod2map_bin / TODView / the diagnostics read."""
    no_jump = SimpleNamespace(is_empty=lambda: True)
    empty_ps = lambda: np.full((1, 1, 100), np.nan, dtype=np.float32)
    return SimpleNamespace(
        noise_params=np.full((1, 1, 1), sigma0), abs_gain=_GAIN, rel_gain=np.zeros(1),
        temporal_gain=np.zeros((1, 1)), jumps=SimpleNamespace(get=lambda iscan, idet: no_jump),
        accept=np.ones((1, 1), dtype=bool), band_unit_factor=1.0, band_unit="uK_RJ",
        chisq_z=np.full((1, 1), np.nan), good_fraction=np.full((1, 1), np.nan),
        TOD_PS_NBIN=100, tod_ps_freqs=empty_ps(), tod_ps_raw=empty_ps(), tod_ps_residual=empty_ps(),
        tod_ps_ncorrsub=empty_ps(), tod_ps_ncorr=empty_ps(), ncorr_tods=None)


def _run(band: DetectorGroupTOD, sky_model: np.ndarray) -> dict[str, np.ndarray]:
    mapmaking = tod_processing.MapmakingConfig(
        mapmaker="bin", num_threads=1,
        include_orbital_dipole_maps=False, include_corr_noise_maps=False,
        include_sky_model_maps=False, include_residual_maps=True, include_hit_maps=True,
        sparse_maps=False, common_res_fwhm=0.0,
    )
    _, maps = tod_processing.tod2map_bin(
        MPI.COMM_SELF, band, sky_model, _fake_tod_samples(), 1, mapmaking,
        tod_processing.CorrelatedNoiseConfig(sample_sigma0=False),
        tod_processing.DataSelectionConfig(),
    )
    return maps


def _project(sky: np.ndarray, pix: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """The IQU sky along a pointing: I + Q cos(2 psi) + U sin(2 psi)."""
    return sky[0, pix] + sky[1, pix]*np.cos(2*psi) + sky[2, pix]*np.sin(2*psi)


def test_residual_map_is_zero_for_a_perfect_noiseless_model(monkeypatch):
    """A TOD that is exactly gain*sky leaves nothing behind, so maps/res must vanish."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    rng = np.random.default_rng(4)
    n = 256   # Long enough for the log-binned diagnostic PSD and the per-pixel IQU (3x3) solve.
    pix = rng.integers(0, _NPIX, n).astype(np.int64)
    psi = rng.uniform(0.0, np.pi, n)
    sky = rng.normal(scale=50.0, size=(3, _NPIX))

    maps = _run(_build_band(pix, psi, _GAIN*_project(sky, pix, psi)), sky)

    # float32 TODs at this signal amplitude, so compare at single precision.
    np.testing.assert_allclose(maps["res"], 0.0, atol=1e-3)
    np.testing.assert_allclose(maps["observed_sky"], sky, rtol=0, atol=1e-3)


def test_residual_map_recovers_an_injected_offset(monkeypatch):
    """Adding a constant to the intensity TOD must show up in maps/res, not just in the signal."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    rng = np.random.default_rng(5)
    n = 256
    pix = rng.integers(0, _NPIX, n).astype(np.int64)
    psi = rng.uniform(0.0, np.pi, n)
    sky = rng.normal(scale=50.0, size=(3, _NPIX))
    offset = 7.0   # In sky units, so the TOD gets gain*offset.

    maps = _run(_build_band(pix, psi, _GAIN*(_project(sky, pix, psi) + offset)), sky)

    # An unpolarized offset lands entirely in I; Q and U see it as a cos/sin average over psi.
    np.testing.assert_allclose(maps["res"][0], offset, rtol=1e-3)
    np.testing.assert_allclose(maps["observed_sky"][0], sky[0] + offset, rtol=1e-3)


def test_hit_map_counts_unflagged_samples_only(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    rng = np.random.default_rng(6)
    n, n_flagged = 256, 40
    pix = rng.integers(0, _NPIX, n + n_flagged).astype(np.int64)
    psi = rng.uniform(0.0, np.pi, n + n_flagged)
    flag = np.concatenate([np.zeros(n, np.int64), np.full(n_flagged, _BITMASK, np.int64)])
    sky = np.zeros((3, _NPIX))

    maps = _run(_build_band(pix, psi, np.zeros(n + n_flagged), flag=flag), sky)

    expected = np.bincount(pix[:n], minlength=_NPIX)
    np.testing.assert_array_equal(maps["nhit"], expected)
    assert maps["nhit"].sum() == n
