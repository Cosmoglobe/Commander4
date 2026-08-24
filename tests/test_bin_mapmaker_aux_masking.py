"""The binned mapmaker's aux maps must bin over the same good samples as the signal / cov maps.

Unlike the gap-filling CG path, ``tod2map_bin`` *drops* flagged samples: the signal and
inverse-variance maps accumulate only good samples, and every aux map (orbital dipole, corr-noise)
is normalized by that same good-sample inverse-variance cov. Accumulating flagged samples into an
aux map's numerator alone would therefore bias it at partially-flagged pixels (numerator over all
samples, denominator over good samples only).

This drives the real ``tod2map_bin`` and asserts the orbital-dipole aux map (and the signal / rms
maps) are invariant to extra *flagged* samples added at already-observed pixels -- i.e. flagged
samples do not leak into the aux maps. The corr-noise aux map is fixed on the same lines with the
same masking, so the orbital-dipole check guards both.
"""
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI

from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.scan_tod import ScanTOD
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.pointing import PixelPointing
import commander4.tod.processing as tod_processing

_BITMASK = 1        # one bad-data bit; a flagged sample has (flag & _BITMASK) != 0
_NSIDE = 1
_NPIX = 12 * _NSIDE**2
_NU = 30.0


def _build_band(pix: np.ndarray, psi: np.ndarray, flag: np.ndarray, tod: np.ndarray) -> DetectorGroupTOD:
    """Build an IQU band whose flag marks bad samples with uncompressed pointing."""
    ntod = pix.size
    pointing = PixelPointing(pix.astype(np.int64), psi.astype(np.float64), np.array([0], np.int64),
                             None, None, _NSIDE, _NSIDE, ntod, ntod)
    orbital_velocity = np.array([1.0, 0.2, -0.3], dtype=np.float32)
    det = DetectorTOD(
        name="d0", det_idx_fullband=0, tod=tod.astype(np.float32), pointing=pointing,
        sampling_rate_hz=1.0, orbital_velocity_m_per_s=orbital_velocity, huffman_tree=None,
        huffman_symbols=None, default_proc_mask=np.ones(_NPIX, bool), specific_proc_masks={},
        flag_encoded=flag.astype(np.int64), bad_data_bitmask=_BITMASK,
        flag_is_compressed=False,
    )
    noise_model = SimpleNamespace(npar=1, params=np.array([np.nan]))
    return DetectorGroupTOD([ScanTOD([det], 0.0, 0)], "EXP", "B", nside=_NSIDE, nu=_NU, fwhm=0.0,
                       fsamp=1.0, ndet=1, pols="IQU", noise_model=noise_model)


def _fake_tod_samples(sigma0: float = 2.0) -> SimpleNamespace:
    """Minimal stand-in exposing exactly the fields tod2map_bin / TODView / the diagnostics read."""
    no_jump = SimpleNamespace(is_empty=lambda: True)
    empty_ps = lambda: np.full((1, 1, 100), np.nan, dtype=np.float32)
    return SimpleNamespace(
        noise_params=np.full((1, 1, 1), sigma0), abs_gain=1.5, rel_gain=np.zeros(1),
        temporal_gain=np.zeros((1, 1)), jumps=SimpleNamespace(get=lambda iscan, idet: no_jump),
        accept=np.ones((1, 1), dtype=bool), band_unit_factor=1.0, band_unit="uK_RJ",
        chisq_z=np.full((1, 1), np.nan), good_fraction=np.full((1, 1), np.nan),
        TOD_PS_NBIN=100, tod_ps_freqs=empty_ps(), tod_ps_raw=empty_ps(), tod_ps_residual=empty_ps(),
        tod_ps_ncorrsub=empty_ps(), tod_ps_ncorr=empty_ps(), ncorr_tods=None)


def _run_bin_mapmaker(band: DetectorGroupTOD) -> dict[str, np.ndarray]:
    """Run binned mapmaking and return the maps selected for chain output."""
    mapmaking = tod_processing.MapmakingConfig(
        mapmaker="bin",
        num_threads=1,
        include_orbital_dipole_maps=True,
        include_corr_noise_maps=False,
        include_sky_model_maps=False,
        sparse_maps=False,
        common_res_fwhm=0.0,
    )
    correlated_noise = tod_processing.CorrelatedNoiseConfig(sample_sigma0=False)
    data_selection = tod_processing.DataSelectionConfig()
    _, maps = tod_processing.tod2map_bin(
        MPI.COMM_SELF, band, np.zeros((3, _NPIX)), _fake_tod_samples(), 1,
        mapmaking, correlated_noise, data_selection,
    )
    return maps


def test_bin_aux_maps_ignore_flagged_samples(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")  # get_s_orb_tod reads this.
    rng = np.random.default_rng(1)
    # Long enough for the log-binned diagnostic PSD and to condition the per-pixel IQU (3x3) solve.
    n = 256
    pix = rng.integers(0, _NPIX, n).astype(np.int64)
    psi = rng.uniform(0.0, np.pi, n)
    tod = rng.normal(size=n)
    good = _run_bin_mapmaker(_build_band(pix, psi, np.zeros(n), tod))

    # Same good samples, plus flagged samples carrying large garbage TOD at already-observed pixels.
    ne = 40
    pix_b = np.concatenate([pix, rng.integers(0, _NPIX, ne)]).astype(np.int64)
    psi_b = np.concatenate([psi, rng.uniform(0.0, np.pi, ne)])
    tod_b = np.concatenate([tod, rng.normal(size=ne) * 500.0])
    flag_b = np.concatenate([np.zeros(n, np.int64), np.full(ne, _BITMASK, np.int64)])
    with_flagged = _run_bin_mapmaker(_build_band(pix_b, psi_b, flag_b, tod_b))

    # All three maps must be unchanged: the flagged samples are dropped everywhere. An aux map that
    # binned them into its numerator alone would be biased against the good-sample cov that
    # normalizes it.
    for key in ("observed_sky", "rms", "orbdipole"):
        np.testing.assert_allclose(with_flagged[key], good[key], rtol=0, atol=1e-9,
                                   err_msg=f"{key} changed when flagged samples were added")
