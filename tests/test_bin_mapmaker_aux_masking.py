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
from pixell.bunch import Bunch

from commander4.data_models.detector_TOD import DetectorTOD
from commander4.data_models.scan_TOD import ScanTOD
from commander4.data_models.detector_group_TOD import DetGroupTOD
from commander4.data_models.pointing import PixelPointing
import commander4.tod_processing as tod_processing

_BITMASK = 1        # one bad-data bit; a flagged sample has (flag & _BITMASK) != 0
_NSIDE = 1
_NPIX = 12 * _NSIDE**2
_NU = 30.0


def _build_band(pix: np.ndarray, psi: np.ndarray, flag: np.ndarray, tod: np.ndarray) -> DetGroupTOD:
    """Real one-detector, one-scan IQU band with `flag` marking bad samples (uncompressed pointing)."""
    ntod = pix.size
    pointing = PixelPointing(pix.astype(np.int64), psi.astype(np.float64), np.array([0], np.int64),
                             None, None, _NSIDE, _NSIDE, ntod, ntod)
    orb_dir_vec = np.array([1.0, 0.2, -0.3], dtype=np.float32)  # arbitrary spacecraft velocity
    det = DetectorTOD("d0", 0, 0, tod.astype(np.float32), pointing, 1.0, orb_dir_vec, None, None,
                      np.ones(_NPIX, bool), {}, ntod, ntod, flag_encoded=flag.astype(np.int64),
                      bad_data_bitmask=_BITMASK, flag_is_compressed=False)
    noise_model = SimpleNamespace(npar=1, params=np.array([np.nan]))
    return DetGroupTOD([ScanTOD([det], 0.0, 0)], "EXP", "B", nside=_NSIDE, nu=_NU, fwhm=0.0,
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


def _run_bin_mapmaker(band: DetGroupTOD, monkeypatch) -> dict[str, np.ndarray]:
    """Drive tod2map_bin (no ncorr, fixed sigma0) and capture the maps it hands to the chain writer."""
    params = Bunch(general=Bunch(common_res_fwhm=0.0, write_orb_dipole_maps_to_chain=True,
                                 write_corr_noise_maps_to_chain=False,
                                 write_sky_model_maps_to_chain=False),
                   experiments=Bunch(EXP=Bunch()))
    ncorr_cfg = Bunch(do_ncorr=False, sample_sigma0=False, do_param=False, sigma0_method="pairwise")
    dataselect_cfg = Bunch(enabled=False, active=False, chisq_abs_threshold=1.0e4,
                           min_good_fraction=0.1)
    captured: dict[str, np.ndarray] = {}
    monkeypatch.setattr(tod_processing, "write_map_chain_to_file",
                        lambda *a: captured.update({k: np.array(v, copy=True)
                                                    for k, v in a[5].items()}))
    tod_processing.tod2map_bin(MPI.COMM_SELF, band, np.zeros((3, _NPIX)), _fake_tod_samples(),
                               params, 0, 0, ncorr_cfg, dataselect_cfg)
    return captured


def test_bin_aux_maps_ignore_flagged_samples(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")  # get_s_orb_TOD reads this.
    rng = np.random.default_rng(1)
    # Long enough for the log-binned diagnostic PSD and to condition the per-pixel IQU (3x3) solve.
    n = 256
    pix = rng.integers(0, _NPIX, n).astype(np.int64)
    psi = rng.uniform(0.0, np.pi, n)
    tod = rng.normal(size=n)
    good = _run_bin_mapmaker(_build_band(pix, psi, np.zeros(n), tod), monkeypatch)

    # Same good samples, plus flagged samples carrying large garbage TOD at already-observed pixels.
    ne = 40
    pix_b = np.concatenate([pix, rng.integers(0, _NPIX, ne)]).astype(np.int64)
    psi_b = np.concatenate([psi, rng.uniform(0.0, np.pi, ne)])
    tod_b = np.concatenate([tod, rng.normal(size=ne) * 500.0])
    flag_b = np.concatenate([np.zeros(n, np.int64), np.full(ne, _BITMASK, np.int64)])
    with_flagged = _run_bin_mapmaker(_build_band(pix_b, psi_b, flag_b, tod_b), monkeypatch)

    # All three maps must be unchanged: the flagged samples are dropped everywhere. (Before the fix
    # the orbital-dipole and corr-noise maps binned flagged samples into the numerator only, biasing
    # them against the good-sample cov used to normalize them.)
    for key in ("map_observed_sky", "map_rms", "map_orbdipole"):
        np.testing.assert_allclose(with_flagged[key], good[key], rtol=0, atol=1e-9,
                                   err_msg=f"{key} changed when flagged samples were added")
