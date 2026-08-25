"""The CG mapmaker on a partial sky: unobserved and degenerate pixels must not poison the solve.

A patch experiment leaves most of the sky unhit and, along the ragged coverage edge, leaves pixels
that *are* hit but only over a narrow range of polarization angles. The second kind is the dangerous
one: a pixel seen only at psi = 45 deg has ``A_QQ = sum w*cos^2(2 psi)`` at rounding level (~1e-33,
not exactly 0) while ``A_II`` and ``A_UU`` are large, so a diagonal preconditioner built as
``1/diag(A)`` hands the CG an entry of ~1e33 and every dot product in the iteration is swamped by
that one pixel. The block-Jacobi preconditioner inverts the whole 3x3 instead and zeroes the pixels
whose 3x3 is singular, which projects them out of the solve.

These tests drive the real ``tod2map_CG`` on a small patch and check that the solved map is zero
exactly where the rms is ``+inf``, and that the well-measured pixels reproduce ``tod2map_bin`` -- with
an identity transfer function the two mapmakers solve the same normal equations, so they must agree.
"""
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI

from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.scan_tod import ScanTOD
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.pointing import PixelPointing
from commander4.tod.mapmaking.preconditioners import BlockInvNPreconditionerIQU,\
    InvNPreconditionerIQU
from commander4.tod.step_config import CGConfig
import commander4.tod.processing as tod_processing

_NSIDE = 4
_NPIX = 12 * _NSIDE**2
_SIGMA0 = 2.0
_N_GOOD_PIX = 30        # pixels 0..29 are seen at four polarization angles
_EDGE_PIX = 30          # seen only at psi = 45 deg, so Q is unconstrained
_NHIT = 24              # samples per observed pixel


def _patch_pointing(pols: str) -> tuple[np.ndarray, np.ndarray]:
    """Pointing over a small patch: well-covered pixels, one degenerate edge pixel, rest unhit."""
    good = np.repeat(np.arange(_N_GOOD_PIX), _NHIT)
    psi_good = np.tile(np.array([0.0, 0.25, 0.5, 0.75]) * np.pi, good.size // 4)
    if pols == "I":
        return good, np.zeros(good.size)
    edge = np.full(_NHIT, _EDGE_PIX)
    return np.concatenate([good, edge]), np.concatenate([psi_good, np.full(_NHIT, 0.25 * np.pi)])


def _build_band(pix: np.ndarray, psi: np.ndarray, tod: np.ndarray, pols: str,
                velocity: tuple[float, float, float] = (1.0, 0.2, -0.3)) -> DetectorGroupTOD:
    """One-detector, one-scan band with uncompressed pointing and no flagged samples.

    `velocity` is the spacecraft velocity direction driving the orbital dipole the mapmakers
    subtract; set it to zero for a test that wants the map to be the plain binned TOD.
    """
    ntod = pix.size
    pointing = PixelPointing(pix.astype(np.int64), psi.astype(np.float64), np.array([0], np.int64),
                             None, None, _NSIDE, _NSIDE, ntod, ntod)
    orbital_velocity = np.array(velocity, dtype=np.float32)
    det = DetectorTOD(
        name="d0", det_idx_fullband=0, tod=tod.astype(np.float32), pointing=pointing,
        sampling_rate_hz=1.0, orbital_velocity_m_per_s=orbital_velocity, huffman_tree=None,
        huffman_symbols=None, default_proc_mask=np.ones(_NPIX, bool), specific_proc_masks={},
        flag_encoded=np.zeros(ntod, np.int64), bad_data_bitmask=1, flag_is_compressed=False,
    )
    noise_model = SimpleNamespace(npar=1, params=np.array([np.nan]))
    return DetectorGroupTOD([ScanTOD([det], 0.0, 0)], "EXP", "B", nside=_NSIDE, nu=30.0, fwhm=0.0,
                            fsamp=1.0, ndet=1, pols=pols, noise_model=noise_model)


def _fake_tod_samples() -> SimpleNamespace:
    """Minimal stand-in exposing exactly the fields the mapmakers / TODView / diagnostics read."""
    no_jump = SimpleNamespace(is_empty=lambda: True)
    empty_ps = lambda: np.full((1, 1, 100), np.nan, dtype=np.float32)
    return SimpleNamespace(
        noise_params=np.full((1, 1, 1), _SIGMA0), abs_gain=1.0, rel_gain=np.zeros(1),
        temporal_gain=np.zeros((1, 1)), jumps=SimpleNamespace(get=lambda iscan, idet: no_jump),
        accept=np.ones((1, 1), dtype=bool), band_unit_factor=1.0, band_unit="uK_RJ",
        chisq_z=np.full((1, 1), np.nan), good_fraction=np.full((1, 1), np.nan),
        TOD_PS_NBIN=100, tod_ps_freqs=empty_ps(), tod_ps_raw=empty_ps(), tod_ps_residual=empty_ps(),
        tod_ps_ncorrsub=empty_ps(), tod_ps_ncorr=empty_ps(), ncorr_tods=None)


def _run_mapmaker(band: DetectorGroupTOD, mapmaker: str) -> dict[str, np.ndarray]:
    """Run one of the two mapmakers on `band` and return the maps selected for chain output."""
    mapmaking = tod_processing.MapmakingConfig(
        mapmaker=mapmaker, num_threads=1, include_orbital_dipole_maps=False,
        include_corr_noise_maps=False, include_sky_model_maps=False, sparse_maps=False,
        common_res_fwhm=0.0, cg=CGConfig(max_iter=20, err_tol=1e-12))
    run = tod_processing.tod2map_CG if mapmaker == "CG" else tod_processing.tod2map_bin
    ncomp = 3 if "QU" in band.pols else 1
    _, maps = run(MPI.COMM_SELF, band, np.zeros((ncomp, _NPIX)), _fake_tod_samples(), 1,
                  mapmaking, tod_processing.CorrelatedNoiseConfig(sample_sigma0=False),
                  tod_processing.DataSelectionConfig())
    return maps


def test_cg_patch_matches_binned_and_zeroes_unsolvable_pixels(monkeypatch):
    """IQU patch: the CG map equals the binned map where solvable, and is zero everywhere else."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")  # get_s_orb_tod reads this.
    rng = np.random.default_rng(4)
    pix, psi = _patch_pointing("IQU")
    sky = np.zeros((3, _NPIX))
    sky[:, :_N_GOOD_PIX + 1] = rng.normal(size=(3, _N_GOOD_PIX + 1)) * 100.0
    tod = (sky[0, pix] + sky[1, pix]*np.cos(2*psi) + sky[2, pix]*np.sin(2*psi)
           + rng.normal(size=pix.size) * _SIGMA0)
    band = _build_band(pix, psi, tod, "IQU")

    cg = _run_mapmaker(band, "CG")
    binned = _run_mapmaker(_build_band(pix, psi, tod, "IQU"), "bin")

    assert np.isfinite(cg["observed_sky"]).all()
    # The edge pixel's 3x3 is singular (single psi), so both mapmakers must give up on it, and the
    # 161 unhit pixels have no data at all.
    solvable = np.isfinite(cg["rms"])
    assert solvable[0].sum() == _N_GOOD_PIX
    assert not solvable[:, _EDGE_PIX].any()
    np.testing.assert_array_equal(cg["observed_sky"][~solvable], 0.0)
    # With T = identity the CG solves exactly the binned normal equations.
    np.testing.assert_allclose(cg["observed_sky"][solvable], binned["observed_sky"][solvable],
                               rtol=1e-5, atol=1e-4)
    np.testing.assert_array_equal(cg["rms"], binned["rms"])


def test_cg_patch_intensity_only(monkeypatch):
    """I-only patch: unobserved pixels get +inf rms and a zero map, the rest the weighted mean."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    rng = np.random.default_rng(5)
    pix, psi = _patch_pointing("I")
    sky = np.zeros(_NPIX)
    sky[:_N_GOOD_PIX] = rng.normal(size=_N_GOOD_PIX) * 100.0
    tod = (sky[pix] + rng.normal(size=pix.size) * _SIGMA0).astype(np.float32)
    # No spacecraft motion, so the mapmaker's orbital-dipole subtraction leaves the TOD alone and
    # the solution is just the binned TOD.
    maps = _run_mapmaker(_build_band(pix, psi, tod, "I", velocity=(0.0, 0.0, 0.0)), "CG")

    signal, map_rms = maps["observed_sky"], maps["rms"]
    assert signal.shape == map_rms.shape == (1, _NPIX)
    observed = np.zeros(_NPIX, dtype=bool)
    observed[:_N_GOOD_PIX] = True
    assert np.isinf(map_rms[0, ~observed]).all()
    np.testing.assert_allclose(map_rms[0, observed], _SIGMA0/np.sqrt(_NHIT), rtol=1e-6)
    np.testing.assert_array_equal(signal[0, ~observed], 0.0)
    # A is diagonal for I-only, so the exact solution is the per-pixel mean of the TOD.
    expected = np.bincount(pix, weights=tod, minlength=_NPIX)[observed] / _NHIT
    np.testing.assert_allclose(signal[0, observed], expected, rtol=1e-6, atol=1e-6)


def _normal_matrix_three_pixels() -> np.ndarray:
    """(6, 3) normal matrix: pixel 0 degenerate (one psi), pixel 1 well covered, pixel 2 unhit."""
    w, n = 1.0/_SIGMA0**2, float(_NHIT)
    cos2, sin2 = np.cos(2*0.25*np.pi), np.sin(2*0.25*np.pi)   # cos2 = 6.1e-17, not 0
    # Pixel 0: every hit at psi = 45 deg, so A = n*w * outer([1, cos2, sin2], [1, cos2, sin2]).
    degenerate = n*w*np.array([1.0, cos2, sin2, cos2*cos2, sin2*cos2, sin2*sin2])
    # Pixel 1: n/4 hits at each of 0, 45, 90, 135 deg, which averages the off-diagonals away.
    covered = np.array([n*w, 0.0, 0.0, 0.5*n*w, 0.0, 0.5*n*w])
    return np.stack([degenerate, covered, np.zeros(6)], axis=1)


def test_preconditioners_drop_degenerate_and_unhit_pixels():
    """Both mapmaking preconditioners must zero a pixel whose 3x3 has no inverse.

    The dangerous case is the ragged coverage edge, not the unhit sky: a pixel seen at a single
    polarization angle has ``A_QQ = n*w*cos^2(2 psi)`` at rounding level rather than exactly zero, so
    guarding only against a zero diagonal leaves the CG with a preconditioner entry of ~1e33.
    """
    A = _normal_matrix_three_pixels()
    assert 0.0 < A[3, 0] < 1e-30 and 1.0/A[3, 0] > 1e29   # the trap a bare 1/diag(A) falls into

    block = BlockInvNPreconditionerIQU(A)
    diagonal = InvNPreconditionerIQU(A)
    np.testing.assert_array_equal(block.inv_N_IQU[:, [0, 2]], 0.0)
    np.testing.assert_array_equal(diagonal.inv_N_IQU[:, [0, 2]], 0.0)
    # The well-covered pixel is diagonal, so both preconditioners give the same exact inverse there.
    expected_diag = 1.0/A[(0, 3, 5), 1]
    np.testing.assert_allclose(diagonal.inv_N_IQU[:, 1], expected_diag, rtol=1e-12)
    np.testing.assert_allclose(block.inv_N_IQU[(0, 3, 5), 1], expected_diag, rtol=1e-12)
    np.testing.assert_allclose(block.inv_N_IQU[(1, 2, 4), 1], 0.0, atol=1e-12)
    # And applying them leaves the dropped pixels at zero.
    m = np.ones((3, 3))
    np.testing.assert_array_equal(block(m)[:, [0, 2]], 0.0)
    np.testing.assert_array_equal(diagonal(m)[:, [0, 2]], 0.0)
