"""The CG mapmaker on a partial sky: unobserved and degenerate pixels must not poison the solve.

A patch experiment leaves most of the sky unhit and, along the ragged coverage edge, leaves pixels
that *are* hit but only over a narrow range of polarization angles. The second kind is the dangerous
one: a pixel seen only at psi = 45 deg has ``A_QQ = sum w*cos^2(2 psi)`` at rounding level (~1e-33,
not exactly 0) while ``A_II`` and ``A_UU`` are large, so a diagonal preconditioner built as
``1/diag(A)`` hands the CG an entry of ~1e33 and every dot product in the iteration is swamped by
that one pixel. The block-Jacobi preconditioner inverts the whole 3x3 instead and zeroes the pixels
whose 3x3 is singular, which projects them out of the solve.

These tests drive the real ``tod2map`` on a small patch and check that the solved CG map is zero
exactly where the rms is ``+inf``, and that the well-measured pixels reproduce the binned map --
with an identity transfer function the two mapmakers solve the same normal equations, so they must
agree. With a transfer function, the CG map is checked against a direct solve instead.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from mpi4py import MPI

from commander4.data_models.detector_tod import DetectorTOD
from commander4.tod.glitches.events import empty_glitch_grid
from commander4.tod.jumps.events import empty_jump_grid
from commander4.data_models.scan_tod import ScanTOD
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.pixel_domain import PixelDomain
from commander4.data_models.pointing import PixelPointing
from commander4.tod.mapmaking.preconditioners import BlockInvNPreconditionerIQU,\
    InvNPreconditionerIQU
from commander4.tod.config import CGConfig
import commander4.tod.processing as tod_processing
from commander4.tod.config import MapmakingConfig, CorrelatedNoiseConfig, DataSelectionConfig
from commander4.tod.view import TODView

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
                velocity: tuple[float, float, float] = (1.0, 0.2, -0.3),
                responses: list[tuple[float, float]] | None = None,
                flagged: np.ndarray | None = None,
                tf_tau_sec: float | None = None) -> DetectorGroupTOD:
    """One-scan band with uncompressed pointing, sampled at 1 Hz.

    Every detector sees the pixels `pix`. `psi` and `tod` have one row per detector, or are 1-D for
    a single detector. `responses` gives each detector's (response_I, response_P); the default is
    (1, 1). `velocity` is the spacecraft velocity direction driving the orbital dipole the
    mapmakers subtract; set it to zero for a test that wants the map to be the plain binned TOD.
    `flagged` marks the samples flagged as bad data in every detector (none by default), and
    `tf_tau_sec` is the bolometer time constant (no transfer function by default).
    """
    ntod = pix.size
    psi, tod = np.atleast_2d(psi), np.atleast_2d(tod)
    responses = [None]*len(tod) if responses is None else responses
    flag = np.zeros(ntod, np.int64) if flagged is None else flagged.astype(np.int64)
    orbital_velocity = np.array(velocity, dtype=np.float32)
    detectors = []
    for idet, (det_psi, det_tod, response) in enumerate(zip(psi, tod, responses)):
        pointing = PixelPointing(pix.astype(np.int64), det_psi.astype(np.float64),
                                 np.array([0], np.int64), None, None, _NSIDE, _NSIDE, ntod, ntod)
        detectors.append(DetectorTOD(
            name=f"d{idet}", det_idx_fullband=idet, tod=det_tod.astype(np.float32),
            pointing=pointing, sampling_rate_hz=1.0, orbital_velocity_m_per_s=orbital_velocity,
            huffman_tree=None, huffman_symbols=None, default_proc_mask=np.ones(_NPIX, bool),
            specific_proc_masks={}, flag_encoded=flag, bad_data_bitmask=1,
            flag_is_compressed=False, response_I_P=response,
        ))
    noise_model = SimpleNamespace(npar=1, params=np.array([np.nan]))
    return DetectorGroupTOD([ScanTOD(detectors, 0.0, 0)], "EXP", "B", nside=_NSIDE, nu=30.0,
                            fwhm=0.0, fsamp=1.0, ndet=len(detectors), pols=pols,
                            noise_model=noise_model, tf_tau_sec=tf_tau_sec)


def _fake_tod_samples(ndet: int = 1) -> SimpleNamespace:
    """Minimal stand-in exposing exactly the fields the mapmakers / TODView / diagnostics read."""
    empty_ps = lambda: np.full((1, ndet, 100), np.nan, dtype=np.float32)
    return SimpleNamespace(
        noise_params=np.full((1, ndet, 1), _SIGMA0), abs_gain=1.0, rel_gain=np.zeros(ndet),
        temporal_gain=np.zeros((1, ndet)), jumps=empty_jump_grid(1, ndet),
        glitches=empty_glitch_grid(1, ndet),
        accept=np.ones((1, ndet), dtype=bool), band_unit_factor=1.0, band_unit="uK_RJ",
        chisq_z=np.full((1, ndet), np.nan), good_fraction=np.full((1, ndet), np.nan),
        TOD_PS_NBIN=100, tod_ps_freqs=empty_ps(), tod_ps_raw=empty_ps(), tod_ps_residual=empty_ps(),
        tod_ps_ncorrsub=empty_ps(), tod_ps_ncorr=empty_ps(), ncorr_tods=None, residual_tods=None,
        scan_runtime=np.zeros(1))


def _run_mapmaker(band: DetectorGroupTOD, mapmaker: str, sparse_maps: bool = False,
                  tod_samples: SimpleNamespace | None = None,
                  cg: CGConfig = CGConfig(max_iter=20, err_tol=1e-12)) -> dict[str, np.ndarray]:
    """Run one of the two mapmakers on `band` and return the maps selected for chain output."""
    mapmaking = MapmakingConfig(
        mapmaker=mapmaker, num_threads=1, include_orbital_dipole_maps=False,
        include_corr_noise_maps=False, include_sky_model_maps=False,
        common_res_fwhm=0.0, cg=cg)
    ncomp = 3 if "QU" in band.pols else 1
    tod_samples = _fake_tod_samples() if tod_samples is None else tod_samples
    # As in a real run, the domain is built first and the sky model holds its local pixels.
    band.pixel_domain = PixelDomain.from_view(TODView(band, tod_samples), MPI.COMM_SELF,
                                              "sparse" if sparse_maps else "full", _NSIDE)
    sky = np.zeros((ncomp, band.pixel_domain.n_local))
    _, maps = tod_processing.tod2map(MPI.COMM_SELF, band, sky, tod_samples, 1, mapmaking,
                                     CorrelatedNoiseConfig(sample_sigma0=False),
                                     DataSelectionConfig())
    return maps


@pytest.mark.parametrize("sparse_maps", [False, True])
def test_cg_patch_matches_binned_and_zeroes_unsolvable_pixels(monkeypatch, sparse_maps):
    """IQU patch: the CG map equals the binned map where solvable, and is zero everywhere else.

    With `sparse_maps` the CG keeps rank-local maps, which the full-sky binned map checks.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "1")  # get_s_orb_tod reads this.
    rng = np.random.default_rng(4)
    pix, psi = _patch_pointing("IQU")
    sky = np.zeros((3, _NPIX))
    sky[:, :_N_GOOD_PIX + 1] = rng.normal(size=(3, _N_GOOD_PIX + 1)) * 100.0
    tod = (sky[0, pix] + sky[1, pix]*np.cos(2*psi) + sky[2, pix]*np.sin(2*psi)
           + rng.normal(size=pix.size) * _SIGMA0)
    band = _build_band(pix, psi, tod, "IQU")

    cg = _run_mapmaker(band, "CG", sparse_maps)
    binned = _run_mapmaker(_build_band(pix, psi, tod, "IQU"), "bin")

    assert np.isfinite(cg["observed_sky"]).all()
    assert cg["observed_sky"].dtype == binned["observed_sky"].dtype == np.float32
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


@pytest.mark.parametrize("sparse_maps", [False, True])
def test_cg_patch_intensity_only(monkeypatch, sparse_maps):
    """I-only patch: unobserved pixels get +inf rms and a zero map, the rest the weighted mean."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    rng = np.random.default_rng(5)
    pix, psi = _patch_pointing("I")
    sky = np.zeros(_NPIX)
    sky[:_N_GOOD_PIX] = rng.normal(size=_N_GOOD_PIX) * 100.0
    tod = (sky[pix] + rng.normal(size=pix.size) * _SIGMA0).astype(np.float32)
    # No spacecraft motion, so the mapmaker's orbital-dipole subtraction leaves the TOD alone and
    # the solution is just the binned TOD.
    maps = _run_mapmaker(_build_band(pix, psi, tod, "I", velocity=(0.0, 0.0, 0.0)), "CG",
                         sparse_maps)

    signal, map_rms = maps["observed_sky"], maps["rms"]
    # The chain format has I, Q, U rows even for an I-only band, with nothing in Q and U.
    assert signal.shape == map_rms.shape == (3, _NPIX)
    np.testing.assert_array_equal(signal[1:], 0.0)
    assert np.isinf(map_rms[1:]).all()
    observed = np.zeros(_NPIX, dtype=bool)
    observed[:_N_GOOD_PIX] = True
    assert np.isinf(map_rms[0, ~observed]).all()
    np.testing.assert_allclose(map_rms[0, observed], _SIGMA0/np.sqrt(_NHIT), rtol=1e-6)
    np.testing.assert_array_equal(signal[0, ~observed], 0.0)
    # A is diagonal for I-only, so the exact solution is the per-pixel mean of the TOD.
    expected = np.bincount(pix, weights=tod, minlength=_NPIX)[observed] / _NHIT
    np.testing.assert_allclose(signal[0, observed], expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("pols", ["IQU", "I"])
def test_cg_matches_binned_for_flags_and_detectors_with_different_gains_and_responses(monkeypatch,
                                                                                        pols):
    """Without a transfer function the CG map equals the binned map. Both use only the good
    samples, and the CG operator projects and weights each sample as the binned maps do: with the
    detector's response in its pointing row, and with the weight (gain/sigma0)^2."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    rng = np.random.default_rng(6)
    pix = np.repeat(np.arange(_N_GOOD_PIX), _NHIT)
    psi = rng.uniform(0.0, np.pi, size=(2, pix.size))
    responses, gains, sigma0s = [(1.0, 1.0), (0.8, 0.6)], np.array([1.0, 1.7]), np.array([2.0, 3.0])
    sky = rng.normal(size=(3, _NPIX)) * 100.0
    if pols == "I":
        sky[1:] = 0.0
    tod = np.stack([gain*(r_I*sky[0, pix] + r_P*(sky[1, pix]*np.cos(2*det_psi)
                                                 + sky[2, pix]*np.sin(2*det_psi)))
                    + rng.normal(size=pix.size)*sigma0
                    for det_psi, (r_I, r_P), gain, sigma0 in zip(psi, responses, gains, sigma0s)])
    # A flagged stretch holding a large glitch, and a few single flagged samples.
    flagged = np.zeros(pix.size, dtype=bool)
    flagged[300:340] = True
    flagged[[11, 12, 500]] = True
    tod[:, flagged] = 1e5
    maps = {}
    for mapmaker in ("CG", "bin"):
        tod_samples = _fake_tod_samples(ndet=2)
        tod_samples.rel_gain = gains - tod_samples.abs_gain
        tod_samples.noise_params[0, :, 0] = sigma0s
        band = _build_band(pix, psi, tod, pols, velocity=(0.0, 0.0, 0.0), responses=responses,
                           flagged=flagged)
        maps[mapmaker] = _run_mapmaker(band, mapmaker, tod_samples=tod_samples)

    # Pixel 13 is seen only during the flagged stretch, so neither mapmaker can solve it.
    solvable = np.isfinite(maps["bin"]["rms"])
    assert solvable[0].sum() == _N_GOOD_PIX - 1 and not solvable[0, 13]
    np.testing.assert_array_equal(maps["CG"]["rms"], maps["bin"]["rms"])
    np.testing.assert_array_equal(maps["CG"]["observed_sky"][~solvable], 0.0)
    np.testing.assert_allclose(maps["CG"]["observed_sky"][solvable],
                               maps["bin"]["observed_sky"][solvable], rtol=1e-5, atol=1e-4)


def _transfer_matrix(ntod: int, tau: float) -> np.ndarray:
    """The CG's transfer-function operator T for a 1 Hz scan of `ntod` samples, as a dense matrix:
    mirror the scan to twice its length, filter with 1/(1 + 2 pi i f tau), keep the first half."""
    H = 1.0/(1.0 + 2j*np.pi*np.fft.rfftfreq(2*ntod, d=1.0)*tau)
    unit = np.eye(ntod)
    mirrored = np.concatenate([unit, unit[::-1]])
    return np.fft.irfft(np.fft.rfft(mirrored, axis=0)*H[:, None], n=2*ntod, axis=0)[:ntod]


def test_cg_with_flags_and_a_transfer_function_matches_a_dense_solve(monkeypatch):
    """With a transfer function T, the CG map is the maximum-likelihood map of the good samples
    alone: the solution of P^T T^T N^-1 T P m = P^T T^T N^-1 d, with N^-1 zero at flagged samples.
    Flagged data (here a large glitch) must not reach the map, even though T smears every sample
    over its neighbours, and a pixel seen only while flagged is left unsolved."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    rng = np.random.default_rng(8)
    tau = 1.5  # seconds; at 1 Hz, T smears each sample over a few neighbours
    # Pixels 0-9 at random angles, with a flagged stretch in the middle that alone sees pixel 10.
    pix = rng.integers(0, 10, 412)
    pix[200:212] = 10
    psi = rng.uniform(0.0, np.pi, pix.size)
    flagged = np.zeros(pix.size, dtype=bool)
    flagged[200:212] = True
    flagged[[37, 38, 301]] = True
    sky = rng.normal(size=(3, _NPIX))*100.0
    T = _transfer_matrix(pix.size, tau)
    tod = T @ (sky[0, pix] + sky[1, pix]*np.cos(2*psi) + sky[2, pix]*np.sin(2*psi))
    tod = (tod + rng.normal(size=pix.size)*_SIGMA0).astype(np.float32)
    tod[flagged] = 1e5
    band = _build_band(pix, psi, tod, "IQU", velocity=(0.0, 0.0, 0.0), flagged=flagged,
                       tf_tau_sec=tau)
    maps = _run_mapmaker(band, "CG", cg=CGConfig(max_iter=300, err_tol=1e-20))

    # The dense solve, over the pixels whose good samples constrain I, Q and U; the rest stay 0.
    P = np.zeros((pix.size, 3, _NPIX))
    P[np.arange(pix.size), 0, pix] = 1.0
    P[np.arange(pix.size), 1, pix] = np.cos(2*psi)
    P[np.arange(pix.size), 2, pix] = np.sin(2*psi)
    TP = T @ P.reshape(pix.size, -1)
    weight = np.where(flagged, 0.0, 1.0/_SIGMA0**2)
    A = TP.T @ (weight[:, None]*TP)
    b = TP.T @ (weight*tod.astype(np.float64))
    solved = (np.arange(3*_NPIX) % _NPIX) < 10
    expected = np.zeros(3*_NPIX)
    expected[solved] = np.linalg.solve(A[np.ix_(solved, solved)], b[solved])
    np.testing.assert_allclose(maps["observed_sky"], expected.reshape(3, _NPIX), rtol=1e-4,
                               atol=1e-3)
    assert np.isinf(maps["rms"][:, 10]).all()


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
