"""The CG mapmaker LHS operator must span the same samples as the RHS.

``apply_LHS`` applies ``P^T T^T N^-1 T P`` and must run over the same per-sample selection that
``accum_to_RHS`` uses; otherwise the CG solves an inconsistent ``(A, b)`` and the map is biased.
Unlike the binned mapmaker (which drops flagged samples), the CG path applies ``apply_T`` -- a
Fourier transfer function that needs a *continuous* TOD -- so it cannot remove flagged samples:
both the LHS and RHS gap-fill them and therefore span **every** sample of each accepted
detector-scan. These tests build a real single-detector band and check that the operator spans all
samples (flagged pixels stay populated, matching the RHS), and that the full-length pointing arrays
are handled correctly by the C ``map2tod``/accumulator pair.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from mpi4py import MPI

from commander4.data_models.detector_TOD import DetectorTOD
from commander4.data_models.scan_TOD import ScanTOD
from commander4.data_models.detector_group_TOD import DetGroupTOD
from commander4.data_models.pointing import PixelPointing
from commander4.data_models.TOD_samples import TODSamples
from commander4.utils.CG_mapmaker import CGMapmakerI, CGMapmakerIQU

_BITMASK = 1  # one bad-data bit; a flagged sample has (flag & _BITMASK) != 0


def _build_band(pix: np.ndarray, bad_idx, nside: int, sigma0: float, pols: str,
                psi: np.ndarray | None = None):
    """Real one-detector, one-scan band with `bad_idx` samples flagged bad (uncompressed pointing)."""
    ntod = pix.size
    npix = 12 * nside**2
    if psi is None:
        psi = np.zeros(ntod, dtype=np.float64)
    flag = np.zeros(ntod, dtype=np.int64)
    flag[list(bad_idx)] = _BITMASK
    dummy_tree = np.array([0], dtype=np.int64)  # PixelPointing.__init__ does tree.astype, needs non-None
    pointing = PixelPointing(pix.astype(np.int64), psi.astype(np.float64), dummy_tree, None, None,
                             nside, nside, ntod, ntod)
    proc_mask = np.ones(npix, dtype=bool)  # all-sky processing mask -> full_mask == good_data_mask
    det = DetectorTOD("d0", 0, 0, np.zeros(ntod, dtype=np.float32), pointing, 1.0, None, None, None,
                      proc_mask, {}, ntod, ntod, flag_encoded=flag, bad_data_bitmask=_BITMASK,
                      flag_is_compressed=False)
    noise_model = SimpleNamespace(npar=1, params=np.array([np.nan]))
    band = DetGroupTOD([ScanTOD([det], 0.0, 0)], "EXP", "B", nside=nside, nu=0.0, fwhm=0.0,
                       fsamp=1.0, ndet=1, pols=pols, noise_model=noise_model)
    ts = TODSamples.__new__(TODSamples)
    ts.accept = np.ones((1, 1), dtype=bool)
    ts.noise_params = np.full((1, 1, 1), sigma0)
    return band, ts


def test_apply_LHS_I_is_full_sample_diagonal():
    """I-only LHS equals the all-sample inverse-variance hit count: A_pp = n_all(p)/sigma0^2.

    With the identity transfer function ``apply_T`` reduces to the identity, so the operator is just
    ``P^T N^-1 P`` summed over *all* samples of the accepted detector-scan (flagged samples included,
    since the CG path gap-fills rather than dropping them -- see the module docstring)."""
    nside, sigma0 = 2, 2.0
    # All-distinct pixels; samples 3 and 7 (pixels 5 and 9) are flagged bad but still contribute.
    pix = np.array([0, 1, 2, 5, 4, 6, 7, 9, 8, 10], dtype=np.int64)
    bad_idx = [3, 7]
    band, ts = _build_band(pix, bad_idx, nside, sigma0, "I")
    cg = CGMapmakerI(band, ts, MPI.COMM_SELF)

    npix = 12 * nside**2
    m = np.random.default_rng(0).normal(size=(1, npix))
    out = cg.apply_LHS(m.copy())

    n_all = np.bincount(pix, minlength=npix)
    expected = (n_all / sigma0**2)[None, :] * m
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-12)
    # Pixels hit only by flagged samples still carry weight (gap-filled, not removed, like the RHS).
    assert out[0, 5] != 0.0 and out[0, 9] != 0.0


def test_apply_LHS_I_flagged_pixels_are_singly_hit():
    """Sanity for the assertions above: pixels 5 and 9 are each hit exactly once (by the flagged
    samples 3 and 7), so their nonzero weight is a genuine consequence of spanning all samples."""
    nside = 2
    pix = np.array([0, 1, 2, 5, 4, 6, 7, 9, 8, 10], dtype=np.int64)
    npix = 12 * nside**2
    n_all = np.bincount(pix, minlength=npix)
    assert n_all[5] == 1 and n_all[9] == 1


def test_finalize_RHS_without_accumulation_contributes_zeros():
    """A rank that accumulated no detector-scans (all its scans rejected during read-in) never
    allocates a local RHS map. ``finalize_RHS`` must still take part in the collective reduce,
    contributing zeros, instead of crashing on the un-allocated (None) buffer."""
    nside = 2
    pix = np.array([0, 1, 2, 3], dtype=np.int64)
    band, ts = _build_band(pix, [], nside, 1.0, "I")
    cg = CGMapmakerI(band, ts, MPI.COMM_SELF)

    # Deliberately skip accum_to_RHS, mimicking a rank with no accepted detector-scans.
    rhs = cg.finalize_RHS()

    assert rhs.shape == (1, 12 * nside**2)
    np.testing.assert_array_equal(rhs, 0.0)


def _dense_operator(op, n):
    """Materialize a length-preserving linear TOD operator ``op`` as its (n, n) matrix."""
    M = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        M[:, j] = op(e)
    return M


def _single_pole(tau):
    """A non-trivial, Hermitian-symmetric transfer function on the normalized (2N) frequency grid."""
    return lambda f: 1.0 / (1.0 + 2j * np.pi * f * tau)


def test_apply_T_identity_is_a_noop():
    """The default identity ``T_omega`` must leave the TOD unchanged despite the mirrored round-trip."""
    nside = 2
    pix = np.arange(12, dtype=np.int64) % (12 * nside**2)
    band, ts = _build_band(pix, [], nside, 1.0, "I")
    cg = CGMapmakerI(band, ts, MPI.COMM_SELF)
    x = np.random.default_rng(0).normal(size=pix.size)
    np.testing.assert_allclose(cg.apply_T(x.copy()), x, atol=1e-12)
    np.testing.assert_allclose(cg.apply_T_adjoint(x.copy()), x, atol=1e-12)


def test_apply_T_adjoint_is_true_transpose():
    """``apply_T_adjoint`` must be the exact transpose of ``apply_T`` for a non-trivial ``T_omega``.

    This is the core-logic fix: the transpose conjugates the filter (``H*``) and pairs reflect-extend
    with zero-pad/fold-back. The previous "flip the frequency array" adjoint is *not* the transpose
    for any non-identity filter, so this test fails against it.
    """
    nside = 2
    pix = np.arange(16, dtype=np.int64) % (12 * nside**2)
    band, ts = _build_band(pix, [], nside, 1.0, "I")
    cg = CGMapmakerI(band, ts, MPI.COMM_SELF, T_omega=_single_pole(0.9))
    n = pix.size
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=n), rng.normal(size=n)
    # Inner-product identity <T x, y> == <x, T^T y>.
    assert np.isclose(np.dot(cg.apply_T(x.copy()), y), np.dot(x, cg.apply_T_adjoint(y.copy())),
                      rtol=1e-10, atol=1e-12)
    # Full dense check: matrix of the adjoint equals the transpose of the forward's matrix.
    T = _dense_operator(cg.apply_T, n)
    Tt = _dense_operator(cg.apply_T_adjoint, n)
    np.testing.assert_allclose(Tt, T.T, atol=1e-10)
    assert not np.allclose(T, np.eye(n))     # the filter genuinely alters the TOD (guards no-op)


def test_apply_LHS_symmetric_with_nontrivial_transfer_function():
    """``P^T T^T N^-1 T P`` stays symmetric when ``T_omega != 1`` -- the point of the exact adjoint.

    At the identity transfer function symmetry is trivial (T = T^T = I), so it cannot detect a wrong
    adjoint; a genuine filter can, and the CG needs a symmetric operator to be well-posed.
    """
    nside, sigma0 = 2, 1.3
    pix = np.array([0, 1, 2, 5, 4, 6, 7, 9, 8, 10], dtype=np.int64)
    band, ts = _build_band(pix, [], nside, sigma0, "I")
    cg = CGMapmakerI(band, ts, MPI.COMM_SELF, T_omega=_single_pole(0.7))
    npix = 12 * nside**2
    rng = np.random.default_rng(5)
    m1, m2 = rng.normal(size=(1, npix)), rng.normal(size=(1, npix))
    Am2, Am1 = cg.apply_LHS(m2.copy()), cg.apply_LHS(m1.copy())
    np.testing.assert_allclose(np.vdot(m1, Am2), np.vdot(m2, Am1), rtol=1e-9, atol=1e-10)


def test_apply_LHS_IQU_symmetric_and_spans_all_samples():
    """IQU LHS stays symmetric (A = A^T); flagged samples are gap-filled rather than dropped, so
    pixels reached only through them stay populated in I/Q/U (matching the RHS)."""
    nside, sigma0 = 2, 1.5
    pix = np.array([0, 1, 2, 5, 4, 6, 7, 9, 8, 10], dtype=np.int64)
    bad_idx = [3, 7]
    rng = np.random.default_rng(1)
    psi = rng.uniform(0, np.pi, pix.size)
    band, ts = _build_band(pix, bad_idx, nside, sigma0, "IQU", psi=psi)
    cg = CGMapmakerIQU(band, ts, MPI.COMM_SELF)

    npix = 12 * nside**2
    m1 = rng.normal(size=(3, npix))
    m2 = rng.normal(size=(3, npix))
    Am2 = cg.apply_LHS(m2.copy())
    Am1 = cg.apply_LHS(m1.copy())
    # Symmetry of A = P^T N^-1 P: <m1, A m2> == <m2, A m1>.
    np.testing.assert_allclose(np.vdot(m1, Am2), np.vdot(m2, Am1), rtol=1e-9, atol=1e-10)
    # Pixels reached only through flagged samples still carry weight (gap-fill, not removal).
    assert np.any(Am1[:, 5] != 0.0) and np.any(Am1[:, 9] != 0.0)
