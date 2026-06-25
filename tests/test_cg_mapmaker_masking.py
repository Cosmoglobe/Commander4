"""The CG mapmaker LHS operator must span the same samples as the RHS.

``apply_LHS`` applies ``P^T T^T N^-1 T P`` and must restrict to the same per-sample selection
(``good_data_mask``) that ``accum_to_RHS`` (and the binned mapmaker) use; otherwise the CG solves
an inconsistent ``(A, b)`` and the map is biased. These tests build a real single-detector band and
check that flagged samples contribute to neither the operator value nor its support, and that the
masked-length pointing arrays are handled correctly by the C ``map2tod``/accumulator pair.
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
                      proc_mask, ntod, ntod, flag_encoded=flag, bad_data_bitmask=_BITMASK,
                      flag_is_compressed=False)
    noise_model = SimpleNamespace(npar=1, params=np.array([np.nan]))
    band = DetGroupTOD([ScanTOD([det], 0.0, 0)], "EXP", "B", nside=nside, nu=0.0, fwhm=0.0,
                       fsamp=1.0, ndet=1, pols=pols, noise_model=noise_model)
    ts = TODSamples.__new__(TODSamples)
    ts.accept = np.ones((1, 1), dtype=bool)
    ts.noise_params = np.full((1, 1, 1), sigma0)
    return band, ts


def test_apply_LHS_I_is_masked_diagonal():
    """I-only LHS equals the good-sample inverse-variance hit count: A_pp = n_good(p)/sigma0^2."""
    nside, sigma0 = 2, 2.0
    # All-distinct pixels; samples 3 and 7 (pixels 5 and 9) are flagged bad.
    pix = np.array([0, 1, 2, 5, 4, 6, 7, 9, 8, 10], dtype=np.int64)
    bad_idx = [3, 7]
    band, ts = _build_band(pix, bad_idx, nside, sigma0, "I")
    cg = CGMapmakerI(band, ts, MPI.COMM_SELF)

    npix = 12 * nside**2
    m = np.random.default_rng(0).normal(size=(1, npix))
    out = cg.apply_LHS(m.copy())

    good = np.ones(pix.size, bool)
    good[bad_idx] = False
    n_good = np.bincount(pix[good], minlength=npix)
    expected = (n_good / sigma0**2)[None, :] * m
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-12)
    # Pixels reachable only through flagged samples must carry zero weight (excluded, like the RHS).
    assert out[0, 5] == 0.0 and out[0, 9] == 0.0


def test_apply_LHS_I_full_samples_would_differ():
    """Sanity: if the masked samples were *not* excluded, pixels 5 and 9 would be nonzero -- so the
    test above is actually exercising the masking, not a vacuous all-zero region."""
    nside = 2
    pix = np.array([0, 1, 2, 5, 4, 6, 7, 9, 8, 10], dtype=np.int64)
    npix = 12 * nside**2
    # Hit counts including the would-be-masked samples: pixels 5 and 9 get a hit.
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


def test_apply_LHS_IQU_symmetric_and_excludes_masked():
    """IQU LHS stays symmetric (A = A^T) and masked-only pixels are zero in all of I, Q, U."""
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
    # Pixels reached only through flagged samples are zero in every polarization component.
    assert np.all(Am1[:, 5] == 0.0) and np.all(Am1[:, 9] == 0.0)
