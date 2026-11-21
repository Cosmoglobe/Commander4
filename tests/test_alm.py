"""Correctness contracts for the alm repacking, including mmax != lmax."""

import healpy as hp
import numpy as np
import pytest

from commander4.math_utils.alm import (alm_complex2real, alm_real2complex,
                                       alm_complex2real_commander3, alm_real2complex_commander3,
                                       _dot_complex_alm_1D_arrays, nalm, almxfl)


def random_complex_alms(lmax: int, mmax: int, seed: int = 0) -> np.ndarray:
    """Draw random alms in the Healpy complex convention, with real m=0 modes."""
    rng = np.random.default_rng(seed)
    alm = rng.normal(size=(2, nalm(lmax, mmax))) + 1j*rng.normal(size=(2, nalm(lmax, mmax)))
    alm[:, :lmax+1] = alm[:, :lmax+1].real
    return alm


@pytest.mark.parametrize("lmax, mmax", [(8, None), (8, 8), (8, 3), (8, 0), (12, 5)])
def test_complex_real_roundtrip_and_lengths(lmax: int, mmax: int | None) -> None:
    alm = random_complex_alms(lmax, lmax if mmax is None else mmax)
    x = alm_complex2real(alm, lmax, mmax)
    assert x.shape[-1] == 2*alm.shape[-1] - (lmax+1)
    assert np.allclose(alm_real2complex(x, lmax, mmax), alm)


@pytest.mark.parametrize("lmax, mmax", [(8, 8), (8, 3), (12, 5)])
def test_real_packing_preserves_the_alm_inner_product(lmax: int, mmax: int) -> None:
    """The sqrt(2) on the m>0 modes makes the plain real dot product equal the alm one."""
    alm = random_complex_alms(lmax, mmax)
    x = alm_complex2real(alm, lmax, mmax)
    for ipol in range(alm.shape[0]):
        assert np.dot(x[ipol], x[ipol]) == pytest.approx(
            _dot_complex_alm_1D_arrays(alm[ipol], alm[ipol], lmax))


def test_mmax_defaults_to_lmax() -> None:
    lmax = 7
    alm = random_complex_alms(lmax, lmax)
    assert alm_complex2real(alm, lmax).shape[-1] == (lmax+1)**2
    assert np.allclose(alm_complex2real(alm, lmax), alm_complex2real(alm, lmax, lmax))


@pytest.mark.parametrize("lmax, mmax", [(8, None), (8, 8), (8, 3), (8, 0), (12, 5)])
def test_commander3_complex_real_roundtrip(lmax: int, mmax: int | None) -> None:
    actual_mmax = lmax if mmax is None else mmax
    alm = random_complex_alms(lmax, actual_mmax)
    x = alm_complex2real_commander3(alm, lmax, mmax)

    assert x.shape[-1] == (lmax+1)**2
    assert np.allclose(alm_real2complex_commander3(x, lmax, mmax), alm)


def test_commander3_real_layout_uses_l_squared_plus_l_plus_m() -> None:
    lmax = 2
    alm = np.zeros(nalm(lmax, lmax), dtype=np.complex128)
    alm[hp.Alm.getidx(lmax, 1, 0)] = 2.0
    alm[hp.Alm.getidx(lmax, 1, 1)] = 3.0 + 4.0j

    x = alm_complex2real_commander3(alm, lmax)

    assert x[1**2 + 1] == 2.0
    assert x[1**2 + 1 + 1] == pytest.approx(np.sqrt(2.0)*3.0)
    assert x[1**2 + 1 - 1] == pytest.approx(np.sqrt(2.0)*4.0)


def test_mismatched_lengths_raise_explicit_exceptions() -> None:
    lmax = 7
    alm = random_complex_alms(lmax, lmax)
    with pytest.raises(ValueError, match="do not match"):
        alm_complex2real(alm, lmax, 3)
    with pytest.raises(ValueError, match="do not match"):
        alm_real2complex(np.zeros((lmax+1)**2), lmax, 3)


@pytest.mark.parametrize("lmax, mmax, dtype", [(7, 7, np.complex64), (40, 40, np.complex128),
                                               (40, 13, np.complex128)])
def test_almxfl_matches_healpy(lmax: int, mmax: int, dtype: type) -> None:
    """`almxfl` scales every (l, m) coefficient by `fl[l]`, also when mmax < lmax."""
    rng = np.random.default_rng(0)
    nalm = hp.Alm.getsize(lmax, mmax)
    alm = (rng.normal(size=nalm) + 1j*rng.normal(size=nalm)).astype(dtype)
    fl = rng.uniform(0.1, 1.0, lmax + 1)

    scaled = almxfl(alm, fl, lmax=lmax, mmax=mmax)
    assert scaled.dtype == dtype
    np.testing.assert_allclose(scaled, hp.almxfl(alm.astype(np.complex128), fl, mmax=mmax),
                               rtol=1e-6)
