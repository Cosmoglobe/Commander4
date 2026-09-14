"""Correctness contracts for the alm repacking and mmax projection, including mmax != lmax."""

import healpy as hp
import numpy as np
import pytest

from commander4.math_utils.alm import (alm_complex2real, alm_real2complex,
                                       alm_complex2real_commander3,
                                       alm_real2complex_commander3, alm_dot_product,
                                       project_alms_mmax, get_mmax, nalm)


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
        assert np.dot(x[ipol], x[ipol]) == pytest.approx(alm_dot_product(alm[ipol], alm[ipol], lmax))


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
    with pytest.raises(ValueError, match="do not match"):
        project_alms_mmax(alm, lmax, 3, 5)


def test_mmax_truncation_keeps_the_low_m_modes_and_padding_undoes_it() -> None:
    lmax, mmax = 8, 3
    alm = random_complex_alms(lmax, lmax)
    cut = project_alms_mmax(alm, lmax, lmax, mmax)
    assert cut.shape[-1] == nalm(lmax, mmax)
    # The m-blocks are contiguous and equally long in both layouts, so this is the leading prefix.
    assert np.array_equal(cut, alm[:, :nalm(lmax, mmax)])
    back = project_alms_mmax(cut, lmax, mmax, lmax)
    assert back.shape == alm.shape
    assert np.array_equal(back[:, :nalm(lmax, mmax)], cut)
    assert np.all(back[:, nalm(lmax, mmax):] == 0.0)


@pytest.mark.parametrize("lmax, mmax", [(8, 8), (8, 3), (8, 0), (512, 100), (512, 512)])
def test_get_mmax_inverts_nalm(lmax: int, mmax: int) -> None:
    assert get_mmax(np.zeros((2, nalm(lmax, mmax)), dtype=np.complex128), lmax) == mmax


def test_get_mmax_rejects_lengths_that_are_not_alm_arrays() -> None:
    with pytest.raises(ValueError, match="do not match any mmax"):
        get_mmax(np.zeros(nalm(8, 3) + 1, dtype=np.complex128), 8)
    with pytest.raises(ValueError, match="do not match any mmax"):
        get_mmax(np.zeros(nalm(9, 9), dtype=np.complex128), 8)


def test_get_mmax_rejects_real_convention_alms() -> None:
    """A real array of lmax=8, mmax=2 has length nalm(8, 5), so it would give a wrong mmax."""
    x = alm_complex2real(random_complex_alms(8, 2), 8, 2)
    assert x.shape[-1] == nalm(8, 5)
    with pytest.raises(TypeError, match="complex64 or complex128"):
        get_mmax(x, 8)


@pytest.mark.parametrize("mmax_in, mmax_out", [(8, 3), (3, 8), (8, 0), (5, 5)])
def test_mmax_projection_is_its_own_adjoint(mmax_in: int, mmax_out: int) -> None:
    """<P a, b> in the output space must equal <a, P b> in the input space."""
    lmax = 8
    a = random_complex_alms(lmax, mmax_in, seed=1)[0]
    b = random_complex_alms(lmax, mmax_out, seed=2)[0]
    lhs = alm_dot_product(project_alms_mmax(a, lmax, mmax_in, mmax_out), b, lmax)
    rhs = alm_dot_product(a, project_alms_mmax(b, lmax, mmax_out, mmax_in), lmax)
    assert lhs == pytest.approx(rhs)
