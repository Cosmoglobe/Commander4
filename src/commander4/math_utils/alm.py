"""Operations on spherical-harmonic coefficients stored in the healpy complex convention.

Everything here works on alm arrays directly: inner products, array sizes, random draws,
multipole-dependent filtering, resolution changes, and conversion between complex alms and the two
real layouts used by Commander. Transforms between alms and maps live in `sht.py`.
"""
from math import sqrt

import healpy as hp
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


@njit(fastmath=True, parallel=True)
def _dot_complex_alm_1D_arrays(alm1: NDArray, alm2: NDArray, lmax: int) -> NDArray:
    """ Function calculating the dot product of two alms, given that they follow the Healpy standard
        where alms are represented as complex numbers, but with the conjugate 'negative' ms missing.
    """
    nm0 = lmax + 1
    return np.sum((alm1[:nm0] * alm2[:nm0]).real)\
           + np.sum((alm1[nm0:] * np.conj(alm2[nm0:])).real * 2)


def nalm(lmax: int, mmax: int) -> int:
    """ Calculates the number of a_lm elements for a spherical harmonic representation up to
        l<=lmax and m<=mmax.
    """
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def gaussian_random_alm(lmax: int, mmax: int, spin: int, ncomp: int,
                        rng: np.random.Generator) -> NDArray:
    """Draw unit-variance white-noise alms in the Healpy complex convention (m<0 dropped).

    "Unit variance" is defined with respect to the inner product these alms are actually used
    under, `_dot_complex_alm_1D_arrays`: <a,b> = sum_{m=0} Re(a b) + 2 sum_{m>0} Re(a b*). Writing
    the corresponding real coordinates (a_l0 for m=0; sqrt(2)Re(a_lm), sqrt(2)Im(a_lm) for m>0),
    <a,a> is their plain sum of squares, so those coordinates must each be N(0,1). That means the
    m=0 entries are real with variance 1, while the m>0 real and imaginary parts each carry
    variance 1/2 (hence the 1/sqrt(2) below). The result satisfies E[<eta,eta>] = (lmax+1)^2, the
    number of real degrees of freedom, which is the property the CG fluctuation term relies on.

    For spin>0 fields the first `spin` multipoles are not defined and are zeroed.
    """
    res = rng.normal(0., 1., (ncomp, nalm(lmax, mmax))) \
     + 1j*rng.normal(0., 1., (ncomp, nalm(lmax, mmax)))
    # make a_lm with m==0 real-valued
    res[:, 0:lmax+1].imag = 0.
    ofs=0
    for s in range(spin):
        res[:, ofs:ofs+spin-s] = 0.
        ofs += lmax+1-s
    res[:, lmax+1:] /= np.sqrt(2.)
    return res


@njit(cache=True, fastmath=True, parallel=True)
def _almxfl_numba(res, lmax, mmax, fl):
    for m in prange(mmax + 1):
        ofs = m*(lmax + 1) - m*(m - 1)//2  # Where the m block starts in healpy's alm ordering.
        res[ofs:ofs + lmax + 1 - m] *= fl[m:lmax + 1]

def almxfl(alm, fl, lmax=None, mmax=None, inplace=False):
    """Multiply the 1D array `alm` by the per-multipole window `fl` (as `hp.almxfl`)."""
    res = alm if inplace else alm.copy()
    lmax = hp.Alm.getlmax(alm.shape[-1]) if lmax is None else lmax
    mmax = lmax if mmax is None else mmax
    _almxfl_numba(res, lmax, mmax, fl)
    return res


@njit(cache=True, fastmath=True)
def _project_alms_numba(alms_in, lmax_in, lmax_out, nalm_out):
    """ Numba helper function to compute _project_alms (see function below)
    """
    alms_out = np.zeros((*alms_in.shape[:-1], nalm_out), dtype=alms_in.dtype)
    # Determine the number of modes to copy
    l_copy = min(lmax_in, lmax_out)
    m_copy = min(lmax_in, lmax_out)
    # Copy alm data up to the minimum lmax
    ofs_in, ofs_out = 0, 0
    for m in range(m_copy + 1):
        alms_out[:, ofs_out:ofs_out+l_copy+1-m] = alms_in[:, ofs_in:ofs_in+l_copy+1-m]
        ofs_in += lmax_in+1-m
        ofs_out += lmax_out+1-m
    return alms_out

def project_alms(alms_in, lmax_out):
    """ Projects alms from one lmax resolution to another, handling truncation or zero-padding.
        Importantly, this function is the adjoint of itself. Takes complex alms as input.
    """
    lmax_in = hp.Alm.getlmax(alms_in.shape[-1])
    if lmax_in == lmax_out:
        return alms_in
    nalm_out = hp.Alm.getsize(lmax_out)
    alms_out = _project_alms_numba(alms_in, lmax_in, lmax_out, nalm_out)
    return alms_out


def alm_complex2real(alm: NDArray[np.complexfloating], lmax: int,
                     mmax: int | None = None) -> NDArray[np.floating]:
    """Repack Healpy complex alms as the real coordinates used by Commander4 solvers.

    The last axis starts with the real ``m=0`` coefficients for every l. It then stores the real and
    imaginary parts of each ``m>0`` Healpy coefficient consecutively, multiplied by ``sqrt(2)``.
    This scaling makes an ordinary dot product of the real coordinates equal the corresponding alm
    inner product.

    Args:
        alm: Complex alms in Healpy m-major order, with alms on the last axis.
        lmax: Maximum multipole.
        mmax: Maximum azimuthal mode. Defaults to ``lmax``.

    Returns:
        Packed real coordinates of length ``2*nalm(lmax, mmax) - (lmax+1)`` on the last axis.
    """
    if alm.dtype not in (np.complex128, np.complex64):
        raise TypeError(f"Input alms must have dtype complex64 or complex128, got {alm.dtype}.")
    mmax = lmax if mmax is None else mmax
    if alm.shape[-1] != nalm(lmax, mmax):
        raise ValueError(f"Complex alms of length {alm.shape[-1]} do not match lmax={lmax}, "
                         f"mmax={mmax} (expected {nalm(lmax, mmax)}).")
    float_dtype = np.float64 if alm.dtype == np.complex128 else np.float32
    # Healpy stores all m=0 coefficients first. Viewing the remaining contiguous complex values as
    # floats produces the required [real, imaginary, real, imaginary, ...] sequence.
    i = lmax + 1
    return np.concatenate([alm[...,:i].real,sqrt(2.0)*alm[...,i:].view(float_dtype)], axis=-1)


def alm_real2complex(x: NDArray[np.floating], lmax: int,
                     mmax: int | None = None) -> NDArray[np.complexfloating]:
    """Convert Commander4's packed real coordinates back to Healpy complex alms.

    This is the inverse of :func:`alm_complex2real`.

    Args:
        x: Packed real coordinates on the last axis.
        lmax: Maximum multipole.
        mmax: Maximum azimuthal mode. Defaults to ``lmax``.

    Returns:
        Complex alms in Healpy m-major order, with length ``nalm(lmax, mmax)``.
    """
    if x.dtype not in (np.float32, np.float64):
        raise TypeError(f"Input map must have dtype float32 or float64, got {x.dtype}.")
    mmax = lmax if mmax is None else mmax
    nelem = nalm(lmax, mmax)
    if x.shape[-1] != 2*nelem - (lmax+1):
        raise ValueError(f"Real alms of length {x.shape[-1]} do not match lmax={lmax}, "
                         f"mmax={mmax} (expected {2*nelem - (lmax+1)}).")
    complex_dtype = np.complex128 if x.dtype == np.float64 else np.complex64
    i = lmax + 1
    oalm = np.zeros((*x.shape[:-1], nelem), complex_dtype)
    oalm[...,:i] = x[...,:i]
    # Pair consecutive floats into complex values, then undo the sqrt(2) scaling of m>0 modes.
    oalm[...,i:] = x[...,i:].view(complex_dtype)/sqrt(2.0)
    return oalm


def alm_complex2real_commander3(alm: NDArray[np.complexfloating], lmax: int,
                                mmax: int | None = None) -> NDArray[np.floating]:
    """Convert Healpy complex alms to the l-major real convention used by Commander3.

    Unlike Commander4's packed layout, Commander3 reserves one slot for every signed m at index
    ``l**2 + l + m``. For ``m>0``, the ``+m`` slot contains ``sqrt(2)*real(a_lm)`` and the ``-m``
    slot contains ``sqrt(2)*imag(a_lm)``. A restricted ``mmax`` therefore leaves zero padding in
    the unused signed-m slots instead of shortening the array.

    Args:
        alm: Complex alms in Healpy m-major order, with alms on the last axis.
        lmax: Maximum multipole.
        mmax: Maximum azimuthal mode. Defaults to ``lmax``.

    Returns:
        Real Commander3 alms, with the padded l-major layout on the last axis.
    """
    if alm.dtype not in (np.complex128, np.complex64):
        raise TypeError(f"Input alms must have dtype complex64 or complex128, got {alm.dtype}.")
    mmax = lmax if mmax is None else mmax
    nelem = nalm(lmax, mmax)
    if alm.shape[-1] != nelem:
        raise ValueError(f"Complex alms of length {alm.shape[-1]} do not match lmax={lmax}, "
                         f"mmax={mmax} (expected {nelem}).")

    float_dtype = np.float64 if alm.dtype == np.complex128 else np.float32
    x = np.zeros((*alm.shape[:-1], (lmax+1)**2), dtype=float_dtype)
    l, m = hp.Alm.getlm(lmax, np.arange(nelem))
    # Locations of the +m and -m real coordinates corresponding to each stored Healpy mode.
    ip = l**2 + l + m
    im = l**2 + l - m
    m0 = m == 0
    x[..., ip[m0]] = alm[..., m0].real
    x[..., ip[~m0]] = sqrt(2.0)*alm[..., ~m0].real
    x[..., im[~m0]] = sqrt(2.0)*alm[..., ~m0].imag
    return x


def alm_real2complex_commander3(x: NDArray[np.floating], lmax: int,
                                mmax: int | None = None) -> NDArray[np.complexfloating]:
    """Convert l-major Commander3 real alms to Healpy complex alms.

    This is the inverse of :func:`alm_complex2real_commander3`. Values in padded slots with
    ``abs(m)>mmax`` are ignored.

    Args:
        x: Real Commander3 alms, with a padded ``(lmax+1)**2`` last axis.
        lmax: Maximum multipole.
        mmax: Maximum azimuthal mode to include. Defaults to ``lmax``.

    Returns:
        Complex alms in Healpy m-major order, with length ``nalm(lmax, mmax)``.
    """
    if x.dtype not in (np.float32, np.float64):
        raise TypeError(f"Input alms must have dtype float32 or float64, got {x.dtype}.")
    mmax = lmax if mmax is None else mmax
    expected_size = (lmax+1)**2
    if x.shape[-1] != expected_size:
        raise ValueError(f"Commander3 alms of length {x.shape[-1]} do not match lmax={lmax} "
                         f"(expected {expected_size}).")

    nelem = nalm(lmax, mmax)
    complex_dtype = np.complex128 if x.dtype == np.float64 else np.complex64
    alm = np.empty((*x.shape[:-1], nelem), dtype=complex_dtype)
    l, m = hp.Alm.getlm(lmax, np.arange(nelem))
    # Read the real and imaginary coordinates for each Healpy mode from the signed-m slots.
    ip = l**2 + l + m
    im = l**2 + l - m
    m0 = m == 0
    alm[..., m0] = x[..., ip[m0]]
    alm[..., ~m0] = (x[..., ip[~m0]] + 1j*x[..., im[~m0]])/sqrt(2.0)
    return alm
