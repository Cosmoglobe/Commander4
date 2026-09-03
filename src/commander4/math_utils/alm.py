"""Operations on spherical-harmonic coefficients stored in the healpy complex convention.

Everything here works on alm arrays directly: their inner product (which must count m>0 twice),
the alm count, random draws, multiplication by an l-dependent filter, resolution changes, and the
complex<->real repacking the samplers need. Transforms between alms and maps live in `sht.py`.
"""
from math import sqrt

import healpy as hp
import numpy as np
from numba import njit
from numpy.typing import NDArray

from commander4.math_utils.arithmetic import dot, inplace_scale_add, inplace_add_scaled_vec


@njit(fastmath=True, parallel=True)
def _dot_complex_alm_1D_arrays(alm1: NDArray, alm2: NDArray, lmax: int) -> NDArray:
    """ Function calculating the dot product of two alms, given that they follow the Healpy standard
        where alms are represented as complex numbers, but with the conjugate 'negative' ms missing.
    """
    nm0 = lmax + 1
    return np.sum((alm1[:nm0] * alm2[:nm0]).real)\
           + np.sum((alm1[nm0:] * np.conj(alm2[nm0:])).real * 2)

###### ALM-LIST FUNCTIONS ######
# These functions are common array operations, but made to work on the alm-lists, which are
# lists of arrays, with each array being the alms of a certain component.

def inplace_almlist_add_scaled_array(list_inplace, list_other, value):
    """ `list_inplace += value*list_other`
    """
    for i in range(len(list_inplace)):
        inplace_add_scaled_vec(list_inplace[i], list_other[i], value)

def inplace_almlist_scale_and_add(list_inplace, list_other, value):
    """ `list_inplace = value*list_inplace + list_other`
    """
    for i in range(len(list_inplace)):
        inplace_scale_add(list_inplace[i], list_other[i], value)

def almlist_dot_complex(alm_list1, alm_list2):
    """ `dot(alm_list1, alm_list2)`. Calculates the correct dot product between two alm lists where
        the alms follow the Healpy convention of not storing negative ms.
    """
    res = 0.0
    for i in range(len(alm_list1)):
        npol, nalm = alm_list1[i].shape
        lmax = hp.Alm.getlmax(nalm)
        for ipol in range(npol):
            res += _dot_complex_alm_1D_arrays(alm_list1[i][ipol], alm_list2[i][ipol], lmax)
    return res

def almlist_dot_real(alm_list1, alm_list2):
    """ `list_inplace = value*list_inplace + list_other`
    """
    res = 0.0
    for i in range(len(alm_list1)):
        npol, nalm = alm_list1[i].shape
        for ipol in range(npol):
            res += dot(alm_list1[i][ipol], alm_list2[i][ipol])
    return res




def nalm(lmax: int, mmax: int) -> int:
    """ Calculates the number of a_lm elements for a spherical harmonic representation up to
        l<=lmax and m<=mmax.
    """
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def gaussian_random_alm(lmax, mmax, spin, ncomp):
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
    res = np.random.normal(0., 1., (ncomp, nalm(lmax, mmax))) \
     + 1j*np.random.normal(0., 1., (ncomp, nalm(lmax, mmax)))
    # make a_lm with m==0 real-valued
    res[:, 0:lmax+1].imag = 0.
    ofs=0
    for s in range(spin):
        res[:, ofs:ofs+spin-s] = 0.
        ofs += lmax+1-s
    res[:, lmax+1:] /= np.sqrt(2.)
    return res


@njit(cache=True, fastmath=True)
def _almxfl_numba(res, lmax, mmax, fl):
    ofs = 0
    for m in range(mmax+1):
        next = ofs + lmax + 1 - m
        res[ofs:next] *= fl[m:lmax+1]
        ofs = next

def almxfl(alm, fl, lmax=None, mmax=None, inplace=False):
    res = alm if inplace else alm.copy()
    lmax = hp.Alm.getlmax(alm.shape[-1]) if lmax is None else lmax
    mmax = lmax if mmax is None else mmax
    _almxfl_numba(res, lmax, mmax, fl)
    return res

# Parallel implementation of almxfl. Feel free to optimize.

# @njit(parallel=True, cache=True, fastmath=True)
# def _almxfl_numba_schedule(alm, lmax, mmax, m_offsets,  fl,  num_threads, inplace=False):
#     res = alm if inplace else alm.copy()

#     for thread_idx in prange(num_threads):
#         for m in range(thread_idx, mmax + 1, num_threads):
#             start = m_offsets[m]
#             end = m_offsets[m+1]
#             num_l = lmax + 1 - m
#             res[start:end] *= fl[m : m + num_l]
#     return res


# def almxfl(alm, fl, lmax=None, mmax=None, inplace=False):
#     res = alm if inplace else alm.copy()
#     lmax = hp.Alm.getlmax(alm.shape[-1]) if lmax is None else lmax
#     mmax = lmax if mmax is None else mmax
#     m_offsets = np.zeros(mmax + 2, dtype=np.int64)
#     for m in range(mmax + 1):
#         m_offsets[m+1] = m_offsets[m] + (lmax - m + 1)
#     n_threads = numba.get_num_threads()
    
#     _almxfl_numba_schedule(alm, lmax, mmax, m_offsets, fl, n_threads, inplace=True)
#     return res


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


def project_alms_mmax(alms_in: NDArray, lmax: int, mmax_in: int, mmax_out: int) -> NDArray:
    """ Projects alms from one mmax to another at fixed lmax, handling truncation or zero-padding.
        Like project_alms, this function is the adjoint of itself. Takes complex alms as input.
        Args:
            alms_in (np.array): Complex alms where the last axis has length nalm(lmax, mmax_in).
            lmax (int): The lmax of the alms, which the projection leaves unchanged.
            mmax_in (int): The mmax of the input alms.
            mmax_out (int): The mmax of the returned alms.
        Returns:
            alms_out (np.array): Complex alms where the last axis has length nalm(lmax, mmax_out).
    """
    if alms_in.shape[-1] != nalm(lmax, mmax_in):
        raise ValueError(f"Alms of length {alms_in.shape[-1]} do not match lmax={lmax}, "
                         f"mmax={mmax_in} (expected {nalm(lmax, mmax_in)}).")
    if mmax_in == mmax_out:
        return alms_in
    # Alms are stored m-major, and at fixed lmax the block for a given m has the same length
    # (lmax+1-m) in both arrays. The shared m <= min(mmax_in, mmax_out) modes are therefore a
    # contiguous prefix of both, so the projection is a single copy rather than a loop over m.
    n_copy = nalm(lmax, min(mmax_in, mmax_out))
    alms_out = np.zeros((*alms_in.shape[:-1], nalm(lmax, mmax_out)), dtype=alms_in.dtype)
    alms_out[..., :n_copy] = alms_in[..., :n_copy]
    return alms_out


def alm_dot_product(alm1: NDArray, alm2: NDArray, lmax: int) -> NDArray:
    """ Function calculating the dot product of two alms, given that they follow the Healpy standard
        where alms are represented as complex numbers, but with the conjugate 'negative' ms missing.
    """
    nm0 = lmax + 1
    return np.sum((alm1[:nm0] * alm2[:nm0]).real)\
        + np.sum((alm1[nm0:] * np.conj(alm2[nm0:])).real * 2)


def alm_complex2real(alm: NDArray[np.complexfloating], lmax: int,
                     mmax: int | None = None) -> NDArray[np.floating]:
    """ Over the last axis of the input array, converts from the complex convention of storing alms
        to the real convention (which is only applicable when the map is real). In the real
        convention, the all m modes are stored, but they are all stored as real values, not complex.
        Args:
            alm (np.array): Complex alm array where the last axis has length nalm(lmax, mmax).
            lmax (int): The lmax of the alm array.
            mmax (int): The mmax of the alm array. Defaults to lmax.
        Returns:
            x (np.array): Real alm array where the last axis has length
                2*nalm(lmax, mmax) - (lmax+1), which is (lmax+1)^2 for the usual mmax = lmax.
    """
    if alm.dtype not in (np.complex128, np.complex64):
        raise TypeError(f"Input alms must have dtype complex64 or complex128, got {alm.dtype}.")
    mmax = lmax if mmax is None else mmax
    if alm.shape[-1] != nalm(lmax, mmax):
        raise ValueError(f"Complex alms of length {alm.shape[-1]} do not match lmax={lmax}, "
                         f"mmax={mmax} (expected {nalm(lmax, mmax)}).")
    float_dtype = np.float64 if alm.dtype == np.complex128 else np.float32
    # Alms are stored m-major, so the real m=0 modes are always the first lmax+1 entries, whatever
    # mmax is. The m>0 entries that follow are split into their real and imaginary parts.
    i = lmax + 1
    return np.concatenate([alm[...,:i].real,sqrt(2.0)*alm[...,i:].view(float_dtype)], axis=-1)


def alm_real2complex(x: NDArray[np.floating], lmax: int,
                     mmax: int | None = None) -> NDArray[np.complexfloating]:
    """ Over the last axis of the input array, converts from the real convention of storing alms
        (which is applicable when the map is real), to the complex convention. In the complex
        convention, the only m>=0 is stored, but are stored as complex numbers (m=0 is always real).
        Args:
            x (np.array): Real alm array where the last axis has length
                2*nalm(lmax, mmax) - (lmax+1), which is (lmax+1)^2 for the usual mmax = lmax.
            lmax (int): The lmax of the alm array.
            mmax (int): The mmax of the alm array. Defaults to lmax.
        Returns:
            oalm (np.array): Complex alm array where the last axis has length nalm(lmax, mmax).
    """
    if x.dtype not in (np.float32, np.float64):
        raise TypeError(f"Input map must have dtype float32 or float64, got {x.dtype}.")
    mmax = lmax if mmax is None else mmax
    nelem = nalm(lmax, mmax)
    if x.shape[-1] != 2*nelem - (lmax+1):
        raise ValueError(f"Real alms of length {x.shape[-1]} do not match lmax={lmax}, "
                         f"mmax={mmax} (expected {2*nelem - (lmax+1)}).")
    complex_dtype = np.complex128 if x.dtype == np.float64 else np.complex64
    # See alm_complex2real: the first lmax+1 entries are the real m=0 modes, the rest are the real
    # and imaginary parts of the m>0 modes.
    i = lmax + 1
    # oalm will have the same shape as x except for the last axis.
    oalm = np.zeros((*x.shape[:-1], nelem), complex_dtype)
    oalm[...,:i] = x[...,:i]
    oalm[...,i:] = x[...,i:].view(complex_dtype)/sqrt(2.0)
    return oalm
