# general idea is to have heavy commonly used math operations
# in this file. For now they are pretty simple calls to outside functions
# but they could be improved and streamlined later for specific implementations.

import numpy as np
from numpy.typing import NDArray
import healpy as hp
import pysm3.units as u
from pixell import curvedsky
import ducc0
from commander4.output.log import logassert
import logging
import os
from math import sqrt
from numba import njit, prange
from scipy.linalg import blas as blas_wrapper

import typing
if typing.TYPE_CHECKING:  # Only import when performing type checking, avoiding circular import during normal runtime.
    from src.python.solvers.comp_sep_solvers import CompSepSolver
    from src.python.sky_models.component import Component


###### NUMPY REPLACEMENTS ######
# Collection of Numba functions and BLAS wrappers for simply array manipulation.
# These exist because 1. Numpy is not threaded, which is a problem on the comp-sep module which has
# a lot of cores available, and 2. Certain Numpy operations create copies, these functions do not.

AXPY_ROUTINES = {
    np.dtype('float32'): blas_wrapper.saxpy,
    np.dtype('float64'): blas_wrapper.daxpy,
    np.dtype('complex64'): blas_wrapper.caxpy,
    np.dtype('complex128'): blas_wrapper.zaxpy,
}
def inplace_axpy(inplace_array, add_array, multiply_value):
    """`inplace_array += add_array*multiply_value`. Performs in-place scaled vector addition using
    BLAS AXPY routines. Support f32, 64, c64, and c128 data types, but all arguments must match.
    NB: Seems to fail for arrays larger than 2**32, which is a bit of an issue...
    """
    if inplace_array.size == 0: return
    assert(inplace_array.shape == add_array.shape)
    assert(inplace_array.dtype == add_array.dtype)
    assert(inplace_array.ndim == add_array.ndim)
    # Select the Correct BLAS Routine
    axpy_func = AXPY_ROUTINES[inplace_array.dtype]
    axpy_func(x=add_array, y=inplace_array, n=inplace_array.size, a=multiply_value)


@njit(fastmath=True, parallel=True)
def inplace_scale_add(arr_main, arr_add, float_mult):
    assert(arr_main.shape==arr_add.shape)
    flat1 = arr_main.ravel()
    flat2 = arr_add.ravel()
    for i in prange(arr_main.size):
        flat1[i] = flat1[i]*float_mult + flat2[i]

@njit(fastmath=True)
def inplace_add_scaled_vec_serial(arr_main, arr_add, float_mult):
    assert(arr_main.shape==arr_add.shape)
    flat1 = arr_main.ravel()
    flat2 = arr_add.ravel()
    for i in range(arr_main.size):
        flat1[i] += flat2[i]*float_mult

@njit(fastmath=True, parallel=True)
def inplace_add_scaled_vec(arr_main, arr_add, float_mult):
    assert(arr_main.shape==arr_add.shape)
    flat1 = arr_main.ravel()
    flat2 = arr_add.ravel()
    for i in prange(arr_main.size):
        flat1[i] += flat2[i]*float_mult

@njit(fastmath=True, parallel=True)
def inplace_arr_add(arr_main, arr_add):
    assert(arr_main.shape==arr_add.shape)
    flat1 = arr_main.ravel()
    flat2 = arr_add.ravel()
    for i in prange(arr_main.size):
        flat1[i] += flat2[i]

@njit(fastmath=True, parallel=True)
def inplace_arr_sub(arr_main, arr_add):
    assert(arr_main.shape==arr_add.shape)
    flat1 = arr_main.ravel()
    flat2 = arr_add.ravel()
    for i in prange(arr_main.size):
        flat1[i] -= flat2[i]

@njit(fastmath=True, parallel=True)
def inplace_arr_prod(arr_main, arr_prod):
    len = arr_main.size
    assert(arr_main.shape==arr_prod.shape)
    flat1 = arr_main.ravel()
    flat2 = arr_prod.ravel()
    for i in prange(len):
        flat1[i] *= flat2[i]

@njit(fastmath=True, parallel=True)
def inplace_arr_truediv(arr_main, arr_prod):
    len = arr_main.size
    assert(arr_main.shape==arr_prod.shape)
    flat1 = arr_main.ravel()
    flat2 = arr_prod.ravel()
    for i in prange(len):
        flat1[i] /= flat2[i]

@njit(fastmath=True, parallel=True)
def inplace_scale(arr_main, scalar_prod):
    len = arr_main.size
    flat1 = arr_main.ravel()
    for i in prange(len):
        flat1[i] *= scalar_prod

@njit(fastmath=True, parallel=True)
def dot(arr1, arr2):
    len = arr1.size
    res = 0.0
    flat1 = arr1.ravel()
    flat2 = arr2.ravel()
    for i in prange(len):
        res += flat1[i]*flat2[i]
    return res


@njit(fastmath=True)
def calculate_sigma0(tod: NDArray, mask: NDArray[np.bool_]) -> float:
    """
    Calcualtes the white noise level of the "tod" array, using only elements where the boolean
    array "mask" is True. Uses the std(tod[1:] - tod[:-1])/sqrt(2) trick to get sigma0.
    Args:
        tod: The input time-ordered data array.
        mask: A boolean mask array of the same size as tod.
    Returns:
        The calculated sigma value, or np.inf if fewer than two valid data points exist.
    """
    assert tod.shape == mask.shape, "Input shapes don't match"
    # Variables in "Welford's online algorithm" for variance calculation
    count = 0
    mean = 0.0
    m2 = 0.0  # Sum of squares of differences from the current mean
    last_valid_val = 0.0
    has_first_val = False  # Track whether we have hit first non-masked value.
    for i in range(tod.size):
        if mask[i]:
            current_val = tod[i]
            if not has_first_val:
                # First valid value found.
                last_valid_val = current_val
                has_first_val = True
            else:
                diff = current_val - last_valid_val
                count += 1
                delta = diff - mean
                mean += delta / count
                delta2 = diff - mean
                m2 += delta * delta2
                # The current value becomes the last valid value for the next pair.
                last_valid_val = current_val
    if count == 0:
        return np.inf
    var = m2 / count
    std_dev = sqrt(var)
    return float(std_dev/sqrt(2.0))


@njit(fastmath=True, parallel=True)
def _dot_complex_alm_1D_arrays(alm1: NDArray, alm2: NDArray, lmax: int) -> NDArray:
    """ Function calculating the dot product of two alms, given that they follow the Healpy standard,
        where alms are represented as complex numbers, but with the conjugate 'negative' ms missing.
    """
    return np.sum((alm1[:lmax]*alm2[:lmax]).real) + np.sum((alm1[lmax:]*np.conj(alm2[lmax:])).real*2)

#Specific function for point sources:
@njit(fastmath=True, parallel=True)
def _numba_proj2map(map, pix_disc_idx_list, beam_disc_val_list, amps, sed_s=None):
    for src_i in prange(len(pix_disc_idx_list)):
        map[pix_disc_idx_list[src_i]] += beam_disc_val_list[src_i] * amps[src_i] * (sed_s[src_i] if sed_s is not None else 1)
    return map

@njit(fastmath=True, parallel=True)
def _numba_eval_from_map(map, pix_disc_idx_list, beam_disc_val_list, amps, sed_s=None):
    for src_i in range(len(pix_disc_idx_list)):
            amps[src_i] = np.sum(map[pix_disc_idx_list[src_i]] * beam_disc_val_list[src_i]) * (sed_s[src_i] if sed_s is not None else 1)
    return amps

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


###### COMP-LIST FUNCTIONS ######
# These functions are common array operations, but made to work on the comp-lists, which are
# lists of Component objects, each containing component-specifically formatted data.

def inplace_complist_add_scaled_array(list_inplace:list["Component"], list_other:list["Component"], scalar):
    """ `list_inplace += scalar*list_other`
    """
    if len(list_inplace) != len(list_other):
        raise ValueError("Component lists must match in length.")
    
    for ci, co in zip(list_inplace, list_other):
        inplace_add_scaled_vec(ci._data, co._data, scalar)

def inplace_complist_scale_and_add(list_inplace:list["Component"], list_other:list["Component"], scalar):
    """ `list_inplace = scalar*list_inplace + list_other`
    """
    if len(list_inplace) != len(list_other):
        raise ValueError("Component lists must match in length.")

    for ci, co in zip(list_inplace, list_other):
        inplace_scale_add(ci._data, co._data, scalar)

def complist_dot(comp_list1:list["Component"], comp_list2:list["Component"]) -> float:
    """ `dot(comp_list1, comp_list2)`. Calculates the correct dot product between two lists of Component objects 
        where the alms follow the Healpy complex storing convention, for components with alms.
        It will automatically handle the correct dot product definition for each type of Component.
    """
    if len(comp_list1) != len(comp_list2):
        raise ValueError("Component lists must match in length.")
    if len(comp_list1) == 0:
        print("WARNING dot prod between empty comp list")
    res = 0.0
    for c1, c2 in zip(comp_list1, comp_list2):
        res += float(c1 @ c2)
    return res

def complist_norm(comp_list:list["Component"]) -> float:
    """ `norm(comp_list1, comp_list2)`. Calculates the Euclidean norm of a lists of Component objects,
        handling it as it was a single vectors of values.
    """
    return complist_dot(comp_list, comp_list)


###### GENERAL MATH STUFF ######

def forward_rfft(data:NDArray[np.floating], nthreads:int = 1):
    """ Forward real Fourier transform, equivalent to scipy.fft.rfft.
        Args:
            data (np.array): Real-valued data array to be Fourier transformed.
            nthreads (int): Number of threads to use.
        Returns:
            data_f (np.array): The Fourier transform of the input.
                               A complex array of length tod.size//2 + 1.
    """
    return ducc0.fft.r2c(data, nthreads=nthreads)

def backward_rfft(data_f:NDArray, ntod:int, nthreads:int = None) -> NDArray[np.floating]:
    """ Backward real Fourier transform, equivalent to scipy.fft.irfft.
        Args:
            data_f (np.array): Complex Fourier coefficients to be converted back to real data.
            ntod (int): The length of the original TOD. This must be provided because a
                           Fourier array of length e.g. 6 could correspond to ntod = 10 or 11.
            nthreads (int): Number of threads to use.
        Returns:
            data (np.array): A real-valued data array of length ntod.
    """
    # If nthreads is not set, put it to how many threads OMP has been given.
    nthreads = int(os.environ["OMP_NUM_THREADS"]) if nthreads is None else nthreads
    # Forward = False makes ducc correctly order the output, as the output order is not
    # symmetric for forward and reverse Fourier when doing rfft as supposed to regular fft.
    # inorm = 2 tells ducc to normalize by dividing by ntod, which is the same as what scipy does.
    return ducc0.fft.c2r(data_f, lastsize=ntod, forward=False, nthreads=nthreads, inorm=2)


##### GENERAL ALM STUFF ############

def nalm(lmax: int, mmax: int) -> int:
    """Calculates the number of a_lm elements for a spherical harmonic representation up to l<=lmax and m<=mmax.
    """
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


# MR FIXME: I'm not absolutely sure that this is fully correct. Please double-check!
def gaussian_random_alm(lmax, mmax, spin, ncomp):
    """Calculates Gaussianly distributed alms for the complex alm convension (not storing m<0 because map is real.)
    """
    res = np.random.normal(0., 1., (ncomp, nalm(lmax, mmax))) \
     + 1j*np.random.normal(0., 1., (ncomp, nalm(lmax, mmax)))
    # make a_lm with m==0 real-valued
    res[:, 0:lmax+1].imag = 0.
    ofs=0
    for s in range(spin):
        res[:, ofs:ofs+spin-s] = 0.
        ofs += lmax+1-s
    res[lmax+1:] *= np.sqrt(2.)
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


def alm_dot_product(alm1: NDArray, alm2: NDArray, lmax: int) -> NDArray:
    """ Function calculating the dot product of two alms, given that they follow the Healpy standard,
        where alms are represented as complex numbers, but with the conjugate 'negative' ms missing.
    """
    return np.sum((alm1[:lmax]*alm2[:lmax]).real) + np.sum((alm1[lmax:]*np.conj(alm2[lmax:])).real*2)


def alm_complex2real(alm: NDArray[np.complexfloating], lmax: int) -> NDArray[np.floating]:
    """ Over the last axis of the input array, converts from the complex convention of storing alms
        to the real convention (which is only applicable when the map is real). In the real
        convention, the all m modes are stored, but they are all stored as real values, not complex.
        Args:
            alm (np.array): Complex alm array where the last axis has length ((lmax+1)*(lmax+2))/2.
            lmax (int): The lmax of the alm array.
        Returns:
            x (np.array): Real alm array where the last axis has length (lmax+1)^2.
    """
    logger = logging.getLogger(__name__)
    logassert(alm.dtype in [np.complex128, np.complex64], "Input alms are not of type np.complex128"
             f" or np.complex64  (they are {alm.dtype})", logger)
    float_dtype = np.float64 if alm.dtype == np.complex128 else np.float32
    ainfo = curvedsky.alm_info(lmax=lmax)
    i = int(ainfo.mstart[1]+1)
    return np.concatenate([alm[...,:i].real,sqrt(2.0)*alm[...,i:].view(float_dtype)], axis=-1)


def alm_real2complex(x: NDArray[np.floating], lmax: int) -> NDArray[np.complexfloating]:
    """ Over the last axis of the input array, converts from the real convention of storing alms
        (which is applicable when the map is real), to the complex convention. In the complex
        convention, the only m>=0 is stored, but are stored as complex numbers (m=0 is always real). 
        Args:
            x (np.array): Real alm array where the last axis has length (lmax+1)^2.
            lmax (int): The lmax of the alm array.
        Returns:
            oalm (np.array): Complex alm array where the last axis has length ((lmax+1)*(lmax+2))/2.
    """
    logger = logging.getLogger(__name__)
    logassert(x.dtype in [np.float32, np.float64], f"Input map is not of type np.float32 or "
              f"np.float64 (it is {x.dtype})", logger)
    complex_dtype = np.complex128 if x.dtype == np.float64 else np.complex64
    ainfo = curvedsky.alm_info(lmax=lmax)
    i    = int(ainfo.mstart[1]+1)
    # oalm will have the same shape as x except for the last axis.
    oalm = np.zeros((*x.shape[:-1], ainfo.nelem), complex_dtype)
    oalm[...,:i] = x[...,:i]
    oalm[...,i:] = x[...,i:].view(complex_dtype)/sqrt(2.0)
    return oalm


############ SHT STUFF ##############

# Cache for geom_info objects ... pretty small, each entry has a size of O(nside)
# This will be mainly beneficial for small SHTs with high nthreads
hp_geominfos = {}

def _prep_input(arr_in, arr_out, nside, spin):
    ndim_in = arr_in.ndim
    if spin == 0 and ndim_in == 1:
        arr_in = arr_in.reshape((1,-1))
        if arr_out is not None:
            arr_out = arr_out.reshape((1,-1))

    if arr_in.ndim !=2 or (arr_out is not None and arr_out.ndim != 2):
        raise RuntimeError("bad array dimensionality") 

    if nside not in hp_geominfos:
        hp_geominfos[nside] = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()

    return arr_in, arr_out, ndim_in


def alm_to_map(alm: NDArray, nside: int, lmax: int, *, spin: int=0,
               nthreads: int=1, out=None, acc: bool=False) -> NDArray:
    use_theta_interpol = nside >= 2048
    alm, out, ndim_in = _prep_input(alm, out, nside, spin)
    if acc:
        if out is None:
            raise RuntimeError("Can not accumulate to None output")
        tmp_out = np.copy(out)
    out = ducc0.sht.synthesis(alm=alm, map=out, lmax=lmax, spin=spin,
                              nthreads=nthreads, **hp_geominfos[nside],
                              theta_interpol=use_theta_interpol)
    if acc:
        inplace_arr_add(out, tmp_out)
    return out if ndim_in == 2 else out.reshape((-1,))


def alm_to_map_adjoint(mp: NDArray, nside: int, lmax: int, *, spin: int=0,
                       nthreads: int=1, out=None, acc: bool=False) -> NDArray:
    use_theta_interpol = nside >= 2048
    mp, out, ndim_in = _prep_input(mp, out, nside, spin)
    if acc:
        if out is None:
            raise RuntimeError("Can not accumulate to None output")
        tmp_out = np.copy(out)
    out = ducc0.sht.adjoint_synthesis(map=mp, alm=out, lmax=lmax, spin=spin,
                                      nthreads=nthreads, **hp_geominfos[nside],
                                      theta_interpol=use_theta_interpol)
    if acc:
        inplace_arr_add(out, tmp_out)
    return out if ndim_in == 2 else out.reshape((-1,))


def map_to_alm(mp: NDArray, nside: int, lmax: int, *, spin: int=0,
                       nthreads: int=1, out=None, acc: bool=False) -> NDArray:
    use_theta_interpol = nside >= 2048
    mp, out, ndim_in = _prep_input(mp, out, nside, spin)
    if acc:
        if out is None:
            raise RuntimeError("Can not accumulate to None output")
        tmp_out = np.copy(out)
    out = ducc0.sht.adjoint_synthesis(map=mp, alm=out, lmax=lmax, spin=spin,
                                      nthreads=nthreads, **hp_geominfos[nside],
                                      theta_interpol=use_theta_interpol)
    out *= 4*np.pi/(12*nside**2)
    if acc:
        inplace_arr_add(out, tmp_out)
    return out if ndim_in == 2 else out.reshape((-1,))


def map_to_alm_adjoint(alm: NDArray, nside: int, lmax: int, *, spin: int=0,
               nthreads: int=1, out=None, acc: bool=False) -> NDArray:
    use_theta_interpol = nside >= 2048
    alm, out, ndim_in = _prep_input(alm, out, nside, spin)
    if acc:
        if out is None:
            raise RuntimeError("Can not accumulate to None output")
        tmp_out = np.copy(out)
    out = ducc0.sht.synthesis(alm=alm, map=out, lmax=lmax, spin=spin,
                              nthreads=nthreads, **hp_geominfos[nside],
                              theta_interpol=use_theta_interpol)
    out *= 4*np.pi/(12*nside**2)
    if acc:
        inplace_arr_add(out, tmp_out)
    return out if ndim_in == 2 else out.reshape((-1,))


def pseudo_alm_to_map_inverse(map: NDArray, nside: int, lmax: int, *, spin: int=0,
               nthreads: int=1, out=None, epsilon: float, maxiter: int) -> NDArray:
    """Tries to extract spherical harmonic coefficients from (sets of) one or two maps
    by using the iterative LSMR algorithm.
    
    Parameters
    ----------
    map: numpy.ndarray(([ncomp,] 12*nside**2), dtype=numpy.float32 or numpy.float64
    nside: int
        nside parameter of the Healpix map
    lmax: int >= 0
        the maximum l moment of the transform (inclusive).
    spin: int >= 0
        the spin to use for the transform.
        If spin==0, ncomp must be 1, otherwise 2
    nthreads: int >= 0
        the number of threads to use for the computation
        if 0, use as many threads as there are hardware threads available on the system
    out: None or numpy.ndarray([ncomp,] (lmax+1)*(lmax+2)//2),
         dtype=numpy.complex of same precision as `map`)
        the set of spherical harmonic coefficients.
        if `None`, a new suitable array is allocated
    epsilon: float > 0
        the relative tolerance used as a stopping criterion
    maxiter: int >= 0
        the maximum number of iterations before stopping the algorithm
    
    Returns
    -------
    numpy.ndarray(([ncomp,] (lmax+1)*(lmax+2)//2), dtype=numpy.complex of same accuracy as `map`)
        the set of spherical harmonic coefficients.
        If `out` was supplied, this will be the same object
    
    int:
        the reason for stopping the iteration
        1: approximate solution to the equation system found
        2: approximate least-squares solution found
        3: condition number of the equation system too large
        7: maximum number of iterations reached
    
    int:
        the iteration count
    
    float:
        the residual norm, divided by the norm of `map`
    
    float:
        the quality of the least-squares solution
    """
    map, out, ndim_in = _prep_input(map, out, nside, spin)
    res = ducc0.sht.pseudo_analysis(map=map, alm=out, lmax=lmax, spin=spin,
                                    nthreads=nthreads, **hp_geominfos[nside],
                                    epsilon=epsilon, maxiter=maxiter)
    out = res[0] if ndim_in == 2 else res[0].reshape((-1,))
    return (out, res[1], res[2], res[3], res[4])
