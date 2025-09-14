# general idea is to have heavy commonly used math operations
# in this file. For now they are pretty simple calls to outside functions
# but they could be improved and streamlined later for specific implementations.

import numpy as np
from numpy.typing import NDArray
import healpy as hp
import pysm3.units as u
from pixell import curvedsky
import ducc0
from src.python.output.log import logassert
import logging
import os
from math import sqrt

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
               nthreads: int=1, out=None) -> NDArray:
    alm, out, ndim_in = _prep_input(alm, out, nside, spin)
    out = ducc0.sht.synthesis(alm=alm, map=out, lmax=lmax, spin=spin,
                              nthreads=nthreads, **hp_geominfos[nside])
    return out if ndim_in == 2 else out.reshape((-1,))


def alm_to_map_adjoint(mp: NDArray, nside: int, lmax: int, *, spin: int=0,
                       nthreads: int=1, out=None) -> NDArray:
    mp, out, ndim_in = _prep_input(mp, out, nside, spin)
    out = ducc0.sht.adjoint_synthesis(map=mp, alm=out, lmax=lmax, spin=spin,
                                      nthreads=nthreads, **hp_geominfos[nside])
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


def spherical_beam_to_bl(fwhm: float, lmax: int) -> NDArray:
    # expects FWHM in units of arcmin
    fwhm = (fwhm*u.arcmin).to('rad').value
    return hp.gauss_beam(fwhm, lmax)


def spherical_beam_applied_to_alm(alm: NDArray, fwhm: float) -> NDArray:
    # expects FWHM in units of arcmin
    fwhm = (fwhm*u.arcmin).to('rad').value
    return hp.smoothalm(alm, fwhm)


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


def forward_rfft(data:NDArray[np.floating], nthreads:int = 1):
    """ Forward real Fourier transform, equivalent to scipy.fft.rfft.
        Args:
            data (np.array): Real-valued data array to be Fourier transformed.
            nthreads (int): Number of threads to use.
        Returns:
            data_fft (np.array): The Fourier transform of the input.
                                 A complex array of length tod.size//2 + 1.
    """
    return ducc0.fft.r2c(data, nthreads=nthreads)

def backward_rfft(data_f:NDArray, ntod:int, nthreads:int = None) -> NDArray[np.floating]:
    """ Backward real Fourier transform, equivalent to scipy.fft.irfft.
        Args:
            data_fft (np.array): Complex Fourier coefficients to be converted back to real data.
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