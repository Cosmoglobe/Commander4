# general idea is to have heavy commonly used math operations
# in this file. For now they are pretty simple calls to outside functions
# but they could be improved and streamlined later for specific implementations.

import numpy as np
from numpy.typing import NDArray
import healpy as hp
import pysm3.units as u
from pixell import curvedsky
import ducc0


def nalm(lmax: int, mmax: int) -> int:
    """ Calculates the number of a_lm elements for a spherical harmonic representation up to l<=lmax and m<=mmax.
    """
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


# MR FIXME: I'm not absolutely sure that this is fully correct. Please double-check!
def gaussian_random_alm(lmax, mmax, spin, ncomp):
    """ Calculates Gaussianly distributed alms for the complex alm convension (not storing m<0 because map is real.)
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

    if arr_in.ndim !=2 or arr_out.ndim != 2:
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


def alm_complex2real(alm: NDArray[np.complex128], lmax: int) -> NDArray[np.float64]:
    """ Coverts from the complex convention of storing alms when the map is real, to the real convention.
        In the real convention, the all m modes are stored, but they are all stored as real values, not complex.
        Args:
            alm (np.array): Complex alm array of length ((lmax+1)*(lmax+2))/2.
            lmax (int): The lmax of the alm array.
        Returns:
            x (np.array): Real alm array of length (lmax+1)^2.
    """
    ainfo = curvedsky.alm_info(lmax=lmax)
    i = int(ainfo.mstart[1]+1)
    return np.concatenate([alm[:i].real,np.sqrt(2.)*alm[i:].view(np.float64)])


def alm_real2complex(x: NDArray[np.float64], lmax: int) -> NDArray[np.complex128]:
    """ Coverts from the real convention of storing alms when the map is real, to the complex convention.
        In the complex convention, the only m>=0 is stored, but are stored as complex numbers (m=0 still always real).
        Args:
            x (np.array): Real alm array of length (lmax+1)^2.
            lmax (int): The lmax of the alm array.
        Returns:
            oalm (np.array): Complex alm array of length ((lmax+1)*(lmax+2))/2.
    """
    ainfo = curvedsky.alm_info(lmax=lmax)
    i    = int(ainfo.mstart[1]+1)
    oalm = np.zeros(ainfo.nelem, np.complex128)
    oalm[:i] = x[:i]
    oalm[i:] = x[i:].view(np.complex128)/np.sqrt(2.)
    return oalm
