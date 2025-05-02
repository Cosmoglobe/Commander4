# general idea is to have heavy commonly used math operations
# in this file. For now they are pretty simple calls to outside functions
# but they could be improved and streamlined later for specific implementations.

import numpy as np
import healpy as hp
import pysm3.units as u
import ducc0


def nalm(lmax, mmax):
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


def alm_to_map(alm: np.array, nside: int, lmax: int, nthreads=1) -> np.array:
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    return ducc0.sht.synthesis(alm=alm.reshape((1,-1)),
                               lmax=lmax,
                               spin=0,
                               nthreads=nthreads, **geom).reshape((-1,))


def alm_to_map_adjoint(mp: np.array, nside: int, lmax: int, nthreads=1) -> np.array:
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    return ducc0.sht.adjoint_synthesis(map=mp.reshape((1,-1)),
                                       lmax=lmax,
                                       spin=0,
                                       nthreads=nthreads, **geom).reshape((-1,))


def spherical_beam_to_bl(fwhm: float, lmax: int) -> np.array:
    # expects FWHM in units of arcmin
    fwhm = (fwhm*u.arcmin).to('rad').value
    return hp.gauss_beam(fwhm, lmax)


def spherical_beam_applied_to_alm(alm: np.array, fwhm: float) -> np.array:
    # expects FWHM in units of arcmin
    fwhm = (fwhm*u.arcmin).to('rad').value
    return hp.smoothalm(alm, fwhm)


def alm_dot_product(alm1: np.array, alm2: np.array, lmax: int) -> np.array:
    """ Function calculating the dot product of two alms, given that they follow the Healpy standard,
        where alms are represented as complex numbers, but with the conjugate 'negative' ms missing.
    """
    return np.sum((alm1[:lmax]*alm2[:lmax]).real) + np.sum((alm1[lmax:]*np.conj(alm2[lmax:])).real*2)

