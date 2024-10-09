# general idea is to have heavy commonly used math operations
# in this file. For now they are pretty simple calls to outside functions
# but they could be improved and streamlined later for specific implementations.

import numpy as np
import healpy as hp
import pysm3.units as u
import ducc0


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

