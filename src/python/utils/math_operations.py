# general idea is to have heavy commonly used math operations
# in this file. For now they are pretty simple calls to outside functions
# but they could be improved and streamlined later for specific implementations.

import numpy as np
import healpy as hp
import pysm3.units as u


def alm_to_map(alm: np.array, nside: int) -> np.array:
    return hp.alm2map(alm, nside)

def map_to_alm(mp: np.array) -> np.array:
    return hp.map2alm(mp)

def spherical_beam_to_bl(fwhm: float, lmax: int) -> np.array:
    # expects FWHM in units of arcmin
    fwhm = (fwhm*u.arcmin).to('rad').value
    return hp.gauss_beam(fwhm, lmax)

def spherical_beam_applied_to_alm(alm: np.array, fwhm: float) -> np.array:
    # expects FWHM in units of arcmin
    fwhm = (fwhm*u.arcmin).to('rad').value
    return hp.smoothalm(alm, fwhm)

