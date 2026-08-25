"""Gaussian beam profiles in map space.

The instrument beam enters in several places - smoothing a map to a common resolution, painting a
point source, building a preconditioner - and all of them want the same normalized Gaussian.
"""
import numpy as np


def fwhm2sigma(fwhm):
    return fwhm/(2*np.sqrt(2*np.log(2)))

def gauss_beam(x, fwhm):
    """
    Gaussian integral-normalized beam in map space.
    Be CAREFUL giving `x` and `fwhm` in the same units of measure. 
    """
    sigma = fwhm2sigma(fwhm)
    return 1/(2*np.pi*sigma**2)*np.exp(-(x)**2 / (2*sigma**2))

def get_gauss_beam_radius(fwhm, frac=1e-4):
    """
    Finds the distance from the center of a gaussian beam corresponding to a fraction 'frac'
    of the peak intensity.
    """
    sigma = fwhm2sigma(fwhm)
    return sigma * np.sqrt( - 2* np.log(frac))
