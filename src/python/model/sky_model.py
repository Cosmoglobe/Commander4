import numpy as np
from .component import CMB

class SkyModel:
    def __init__(self, components):
        # components = list of Component objects
        self.components = components

    def get_sky(self, band):
        """ Get sky from a bandpass.
        """

    def get_sky_at_nu(self, nu, nside, fwhm=None):
        """ Get sky at specific frequency.
        """
        npix = 12*nside**2
        skymap = np.zeros((npix))
        for component in self.components:
            skymap += component.get_sky(nu, nside, fwhm)
        return skymap
    
    def get_foreground_sky_at_nu(self, nu, nside, fwhm=None):
        """ Get sky, excluding the cmb, at specific frequency.
        """
        npix = 12*nside**2
        skymap = np.zeros((npix))
        for component in self.components:
            if not isinstance(component, CMB):
                skymap += component.get_sky(nu, nside, fwhm)
        return skymap
    
    def get_cmb_sky_at_nu(self, nu, nside, fwhm=None):
        """ Get sky, excluding the cmb, at specific frequency.
        """
        npix = 12*nside**2
        skymap = np.zeros((npix))
        for component in self.components:
            if isinstance(component, CMB):
                skymap += component.get_sky(nu, nside, fwhm)
        return skymap