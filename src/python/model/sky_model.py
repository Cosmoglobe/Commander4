import numpy as np

class SkyModel:
    def __init__(self, components):
        # components = list of Component objects
        self.components = components

    def get_sky(self, band):
        """ Get sky from a bandpass.
        """

    def get_sky_at_nu(self, nu, npix):
        """ Get sky at specific frequency.
        """
        skymap = np.zeros((npix))
        for component in self.components:
            skymap += component.get_sky(nu)
        return skymap