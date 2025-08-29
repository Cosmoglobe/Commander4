import numpy as np
from .component import CMB


class SkyModel:
    def __init__(self, components):
        # components = list of Component objects
        self._components = components

    def get_sky(self, band):
        """ Get sky from a bandpass.
        """
        raise NotImplementedError

    def get_sky_at_nu(self, nu, nside, fwhm=None, pol=(True, True, True)):
        """ Get sky at specific frequency.
        """
        npix = 12*nside**2
        npol = np.sum(pol)
        skymap = np.zeros((npol, npix))
        for component in self._components:
            skymap[0] += component.get_sky(nu, nside, False, fwhm)[0]
            if component.polarized:
                skymap[1:] += component.get_sky(nu, nside, True, fwhm)
        return skymap
    
# class SkyModel:
#     def __init__(self, components_I, components_Q=None, components_U=None):
#         # components = list of Component objects
#         self.components_I = components_I
#         self.components_Q = components_Q
#         self.components_U = components_U
#         self.all_components = [components_I, components_Q, components_U]
#         self.comp_names = ["I", "Q", "U"]

#     def get_sky(self, band):
#         """ Get sky from a bandpass.
#         """
#         raise NotImplementedError

#     def get_sky_at_nu(self, nu, nside, fwhm=None, pol=(True,True,True)):
#         """ Get sky at specific frequency.
#         """
#         npix = 12*nside**2
#         skymap = np.zeros((np.sum(pol), npix))
#         idx = 0
#         for ipol in range(3):
#             if pol[ipol]:
#                 if self.all_components[ipol] is None:
#                     raise ValueError("Attempted to create sky model containing "
#                                      f"{self.comp_names[ipol]} but this component is not set.")
#                 for component in self.all_components[ipol]:
#                     skymap[idx] += component.get_sky(nu, nside, ipol, fwhm)
#                 idx += 1
#         return skymap
    
    # def get_foreground_sky_at_nu(self, nu, nside, fwhm=None):
    #     """ Get sky, excluding the cmb, at specific frequency.
    #     """
    #     npix = 12*nside**2
    #     skymap = np.zeros((npix))
    #     for component in self.components:
    #         if not isinstance(component, CMB):
    #             skymap += component.get_sky(nu, nside, fwhm)
    #     return skymap
    
    # def get_cmb_sky_at_nu(self, nu, nside, fwhm=None):
    #     """ Get sky, excluding the cmb, at specific frequency.
    #     """
    #     npix = 12*nside**2
    #     skymap = np.zeros((npix))
    #     for component in self.components:
    #         if isinstance(component, CMB):
    #             skymap += component.get_sky(nu, nside, fwhm)
    #     return skymap