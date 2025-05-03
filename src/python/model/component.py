import astropy.units as u
import astropy.constants as c
import numpy as np
import pysm3.units as pysm3u
import healpy as hp
from pixell.bunch import Bunch
import logging

from src.python.output import log
from src.python.utils.math_operations import alm_to_map


A = (2*c.h*u.GHz**3/c.c**2).to('MJy').value
h_over_k = (c.h/c.k_B/(1*u.K)).to('GHz-1').value
h_over_kTCMB = (c.h/c.k_B/(2.7255*u.K)).to('GHz-1').value
def blackbody(nu, T):
    return A*nu**3/np.expm1(nu*h_over_k/T)
def g(nu):
    # From uK_CMB to MJy/sr
    x = nu*h_over_kTCMB
    return np.expm1(x)**2/(x**4*np.exp(x))


# First tier component classes
class Component:
    def __init__(self, params: Bunch):
        self.params = params
        self.lmax = params.lmax
        self.longname = params.longname if "longname" in params else "Unknown Component"
        self.shortname = params.shortname if "shortname" in params else "comp"

# Second tier component classes
class DiffuseComponent(Component):
    def __init__(self, params: Bunch):
        super().__init__(params)
        self.component_alms = None
        # self.nside_comp_map = 2048
        # self.prior_l_power_law = 0 # l^alpha as in S^-1 in comp sep

    def get_component_map(self, nside:int, fwhm:int=0):
        if self.component_alms is None:
            raise ValueError("component_alms property not set.")
        if fwhm == 0:
            return alm_to_map(self.component_alms, nside, self.lmax)
        else:
            return alm_to_map(hp.smoothalm(self.component_alms, fwhm), nside, self.lmax)

    def get_sky(self, nu, nside, fwhm=None):
        return self.get_component_map(nside, fwhm)*self.get_sed(nu)
    
    def get_sed(self, nu):
        logger = logging.getLogger(__name__)
        log.lograise(NotImplementedError, "", logger)

class PointSourceComponent(Component):
    pass

class TemplateComponent(Component):
    pass

# Third tier component classes
class CMB(DiffuseComponent):
    def __init__(self, params: Bunch):
        super().__init__(params)
        self.longname = params.longname if "longname" in params else "CMB"
        self.shortname = params.shortname if "shortname" in params else "cmb"

    def get_sed(self, nu):
        # Assuming we are working in uK_CMB units
        return (np.ones_like(nu)*pysm3u.uK_CMB).to("uK_RJ", equivalencies=pysm3u.cmb_equivalencies(nu*u.GHz)).value

class RadioSource(PointSourceComponent):
    pass

class CMBRelQuad(TemplateComponent):
    pass

class ThermalDust(DiffuseComponent):
    def __init__(self, params: Bunch):
        super().__init__(params)
        self.beta = params.beta
        self.T = params.T
        self.nu0 = params.nu0
        self.prior_l_power_law = 2.5
        self.longname = params.longname if "longname" in params else "Thermal Dust"
        self.shortname = params.shortname if "shortname" in params else "dust"


    def get_sed(self, nu):
        # Modified blackbody, in uK_CMB
        return ((nu/self.nu0)**self.beta * blackbody(nu, self.T)/blackbody(self.nu0, self.T)*pysm3u.uK_CMB).to("uK_RJ", equivalencies=pysm3u.cmb_equivalencies(nu*u.GHz)).value


class Synchrotron(DiffuseComponent):
    def __init__(self, params: Bunch):
        super().__init__(params)
        self.beta = params.beta
        self.nu0 = params.nu0
        self.nside_comp_map = 512
        self.prior_l_power_law = -3
        self.longname = params.longname if "longname" in params else "Synchrotron"
        self.shortname = params.shortname if "shortname" in params else "sync"

    def get_sed(self, nu):
        # power law with spectral index beta
        return ((nu/self.nu0)**self.beta*pysm3u.uK_CMB).to("uK_RJ", equivalencies=pysm3u.cmb_equivalencies(nu*u.GHz)).value
