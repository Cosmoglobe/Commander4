import astropy.units as u
import astropy.constants as c
import numpy as np
import pysm3.units as pysm3u

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
    def __init__(self, params):
        self.params = params
        self.longname = params.longname if "longname" in params else "Unknown Component"
        self.shortname = params.shortname if "shortname" in params else "comp"

# Second tier component classes
class DiffuseComponent(Component):
    def __init__(self, params):
        super().__init__(params)
        self._component_map = None
        self.nside_comp_map = 2048
        self.prior_l_power_law = 0 # l^alpha as in S^-1 in comp sep

    @property
    def component_map(self):
        if self._component_map is None:
            raise ValueError("component_map property not set.")
        return self._component_map

    @component_map.setter
    def component_map(self, map):
        # if not self._component_map is None:  # Temporarily disabled, might want to add back later.
        #     raise ValueError("DiffuseComponent does not allow for overwriting already set component_map parameter.")
        self._component_map = map

    def get_sky(self, nu):
        return self.component_map*self.get_sed(nu)

class PointSourceComponent(Component):
    pass

class TemplateComponent(Component):
    pass

# Third tier component classes
class CMB(DiffuseComponent):
    def __init__(self, params):
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
    def __init__(self, params):
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
    def __init__(self, params):
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
