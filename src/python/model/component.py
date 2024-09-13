import astropy.units as u
import astropy.constants as c
import numpy as np

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
    def __init__(self):
        pass

# Second tier component classes
class DiffuseComponent(Component):
    def __init__(self):
        self._component_map = None

    @property
    def component_map(self):
        if self._component_map is None:
            raise ValueError("component_map property not set.")
        return self._component_map

    @component_map.setter
    def component_map(self, map):
        if not self._component_map is None:
            raise ValueError("DiffuseComponent does not allow for overwriting already set component_map parameter.")
        self._component_map = map

    def get_sky(self, nu):
        return self.component_map*self.get_sed(nu)

class PointSourceComponent(Component):
    pass

class TemplateComponent(Component):
    pass

# Third tier component classes
class CMB(DiffuseComponent):
    def get_sed(self, nu):
        # Assuming we are working in uK_CMB units
        return np.ones_like(nu)

class RadioSource(PointSourceComponent):
    pass

class CMBRelQuad(TemplateComponent):
    pass

class ThermalDust(DiffuseComponent):
    def __init__(self):
        super().__init__()
        self.beta = 1.5
        self.T = 20.0
        self.nu0 = 857.0

    def get_sed(self, nu):
        # Modified blackbody, in uK_CMB
        return g(nu)/g(self.nu0) * (nu/self.nu0)**self.beta * blackbody(nu, self.T)/blackbody(self.nu0, self.T)