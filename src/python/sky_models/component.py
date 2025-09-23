import astropy.units as u
import astropy.constants as c
import numpy as np
import pysm3.units as pysm3u
import healpy as hp
from pixell.bunch import Bunch
import logging
from scipy.interpolate import interp1d
from numpy.typing import NDArray

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
        self.longname = params.longname if "longname" in params else "Unknown Component"
        self.shortname = params.shortname if "shortname" in params else "comp"

# Second tier component classes
class DiffuseComponent(Component):
    def __init__(self, params: Bunch):
        super().__init__(params)
        self.polarized = params.polarized
        self.spatially_varying_MM = params.spatially_varying_MM
        self.lmax = params.lmax
        self.smoothing_prior_FWHM = params.smoothing_prior_FWHM
        self.smoothing_prior_amplitude = params.smoothing_prior_amplitude
        self._component_alms_intensity = None
        self._component_alms_polarization = None
        # self.nside_comp_map = 2048
        # self.prior_l_power_law = 0 # l^alpha as in S^-1 in comp sep

    @property
    def component_alms_intensity(self):
        if self._component_alms_intensity is not None:
            return self._component_alms_intensity
        else:
            raise ValueError("Intensity alms not yet set.")

    @property
    def component_alms_polarization(self):
        if self._component_alms_polarization is not None:
            return self._component_alms_polarization
        else:
            raise ValueError("Polarization alms not yet set.")

    @component_alms_intensity.setter
    def component_alms_intensity(self, alms):
        if alms.ndim == 2 and alms.shape[0] == 1:
            self._component_alms_intensity = alms
        else:
            raise ValueError("Trying to set alms with unexpected number of dimensions"
                             f"{alms.ndim} != 2 OR wrong first axis {alms.shape[0]} != 1")

    @component_alms_polarization.setter
    def component_alms_polarization(self, alms):
        if alms.ndim == 2 and alms.shape[0] == 2:
            self._component_alms_polarization = alms
        else:
            raise ValueError("Trying to set alms with unexpected number of dimensions"
                             f"{alms.ndim} != 2 OR wrong first axis {alms.shape[0]} != 2")

    @property
    def P_smoothing_prior(self):
        fwhm_rad = np.deg2rad(self.smoothing_prior_FWHM / 60.0)
        sigma = fwhm_rad / np.sqrt(8 * np.log(2))
        ells = np.arange(self.lmax + 1)
        prior_amplitude = self.smoothing_prior_amplitude
        prior_exponential = -ells * (ells + 1) * sigma**2
        if np.abs(prior_exponential[-1]) > 50:
            logger = logging.getLogger(__name__)
            logger.warning(f"Exponent in smoothing prior goes up to {prior_exponential[-1]:.4e} "
                           f"for {self.longname} with lmax {self.lmax} and smoothing prior FWHM = "
                           f"{self.smoothing_prior_FWHM}rad. CG will likely be unstable.")
        return prior_amplitude * np.exp(prior_exponential)

    @property
    def P_smoothing_prior_inv(self):
        P = self.P_smoothing_prior
        P_inv = np.zeros_like(P)
        P_inv[P != 0] = 1.0/P[P != 0]
        return P_inv

    def get_component_map(self, nside:int, pol:bool=False, fwhm:int=0):
        component_alms = self.component_alms_polarization if pol else self.component_alms_intensity
        if component_alms is None:
            raise ValueError("component_alms property not set.")
        if fwhm == 0:
            return alm_to_map(component_alms, nside, self.lmax, spin = 2 if pol else 0)
        else:
            return alm_to_map(hp.smoothalm(component_alms, fwhm, inplace=False), nside, self.lmax, spin = 2 if pol else 0)

    def get_sky(self, nu, nside, pol=False, fwhm=0):
        return self.get_component_map(nside, pol, fwhm)*self.get_sed(nu)
    
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
        """Calculates the spectral energy distribution (SED) for CMB emission.
           The result is unitless, but meant to be multiplied by a RJ brightness temperature.
        Args:
            nu (float or np.ndarray): Frequency in GHz at which to evaluate the SED.            
        Returns:
            The SED scaling factor (float or np.ndarray).
        """
        return (np.ones_like(nu)*pysm3u.uK_CMB).to(pysm3u.uK_RJ,equivalencies=
                                                   pysm3u.cmb_equivalencies(nu*u.GHz)).value
    
    def get_sky_anisotropies(self, nu, nside, pol=False, fwhm=0):
        component_alms = self.component_alms_polarization if pol else self.component_alms_intensity
        if component_alms is None:
            raise ValueError("component_alms property not set.")
        component_alms = component_alms.copy()
        # Zero out monopole (l=0)
        component_alms[:,hp.Alm.getidx(self.lmax, 0, 0)] = 0.0 + 0.0j
        # Zero out the dipole (l=1)
        for m in range(2):  # m = 0, 1
            component_alms[:,hp.Alm.getidx(self.lmax, 1, m)] = 0.0 + 0.0j
        # Zero out the quadrupole (l=2)
        for m in range(3):  # m = 0, 1, 2
            component_alms[:,hp.Alm.getidx(self.lmax, 2, m)] = 0.0 + 0.0j
        if fwhm == 0:
            return alm_to_map(component_alms, nside, self.lmax, spin = 2 if pol else 0)*self.get_sed(nu)
        else:
            return alm_to_map(hp.smoothalm(component_alms, fwhm, inplace=False), nside, self.lmax, spin = 2 if pol else 0)*self.get_sed(nu)



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
        """Calculates the spectral energy distribution (SED) for Thermal Dust emission.
           The result is unitless, but meant to be multiplied by a RJ brightness temperature.
        Args:
            nu (float or np.ndarray): Frequency in GHz at which to evaluate the SED.            
        Returns:
            The SED scaling factor (float or np.ndarray).
        """
        # Modified blackbody, in uK_CMB
        x = (h_over_k*nu)/(self.T)
        x0 = (h_over_k*self.nu0)/(self.T)
        return (nu / self.nu0)**(self.beta + 1.0) * np.expm1(x0) / np.expm1(x)


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
        """Calculates the spectral energy distribution (SED) for Synchrotron emission.
           The result is unitless, but meant to be multiplied by a RJ brightness temperature.
        Args:
            nu (float or np.ndarray): Frequency in GHz at which to evaluate the SED.            
        Returns:
            The SED scaling factor (float or np.ndarray).
        """
        return (nu/self.nu0)**self.beta


class FreeFree(DiffuseComponent):
    def __init__(self, params: Bunch):
        super().__init__(params)
        self.T = params.T  # Electron temperature in K
        self.nu0 = params.nu0 # Reference frequency in GHz
        self.longname = params.longname if "longname" in params else "Free-Free"
        self.shortname = params.shortname if "shortname" in params else "ff"

    def _gaunt_factor(self, nu, T):
        """Calculates the Gaunt factor for free-free emission, as per Eq. 18 in BP1.
        Args:
            nu (float or np.ndarray): Frequency in GHz.
            T (float): Electron temperature in Kelvin.
        Returns:
            The Gaunt factor (float or np.ndarray)
        """
        T4 = T / 1e4
        log_arg = nu * (T4**(-1.5))
        inner_exp = 5.960 - (np.sqrt(3) / np.pi) * np.log(log_arg)
        return np.log(np.exp(inner_exp) + np.e)

    def get_sed(self, nu):
        """Calculates the spectral energy distribution (SED) for Free-Free emission.
           The result is unitless, but meant to be multiplied by a RJ brightness temperature.
        Args:
            nu (float or np.ndarray): Frequency in GHz at which to evaluate the SED.            
        Returns:
            The SED scaling factor (float or np.ndarray).
        """
        gaunt_nu = self._gaunt_factor(nu, self.T)
        gaunt_nu0 = self._gaunt_factor(self.nu0, self.T)
        
        # The scaling is proportional to nu^-2 * g_ff(nu), normalized to 1 at nu0.
        sed = (self.nu0 / nu)**2 * (gaunt_nu / gaunt_nu0)
        return sed


class SpinningDust(DiffuseComponent):
    """
    Spinning Dust component spectral model, based on spinning dust.
    The SED is derived from the SpDust2 code template for the Cold Neutral Medium.
    """
    # SpDust2 template data for Cold Neutral Medium (CNM)
    # This template has an intensity peak at 30 GHz.
    # Columns: Frequency (GHz), Emissivity (proportional to Intensity)

    def __init__(self, params: Bunch):
        """
        Args:
            nu_peak (float): The peak frequency of the spinning dust component in GHz.
            nu_0 (float): The reference frequency of the spinning dust template in GHz.
                          This will not impact the shape of the SED, just the absolute scaling.
        """
        super().__init__(params)
        # Read SpDust2 template data. This is a simulation of what the spectral shape of
        # spinning dust emission should look like if it happens to peak at 30 GHz.
        freqs, SED = np.loadtxt(params.template_path).T
        self.nu_peak_ref = 30.0  # The reference peak frequency of 30 GHz.
        self.nu_peak_eval = params.nu_peak
        self.nu_0 = params.nu_0  # Reference frequency for the amplitude map in GHz

        # Create an logarithmic interpolation function from the SpDust2 template
        log_nu = np.log(freqs)
        log_SED = np.log(SED)
        self._log_j_interp = interp1d(log_nu, log_SED, kind='cubic',
                                      bounds_error=False, fill_value=-np.inf)


    def _get_template_emissivity(self, nu):
        """Calculates the template emissivity at a given frequency using interpolation."""
        return np.exp(self._log_j_interp(np.log(nu)))


    def get_sed(self, nu: float|NDArray[np.floating]):
        """
        Calculates the spinning dust SED scaling factor based on the spinning dust model.
        This factor scales an amplitude map from its reference frequency (22 GHz)
        to the target frequency nu.

        Args:
            nu (float|array): Frequency at which to get the SED, in GHz.
        Returns:
            float|array: The unitless SED scaling factor.
        """
        # Numerator: template evaluated at the shifted frequency
        nu_shifted_eval = nu * self.nu_peak_ref / self.nu_peak_eval
        SED_eval = self._get_template_emissivity(nu_shifted_eval)

        # Denominator: template evaluated at the shifted reference frequency for normalization
        nu_shifted_ref = self.nu_0 * self.nu_peak_ref / self.nu_peak_eval
        SED_ref = self._get_template_emissivity(nu_shifted_ref)

        # Shifting the SED spectrum from the reference frequency to the given peak frequency.
        SED_at_eval_freq = SED_eval / SED_ref

        # Converting from intensity to brightness temperature.
        SED_uK_RJ = (self.nu_0 / nu)**2 * SED_at_eval_freq
        return SED_uK_RJ