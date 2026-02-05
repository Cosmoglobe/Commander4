import astropy.units as u
import astropy.constants as c
import numpy as np
import pysm3.units as pysm3u
import healpy as hp
from pixell.bunch import Bunch
import logging
from copy import deepcopy
from scipy.interpolate import interp1d
from numpy.typing import NDArray
from typing import Callable
from mpi4py import MPI

from src.python.output import log
from src.python.utils.math_operations import alm_to_map, map_to_alm, project_alms, inplace_scale,\
        inplace_add_scaled_vec, map_to_alm_adjoint, alm_to_map_adjoint, almxfl,\
        inplace_arr_add, inplace_arr_sub, inplace_arr_prod, inplace_arr_truediv, dot, _dot_complex_alm_1D_arrays
from src.python.utils.map_utils import fwhm2sigma, gauss_beam, get_gauss_beam_radius
from src.python.data_models.band import Band

MPI_LIMIT_32BIT = 2**31 - 1

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
    def __init__(self, comp_params: Bunch, global_params: Bunch):
        self.comp_params = comp_params
        self.global_params = global_params
        self.longname = comp_params.longname if "longname" in comp_params else "Unknown Component"
        self.shortname = comp_params.shortname if "shortname" in comp_params else "comp"
        self._data = None

    @property
    def pol(self):
        if self._data is not None:
            return False if self._data.shape[0] == 1 else True
        else:
            raise ValueError("Component {self.longname} has with no data.")
        
    @property
    def npol(self):
        if self._data is not None:
            return self._data.shape[0]
        else:
            raise ValueError("Component {self.longname} has with no data.")
    
    def __add__(self, other):
        if type(self) is not type(other):
            raise TypeError("Both operands must be of the same Component type.")
        if self._data is None or other._data is None:
            raise ValueError("Cannot add Components with no data.")
        if self._data.shape != other._data.shape:
            raise ValueError("Data arrays of the two Components must match in size.")
        out = deepcopy(self)
        inplace_arr_add(out._data, other._data)
        return out
    
    def __iadd__(self, other):
        if type(self) is not type(other):
            raise TypeError("Both operands must be of the same Component type.")
        if self._data is None or other._data is None:
            raise ValueError("Cannot add Components with no data.")
        if self._data.shape != other._data.shape:
            raise ValueError("Data arrays of the two Components must match in size.")
        inplace_arr_add(self._data, other._data)
        return self
    
    def __sub__(self, other):
        if type(self) is not type(other):
            raise TypeError("Both operands must be of the same Component type.")
        if self._data is None or other._data is None:
            raise ValueError("Cannot add Components with no data.")
        if self._data.shape != other._data.shape:
            raise ValueError("Data arrays of the two Components must match in size.")
        out = deepcopy(self)
        inplace_arr_sub(out._data, other._data)
        return out
    
    def __isub__(self, other):
        if type(self) is not type(other):
            raise TypeError("Both operands must be of the same Component type.")
        if self._data is None or other._data is None:
            raise ValueError("Cannot add Components with no data.")
        if self._data.shape != other._data.shape:
            raise ValueError("Data arrays of the two Components must match in size.")
        inplace_arr_sub(self._data, other._data)
        return self
    
    def __mul__(self, other):
        if type(self) is not type(other):
            raise TypeError("Both operands must be of the same Component type.")
        if self._data is None or other._data is None:
            raise ValueError("Cannot add Components with no data.")
        if self._data.shape != other._data.shape:
            raise ValueError("Data arrays of the two Components must match in size.")
        out = deepcopy(self)
        inplace_arr_prod(out._data, other._data)
        return out
    
    def __imul__(self, other):
        if type(self) is not type(other):
            raise TypeError("Both operands must be of the same Component type.")
        if self._data is None or other._data is None:
            raise ValueError("Cannot add Components with no data.")
        if self._data.shape != other._data.shape:
            raise ValueError("Data arrays of the two Components must match in size.")
        inplace_arr_prod(self._data, other._data)
        return self
    
    def __truediv__(self, other):
        if type(self) is not type(other):
            raise TypeError("Both operands must be of the same Component type.")
        if self._data is None or other._data is None:
            raise ValueError("Cannot add Components with no data.")
        if self._data.shape != other._data.shape:
            raise ValueError("Data arrays of the two Components must match in size.")
        out = deepcopy(self)
        inplace_arr_truediv(out._data, other._data)
        return out
    
    def __itruediv__(self, other):
        if type(self) is not type(other):
            raise TypeError("Both operands must be of the same Component type.")
        if self._data is None or other._data is None:
            raise ValueError("Cannot add Components with no data.")
        if self._data.shape != other._data.shape:
            raise ValueError("Data arrays of the two Components must match in size.")
        inplace_arr_add(self._data, other._data)
        return self
    
    def __matmul__(self, other):
        if type(self) is not type(other):
            raise TypeError("Both operands must be of the same Component type.")
        if self._data is None or other._data is None:
            raise ValueError("Cannot add Components with no data.")
        if self._data.shape != other._data.shape:
            raise ValueError("Data arrays of the two Components must match in size.")
        return dot(self._data, other._data)

    def bcast_data_blocking(self, comm:MPI.Comm, root=0):
        """
        Broadcasts the data object of the component stored on the root MPI rank.
        """
        logger = logging.getLogger(__name__)
        log.logassert(isinstance(self._data, np.ndarray), "the data object must be an array", logger)
        comm.Bcast(self._data, root=root)

    def bcast_data_non_blocking(self, comm:MPI.Comm, root=0):
        """
        Broadcasts the data object of the component stored on the root MPI rank, it only returns the request.
        """
        logger = logging.getLogger(__name__)
        log.logassert(isinstance(self._data, np.ndarray), "the data object must be an array", logger)
        req = comm.Ibcast(self._data, root=root)
        return req

    def accum_data_blocking(self, comm:MPI.Comm, root=0):
        """
        Accumulates on the root rank the data object of the component through and MPI reduce with a sum.
        """
        logger = logging.getLogger(__name__)
        log.logassert(isinstance(self._data, np.ndarray), "the data object must be an array", logger)
        myrank=comm.Get_rank()
        send, recv = (MPI.IN_PLACE, self._data) if myrank == root else (self._data, None)
        comm.Reduce(send, recv, op=MPI.SUM, root=root)

    def accum_data_non_blocking(self, comm:MPI.Comm, root=0):
        """
        Accumulates on the root rank the data object of the component through and MPI reduce with a sum, it only returns the request.
        """
        logger = logging.getLogger(__name__)
        log.logassert(isinstance(self._data, np.ndarray), "the data object must be an array", logger)
        myrank=comm.Get_rank()
        send, recv = (MPI.IN_PLACE, self._data) if myrank == root else (self._data, None)
        req = comm.Ireduce(send, recv, op=MPI.SUM, root=root)
        return req

    def __array_function__(self, func, types, args, kwargs):
        #for numpy func overloads
        if not all(issubclass(t, Component) for t in types):
            return NotImplemented

        if func is np.zeros_like:
            return self._zeros_like(*args, **kwargs)

        return NotImplemented

    def _zeros_like(self, other, dtype=None, order='K', subok=True, shape=None):
        zeros = np.zeros_like(
            other._data,
            dtype=dtype,
            order=order,
            subok=subok,
            shape=shape,
        )
        out = deepcopy(other)
        out._data = zeros
        return out

# Second tier component classes
class DiffuseComponent(Component):
    def __init__(self, comp_params: Bunch, global_params: Bunch):
        super().__init__(comp_params, global_params)
        self.spatially_varying_MM = comp_params.spatially_varying_MM
        self.lmax = comp_params.lmax
        self.smoothing_prior_FWHM = comp_params.smoothing_prior_FWHM
        self.smoothing_prior_amplitude = comp_params.smoothing_prior_amplitude
        self.double_prec = False if global_params.CG_float_precision == "single" else True
        self._data = np.empty((2 if comp_params.polarized else 1, self.alm_len_complex), dtype = (np.complex128 if self.double_prec else np.complex64))

    @property
    def alms(self):
        return self._data

    @alms.setter
    def alms(self, alms):
        if alms.ndim == 2:
            if alms.shape[0] in [1,2]:
                self._data = alms
            else:
                raise ValueError(f"Trying to set alms with wrong first axis length {alms.shape[0]} != 1 or 2")
        else:
            raise ValueError(f"Trying to set alms with unexpected number of dimensions: {alms.ndim} != 2")
            
    @property
    def dtype(self):
        return self._data.dtype

    @property
    def alm_len_complex(self):
        return ((self.lmax+1)*(self.lmax+2))//2

    @property
    def P_smoothing_prior(self):
        fwhm_rad = np.deg2rad(self.smoothing_prior_FWHM / 60.0)
        sigma = fwhm_rad / np.sqrt(8 * np.log(2))
        ells = np.arange(self.lmax + 1)
        prior_amplitude = self.smoothing_prior_amplitude
        prior_exponential = -ells * (ells + 1) * sigma**2
        return prior_amplitude * np.exp(prior_exponential)

    @property
    def P_smoothing_prior_inv(self):
        P = self.P_smoothing_prior
        P_inv = np.zeros_like(P)
        P_inv[P != 0] = 1.0/P[P != 0]
        return P_inv
    
    def __repr__(self):
        return f"Diffuse Component {self.shortname} {'Polarization' if self.pol else 'Intensity'} \n   lmax = {self.lmax} \n   alms: {self.alms}"

    def apply_smoothing_prior_sqrt(self):
        """
        Applies in-place the square root of the smoothing prior to the alms in-place, which are also returned.
        """
        smooth_p_sqrt = np.sqrt(self.P_smoothing_prior)
        for ipol in range(self.npol):
            # S^{1/2} a
            almxfl(self._data[ipol], smooth_p_sqrt, inplace=True)
        return self._data

    def get_component_map(self, nside:int, fwhm:int=0):
        component_alms = self.alms
        if component_alms is None:
            raise ValueError("component_alms property not set.")
        if fwhm == 0:
            return alm_to_map(component_alms, nside, self.lmax, spin = 2 if self.pol else 0)
        else:
            return alm_to_map(hp.smoothalm(component_alms, fwhm, inplace=False), nside, self.lmax, spin = 2 if self.pol else 0)

    def get_sky(self, nu, nside, fwhm=0):
        return self.get_component_map(nside, fwhm)*self.get_sed(nu)
    
    def get_sed(self, nu):
        logger = logging.getLogger(__name__)
        log.lograise(NotImplementedError, "", logger)

    #overwrite of the dot product, as the diffuse component will have alm _data with complex encoding.
    def __matmul__(self, other):
        if type(self) is not type(other):
            raise TypeError("Both operands must be of the same Component type.")
        if self._data is None or other._data is None:
            raise ValueError("Cannot add Components with no data.")
        if self._data.shape != other._data.shape:
            raise ValueError("Data arrays of the two Components must match in size.")
        res = 0.0
        for ipol in range(self.npol):
            res += _dot_complex_alm_1D_arrays(self._data[ipol], other._data[ipol], self.lmax)
        return res

    def project_comp_to_band(self, band:Band, nthreads: int = 1):
        """
        Project the component to the given band in-place, summing its contribution to the alms array of the passed band object.

        NB: this function does not include the beam smoothing.
        """
        if self.pol:
            assert band.pol, "Polarized component can only be projected to polarized band alms"
            npol = 2
            spin = 2
        else:
            assert not band.pol, "Intensity component can only be projected to intensity band alms"
            npol = 1
            spin = 0

        alm_in_band_space = project_alms(self.alms, band.lmax)
        if self.spatially_varying_MM:  # If this component has a MM that is pixel-depnedent.
            # Y a
            comp_map = alm_to_map(alm_in_band_space, band.nside, band.lmax, spin=spin, nthreads=nthreads)
            # M Y a
            for ipol in range(npol):
                inplace_scale(comp_map[ipol], self.get_sed(band.nu)) 
            # Y^-1 M Y a
            band.alms = map_to_alm(comp_map, band.nside, band.lmax, spin=spin, acc=True, nthreads=nthreads)
        else:
            for ipol in range(npol):
                inplace_add_scaled_vec(band.alms[ipol], alm_in_band_space[ipol], self.get_sed(band.nu))
        return band.alms

    def eval_comp_from_band(self, band:Band, nthreads: int = 1, inplace=True):
        """
        Evaluate the band's alm contribution to the component, stores it in-place by default and retruns it.

        All the contributions will be summed to the total proper amplitudes by the master node.

        NB: this function does not include the beam smoothing.
        """
        if self.pol:
            assert band.pol, "Polarized component can only be evaluated from polarized band alms"
            npol = 2
            spin = 2
        else:
            assert not band.pol, "Intensity component can only be evaluated from intensity band alms"
            npol = 1
            spin = 0

        if self.spatially_varying_MM:  # If this component has a MM that is pixel-depnedent.
            # Y^-1^T B^T a
            band_map = map_to_alm_adjoint(band.alm, band.nside, band.lmax, spin=spin, out=None, nthreads=nthreads)

            # M^T Y^-1 B^T a
            for ipol in range(npol):
                inplace_scale(band_map[ipol], self.get_sed(band.nu))

            # Y^T M^T Y^-1^T B^T a
            tmp_alm = alm_to_map_adjoint(band_map, band.nside, band.lmax, spin=spin, out=None, nthreads=nthreads)

        else:
            tmp_alm = band.alms.copy()
            for ipol in range(npol):
                inplace_scale(tmp_alm[ipol], self.get_sed(band.nu))
            
        # Project alm from band to component lmax.
        contrib_to_comp_alm = project_alms(tmp_alm, self.lmax)
        
        if inplace:
            self.alms = contrib_to_comp_alm

        return contrib_to_comp_alm
    

class PointSourceComponent(Component):
    pass

class TemplateComponent(Component):
    pass

# Third tier component classes
class CMB(DiffuseComponent):
    def __init__(self, comp_params: Bunch, global_params: Bunch):
        super().__init__(comp_params, global_params)
        self.longname = comp_params.longname if "longname" in comp_params else "CMB"
        self.shortname = comp_params.shortname if "shortname" in comp_params else "cmb"

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
    
    def get_sky_anisotropies(self, nu, nside, fwhm=0):
        if self.alms is None:
            raise ValueError("component_alms property not set.")
        component_alms = self.alms.copy()
        # Zero out monopole (l=0)
        component_alms[:,hp.Alm.getidx(self.lmax, 0, 0)] = 0.0 + 0.0j
        # Zero out the dipole (l=1)
        for m in range(2):  # m = 0, 1
            component_alms[:,hp.Alm.getidx(self.lmax, 1, m)] = 0.0 + 0.0j
        # Zero out the quadrupole (l=2)
        for m in range(3):  # m = 0, 1, 2
            component_alms[:,hp.Alm.getidx(self.lmax, 2, m)] = 0.0 + 0.0j
        if fwhm == 0:
            return alm_to_map(component_alms, nside, self.lmax, spin = 2 if self.pol else 0)*self.get_sed(nu)
        else:
            return alm_to_map(hp.smoothalm(component_alms, fwhm, inplace=False), nside, self.lmax, spin = 2 if self.pol else 0)*self.get_sed(nu)

class CMBRelQuad(TemplateComponent):
    pass

class ThermalDust(DiffuseComponent):
    def __init__(self, comp_params: Bunch, global_params: Bunch):
        super().__init__(comp_params, global_params)
        self.beta = comp_params.beta
        self.T = comp_params.T
        self.nu0 = comp_params.nu0
        self.prior_l_power_law = 2.5
        self.longname = comp_params.longname if "longname" in comp_params else "Thermal Dust"
        self.shortname = comp_params.shortname if "shortname" in comp_params else "dust",

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
    def __init__(self, comp_params: Bunch, global_params: Bunch):
        super().__init__(comp_params, global_params)
        self.beta = comp_params.beta
        self.nu0 = comp_params.nu0
        self.nside_comp_map = 512
        self.prior_l_power_law = -3
        self.longname = comp_params.longname if "longname" in comp_params else "Synchrotron"
        self.shortname = comp_params.shortname if "shortname" in comp_params else "sync"

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
    def __init__(self, comp_params: Bunch, global_params: Bunch):
        super().__init__(comp_params, global_params)
        self.T = comp_params.T  # Electron temperature in K
        self.nu0 = comp_params.nu0 # Reference frequency in GHz
        self.longname = comp_params.longname if "longname" in comp_params else "Free-Free"
        self.shortname = comp_params.shortname if "shortname" in comp_params else "ff"

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

    def __init__(self, comp_params: Bunch, global_params: Bunch):
        """
        Args:
            nu_peak (float): The peak frequency of the spinning dust component in GHz.
            nu_0 (float): The reference frequency of the spinning dust template in GHz.
                          This will not impact the shape of the SED, just the absolute scaling.
        """
        super().__init__(comp_params, global_params)
        # Read SpDust2 template data. This is a simulation of what the spectral shape of
        # spinning dust emission should look like if it happens to peak at 30 GHz.
        freqs, SED = np.loadtxt(comp_params.template_path).T
        self.nu_peak_ref = 30.0  # The reference peak frequency of 30 GHz.
        self.nu_peak_eval = comp_params.nu_peak
        self.nu_0 = comp_params.nu_0  # Reference frequency for the amplitude map in GHz

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
    

# NON DIFFUSE COMPONENTS

class PointSourcesComponent(Component):
    def __init__(self, comp_params: Bunch, global_params: Bunch):
        super().__init__(comp_params, global_params)
        self.longname = comp_params.longname if "longname" in comp_params else "Unknown PointSourceComp"
        self.shortname = comp_params.shortname if "shortname" in comp_params else "pscomp"

    @property
    def pol(self) -> bool:
        return False
    
    @property
    def npol(self) -> int:
        return 1

class RadioSources(PointSourcesComponent):
    def __init__(self, comp_params: Bunch, global_params: Bunch):
        super().__init__(comp_params, global_params)
        self._data = comp_params.amp_s               #per-source amplitudes
        self.nu0 = comp_params.nu0                   #reference frequency
        self.alpha_s = comp_params.alpha_s           #per-source spectral indexes
        self.lonlat_s = comp_params.lonlat_s         #per-source list of coordinates
        self.pol = False                        #can point sources be polarized?
        self.fwhm_r = None                      #fwhm used for the computation of pix and beam discs
        self.pix_discs_i_s = None               #per-source list of indexes of the pixel forming the disc
        self.beam_discs_val_s = None            #per-source list of beam values, for each pix_i_s 

        self.longname = comp_params.longname if "longname" in comp_params else "RadioPointSources"
        self.shortname = comp_params.shortname if "shortname" in comp_params else "radsources"
        self.mJysr_to_uKRJ = (pysm3u.mJy / pysm3u.steradian).to(pysm3u.uK_RJ, equivalencies=pysm3u.cmb_equivalencies(self.band_nu*pysm3u.GHz))
        self.uKRJ_to_mJysr = (pysm3u.uK_RJ).to(pysm3u.mJy / pysm3u.steradian, equivalencies=pysm3u.cmb_equivalencies(self.band_nu*pysm3u.GHz))

    def compute_pix_beams(self, fwhm_r):
        """
        Computes the beams values, based on fwhm_r, in map space around all the point sources and updates in-place
        in self.pix_discs_i_s, beam_discs_val_s and fwhm_r self.members
        """
        self.pix_discs_i_s = []
        self.beam_discs_val_s = []
        self.fwhm_r = fwhm_r
        for i in range(self.lonlat_s.shape[0]):      #compute and load the beams, these will reamain untouched. The FWHM depends on the band.
            disc_pix_i_s = hp.query_disc(self.band_nside, hp.ang2vec(self.lonlat_s[i,0], self.lonlat_s[i,1], lonlat=True), get_gauss_beam_radius(self.fwhm_r))
            self.pix_discs_i_s.append(disc_pix_i_s)
            beam_disc = gauss_beam(hp.rotator.angdist(self.lonlat_s[i,:], hp.pix2ang(self.band_nside, disc_pix_i_s, lonlat=True), lonlat=True), self.fwhm_r)
            self.beam_discs_val_s.append(beam_disc)

    def get_sed(self, nu):
        """
        Returns a list of sed's, one per `alpha_s`, evaluated at `nu`, with ref frequency `nu0`. 
        Freq. are in GHz.
        """
        return (nu/self.nu0)**(self.alpha_s - 2)
    
    def get_sky(self, nu, nside, pol=False, fwhm=0):
        raise NotImplementedError
    
    def _project_to_band_map(self, map, nu):
        """
        Computes the point source contribution in uK_RJ for band's frequency and beam, and sums it to `map`.
        """
        for src_i in range(len(self.pix_discs_i_s)):
            map[self.disc_pix_i_s[src_i]] += self.mJysr_to_uKRJ * self.beam_discs_val_s[src_i] * self._data[src_i] * self.sed(nu)
    
    def _eval_from_band_map(self, map, nu):
        """
        Computes the amplitude contribution from the local band to each point source, given `map`.

        All the contributions will be summed to the total proper amplitudes by the master node.
        """
        for src_i in range(len(self.pix_discs_i_s)):
            self._data[src_i] = self.mJysr_to_uKRJ * np.sum(map[self.disc_pix_i_s[src_i]] * self.beam_discs_val_s[src_i]) * self.sed(nu)

    def project_comp_to_band(self, band:Band, nthreads: int = 1):
        """
        Project the point sources contribution to the given band in-place, summing its contribution to the alm array of the band object.

        NB: this function does not include the beam smoothing.
        """

        assert not band.pol, "Point sources component can only be projected to intensity band alms"
        # the point-source correspondent of: M Y a
        ps_map = np.zeros(hp.nside2npix(band.nside)) #empty band map
        self._project_to_band_map(ps_map, band.nu)

        # Y^-1 M Y a
        map_to_alm(ps_map, band.nside, band.lmax, spin=0, out=band.alm, acc=True, nthreads=nthreads)
        
        return band.alm

    def eval_comp_from_band(self, band:Band, nthreads: int = 1):
        """
        Evaluate the band's alm contribution to the point sources' amplitudes and stores it in the amp_s member, 
        as well as returning it.

        All the contributions will be summed to the total proper amplitudes when reducing on the master node.

        NB: this function does not include the beam smoothing.
        """

        assert not band.pol, "Point sources component can only be evaluated from intensity band alms"
        # Y^-1^T B^T a
        band_map = map_to_alm_adjoint(band.alm, band.nside, band.lmax, spin=0, out=None, nthreads=nthreads)

        # M^T Y^-1 B^T a
        self._eval_from_band_map(band_map, band.nu) #updates amp_s in-place

        #return band_alm's contribution to point sources amplitudes
        return self._data
    
    def apply_smoothing_prior_sqrt(self):
        """
        In the case of point sources this is just a dummy, the data object is simply returned.
        """
        return self._data

    def __repr__(self):
        return f"Radio Source \n amps: {self._data}"
    

#FIXME: this will go within ComponentList object when implemented
def split_complist(comp_list: list[Component], color:int, IvsQU_colors:tuple = (0,1)) -> list[Component]:
    """
    Extracts from `comp_list` only the components containing the correct Stokes parameter based
    on the passed `color` of the local MPI rank. By default, color=0 will treat Intensity and
    color=1 polarization. A list with the relevant components is returned.
    """
    out_comp_list = []
    IvsQU_colors = IvsQU_colors[:2] #cut off eventual elements in excess
    if color not in IvsQU_colors:
        logging.warning(f"Color {color} not in colors assigned to I or QU ({IvsQU_colors})!")
    else:
        for comp in comp_list:
            if comp.pol == (color == IvsQU_colors[1]):
                # print("Comp", comp.shortname, comp.pol)
                out_comp_list.append(comp)

    return out_comp_list