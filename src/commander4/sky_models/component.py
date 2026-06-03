import astropy.units as u
import astropy.constants as c
import numpy as np
import pysm3.units as pysm3u
import healpy as hp
import logging
from copy import deepcopy
from scipy.interpolate import interp1d
from numpy.typing import NDArray
from mpi4py import MPI
from pixell.bunch import Bunch

import commander4.sky_models.component as component_lib
from commander4.output import log
from commander4.utils.math_operations import alm_to_map, map_to_alm, project_alms, inplace_scale,\
        inplace_add_scaled_vec, map_to_alm_adjoint, alm_to_map_adjoint, almxfl,\
        inplace_arr_add, inplace_arr_sub, inplace_arr_prod, inplace_arr_truediv, dot, \
        _dot_complex_alm_1D_arrays, _numba_proj2map, _numba_eval_from_map, inplace_scale_add
from commander4.utils.map_utils import gauss_beam, get_gauss_beam_radius, get_npol, assert_pol_supported
from commander4.data_models.band import Band
from commander4.utils.execution_ids import EXECUTION_POLS

logger = logging.getLogger(__name__)


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
    default_shortname = "comp"
    legal_pols: tuple[str, ...] = ("I", "QU", "IQU")
    requires_defined_pol = False

    @classmethod
    def _assert_legal_pol(cls, pol: str | None, *, role: str, required: bool = False) -> None:
        if pol is None:
            log.logassert(
                not required,
                f"{cls.__name__} requires a defined polarization mode.",
                logger,
            )
            return
        assert_pol_supported(pol)
        log.logassert(
            pol in cls.legal_pols,
            f"{cls.__name__} does not support {role} polarization {pol!r}. "
            f"Allowed polarizations: {cls.legal_pols!r}.",
            logger,
        )

    def __init__(self, comp_params: Bunch, global_params: Bunch, *,
                 shortname: str | None = None, comp_name: str | None = None,
                 eval_pol: str | None = None, allocate_empty_alms: bool = False):
        self.comp_params = comp_params
        self.global_params = global_params
        self.shortname = (
            shortname
            if shortname is not None
            else comp_params.shortname if "shortname" in comp_params
            else self.default_shortname
        )
        self.comp_name = comp_params._name if comp_name is None else comp_name
        self.defined_pol = comp_params.polarization if "polarization" in comp_params else None
        type(self)._assert_legal_pol(
            self.defined_pol,
            role="defined",
            required=type(self).requires_defined_pol,
        )
        self.eval_pol = self.defined_pol if eval_pol is None else eval_pol
        type(self)._assert_legal_pol(self.eval_pol, role="evaluation")
        self.double_prec = False if global_params.CG_float_precision == "single" else True
        self._data = None

    @property
    def logical_id(self) -> str:
        return self.comp_name

    @property
    def logical_key(self) -> tuple[type["Component"], str]:
        return (type(self), self.logical_id)

    @property
    def execution_key(self) -> tuple[type["Component"], str, str | None]:
        return (type(self), self.logical_id, self.eval_pol)

    @property
    def is_split_view(self) -> bool:
        return self.defined_pol == "IQU" and self.eval_pol in ("I", "QU")

    @property
    def execution_label(self) -> str:
        if self.eval_pol is None or not self.is_split_view:
            return self.shortname
        return f"{self.shortname}[{self.eval_pol}]"

    def _assert_consistent_comp(self, other: "Component") -> None:
        if not isinstance(other, Component):
            raise TypeError("Both operands must be Component objects.")
        if type(self) is not type(other):
            raise TypeError("Both operands must be of the same Component type.")
        mismatched = [
            attr for attr in (
                "comp_name",
                "shortname",
                "defined_pol",
                "eval_pol",
            )
            if getattr(self, attr) != getattr(other, attr)
        ]
        if mismatched:
            raise ValueError(
                "Components must represent the same execution view. "
                f"Mismatched fields: {', '.join(mismatched)}"
            )
        if self._data is None or other._data is None:
            raise ValueError("Cannot operate on Components with no data.")
        if self._data.shape != other._data.shape:
            raise ValueError("Data arrays of the two Components must match in size.")

    def join_split_views(self, other: "Component") -> "Component":
        if not isinstance(other, Component):
            raise TypeError("Can only join Component objects.")
        if type(self) is not type(other):
            raise TypeError("Split views must be of the same Component type.")
        if self.defined_pol != "IQU" or other.defined_pol != "IQU":
            raise ValueError("Only IQU-defined components can be joined.")
        if not self.is_split_view or not other.is_split_view:
            raise ValueError("Only split component views can be joined.")
        if {self.eval_pol, other.eval_pol} != {"I", "QU"}:
            raise ValueError("Joining requires one intensity view and one QU view.")
        mismatched = [
            attr for attr in ("comp_name", "shortname", "defined_pol")
            if getattr(self, attr) != getattr(other, attr)
        ]
        if mismatched:
            raise ValueError(
                "Split views must refer to the same logical component. "
                f"Mismatched fields: {', '.join(mismatched)}"
            )
        if self._data is None or other._data is None:
            raise ValueError("Cannot join split views with no data.")
        intensity_comp, pol_comp = (self, other) if self.eval_pol == "I" else (other, self)
        if intensity_comp._data.shape[1:] != pol_comp._data.shape[1:]:
            raise ValueError("Split views must have compatible alm dimensions.")
        joined = deepcopy(intensity_comp)
        joined.eval_pol = joined.defined_pol
        joined._data = np.concatenate((intensity_comp._data, pol_comp._data), axis=0)
        return joined

    def _apply_array_op(self, other: "Component", arr_op, *, inplace: bool) -> "Component":
        self._assert_consistent_comp(other)
        target = self if inplace else deepcopy(self)
        arr_op(target._data, other._data)
        return target

    def __add__(self, other):
        return self._apply_array_op(other, inplace_arr_add, inplace=False)
    
    def __iadd__(self, other):
        return self._apply_array_op(other, inplace_arr_add, inplace=True)
    
    def __sub__(self, other):
        return self._apply_array_op(other, inplace_arr_sub, inplace=False)
    
    def __isub__(self, other):
        return self._apply_array_op(other, inplace_arr_sub, inplace=True)
    
    def __mul__(self, other):
        return self._apply_array_op(other, inplace_arr_prod, inplace=False)
    
    def __imul__(self, other):
        return self._apply_array_op(other, inplace_arr_prod, inplace=True)
    
    def __truediv__(self, other):
        return self._apply_array_op(other, inplace_arr_truediv, inplace=False)
    
    def __itruediv__(self, other):
        return self._apply_array_op(other, inplace_arr_truediv, inplace=True)
    
    def __matmul__(self, other):
        self._assert_consistent_comp(other)
        return dot(self._data, other._data)

    def bcast_data_blocking(self, comm:MPI.Comm, root=0):
        """
        Broadcasts the data object of the component stored on the root MPI rank.
        """
        log.logassert(isinstance(self._data, np.ndarray), "data object must be an array", logger)
        comm.Bcast(self._data, root=root)

    def bcast_data_non_blocking(self, comm:MPI.Comm, root=0):
        """
        Broadcasts the data object of the component stored on the root MPI rank,
        it only returns the request.
        """
        log.logassert(isinstance(self._data, np.ndarray), "data object must be an array", logger)
        req = comm.Ibcast(self._data, root=root)
        return req

    def accum_data_blocking(self, comm:MPI.Comm, root=0):
        """
        Accumulates on the root rank the data object of the component through and
        MPI reduce with a sum.
        """
        log.logassert(isinstance(self._data, np.ndarray), "data object must be an array", logger)
        myrank=comm.Get_rank()
        send, recv = (MPI.IN_PLACE, self._data) if myrank == root else (self._data, None)
        comm.Reduce(send, recv, op=MPI.SUM, root=root)

    def accum_data_non_blocking(self, comm:MPI.Comm, root=0):
        """
        Accumulates on the root rank the data object of the component through and MPI reduce with
        a sum, it only returns the request.
        """
        log.logassert(isinstance(self._data, np.ndarray), "data object must be an array", logger)
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
    requires_defined_pol = True

    def __init__(self, comp_params: Bunch, global_params: Bunch, 
                 allocate_empty_alms=False, eval_pol:None|str=None,
                 comp_name: str | None = None, shortname: str | None = None):
        super().__init__(
            comp_params,
            global_params,
            shortname=shortname,
            comp_name=comp_name,
            eval_pol=eval_pol,
            allocate_empty_alms=allocate_empty_alms,
        )
        self.spatially_varying_MM = comp_params.spatially_varying_MM
        self.lmax = comp_params.lmax
        self.smoothing_prior_FWHM = comp_params.smoothing_prior_FWHM
        self.smoothing_prior_amplitude = comp_params.smoothing_prior_amplitude
        self._data = None  # Alm data is not allocated by default.
        if allocate_empty_alms:
            self.allocate_empty_alms()

    @property
    def npol(self):
        return get_npol(self.eval_pol)

    @property
    def spin(self):
        if self.eval_pol == "I":
            return 0
        if self.eval_pol in ("QU", "IQU"):
            return 2
        raise ValueError(f"Unsupported polarization '{self.eval_pol}'.")
    
    @property
    def is_pol(self):
        if self.eval_pol == "I":
            return False
        elif self.eval_pol == "QU":
            return True
        else:
            raise ValueError("Specific polarization undefined, or set as IQU.")

    @property
    def alms(self):
        if self._data is None:
            raise ValueError("Trying to access un-initialized diffuse component alms.")
        return self._data

    @alms.setter
    def alms(self, alms):
        if alms.ndim == 2:
            if alms.shape[0] == self.npol:
                self._data = alms
            else:
                raise ValueError("Trying to set alms with wrong first axis length "
                                 f"{alms.shape[0]} != 1 or 2")
        else:
            raise ValueError("Trying to set alms with unexpected number of dimensions: "
                             f"{alms.ndim} != 2")
    
    def allocate_empty_alms(self):
        """ Allocates empty alm array of correct shape. Usefull for e.g. MPI receiving.
        """
        self._data = np.zeros((self.npol, self.alm_len_complex),
                               dtype = (np.complex128 if self.double_prec else np.complex64))
            
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
        return f"Diffuse Component {self.shortname}, with polarization: {self.eval_pol}"\
                f" (originally defined as {self.defined_pol})" \
                f"\n   lmax = {self.lmax} \n   alms: {self.alms}"

    def apply_smoothing_prior_sqrt(self):
        """
        Applies in-place the square root of the smoothing prior to the alms in-place,
        which are also returned.
        """
        smooth_p_sqrt = np.sqrt(self.P_smoothing_prior)
        for ipol in range(self.npol):
            # S^{1/2} a
            almxfl(self._data[ipol], smooth_p_sqrt, inplace=True)
        return self._data

    def _realize_alms_as_map(self, component_alms, nside: int, fwhm: float = 0):
        """Realize component alms as a map.

        Joined `IQU` components still need separate intensity and spin-2 synthesis calls, since
        DUCC does not accept a 3-row alm block in one call.
        """
        component_alms = hp.smoothalm(component_alms, fwhm, inplace=False)
        if self.eval_pol != "IQU":
            return alm_to_map(component_alms, nside, self.lmax, spin=self.spin)
        intensity_map = alm_to_map(component_alms[:1], nside, self.lmax, spin=0)
        pol_map = alm_to_map(component_alms[1:], nside, self.lmax, spin=2)
        return np.concatenate((intensity_map, pol_map), axis=0)

    def get_component_map(self, nside:int, fwhm:int=0):
        component_alms = self.alms
        if component_alms is None:
            raise ValueError("component_alms property not set.")
        return self._realize_alms_as_map(component_alms, nside, fwhm)

    def get_sky(self, nu, nside, fwhm=0):
        return self.get_component_map(nside, fwhm)*self.get_sed(nu)
    
    def get_sed(self, nu):
        log.lograise(NotImplementedError, "", logger)

    #overwrite of the dot product as the diffuse component will have alm _data with complex encoding
    def __matmul__(self, other):
        self._assert_consistent_comp(other)
        res = 0.0
        for ipol in range(self.npol):
            res += _dot_complex_alm_1D_arrays(self._data[ipol], other._data[ipol], self.lmax)
        return res

    def project_comp_to_band(self, band:Band, nthreads: int = 1):
        """
        Project the component to the given band in-place, summing its contribution to the alms
        array of the passed band object.

        NB: this function does not include the beam smoothing.
        """
        log.logassert(self.is_pol == band.is_pol, "Band and component polarization must match",
                      logger)

        alm_in_band_space = project_alms(self.alms, band.lmax)
        if self.spatially_varying_MM:  # If this component has a MM that is pixel-depnedent.
            # Y a
            comp_map = alm_to_map(alm_in_band_space, band.nside, band.lmax, spin=self.spin,
                                  nthreads=nthreads)
            # M Y a
            for ipol in range(self.npol):
                inplace_scale(comp_map[ipol], self.get_sed(band.nu)) 
            # Y^-1 M Y a
            band.alms = map_to_alm(comp_map, band.nside, band.lmax, spin=self.spin, out=band.alms,
                                   acc=True, nthreads=nthreads)
        else:
            for ipol in range(self.npol):
                inplace_add_scaled_vec(band.alms[ipol], alm_in_band_space[ipol],
                                       self.get_sed(band.nu))
        return band.alms

    def eval_comp_from_band(self, band:Band, nthreads: int = 1, inplace=True):
        """
        Evaluate the band's alm contribution to the component, stores it in-place by default and
        retruns it.

        All the contributions will be summed to the total proper amplitudes by the master node.

        NB: this function does not include the beam smoothing.
        """

        log.logassert(self.is_pol == band.is_pol, "Band and component polarization must match",
                      logger)

        if self.spatially_varying_MM:  # If this component has a MM that is pixel-depnedent.
            # Y^-1^T B^T a
            band_map = map_to_alm_adjoint(band.alms, band.nside, band.lmax, spin=self.spin, out=None,
                                          nthreads=nthreads)

            # M^T Y^-1 B^T a
            for ipol in range(self.npol):
                inplace_scale(band_map[ipol], self.get_sed(band.nu))

            # Y^T M^T Y^-1^T B^T a
            tmp_alm = alm_to_map_adjoint(band_map, band.nside, band.lmax, spin=self.spin, out=None,
                                         nthreads=nthreads)

        else:
            tmp_alm = band.alms.copy()
            for ipol in range(self.npol):
                inplace_scale(tmp_alm[ipol], self.get_sed(band.nu))
            
        # Project alm from band to component lmax.
        contrib_to_comp_alm = project_alms(tmp_alm, self.lmax)
        
        if inplace:
            self.alms = contrib_to_comp_alm

        return contrib_to_comp_alm
    

class TemplateComponent(Component):
    pass

# Third tier component classes
class CMB(DiffuseComponent):
    default_shortname = "cmb"

    def __init__(self, comp_params: Bunch, global_params: Bunch, allocate_empty_alms=False,
                 shortname = None, eval_pol = None, comp_name: str | None = None):
        super().__init__(
            comp_params,
            global_params,
            allocate_empty_alms=allocate_empty_alms,
            eval_pol=eval_pol,
            comp_name=comp_name,
            shortname=shortname,
        )

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
        return self._realize_alms_as_map(component_alms, nside, fwhm) * self.get_sed(nu)

class CMBRelQuad(TemplateComponent):
    pass

class ThermalDust(DiffuseComponent):
    default_shortname = "term-dust"

    def __init__(self, comp_params: Bunch, global_params: Bunch, allocate_empty_alms=False,
                 shortname = None, eval_pol = None, comp_name: str | None = None):
        super().__init__(
            comp_params,
            global_params,
            allocate_empty_alms=allocate_empty_alms,
            eval_pol=eval_pol,
            comp_name=comp_name,
            shortname=shortname,
        )
        self.beta = comp_params.beta
        self.T = comp_params.T
        self.nu0 = comp_params.nu0
        self.prior_l_power_law = 2.5

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
    default_shortname = "sync"

    def __init__(self, comp_params: Bunch, global_params: Bunch, allocate_empty_alms=False,
                 shortname = None, eval_pol = None, comp_name: str | None = None):
        super().__init__(
            comp_params,
            global_params,
            allocate_empty_alms=allocate_empty_alms,
            eval_pol=eval_pol,
            comp_name=comp_name,
            shortname=shortname,
        )
        self.beta = comp_params.beta
        self.nu0 = comp_params.nu0
        self.nside_comp_map = 512
        self.prior_l_power_law = -3

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
    default_shortname = "ff"

    def __init__(self, comp_params: Bunch, global_params: Bunch, allocate_empty_alms=False,
                 shortname = None, eval_pol = None, comp_name: str | None = None):
        super().__init__(
            comp_params,
            global_params,
            allocate_empty_alms=allocate_empty_alms,
            eval_pol=eval_pol,
            comp_name=comp_name,
            shortname=shortname,
        )
        self.T = comp_params.T  # Electron temperature in K
        self.nu0 = comp_params.nu0 # Reference frequency in GHz

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
    default_shortname = "spin-dust"

    """
    Spinning Dust component spectral model, based on spinning dust.
    The SED is derived from the SpDust2 code template for the Cold Neutral Medium.
    """
    # SpDust2 template data for Cold Neutral Medium (CNM)
    # This template has an intensity peak at 30 GHz.
    # Columns: Frequency (GHz), Emissivity (proportional to Intensity)

    def __init__(self, comp_params: Bunch, global_params: Bunch, allocate_empty_alms=False,
                 shortname = None, eval_pol = None, comp_name: str | None = None):
        """
        Args:
            nu_peak (float): The peak frequency of the spinning dust component in GHz.
            nu_0 (float): The reference frequency of the spinning dust template in GHz.
                          This will not impact the shape of the SED, just the absolute scaling.
        """
        super().__init__(
            comp_params,
            global_params,
            allocate_empty_alms=allocate_empty_alms,
            eval_pol=eval_pol,
            comp_name=comp_name,
            shortname=shortname,
        )

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
    default_shortname = "pscomp"
    legal_pols: tuple[str, ...] = ("I",)

    def __init__(self, comp_params: Bunch, global_params: Bunch, *,
                 shortname: str | None = None, comp_name: str | None = None,
                 eval_pol: str | None = None, allocate_empty_alms: bool = False):
        super().__init__(
            comp_params,
            global_params,
            shortname=shortname,
            comp_name=comp_name,
            eval_pol="I" if eval_pol is None else eval_pol,
            allocate_empty_alms=allocate_empty_alms,
        )
        self.defined_pol = "I"
        self.eval_pol = "I"

    @property
    def is_pol(self) -> bool:
        return False
    
    @property
    def npol(self) -> int:
        return 1

class RadioSources(PointSourcesComponent):
    default_shortname = "radsources"

    def __init__(self, comp_params: Bunch, global_params: Bunch, *,
                 shortname: str | None = None, comp_name: str | None = None,
                 eval_pol: str | None = None, allocate_empty_alms: bool = False):
        super().__init__(
            comp_params,
            global_params,
            shortname=shortname,
            comp_name=comp_name,
            eval_pol=eval_pol,
            allocate_empty_alms=allocate_empty_alms,
        )
        #reference frequency
        self.nu0 = comp_params.nu_0
        #tabulated data
        ps_bunch = self.read_dat_to_bunch(comp_params.template_path)
        #per-source amplitudes
        self._data = np.array(ps_bunch['I(mJy)'], dtype=np.float32).reshape((1,-1))
        #per-source spectral indexes
        self.alpha_arr = np.array(ps_bunch['alpha_I'], dtype=np.float32)
        self.lonlat_arr = np.array((ps_bunch['Glon(deg)'], ps_bunch['Glat(deg)']),
                                   dtype=np.float32).T #per-source list of coordinates
        del ps_bunch
        if self.alpha_arr.shape[0] != self.lonlat_arr.shape[0]\
        or self.alpha_arr.shape[0] != self._data.shape[1]:
            raise RuntimeError("Point Source tabulated data must be uniform in length.")

        self.pix_disc_idx_list = None   #per-source list of indexes of the pixel forming the disc
        self.beam_disc_val_list = None    #per-source list of beam values, for each pix_i_s
        self.band_eval_nside = None     #nside used for the computation of pix and beam discs
        self.band_fwhm_r = None         #they depend on the bands

    def read_dat_to_bunch(self, file_path):
        """
        Reads a .dat point source raw table and stores it in a Bunch object which is returned.
        """
        rows = []
        head = []
        with open(file_path, 'r') as file:
            # i=0
            for line in file:
                if line.split()[0].startswith("#"):
                    head.append(line.split()[1:])
                else:
                    rows.append(line.split())
        head = head[-1]
        rows = np.array(rows)       
        return Bunch(zip(head,[rows[:,i] for i in range(rows.shape[1])]))

    def compute_pix_beams(self, band_fwhm_r, band_nside, recompute=False):
        """
        Computes the beams values, based on nside and fwhm of the band, in map space around all the
        point sources and updates in-place pix_disc_idx_list, beam_disc_val_list and fwhm_r members.

        If recompute is True will always perform the computation otherwise it does so only if the
        beam lists are not initialized or if the band specs have changed.
        """
        if band_fwhm_r != self.band_fwhm_r \
        or band_nside != self.band_eval_nside \
        or self.pix_disc_idx_list is None \
        or self.beam_disc_val_list is None \
        or recompute:
            self.pix_disc_idx_list = []
            self.beam_disc_val_list = []
            self.band_fwhm_r = band_fwhm_r
            self.band_eval_nside = band_nside
            #compute and load the beams, these will reamain untouched. The FWHM depends on the band.
            for i in range(self.lonlat_arr.shape[0]):
                disc_pix_i_s = hp.query_disc(self.band_eval_nside, hp.ang2vec(self.lonlat_arr[i,0],
                        self.lonlat_arr[i,1], lonlat=True), get_gauss_beam_radius(self.band_fwhm_r))
                self.pix_disc_idx_list.append(disc_pix_i_s)
                beam_disc = gauss_beam(hp.rotator.angdist(self.lonlat_arr[i,:],
                            hp.pix2ang(self.band_eval_nside, disc_pix_i_s, lonlat=True),
                            lonlat=True), self.band_fwhm_r)
                self.beam_disc_val_list.append(beam_disc)
            return True
        else:
            return False

    def get_sed(self, nu:float):
        """
        Returns a list of sed's, one per `alpha_list`, evaluated at `nu`, with ref frequency `nu0`. 
        Freq. are in GHz.
        """
        return (nu/self.nu0)**(self.alpha_arr - 2)
    
    def get_sky(self, nu:float, nside:int, fwhm:float=0.0):
        """
        Returns the sky component given by the point sources at a certain `nu` with a certain
        `nside` and `fwhm` smoothing.
        """
        self.compute_pix_beams(np.deg2rad(fwhm/60), nside)
        map = np.zeros((1, hp.nside2npix(nside)),
                       dtype=np.float64 if self.double_prec else np.float32)
        mJysr_to_uKRJ = (pysm3u.mJy / pysm3u.steradian).to(pysm3u.uK_RJ,
                                            equivalencies=pysm3u.cmb_equivalencies(nu*pysm3u.GHz))
        sed_s = self.get_sed(nu)
        _numba_proj2map(map[0,:], self.pix_disc_idx_list, self.beam_disc_val_list,
                        self._data[0,:],sed_s)
        map*=mJysr_to_uKRJ
        if fwhm == 0.0:
            pass
        else:
            map[0,:] = hp.smoothing(map[0,:], np.deg2rad(fwhm/60))
        return map

    def get_component_map(self, nside:int, fwhm:float=0.0):
        """
        Returns the map of the point sources component with a certain `nside` and `fwhm` smoothing.
        """
        self.compute_pix_beams(np.deg2rad(fwhm/60), nside)
        map = np.zeros((1, hp.nside2npix(nside)),
                       dtype=np.float64 if self.double_prec else np.float32)
        mJysr_to_uKRJ = (pysm3u.mJy / pysm3u.steradian).to(pysm3u.uK_RJ,
                                        equivalencies=pysm3u.cmb_equivalencies(self.nu*pysm3u.GHz))
        _numba_proj2map(map[0,:], self.pix_disc_idx_list, self.beam_disc_val_list, self._data[0,:])
        map*=mJysr_to_uKRJ
        if fwhm == 0.0:
            pass
        else:
            map[0,:] = hp.smoothing(map[0,:], np.deg2rad(fwhm/60))
        return map
    
    def _project_to_band_map(self, map:NDArray, nu:float):
        """
        Computes the point source contribution in uK_RJ for band's frequency and beam,
        and sums it to `map`.

        the `map` array should have shape [1,npix].
        """
        mJysr_to_uKRJ = (pysm3u.mJy / pysm3u.steradian).to(pysm3u.uK_RJ,
                                            equivalencies=pysm3u.cmb_equivalencies(nu*pysm3u.GHz))
        sed_s = self.get_sed(nu)

        _numba_proj2map(map[0,:], self.pix_disc_idx_list, self.beam_disc_val_list,
                        self._data[0,:], sed_s = sed_s)
        map*=mJysr_to_uKRJ
    
    def _eval_from_band_map(self, map, nu):
        """
        Computes the amplitude contribution from the local band to each point source, given `map`.

        All the contributions will be summed to the total proper amplitudes by the master node.

        the `map` array should have shape [1,npix].
        """
        mJysr_to_uKRJ = (pysm3u.mJy / pysm3u.steradian).to(pysm3u.uK_RJ,
                                            equivalencies=pysm3u.cmb_equivalencies(nu*pysm3u.GHz))
        sed_s = self.get_sed(nu)
        _numba_eval_from_map(map[0,:], self.pix_disc_idx_list,
                             self.beam_disc_val_list, self._data[0,:], sed_s = sed_s)
        self._data *= mJysr_to_uKRJ

    def project_comp_to_band(self, band:Band, nthreads: int = 1):
        """
        Project the point sources contribution to the given band in-place,
        summing its contribution to the alm array of the band object.

        NB: this function does not include the beam smoothing.
        """
        assert not band.is_pol, "Point sources component can only be projected to intensity band alms"
        band_fwhm_r, band_nside = np.deg2rad(band.fwhm/60.0), band.nside
        #if not initialized or if band's characteristics changed, recompute the arrays.
        self.compute_pix_beams(band_fwhm_r, band_nside)

        # the point-source correspondent of: M Y a
        ps_map = np.zeros((1,hp.nside2npix(band_nside)),   #empty band map
            dtype=(np.float32 if self.global_params.CG_float_precision == "single" else np.float64))
        self._project_to_band_map(ps_map, band.nu)

        # Y^-1 M Y a
        map_to_alm(ps_map, band_nside, band.lmax, spin=0, out=band.alms, acc=True,
                   nthreads=nthreads)
        
        return band.alms

    def eval_comp_from_band(self, band:Band, nthreads: int = 1):
        """
        Evaluate the band's alm contribution to the point sources' amplitudes and stores it in the
        amp_s member, as well as returning it.

        All the contributions will be summed to the total proper amplitudes when reducing on the
        master node.

        NB: this function does not include the beam smoothing.
        """

        assert not band.is_pol, "Point sources comps can only be evaluated from intensity band alms"
        band_fwhm_r, band_nside = np.deg2rad(band.fwhm/60.0), band.nside
        #if not initialized or if band's characteristics changed, recompute the arrays.
        self.compute_pix_beams(band_fwhm_r, band_nside)

        # Y^-1^T B^T a
        band_map = map_to_alm_adjoint(band.alms, band.nside, band.lmax, spin=0, out=None,
                                      nthreads=nthreads)

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
    
class CompList:
    def __init__(self, comp_list:list[Component]):
        self._validate_comp_list(comp_list)
        self.comp_list = comp_list

    @staticmethod
    def _group_by_logical_key(
        comp_list: list[Component],
    ) -> list[tuple[tuple[type["Component"], str], list[Component]]]:
        grouped_components = {}
        logical_order = []
        for comp in comp_list:
            if comp.logical_key not in grouped_components:
                grouped_components[comp.logical_key] = []
                logical_order.append(comp.logical_key)
            grouped_components[comp.logical_key].append(comp)
        return [(logical_key, grouped_components[logical_key]) for logical_key in logical_order]

    @staticmethod
    def _partition_execution_views(
        group: list[Component],
    ) -> tuple[list[Component], list[Component]]:
        split_views = [comp for comp in group if comp.is_split_view]
        unsplit_views = [comp for comp in group if not comp.is_split_view]
        return split_views, unsplit_views

    @staticmethod
    def _validate_comp_list(comp_list: list[Component]) -> None:
        """Check that a component list has a coherent logical and execution-view layout."""
        if not isinstance(comp_list, list):
            raise TypeError("comp_list must be a list of Component objects.")

        shortname_to_comp_name = {}
        for idx, comp in enumerate(comp_list):
            if not isinstance(comp, Component):
                raise TypeError(f"comp_list[{idx}] must be a Component.")
            if comp.defined_pol is not None:
                assert_pol_supported(comp.defined_pol)
            if comp.eval_pol is not None:
                assert_pol_supported(comp.eval_pol)

            prev_comp_name = shortname_to_comp_name.get(comp.shortname)
            if prev_comp_name is not None and prev_comp_name != comp.comp_name:
                raise ValueError(
                    f"Shortname {comp.shortname!r} is used for both {prev_comp_name!r} and "
                    f"{comp.comp_name!r}."
                )
            shortname_to_comp_name[comp.shortname] = comp.comp_name

        for logical_key, group in CompList._group_by_logical_key(comp_list):
            comp_name = logical_key[1]
            component_types = {type(comp) for comp in group}
            if len(component_types) > 1:
                raise ValueError(
                    f"Component name {comp_name!r} is shared across multiple component classes."
                )

            shortnames = {comp.shortname for comp in group}
            if len(shortnames) > 1:
                raise ValueError(
                    f"Component name {comp_name!r} is associated with multiple shortnames: "
                    f"{sorted(shortnames)!r}."
                )

            split_views, unsplit_views = CompList._partition_execution_views(group)
            if split_views and unsplit_views:
                raise ValueError(
                    f"Component name {comp_name!r} mixes split and unsplit execution views."
                )
            if len(unsplit_views) > 1:
                raise ValueError(f"Duplicate logical component {comp_name!r}.")
            if len(split_views) > 2:
                raise ValueError(f"Component {comp_name!r} has too many split execution views.")
            split_pols = [comp.eval_pol for comp in split_views]
            if len(set(split_pols)) != len(split_pols):
                raise ValueError(f"Component {comp_name!r} repeats a split execution view.")

    @classmethod
    def init_from_params(cls, components:Bunch, params:Bunch):
        # An execution view for a given polarization is only created if there are CompSep MPI ranks
        # assigned to process it. This lets the same component configuration run in intensity-only,
        # QU-only, or joint IQU setups; any requested polarization with no ranks is dropped (with a
        # warning), and a component left with no views at all is skipped entirely.
        pol_has_ranks = {
            "I": params.general.MPI_config.ntask_compsep_I > 0,
            "QU": params.general.MPI_config.ntask_compsep_QU > 0,
        }
        comp_list = []
        for component_str in components:
            component = components[component_str]
            if not component.enabled:
                continue
            component_cls = getattr(component_lib, component.component_class)
            if "lmax" in component.params and component.params.lmax == "full":
                component.params.lmax = (params.general.nside*5)//2
            component_pol = component.params.polarization if "polarization" in component.params \
                else "I"
            if component_pol not in EXECUTION_POLS:
                raise ValueError(
                    f"Unrecognized polarization in parameter file for component {component_str}")
            requested_pols = EXECUTION_POLS[component_pol]
            active_pols = [eval_pol for eval_pol in requested_pols if pol_has_ranks[eval_pol]]
            if not active_pols:
                logging.warning(f"Component '{component_str}' is specified as {component_pol} but "
                                f"no CompSep ranks are assigned to its polarization(s) "
                                f"({'/'.join(requested_pols)}). Skipping.")
                continue
            if len(active_pols) < len(requested_pols):
                skipped = [pol for pol in requested_pols if pol not in active_pols]
                logging.warning(f"Component '{component_str}' is specified as {component_pol} but "
                                f"no CompSep ranks are assigned to {'/'.join(skipped)}; only the "
                                f"{'/'.join(active_pols)} part will be used.")
            for eval_pol in active_pols:
                comp_list.append(component_cls(component.params, params.general, eval_pol=eval_pol,
                                               comp_name=component._name, allocate_empty_alms=True))
        return cls(comp_list)

    def _assert_consistent_comps(self, other: "CompList") -> None:
        if not isinstance(other, CompList):
            raise TypeError("Both operands must be CompList objects.")
        if len(self.comp_list) != len(other.comp_list):
            raise ValueError("Component lists must match in length.")
        self_keys = [comp.execution_key for comp in self.comp_list]
        other_keys = [comp.execution_key for comp in other.comp_list]
        if self_keys != other_keys:
            raise ValueError("Component lists must contain the same execution views in the same order.")

    def components_for_eval_pol(self, target_pol: str) -> list[Component]:
        assert_pol_supported(target_pol)
        return [comp for comp in self.comp_list if comp.eval_pol == target_pol]

    def split_for_eval_pol(self, target_pol: str) -> "CompList":
        """Return the execution-view subset evaluated for one polarization stream."""
        return CompList(self.components_for_eval_pol(target_pol))

    def copy_matching_data_from(self, other: "CompList") -> None:
        if not isinstance(other, CompList):
            raise TypeError("Input must be a CompList.")
        other_by_key = {}
        for comp in other.comp_list:
            if comp.execution_key in other_by_key:
                raise ValueError(f"Duplicate component execution key {comp.execution_key!r}.")
            other_by_key[comp.execution_key] = comp
        self_keys = {comp.execution_key for comp in self.comp_list}
        extra_keys = [key for key in other_by_key if key not in self_keys]
        if extra_keys:
            raise ValueError(f"Found unknown components in source CompList: {extra_keys!r}")
        for comp in self.comp_list:
            other_comp = other_by_key.get(comp.execution_key)
            if other_comp is None:
                continue
            comp._assert_consistent_comp(other_comp)
            np.copyto(comp._data, other_comp._data)

    def broadcast_pol_views(self, comm: MPI.Comm, *, eval_pol: str, source: int) -> None:
        """Broadcast all execution views of `eval_pol` from `source` to every rank in `comm`.

        Used after a sampling step: only the ranks that actually solved a given polarization hold
        the updated component data, so broadcasting that polarization's views from one authoritative
        `source` rank restores a globally consistent component list.
        """
        for comp in self.components_for_eval_pol(eval_pol):
            comp.bcast_data_blocking(comm, root=source)

    def joined(self) -> "CompList":
        """Collapse split execution views back to one logical component per `comp_name`."""
        joined_components = []
        for logical_key, group in self._group_by_logical_key(self.comp_list):
            split_views, unsplit_views = self._partition_execution_views(group)
            if unsplit_views and split_views:
                raise ValueError(
                    f"Logical component {logical_key[1]!r} mixes split and unsplit execution views."
                )
            if len(unsplit_views) > 1:
                raise ValueError(f"Duplicate unsplit component {logical_key[1]!r}.")
            if unsplit_views:
                joined_components.append(deepcopy(unsplit_views[0]))
                continue
            if len(split_views) == 1:
                joined_components.append(deepcopy(split_views[0]))
                continue
            if len(split_views) != 2:
                raise ValueError(
                    f"Expected one or two execution views for {logical_key[1]!r}, got {len(group)}."
                )
            joined_components.append(split_views[0].join_split_views(split_views[1]))

        return CompList(joined_components)
    
    @property
    def components(self):
        return self.comp_list
    
    def __len__(self):
        return len(self.comp_list)
    
    def __matmul__(self, other) -> float:
        """ `dot(comp_list1, comp_list2)`. Calculates the correct dot product between two lists of
            Component objects where the alms follow the Healpy complex storing convention, for
            components with alms. It will automatically handle the correct dot product definition for
            each type of Component.
        """
        self._assert_consistent_comps(other)
        res = 0.0
        for c1, c2 in zip(self.components, other.components):
            res += float(c1 @ c2)
        return res

    def _apply_componentwise_op(self, other: "CompList", component_op, *, inplace: bool) -> "CompList":
        self._assert_consistent_comps(other)
        target = self if inplace else deepcopy(self)
        for target_comp, other_comp in zip(target.components, other.components):
            component_op(target_comp, other_comp)
        return target
    
    def __add__(self, other):
        return self._apply_componentwise_op(other, Component.__iadd__, inplace=False)

    def __iadd__(self, other):
        return self._apply_componentwise_op(other, Component.__iadd__, inplace=True)

    def __sub__(self, other):
        return self._apply_componentwise_op(other, Component.__isub__, inplace=False)

    def __isub__(self, other):
        return self._apply_componentwise_op(other, Component.__isub__, inplace=True)

    def __mul__(self, other):
        return self._apply_componentwise_op(other, Component.__imul__, inplace=False)

    def __imul__(self, other):
        return self._apply_componentwise_op(other, Component.__imul__, inplace=True)

    def __truediv__(self, other):
        return self._apply_componentwise_op(other, Component.__itruediv__, inplace=False)

    def __itruediv__(self, other):
        return self._apply_componentwise_op(other, Component.__itruediv__, inplace=True)

    def __getitem__(self, index):
        return self.comp_list[index]

    def __iter__(self):
        for item in self.comp_list:
            yield item

    def __array_function__(self, func, types, args, kwargs):
        #for numpy func overloads
        if not all(issubclass(t, CompList) for t in types):
            return NotImplemented

        if func is np.zeros_like:
            return self._zeros_like(*args, **kwargs)

        return NotImplemented

    def _zeros_like(self, other, dtype=None, order='K', subok=True, shape=None):
        out = deepcopy(other)
        out.comp_list = [
            np.zeros_like(c,
            dtype=dtype,
            order=order,
            subok=subok,
            shape=shape) for c in other
            ]
        return out

    #MPI functions
    def bcast_data_blocking(self, comm:MPI.Comm, root:int=0):
        for comp in self.comp_list:
            comp.bcast_data_blocking(comm, root=root)
    
    def accum_data_blocking(self, comm:MPI.Comm, root:int=0):
        for comp in self.comp_list:
            comp.accum_data_blocking(comm, root=root)

    def accum_data_non_blocking(self, comm:MPI.Comm, root:int=0) -> list[MPI.Request]:
        requests = []
        for comp in self.comp_list:
            req = comp.accum_data_non_blocking(comm, root=root)
            requests.append(req)
        return requests

    #CompSep solver functions
    def eval_comp_from_band(self, band_in:Band, nthreads:int=1):
        """ Evaluates the band_in's contribution to all the comp_list_out objects, and stores them
            in-place.
        """
        for comp in self.comp_list:
            comp.eval_comp_from_band(band_in, nthreads=nthreads)
    
    def project_comp_to_band(self, band_out:Band, nthreads:int=1) -> NDArray[np.complexfloating]:
        """
        Projects all the components in `comp_list_in`, overwriting the `band_out` object's alms. 
        """
        band_out.alms = np.zeros_like(band_out.alms)
        for comp in self.comp_list:
            comp.project_comp_to_band(band_out, nthreads=nthreads)

    def apply_smoothing_prior_sqrt(self):
        """
        Applies per-component the corresponding smoothing prior.
        """
        for comp in self.comp_list:
            comp.apply_smoothing_prior_sqrt()

    def inplace_add_scaled(self, list_other, scalar):
        """ `list_inplace += scalar*list_other`
        """
        self._assert_consistent_comps(list_other)
        
        for ci, co in zip(self.comp_list, list_other.comp_list):
            inplace_add_scaled_vec(ci._data, co._data, scalar)
    
    def inplace_scale_and_add(self, list_other, scalar):
        """ `list_inplace = scalar*list_inplace + list_other`
        """
        self._assert_consistent_comps(list_other)

        for ci, co in zip(self.comp_list, list_other.comp_list):
            inplace_scale_add(ci._data, co._data, scalar)