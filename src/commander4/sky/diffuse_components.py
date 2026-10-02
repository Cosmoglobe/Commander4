"""Diffuse sky components: the `DiffuseComponent` base and every component built on it.

A diffuse component stores its amplitudes as spherical-harmonic coefficients and evaluates a
spectral energy distribution (SED) per band. `DiffuseComponent` holds everything common to that
representation; the classes after it differ only in their SED and its spectral parameters.
"""
import astropy.constants as c
import astropy.units as u
import healpy as hp
import numpy as np
from numpy.typing import NDArray
from pixell.bunch import Bunch
from scipy.interpolate import interp1d

from commander4.data_models.band import Band
from commander4.sky.component import Component
from commander4.polarization import get_npol
from commander4.parameters.schema import resolve_param
from commander4.units import SUPPORTED_UNITS, unit_factor
from commander4.math_utils.arithmetic import inplace_scale, inplace_add_scaled_vec
from commander4.math_utils.alm import project_alms, almxfl, _dot_complex_alm_1D_arrays
from commander4.math_utils.sht import alm_to_map, map_to_alm, alm_to_map_adjoint, map_to_alm_adjoint

# Blackbody constants shared by the SEDs below.
A = (2*c.h*u.GHz**3/c.c**2).to('MJy').value
h_over_k = (c.h/c.k_B/(1*u.K)).to('GHz-1').value
def blackbody(nu, T):
    return A*nu**3/np.expm1(nu*h_over_k/T)



class DiffuseComponent(Component):
    """A sky component stored as spherical-harmonic coefficients with a per-band SED.

    Holds everything common to that representation: the alm buffer, the C(l) amplitude prior, unit
    conversion of init maps, the SED call that turns an amplitude into a band signal, and the SHT
    projections onto a band. Subclasses supply only `sed_rj` and its spectral parameters, and set
    `nu_ref`.
    """

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
        if ("smoothing_prior_FWHM" in comp_params
                or "smoothing_prior_amplitude" in comp_params):
            raise ValueError(
                f"Component {self.comp_name!r}: the 'smoothing_prior_*' parameters were replaced "
                "by the C3-equivalent 'Cl_prior_*' parameters, which are defined in D_l space "
                "(see DiffuseComponent.P_Cl_prior). Update the parameter file.")
        # C(l) prior (C3 'power_law_gauss' equivalent, see P_Cl_prior). Each parameter may be a
        # scalar or an [I, QU] pair, resolved per execution view like nu_ref; amplitude None
        # disables the prior.
        self.Cl_prior_amplitude = self._per_pol(comp_params.Cl_prior_amplitude)
        self.Cl_prior_beta = self._per_pol(
            comp_params.Cl_prior_beta if "Cl_prior_beta" in comp_params else 0.0)
        self.Cl_prior_FWHM = self._per_pol(
            comp_params.Cl_prior_FWHM if "Cl_prior_FWHM" in comp_params else 0.0)
        self.Cl_prior_l_pivot = (
            comp_params.Cl_prior_l_pivot if "Cl_prior_l_pivot" in comp_params else 50)
        # C3's COMP_L_APOD: the multipole above which the prior is tapered towards zero. Defaults
        # to this component's lmax, which makes the taper a no-op (C3's own parameter files almost
        # always set it that way too).
        self.Cl_prior_l_apod = self._per_pol(
            comp_params.Cl_prior_l_apod if "Cl_prior_l_apod" in comp_params else self.lmax)
        # The unit the amplitude alms are stored in: the run's global unit, referenced to `nu_ref`.
        # It also sets the unit of the C(l) prior (its square) and of the alms in the compsep chain.
        self.amplitude_unit = resolve_param(global_params, "global_unit", ("",), default="uK_RJ",
                                            legal_values=SUPPORTED_UNITS)
        # Unit of an init_from or amp_prior_mean_map sky map for this component (None -> assume it
        # is already in `amplitude_unit`). Only used when reading FITS maps, not compsep chains.
        self.units = comp_params.units if "units" in comp_params else None
        # Cached prior mean mu, filled from `amp_prior_mean_map` by CompList.load_amp_prior_means;
        # None means a zero-mean prior. See the amp_prior_mean property.
        self._amp_prior_mean_alms = None
        self._data = None  # Alm data is not allocated by default.
        if allocate_empty_alms:
            self.allocate_empty_alms()

    def _per_pol(self, value):
        """Resolve a scalar-or-``[I, QU]`` parameter to this view's value (I -> first entry)."""
        if isinstance(value, (list, tuple)):
            return value[0] if self.eval_pol == "I" else value[1]
        return value

    def _reference_frequency(self, comp_params: Bunch) -> float:
        """Reference frequency (GHz) for this view's polarization.

        ``nu_ref`` is either a scalar (shared by I and QU) or a 2-element list ``[nu_I, nu_QU]``.
        """
        return self._per_pol(comp_params.nu_ref)

    def init_map_to_amplitude(self, sky_map: NDArray) -> NDArray:
        """Convert an init sky map (in ``self.units``) to this component's amplitude unit.

        The conversion is done at the component's reference frequency (``self.nu_ref``). It is a
        no-op when the units are unspecified or already equal to the amplitude unit.
        """
        if self.units is None or self.units == self.amplitude_unit:
            return sky_map
        return sky_map*unit_factor(self.nu_ref, self.units, self.amplitude_unit)

    def get_sed(self, nu: float, unit: str) -> float:
        """Factor from this component's amplitude to its signal at `nu` (GHz), expressed in `unit`.

        The amplitude is in `amplitude_unit` at `nu_ref`. It is converted to uK_RJ at `nu_ref`,
        scaled to `nu` by the spectral shape `sed_rj`, and converted to `unit` at `nu`. This is the
        one place a component amplitude becomes a band signal, so `unit` is always the unit of the
        band asking.
        """
        amplitude_to_rj = unit_factor(self.nu_ref, self.amplitude_unit, "uK_RJ")
        return amplitude_to_rj*self.sed_rj(nu)*unit_factor(nu, "uK_RJ", unit)

    def sed_rj(self, nu: float) -> float:
        """The spectral shape: RJ brightness at `nu` (GHz) relative to that at `nu_ref`."""
        raise NotImplementedError(f"{type(self).__name__}.sed_rj() is not implemented.")

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
        """ Allocates a zeroed alm array of the correct shape. Useful for e.g. MPI receiving.
        """
        self._data = np.zeros((self.npol, self.alm_len_complex),
                               dtype = (np.complex128 if self.double_prec else np.complex64))
            
    @property
    def dtype(self):
        return self._data.dtype

    Cl_prior_param_names = ("Cl_prior_amplitude", "Cl_prior_beta", "Cl_prior_FWHM",
                            "Cl_prior_l_pivot", "Cl_prior_l_apod")

    @property
    def alm_len_complex(self):
        return ((self.lmax+1)*(self.lmax+2))//2

    @property
    def sigma_l(self) -> NDArray[np.floating]:
        """Realized angular power spectrum of this component's own amplitudes, per polarization.

        Commander3's `sigma_l` (comm_diffuse_comp_smod.f90). Shape `(npol, lmax+1)`, one auto
        spectrum per stored alm row: T for an intensity view, E and B for a QU one, T/E/B for a
        joined IQU component. Cheap -- the alms are already in memory, so this is no transform.
        """
        # healpy's alm2cl only accepts complex128, while the alms are complex64 unless
        # `compsep.double_precision` is set.
        alms = np.ascontiguousarray(self.alms, dtype=np.complex128)
        return np.array([hp.alm2cl(alms[ipol], lmax=self.lmax) for ipol in range(alms.shape[0])])

    @property
    def P_Cl_prior(self) -> NDArray[np.floating]:
        """Prior angular power spectrum C_l for this component's amplitude alms.

        This is the S in the CG system (1 + S^{1/2} A^T N^-1 A S^{1/2}). It is a Gaussian prior
        constraining the alms to N(0, C_l). Equivalent to C3's 'power_law_gauss' (comm_cl_mod.f90),
        which contains its 'power_law' (FWHM=0) and 'gauss' (beta=0) types as special cases.
        Defined in D_l space, where CMB-like spectra are roughly flat:

            D_l = amplitude * (l / l_pivot)^beta * max(exp(-l(l+1) sigma^2), 1e-10),
            C_l = 2 pi D_l / (l(l+1)) * f_apod(l)^2,

        where sigma is the Gaussian width of Cl_prior_FWHM (arcmin; 0 disables the rolloff). The
        1e-10 floor (relative to the power law) keeps C_l strictly positive so 1/C_l is safe for the
        preconditioners. Units are (amplitude_unit @ nu_ref)^2, i.e. the units of the alms
        themselves, so `Cl_prior_amplitude` is quoted in the square of `compsep.global_unit` (C3
        instead defines the prior in the component's native unit and converts internally).

        f_apod is C3's high-l apodization (`get_Cl_apod` in comm_cl_mod.f90, parameter COMP_L_APOD):
        unity up to l_apod, then exp(-ln(1000) (l-l_apod)^2 / (lmax-l_apod+1)^2), an amplitude taper
        reaching 1e-3 (1e-6 in power) at the component's own lmax. Setting l_apod at or below the
        highest band lmax keeps a component whose lmax exceeds what the data can see from filling
        those multipoles with a full-strength prior draw. It defaults to the component lmax, where
        the taper does nothing.
        """
        if self.Cl_prior_amplitude is None:
            return np.ones(self.lmax + 1)
        sigma = np.deg2rad(self.Cl_prior_FWHM / 60.0) / np.sqrt(8.0 * np.log(2.0))
        ells = np.arange(1, self.lmax + 1)
        Dl = np.empty(self.lmax + 1)
        Dl[1:] = self.Cl_prior_amplitude * (ells / self.Cl_prior_l_pivot)**self.Cl_prior_beta \
            * np.maximum(np.exp(-ells * (ells + 1) * sigma**2), 1e-10)
        Dl[0] = Dl[1]
        Cl = np.empty(self.lmax + 1)
        Cl[1:] = Dl[1:] * 2.0 * np.pi / (ells * (ells + 1))
        Cl[0] = Dl[0]
        return Cl * self.Cl_prior_apodization**2

    @property
    def Cl_prior_apodization(self) -> NDArray[np.floating]:
        """C3's `get_Cl_apod` factor over l = 0..lmax, applied to the prior as f_apod^2.

        Unity up to `Cl_prior_l_apod`, then a Gaussian taper falling to 1e-3 at the component lmax.
        Only C3's positive-`l_apod` branch (the high-l taper) is implemented; C3's negative branch
        suppresses the *low* multipoles instead, which is a separate feature.
        """
        ells = np.arange(self.lmax + 1)
        l_apod = min(self.Cl_prior_l_apod, self.lmax)
        f = np.exp(-np.log(1e3) * (ells - l_apod)**2 / (self.lmax - l_apod + 1)**2)
        f[:l_apod + 1] = 1.0
        return f

    @property
    def P_Cl_prior_inv(self) -> NDArray[np.floating]:
        # P_Cl_prior is strictly positive by construction (1e-10 floor), so plain inversion is safe.
        return 1.0 / self.P_Cl_prior

    def __repr__(self):
        return f"Diffuse Component {self.shortname}, with polarization: {self.eval_pol}"\
                f" (originally defined as {self.defined_pol})" \
                f"\n   lmax = {self.lmax} \n   alms: {self.alms}"

    def apply_Cl_prior_sqrt(self, alms: NDArray[np.complexfloating]) \
            -> NDArray[np.complexfloating]:
        """Multiplies `alms` by the C_l prior square root S^{1/2}, in place, and returns them.

        The target is always explicit, so that scaling the component's own amplitudes and scaling
        something else that lives in its alm space read differently at the call site. To apply this
        to the component's own alms (the usual case, the CG's a = S^{1/2}x reparameterization)
        call ``comp.apply_Cl_prior_sqrt(comp.alms)``.
        """
        prior_sqrt = np.sqrt(self.P_Cl_prior)
        for ipol in range(self.npol):
            almxfl(alms[ipol], prior_sqrt, inplace=True)
        return alms

    def apply_Cl_prior_inv_sqrt(self, alms: NDArray[np.complexfloating]) \
            -> NDArray[np.complexfloating]:
        """Multiplies `alms` by S^{-1/2}, in place, and returns them; the inverse of the above.

        Same contract as `apply_Cl_prior_sqrt`, including the explicit target: use
        ``comp.apply_Cl_prior_inv_sqrt(comp.alms)`` to act on the component's own amplitudes.
        P_Cl_prior is strictly positive by construction (it carries a 1e-10 floor), so the inversion
        needs no guard.
        """
        prior_inv_sqrt = np.sqrt(self.P_Cl_prior_inv)
        for ipol in range(self.npol):
            almxfl(alms[ipol], prior_inv_sqrt, inplace=True)
        return alms

    @property
    def amp_prior_mean(self) -> NDArray[np.complexfloating] | None:
        """The prior mean mu of this component's amplitude alms, in the units of `alms`.

        The Gaussian amplitude prior is a ~ N(mu, S), with S given by `P_Cl_prior`; mu is where the
        solution is pulled in the absence of informative data. Returns None (meaning a zero-mean
        prior, and letting the caller skip the whole term) unless the component sets
        `amp_prior_mean_map`, C3's ``COMP_AMP_PRIOR_MAP``.

        When set, always returns a fresh copy: the CG right-hand side applies S^{-1/2} to it *in
        place*, so handing out the cached mu itself would re-scale the cache on every iteration.
        """
        if self._amp_prior_mean_alms is None:
            return None
        return self._amp_prior_mean_alms.copy()

    @amp_prior_mean.setter
    def amp_prior_mean(self, alms: NDArray[np.complexfloating]) -> None:
        if alms.shape != (self.npol, self.alm_len_complex):
            raise ValueError(f"Component {self.comp_name!r}: prior mean has shape {alms.shape}, "
                             f"expected {(self.npol, self.alm_len_complex)}.")
        self._amp_prior_mean_alms = alms

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

    def get_sky(self, nu: float, unit: str, nside: int, fwhm: float = 0):
        """ Realize this component at frequency `nu` (GHz) in `unit`, at beam resolution `fwhm`
            (radians).
            Note that if the component amplitudes already carry beam-smoothing (which happens when
            the per-pix common-resolution amplitude solver is used), only the effective fwhm
            difference is applied.
        """
        target_fwhm = 0.0 if fwhm is None else fwhm
        applied_fwhm = np.sqrt(max(target_fwhm**2 - self.amp_fwhm_rad**2, 0.0))
        return self.get_component_map(nside, applied_fwhm)*self.get_sed(nu, unit)

    # Overrides the base-class dot product: diffuse-component _data holds complex alms, whose inner
    # product must account for the m>0 coefficients each standing for two real degrees of freedom.
    def __matmul__(self, other):
        self._assert_consistent_comp(other)
        res = 0.0
        for ipol in range(self.npol):
            res += _dot_complex_alm_1D_arrays(self._data[ipol], other._data[ipol], self.lmax)
        return res

    def project_comp_to_band(self, band:Band, nthreads: int = 1):
        """Project the component to the given band in-place, summing its contribution into the alms
           array of the passed band object.

        NB: this function does not include the beam smoothing.
        """
        if self.is_pol != band.is_pol:
            raise ValueError("Band and component polarization must match.")

        alm_in_band_space = project_alms(self.alms, band.lmax)
        # The mixing factor also converts the amplitude to the band's own unit.
        sed = self.get_sed(band.nu, band.unit)
        if self.spatially_varying_MM:  # If this component's mixing matrix is pixel-dependent.
            # Y a
            comp_map = alm_to_map(alm_in_band_space, band.nside, band.lmax, spin=self.spin,
                                  nthreads=nthreads)
            # M Y a
            for ipol in range(self.npol):
                inplace_scale(comp_map[ipol], sed)
            # Y^-1 M Y a
            band.alms = map_to_alm(comp_map, band.nside, band.lmax, spin=self.spin, out=band.alms,
                                   acc=True, nthreads=nthreads)
        else:
            for ipol in range(self.npol):
                inplace_add_scaled_vec(band.alms[ipol], alm_in_band_space[ipol], sed)
        return band.alms

    def eval_comp_from_band(self, band:Band, nthreads: int = 1, inplace=True):
        """Evaluate the band's alm contribution to the component, storing it in-place by default,
           and return it.

        All the contributions will be summed to the total proper amplitudes by the master node.

        NB: this function does not include the beam smoothing.
        """
        if self.is_pol != band.is_pol:
            raise ValueError("Band and component polarization must match.")

        sed = self.get_sed(band.nu, band.unit)
        if self.spatially_varying_MM:  # If this component's mixing matrix is pixel-dependent.
            # Y^-1^T B^T a
            band_map = map_to_alm_adjoint(band.alms, band.nside, band.lmax, spin=self.spin, out=None,
                                          nthreads=nthreads)

            # M^T Y^-1 B^T a
            for ipol in range(self.npol):
                inplace_scale(band_map[ipol], sed)

            # Y^T M^T Y^-1^T B^T a
            tmp_alm = alm_to_map_adjoint(band_map, band.nside, band.lmax, spin=self.spin, out=None,
                                         nthreads=nthreads)

        else:
            tmp_alm = band.alms.copy()
            for ipol in range(self.npol):
                inplace_scale(tmp_alm[ipol], sed)
            
        # Project alm from band to component lmax.
        contrib_to_comp_alm = project_alms(tmp_alm, self.lmax)
        
        if inplace:
            self.alms = contrib_to_comp_alm

        return contrib_to_comp_alm
    


class CMB(DiffuseComponent):
    """The CMB, whose SED is flat in thermodynamic units."""

    default_shortname = "cmb"
    sed_param_names = ("nu_ref",)
    # Like all diffuse components, the CMB amplitude is stored in the global unit, referenced to
    # `nu_ref`. With a global unit of uK_CMB the amplitude is exactly uK_CMB whatever `nu_ref` is.
    # With the default uK_RJ, the default `nu_ref` of 1 GHz keeps it close to uK_CMB.

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
        # The CMB blackbody is polarization-independent, so a scalar reference suffices. The choice
        # is arbitrary (the sky is invariant to it); 1 GHz keeps stored amplitudes ~= uK_CMB.
        self.nu_ref = self._reference_frequency(comp_params) if "nu_ref" in comp_params else 1.0

    def sed_rj(self, nu: float) -> float:
        """The CMB spectral shape: the thermodynamic-to-RJ conversion at `nu` relative to `nu_ref`.

        A CMB fluctuation of fixed thermodynamic temperature has an RJ brightness proportional to
        the uK_CMB -> uK_RJ factor at each frequency, so the ratio of the two factors is the shape.
        """
        return unit_factor(nu, "uK_CMB", "uK_RJ")/unit_factor(self.nu_ref, "uK_CMB", "uK_RJ")

    def get_sky_anisotropies(self, nu: float, unit: str, nside: int, fwhm: float = 0):
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
        return self._realize_alms_as_map(component_alms, nside, fwhm)*self.get_sed(nu, unit)


class ThermalDust(DiffuseComponent):
    """Thermal dust, a modified blackbody with emissivity index `beta` and temperature `T`."""

    default_shortname = "term-dust"
    sed_param_names = ("beta", "T", "nu_ref")

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
        self.nu_ref = self._reference_frequency(comp_params)

    def sed_rj(self, nu):
        """Calculates the spectral energy distribution (SED) for Thermal Dust emission.
           The result is unitless, but meant to be multiplied by a RJ brightness temperature.
        Args:
            nu (float or np.ndarray): Frequency in GHz at which to evaluate the SED.
        Returns:
            The SED scaling factor (float or np.ndarray).
        """
        # Modified blackbody, as an RJ brightness ratio.
        x = (h_over_k*nu)/(self.T)
        x0 = (h_over_k*self.nu_ref)/(self.T)
        return (nu / self.nu_ref)**(self.beta + 1.0) * np.expm1(x0) / np.expm1(x)


class Synchrotron(DiffuseComponent):
    """Synchrotron emission, a power law in RJ brightness with spectral index `beta`."""

    default_shortname = "sync"
    sed_param_names = ("beta", "nu_ref")

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
        self.nu_ref = self._reference_frequency(comp_params)
        self.nside_comp_map = 512

    def sed_rj(self, nu):
        """Calculates the spectral energy distribution (SED) for Synchrotron emission.
           The result is unitless, but meant to be multiplied by a RJ brightness temperature.
        Args:
            nu (float or np.ndarray): Frequency in GHz at which to evaluate the SED.            
        Returns:
            The SED scaling factor (float or np.ndarray).
        """
        return (nu/self.nu_ref)**self.beta


class FreeFree(DiffuseComponent):
    """Free-free (bremsstrahlung) emission from ionized gas at electron temperature `T`."""

    default_shortname = "ff"
    sed_param_names = ("T", "nu_ref")

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
        self.nu_ref = self._reference_frequency(comp_params) # Reference frequency in GHz

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

    def sed_rj(self, nu):
        """Calculates the spectral energy distribution (SED) for Free-Free emission.
           The result is unitless, but meant to be multiplied by a RJ brightness temperature.
        Args:
            nu (float or np.ndarray): Frequency in GHz at which to evaluate the SED.            
        Returns:
            The SED scaling factor (float or np.ndarray).
        """
        gaunt_nu = self._gaunt_factor(nu, self.T)
        gaunt_nu_ref = self._gaunt_factor(self.nu_ref, self.T)

        # The scaling is proportional to nu^-2 * g_ff(nu), normalized to 1 at nu_ref.
        sed = (self.nu_ref / nu)**2 * (gaunt_nu / gaunt_nu_ref)
        return sed


class SpinningDust(DiffuseComponent):
    """Spinning dust, whose spectral shape comes from a tabulated SpDust2 template.

    The template is the Cold Neutral Medium model, which peaks at 30 GHz. It is shifted in
    frequency so its peak lands at `nu_peak` (`comp_params.nu_peak`), which is the one shape
    parameter; the `nu_0` parameter only sets where the amplitude map is normalized, and is stored
    as `nu_ref` like every other component's reference frequency.
    """

    default_shortname = "spin-dust"
    sed_param_names = ("nu_peak_eval", "nu_peak_ref", "nu_ref")

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

        # Two-column SpDust2 template: frequency (GHz) and emissivity (proportional to intensity).
        freqs, SED = np.loadtxt(comp_params.template_path).T
        self.nu_peak_ref = 30.0  # Peak frequency of the template as tabulated.
        self.nu_peak_eval = comp_params.nu_peak
        self.nu_ref = comp_params.nu_0  # Reference frequency for the amplitude map in GHz

        # Interpolate in log-log space, where the template is smooth and spans many decades.
        log_nu = np.log(freqs)
        log_SED = np.log(SED)
        self._log_j_interp = interp1d(log_nu, log_SED, kind='cubic',
                                      bounds_error=False, fill_value=-np.inf)

    def _get_template_emissivity(self, nu):
        """Template emissivity at frequency `nu` (GHz), by log-log interpolation."""
        return np.exp(self._log_j_interp(np.log(nu)))

    def sed_rj(self, nu: float|NDArray[np.floating]):
        """Calculates the spinning dust SED scaling factor.

        Scales an amplitude map from its reference frequency `nu_ref` to the target frequency `nu`.

        Args:
            nu (float|array): Frequency at which to get the SED, in GHz.
        Returns:
            float|array: The unitless SED scaling factor.
        """
        # Numerator: template evaluated at the shifted frequency
        nu_shifted_eval = nu * self.nu_peak_ref / self.nu_peak_eval
        SED_eval = self._get_template_emissivity(nu_shifted_eval)

        # Denominator: template evaluated at the shifted reference frequency for normalization
        nu_shifted_ref = self.nu_ref * self.nu_peak_ref / self.nu_peak_eval
        SED_ref = self._get_template_emissivity(nu_shifted_ref)

        # Shifting the SED spectrum from the reference frequency to the given peak frequency.
        SED_at_eval_freq = SED_eval / SED_ref

        # Converting from intensity to brightness temperature.
        SED_uK_RJ = (self.nu_ref / nu)**2 * SED_at_eval_freq
        return SED_uK_RJ
    
