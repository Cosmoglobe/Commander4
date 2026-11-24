"""Diffuse components: `DiffuseComponent`, stored as alms, and the components built on it.

The components after `DiffuseComponent` differ only in their SED and its parameters.
"""
import logging

import astropy.constants as c
import astropy.units as u
import h5py
import healpy as hp
import numpy as np
import pysm3.units as pysm3u
from numpy.typing import NDArray
from pixell.bunch import Bunch
from scipy.interpolate import interp1d

from commander4.data_models.band import Band
from commander4.sky.component import Component
from commander4.sky.comp_io import _pol_row_indices, _read_view_alms_from_fits
from commander4.math_utils.arithmetic import inplace_scale
from commander4.math_utils.alm import project_alms, almxfl, _dot_complex_alm_1D_arrays
from commander4.math_utils.sht import alm_to_map, map_to_alm, alm_to_map_adjoint, map_to_alm_adjoint

logger = logging.getLogger(__name__)

# h/k_B in GHz^-1 K, for the blackbody of the thermal-dust SED.
h_over_k = (c.h/c.k_B/(1*u.K)).to('GHz-1').value


class DiffuseComponent(Component):
    """A component stored as alms, in uK_RJ at `nu_ref`, with a C(l) prior.

    Subclasses define `get_sed` and its parameters.
    """

    DEFAULTS = {
        **Component.DEFAULTS,
        "lmax": None,                   # The highest multipole of the alms. Must be set.
        # C3's COMP_UNIT: the unit of the `init_from` map, the prior mean map and
        # `Cl_prior_amplitude`. None means `amplitude_unit`, the unit of the chains.
        "units": None,
        # C3's COMP_AMP_PRIOR_MAP: a sky map with the mean of the amplitude prior.
        "amp_prior_mean_map": None,
        # The C(l) prior (see `P_Cl_prior`).
        "Cl_prior_amplitude": None,     # None is no prior (C3's `CL_TYPE = none`).
        "Cl_prior_beta": [0.0, 0.0],
        "Cl_prior_FWHM": [0.0, 0.0],
        "Cl_prior_l_pivot": 50,
        # C3's COMP_L_APOD: the prior is tapered towards zero above it. None is the component
        # lmax, where the taper does nothing. Set it at or below the highest band lmax when the
        # component lmax is above it, so that the multipoles no band sees get no full-strength draw.
        "Cl_prior_l_apod": None,
        # C3's COMP_AMP_PRIOR_LMAX: the prior is tightened below it, which holds the large scales
        # to the prior mean. 0 is no tightening.
        "Cl_prior_l_tighten": [0, 0]}

    amplitude_unit = "uK_RJ"  # Also for the CMB.

    def __init__(self, params: Bunch, comp_name: str, pol: str):
        """One polarization view of the component `comp_name`, with its alms set to zero."""
        super().__init__(params, comp_name, pol)
        self.lmax = self.cfg.lmax
        # Attributes, like the SED parameters: this view's values, written to the chain.
        self.Cl_prior_amplitude = self._per_pol(self.cfg.Cl_prior_amplitude)
        self.Cl_prior_beta = self._per_pol(self.cfg.Cl_prior_beta)
        self.Cl_prior_FWHM = self._per_pol(self.cfg.Cl_prior_FWHM)
        self.Cl_prior_l_pivot = self.cfg.Cl_prior_l_pivot
        self.Cl_prior_l_apod = self._per_pol(self.cfg.Cl_prior_l_apod)
        if self.Cl_prior_l_apod is None:
            self.Cl_prior_l_apod = self.lmax
        self.Cl_prior_l_tighten = self._per_pol(self.cfg.Cl_prior_l_tighten)
        self._amp_prior_mean_alms = None  # Set by `load_amp_prior_mean`.
        self.amps = np.zeros((self.npol, hp.Alm.getsize(self.lmax)),
                             dtype=np.complex128 if self.double_precision else np.complex64)

    def init_map_to_amplitude(self, sky_map: NDArray) -> NDArray:
        """Convert a map from the `units` setting to `amplitude_unit`, at `nu_ref`."""
        if self.cfg.units is None or self.cfg.units == self.amplitude_unit:
            return sky_map
        factor = (1*pysm3u.Unit(self.cfg.units)).to(
            pysm3u.Unit(self.amplitude_unit),
            equivalencies=pysm3u.cmb_equivalencies(self.nu_ref*pysm3u.GHz)).value
        return sky_map * factor

    Cl_prior_param_names = ("Cl_prior_amplitude", "Cl_prior_beta", "Cl_prior_FWHM",
                            "Cl_prior_l_pivot", "Cl_prior_l_apod", "Cl_prior_l_tighten")

    @property
    def sigma_l(self) -> NDArray[np.floating]:
        """The power spectrum of the alms, one per row of `amps` (T; E, B; or T, E, B), shape
        (npol, lmax+1)."""
        # healpy's alm2cl only accepts complex128, while the alms are complex64 unless
        # `compsep.double_precision` is set.
        alms = np.ascontiguousarray(self.amps, dtype=np.complex128)
        return np.array([hp.alm2cl(alms[ipol], lmax=self.lmax) for ipol in range(alms.shape[0])])

    @property
    def P_Cl_prior(self) -> NDArray[np.floating]:
        """The prior power spectrum C_l of the alms (the S of the CG), shape (lmax+1,).

        C3's 'power_law_gauss' prior, defined in D_l:

            D_l = amplitude * (l / l_pivot)^beta * max(exp(-l(l+1) sigma^2), 1e-10),
            C_l = 2 pi D_l / (l(l+1)) * f_apod(l)^2,

        with sigma the Gaussian width of `Cl_prior_FWHM` (arcmin; 0 is no rolloff), and f_apod from
        `Cl_prior_apodization`. `Cl_prior_amplitude` is in the component's `units` squared, and
        C_l in uK_RJ^2. Without a prior (`Cl_prior_amplitude: null`) this returns ones, so that
        S^{1/2} does nothing.
        """
        if self.Cl_prior_amplitude is None:
            return np.ones(self.lmax + 1)
        sigma = np.deg2rad(self.Cl_prior_FWHM / 60.0) / np.sqrt(8.0 * np.log(2.0))
        ells = np.arange(1, self.lmax + 1)
        Dl = np.empty(self.lmax + 1)
        # The unit factor enters squared, because D_l is a power.
        amplitude = self.Cl_prior_amplitude * self.init_map_to_amplitude(1.0)**2
        # The 1e-10 floor keeps C_l above zero, so that 1/C_l is safe.
        Dl[1:] = amplitude * (ells / self.Cl_prior_l_pivot)**self.Cl_prior_beta \
            * np.maximum(np.exp(-ells * (ells + 1) * sigma**2), 1e-10)
        Dl[0] = Dl[1]
        Cl = np.empty(self.lmax + 1)
        Cl[1:] = Dl[1:] * 2.0 * np.pi / (ells * (ells + 1))
        Cl[0] = Dl[0]
        return Cl * self.Cl_prior_apodization**2

    @property
    def Cl_prior_apodization(self) -> NDArray[np.floating]:
        """The factor f_apod(l), l = 0..lmax, that `P_Cl_prior` applies squared.

        1 up to `Cl_prior_l_apod`, then a Gaussian taper down to 1e-3 at lmax. Below
        `Cl_prior_l_tighten` it is also multiplied by sin^4(pi l / (2 l_tighten)), which goes from 1
        at l_tighten towards 0 at l = 1.
        """
        # Only C3's positive l_apod (the high-l taper) is implemented.
        ells = np.arange(self.lmax + 1)
        l_apod = min(self.Cl_prior_l_apod, self.lmax)
        f = np.exp(-np.log(1e3) * (ells - l_apod)**2 / (self.lmax - l_apod + 1)**2)
        f[:l_apod + 1] = 1.0
        # l = 0 gets the factor of l = 1, so that C_l stays above zero.
        tightened = ells < self.Cl_prior_l_tighten
        f[tightened] *= np.sin(0.5*np.pi*np.maximum(ells[tightened], 1)/self.Cl_prior_l_tighten)**4
        return f

    @property
    def P_Cl_prior_inv(self) -> NDArray[np.floating]:
        """1/`P_Cl_prior`."""
        return 1.0 / self.P_Cl_prior

    def __repr__(self):
        return f"Diffuse Component {self.comp_name}, with polarization: {self.pol}"\
                f" (originally defined as {self.cfg.polarization})" \
                f"\n   lmax = {self.lmax} \n   alms: {self.amps}"

    def apply_Cl_prior_sqrt(self, alms: NDArray[np.complexfloating]) \
            -> NDArray[np.complexfloating]:
        """Multiply `alms` by S^{1/2} in place, and return them."""
        # `alms` is an argument, not `self.amps`, because the CG also scales other vectors in this
        # component's alm space.
        prior_sqrt = np.sqrt(self.P_Cl_prior)
        for ipol in range(self.npol):
            almxfl(alms[ipol], prior_sqrt, inplace=True)
        return alms

    def apply_Cl_prior_inv_sqrt(self, alms: NDArray[np.complexfloating]) \
            -> NDArray[np.complexfloating]:
        """Multiply `alms` by S^{-1/2} in place, and return them."""
        prior_inv_sqrt = np.sqrt(self.P_Cl_prior_inv)
        for ipol in range(self.npol):
            almxfl(alms[ipol], prior_inv_sqrt, inplace=True)
        return alms

    @property
    def amp_prior_mean(self) -> NDArray[np.complexfloating] | None:
        """The mean mu of the amplitude prior a ~ N(mu, S), or None for a zero mean.

        Set from `amp_prior_mean_map`. Returns a copy.
        """
        if self._amp_prior_mean_alms is None:
            return None
        # A copy, because the CG applies S^{-1/2} to it in place.
        return self._amp_prior_mean_alms.copy()

    @amp_prior_mean.setter
    def amp_prior_mean(self, alms: NDArray[np.complexfloating]) -> None:
        if alms.shape != self.amps.shape:
            raise ValueError(f"Component {self.comp_name!r}: prior mean has shape {alms.shape}, "
                             f"expected {self.amps.shape}.")
        self._amp_prior_mean_alms = alms

    def load_amp_prior_mean(self) -> None:
        """Read the prior mean mu from `amp_prior_mean_map`, if set.

        The map is converted like an `init_from` map. The CG applies S^{-1/2} to mu in each solve,
        because S may change between iterations.
        """
        source_path = self.cfg.amp_prior_mean_map
        if not source_path:
            return  # Zero-mean prior.
        if not str(source_path).lower().endswith(".fits"):
            raise ValueError(
                f"Component {self.comp_name!r}: 'amp_prior_mean_map' must be a .fits sky map, "
                f"but got {source_path!r}.")
        if self.Cl_prior_amplitude is None:
            raise ValueError(
                f"Component {self.comp_name!r} ({self.pol}): 'amp_prior_mean_map' needs a "
                "prior to be the mean of, but 'Cl_prior_amplitude' is null (no prior).")
        # The iterative inverse transform, as for `init_from`. C3 uses one non-iterative YtW here,
        # which is less accurate on a HEALPix grid.
        view_alms = _read_view_alms_from_fits(self, source_path)
        if view_alms is not None:
            self.amp_prior_mean = view_alms.astype(self.amps.dtype, copy=False)
            logger.info(f"Component {self.comp_name!r} ({self.pol}): prior mean read from "
                        f"{source_path!r}.")

    def read_amps_from_map(self, map_file: str) -> None:
        """Set this view's alms from a FITS sky map, given in the component's `units`.

        If the map lacks this view's polarization, the alms keep their current value.
        """
        view_alms = _read_view_alms_from_fits(self, map_file)
        if view_alms is not None:
            self.amps = view_alms.astype(self.amps.dtype, copy=False)

    def write_chain(self, group: h5py.Group, band_frequencies: dict[str, float]) -> None:
        """As `Component.write_chain`, plus the alms, their power spectrum and the prior
        parameters."""
        super().write_chain(group, band_frequencies)
        group["alms"] = self.amps
        group["sigma_l"] = self.sigma_l
        group["lmax"] = self.lmax
        # The prior's parameters, not the evaluated spectrum. Without a prior no amplitude is
        # written.
        for param_name in self.Cl_prior_param_names:
            value = getattr(self, param_name)
            if value is not None:
                group[f"Cl_prior/{param_name}"] = value

    def read_chain(self, group: h5py.Group, restore_beta: bool) -> None:
        """As `Component.read_chain`, plus this view's rows of the alms, cut or padded to `lmax`.

        Alms the file lacks for this polarization keep their current value.
        """
        super().read_chain(group, restore_beta)
        stored_alms = group["alms"][()]
        rows = _pol_row_indices(stored_alms, self.pol, self.comp_name, group.file.filename)
        if rows is None:
            logger.debug(f"{group.file.filename!r} has no {self.pol!r} alms for component "
                         f"{self.comp_name!r}; leaving those alms unchanged.")
            return
        view_alms = project_alms(np.ascontiguousarray(stored_alms[rows]), self.lmax)
        self.amps = view_alms.astype(self.amps.dtype, copy=False)

    def get_component_map(self, nside: int, fwhm_rad: float = 0.0, lmin: int = 0) -> NDArray:
        """The amplitude map at `nside`, smoothed by a Gaussian beam of `fwhm_rad` radians, without
        the multipoles below `lmin`. For plotting; a band's sky comes from `CompList.get_sky_at_nu`.
        """
        lmax = min(self.lmax, 3*nside - 1)
        window = hp.gauss_beam(fwhm_rad, lmax=lmax)
        window[:lmin] = 0.0
        alms = np.array([almxfl(alm, window) for alm in project_alms(self.amps, lmax)])
        if self.pol != "IQU":
            return alm_to_map(alms, nside, lmax, spin=self.spin)
        # A joined IQU component needs separate spin-0 and spin-2 synthesis calls.
        return np.concatenate((alm_to_map(alms[:1], nside, lmax, spin=0),
                               alm_to_map(alms[1:], nside, lmax, spin=2)), axis=0)

    def get_sky(self, nu: float, nside: int, fwhm_rad: float = 0.0) -> NDArray:
        """The map at frequency `nu` (GHz), through a total beam of `fwhm_rad`, for plotting."""
        # Amplitudes that already carry a beam get only the rest of it.
        applied_fwhm_rad = np.sqrt(max(fwhm_rad**2 - self.amp_fwhm_rad**2, 0.0))
        return self.get_component_map(nside, applied_fwhm_rad)*self.get_sed(nu)

    def get_sed(self, nu):
        """The SED at `nu` (GHz, a number or an array): the factor that takes the amplitude at
        `nu_ref` to the sky at `nu`, both in uK_RJ."""
        raise NotImplementedError(f"{type(self).__name__}.get_sed() is not implemented.")

    def dot(self, other: "DiffuseComponent") -> float:
        """The dot product of the alms with those of `other`, the same view."""
        # Only m >= 0 is stored, so each m > 0 coefficient stands for two and counts twice.
        res = 0.0
        for ipol in range(self.npol):
            res += _dot_complex_alm_1D_arrays(self.amps[ipol], other.amps[ipol], self.lmax)
        return res

    def _band_beam(self, band: Band) -> NDArray:
        """The band's Gaussian beam window, up to the band lmax, less the beam the amplitudes carry.
        """
        # Gaussian beams multiply, so their widths add in quadrature.
        fwhm_rad = np.sqrt(max(band.fwhm_rad**2 - self.amp_fwhm_rad**2, 0.0))
        return hp.gauss_beam(fwhm_rad, lmax=band.lmax)

    def comp2band(self, band: Band, nthreads: int = 1):
        """Add B M a to the band's alms: the alms cut or padded to the band lmax, times the SED,
        smoothed by the band beam.

        A joined IQU component gives its T row to an intensity band, and E and B to a polarization
        band.
        """
        alms = self.amps
        if self.pol == "IQU":
            alms = alms[1:] if band.pol == "QU" else alms[:1]
        elif self.pol != band.pol:
            raise ValueError("Band and component polarization must match.")
        spin = 0 if band.pol == "I" else 2
        alms = project_alms(alms, band.lmax)
        sed = self.get_sed(band.nu)
        # The SED is one number when the spectral parameters are the same over the whole sky, and a
        # map (at the band's nside) when they vary between pixels. Then M acts in pixel space.
        if np.ndim(sed) > 0:  # Y^-1 M Y a
            comp_map = alm_to_map(alms, band.nside, band.lmax, spin=spin, nthreads=nthreads)
            comp_map *= sed
            alms = map_to_alm(comp_map, band.nside, band.lmax, spin=spin, nthreads=nthreads)
        else:
            # M a. `float` keeps the precision of the alms: a NumPy float64 would raise
            # single-precision alms to double precision, and everything after would run slower.
            alms = alms*float(sed)
        beam = self._band_beam(band)
        for ipol in range(len(alms)):
            # B M a. `alms` is a new array in both branches above, so it can be changed in place.
            band.alms[ipol] += almxfl(alms[ipol], beam, inplace=True)

    def band2comp(self, band: Band, nthreads: int = 1):
        """The transpose of `comp2band`: replace `amps` with M^T B^T applied to the band's alms."""
        if self.pol != band.pol:
            raise ValueError("Band and component polarization must match.")
        beam = self._band_beam(band)
        tmp_alm = band.alms.copy()
        for ipol in range(self.npol):
            almxfl(tmp_alm[ipol], beam, inplace=True)  # B^T a (B^T = B: diagonal in l).

        sed = self.get_sed(band.nu)
        if np.ndim(sed) > 0:  # The SED varies between pixels (see `comp2band`).
            # Y^-1^T B^T a
            pix_map = map_to_alm_adjoint(tmp_alm, band.nside, band.lmax, spin=self.spin,
                                         out=None, nthreads=nthreads)
            pix_map *= sed  # M^T Y^-1^T B^T a
            # Y^T M^T Y^-1^T B^T a
            tmp_alm = alm_to_map_adjoint(pix_map, band.nside, band.lmax, spin=self.spin, out=None,
                                         nthreads=nthreads)
        else:
            for ipol in range(self.npol):  # M^T B^T a
                inplace_scale(tmp_alm[ipol], sed)

        self.amps = project_alms(tmp_alm, self.lmax)


class CMB(DiffuseComponent):
    """The CMB, flat in thermodynamic units.

    The default `nu_ref` of 1 GHz keeps the amplitudes (in uK_RJ) close to uK_CMB.
    """

    DEFAULTS = {**DiffuseComponent.DEFAULTS, "nu_ref": [1.0, 1.0]}
    sed_param_names = ("nu_ref",)

    def get_sed(self, nu):
        """The thermodynamic-to-RJ conversion at `nu`, relative to that at `nu_ref`."""
        def cmb_to_rj(f):
            return (np.ones_like(f)*pysm3u.uK_CMB).to(
                pysm3u.uK_RJ, equivalencies=pysm3u.cmb_equivalencies(f*u.GHz)).value
        return cmb_to_rj(nu) / cmb_to_rj(self.nu_ref)

    def get_sky_anisotropies(self, nu, nside, fwhm_rad=0.0):
        """The CMB map at `nu` without monopole, dipole and quadrupole, for plotting."""
        return self.get_component_map(nside, fwhm_rad, lmin=3) * self.get_sed(nu)


class ThermalDust(DiffuseComponent):
    """Thermal dust, a modified blackbody with emissivity index `beta` and temperature `T`."""

    # `T` is the dust temperature in K, and `nu_ref` the reference frequency in GHz.
    DEFAULTS = {**DiffuseComponent.DEFAULTS, "beta": None, "T": None, "nu_ref": None}
    sed_param_names = ("beta", "T", "nu_ref")

    def get_sed(self, nu):
        """nu^beta B_nu(T), divided by the nu^2 of the RJ relation, and 1 at `nu_ref`."""
        x = (h_over_k*nu)/(self.T)
        x0 = (h_over_k*self.nu_ref)/(self.T)
        return (nu / self.nu_ref)**(self.beta + 1.0) * np.expm1(x0) / np.expm1(x)


class Synchrotron(DiffuseComponent):
    """Synchrotron emission, a power law in RJ brightness with spectral index `beta`."""

    DEFAULTS = {**DiffuseComponent.DEFAULTS, "beta": None, "nu_ref": None}
    sed_param_names = ("beta", "nu_ref")

    def get_sed(self, nu):
        """A power law with index `beta`, 1 at `nu_ref`."""
        return (nu/self.nu_ref)**self.beta


class FreeFree(DiffuseComponent):
    """Free-free (bremsstrahlung) emission from ionized gas at electron temperature `T`."""

    # `T` is the electron temperature in K, and `nu_ref` the reference frequency in GHz.
    DEFAULTS = {**DiffuseComponent.DEFAULTS, "T": None, "nu_ref": None}
    sed_param_names = ("T", "nu_ref")

    def _gaunt_factor(self, nu, T):
        """The Gaunt factor at `nu` (GHz) and electron temperature `T` (K), eq. 18 of BP1."""
        T4 = T / 1e4
        log_arg = nu * (T4**(-1.5))
        inner_exp = 5.960 - (np.sqrt(3) / np.pi) * np.log(log_arg)
        return np.log(np.exp(inner_exp) + np.e)

    def get_sed(self, nu):
        """nu^-2 times the Gaunt factor, 1 at `nu_ref`."""
        gaunt_nu = self._gaunt_factor(nu, self.T)
        gaunt_nu_ref = self._gaunt_factor(self.nu_ref, self.T)
        return (self.nu_ref / nu)**2 * (gaunt_nu / gaunt_nu_ref)


class SpinningDust(DiffuseComponent):
    """Spinning dust, shaped by a tabulated SpDust2 template.

    The template (the Cold Neutral Medium model, peaking at 30 GHz) is shifted in frequency to
    peak at `nu_peak`. The amplitudes are given at `nu_ref`.
    """

    DEFAULTS = {**DiffuseComponent.DEFAULTS, "template_path": None, "nu_peak": None,
                "nu_ref": None}
    sed_param_names = ("nu_peak_eval", "nu_peak_ref", "nu_ref")

    def __init__(self, params: Bunch, comp_name: str, pol: str):
        """One polarization view of the component `comp_name`, with its template read."""
        super().__init__(params, comp_name, pol)
        # Two-column SpDust2 template: frequency (GHz) and emissivity (proportional to intensity).
        freqs, SED = np.loadtxt(self.cfg.template_path).T
        self.nu_peak_ref = 30.0  # Peak frequency of the template as tabulated.
        self.nu_peak_eval = self.cfg.nu_peak
        # The SED divides by the template at the shifted `nu_ref`, which is zero outside the
        # tabulated range.
        nu_ref_shifted = self.nu_ref*self.nu_peak_ref/self.nu_peak_eval
        if not freqs.min() <= nu_ref_shifted <= freqs.max():
            raise ValueError(f"{comp_name}: nu_ref = {self.nu_ref} GHz, shifted to the template's "
                             f"peak, is {nu_ref_shifted:.3g} GHz, outside the template range "
                             f"{freqs.min():.3g}-{freqs.max():.3g} GHz ({self.cfg.template_path}).")

        # Interpolate in log-log space, where the template is smooth and spans many decades.
        log_nu = np.log(freqs)
        log_SED = np.log(SED)
        self._log_j_interp = interp1d(log_nu, log_SED, kind='cubic',
                                      bounds_error=False, fill_value=-np.inf)

    def _get_template_emissivity(self, nu):
        """Template emissivity at frequency `nu` (GHz), by log-log interpolation."""
        return np.exp(self._log_j_interp(np.log(nu)))

    def get_sed(self, nu: float|NDArray[np.floating]):
        """The template shifted to peak at `nu_peak`, in RJ temperature, 1 at `nu_ref`."""
        nu_shifted_eval = nu * self.nu_peak_ref / self.nu_peak_eval
        nu_shifted_ref = self.nu_ref * self.nu_peak_ref / self.nu_peak_eval
        SED = (self._get_template_emissivity(nu_shifted_eval)
               / self._get_template_emissivity(nu_shifted_ref))
        # From intensity to RJ temperature.
        return (self.nu_ref / nu)**2 * SED
