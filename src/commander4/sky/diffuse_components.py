"""Diffuse sky components: the `DiffuseComponent` base and every component built on it.

A diffuse component stores its amplitudes as spherical-harmonic coefficients and evaluates a
spectral energy distribution (SED) per band. `DiffuseComponent` holds everything common to that
representation; the classes after it differ only in their SED and its spectral parameters.
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

# Blackbody and thermodynamic-to-brightness conversions shared by the SEDs below.
A = (2*c.h*u.GHz**3/c.c**2).to('MJy').value
h_over_k = (c.h/c.k_B/(1*u.K)).to('GHz-1').value
h_over_kTCMB = (c.h/c.k_B/(2.7255*u.K)).to('GHz-1').value
def blackbody(nu, T):
    return A*nu**3/np.expm1(nu*h_over_k/T)
def g(nu):
    # From uK_CMB to MJy/sr
    x = nu*h_over_kTCMB
    return np.expm1(x)**2/(x**4*np.exp(x))



class DiffuseComponent(Component):
    """A sky component stored as spherical-harmonic coefficients with a per-band SED.

    Holds everything common to that representation: the alm buffer, the C(l) amplitude prior, unit
    conversion of init maps, and the SHT projections onto a band. Subclasses supply only `get_sed`
    and its spectral parameters.
    """

    DEFAULTS = {
        **Component.DEFAULTS,
        "lmax": None,                   # The highest multipole of the alms. Must be set.
        "spatially_varying_MM": False,  # Whether the mixing matrix differs between pixels.
        # C3's COMP_UNIT: the unit of what the parameter file gives for this component, which is
        # an init_from sky map, the prior mean map and `Cl_prior_amplitude`. None means that they
        # are in `amplitude_unit`. Compsep chains are always in `amplitude_unit`.
        "units": None,
        "amp_prior_mean_map": None,     # A sky map with the mean of the amplitude prior.
        # The C(l) prior (C3's 'power_law_gauss', see `P_Cl_prior`).
        "Cl_prior_amplitude": None,     # None is no prior (C3's `CL_TYPE = none`).
        "Cl_prior_beta": [0.0, 0.0],
        "Cl_prior_FWHM": [0.0, 0.0],
        "Cl_prior_l_pivot": 50,
        # C3's COMP_L_APOD: the multipole above which the prior is tapered towards zero. None is
        # the component's lmax, where the taper does nothing.
        "Cl_prior_l_apod": None,
        # C3's COMP_AMP_PRIOR_LMAX: the multipole below which the prior is tightened, so that the
        # large scales are held to the prior mean. 0 is no tightening.
        "Cl_prior_l_tighten": [0, 0]}

    # The unit the amplitude alms are internally represented in, always uK_RJ for diffuse
    # components (including the CMB). Init sky maps are converted to it from their own ``units``
    # (at the component's reference frequency); chain alms are already stored in it.
    amplitude_unit = "uK_RJ"

    def __init__(self, params: Bunch, comp_name: str, pol: str):
        """One polarization view of the component `comp_name`, with its alms set to zero."""
        super().__init__(params, comp_name, pol)
        self.lmax = self.cfg.lmax
        # The C(l) prior parameters are attributes, as the SED parameters, because they are
        # resolved to this view's value and written to the chain.
        self.Cl_prior_amplitude = self._per_pol(self.cfg.Cl_prior_amplitude)
        self.Cl_prior_beta = self._per_pol(self.cfg.Cl_prior_beta)
        self.Cl_prior_FWHM = self._per_pol(self.cfg.Cl_prior_FWHM)
        self.Cl_prior_l_pivot = self.cfg.Cl_prior_l_pivot
        self.Cl_prior_l_apod = self._per_pol(self.cfg.Cl_prior_l_apod)
        if self.Cl_prior_l_apod is None:
            self.Cl_prior_l_apod = self.lmax
        self.Cl_prior_l_tighten = self._per_pol(self.cfg.Cl_prior_l_tighten)
        # Cached prior mean mu, filled from `amp_prior_mean_map` by `load_amp_prior_mean`;
        # None means a zero-mean prior. See the amp_prior_mean property.
        self._amp_prior_mean_alms = None
        self.amps = np.zeros((self.npol, self.alm_len_complex),
                             dtype=np.complex128 if self.double_prec else np.complex64)

    def init_map_to_amplitude(self, sky_map: NDArray) -> NDArray:
        """Convert an init sky map (in the `units` setting) to this component's amplitude unit.

        The conversion is done at the component's reference frequency (``self.nu_ref``) using pysm3's
        CMB equivalencies. It is a no-op when the units are unspecified or already equal to the
        amplitude unit.
        """
        if self.cfg.units is None or self.cfg.units == self.amplitude_unit:
            return sky_map
        factor = (1*pysm3u.Unit(self.cfg.units)).to(
            pysm3u.Unit(self.amplitude_unit),
            equivalencies=pysm3u.cmb_equivalencies(self.nu_ref*pysm3u.GHz)).value
        return sky_map * factor

    Cl_prior_param_names = ("Cl_prior_amplitude", "Cl_prior_beta", "Cl_prior_FWHM",
                            "Cl_prior_l_pivot", "Cl_prior_l_apod", "Cl_prior_l_tighten")

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
        alms = np.ascontiguousarray(self.amps, dtype=np.complex128)
        return np.array([hp.alm2cl(alms[ipol], lmax=self.lmax) for ipol in range(alms.shape[0])])

    @property
    def P_Cl_prior(self) -> NDArray[np.floating]:
        """Prior angular power spectrum C_l for this component's amplitude alms.

        This is the S in the CG system (1 + S^{1/2} A^T N^-1 A S^{1/2}). It is a Gaussian prior
        constraining the alms to N(mu, C_l). Equivalent to C3's 'power_law_gauss' (comm_cl_mod.f90),
        which contains its 'power_law' (FWHM=0) and 'gauss' (beta=0) types as special cases.
        Defined in D_l space, where CMB-like spectra are roughly flat:

            D_l = amplitude * (l / l_pivot)^beta * max(exp(-l(l+1) sigma^2), 1e-10),
            C_l = 2 pi D_l / (l(l+1)) * f_apod(l)^2,

        where sigma is the Gaussian width of Cl_prior_FWHM (arcmin; 0 disables the rolloff). The
        1e-10 floor (relative to the power law) keeps C_l strictly positive so 1/C_l is safe for the
        preconditioners. As in C3, `Cl_prior_amplitude` is given in the component's `units` squared
        (C3's COMP_UNIT; uK_RJ if unset). The returned C_l is in (uK_RJ at nu_ref)^2, the unit of
        the alms.

        With `Cl_prior_amplitude: null` the component has no prior (C3's `CL_TYPE = none`). This
        returns ones, so that the S^{1/2} scaling does nothing, and the CG solver leaves out the
        prior terms: the "1 +" above, the prior mean and the prior draw.

        f_apod is C3's high-l apodization (`get_Cl_apod` in comm_cl_mod.f90, parameter COMP_L_APOD):
        unity up to l_apod, then exp(-ln(1000) (l-l_apod)^2 / (lmax-l_apod+1)^2), an amplitude taper
        reaching 1e-3 (1e-6 in power) at the component's own lmax. Setting l_apod at or below the
        highest band lmax keeps a component whose lmax exceeds what the data can see from filling
        those multipoles with a full-strength prior draw. It defaults to the component lmax, where
        the taper does nothing.

        f_apod also holds the low-l tightening of `Cl_prior_l_tighten` (C3's COMP_AMP_PRIOR_LMAX):
        below that multipole C_l shrinks towards zero, which holds the large scales to the prior
        mean (`amp_prior_mean_map`, or zero without one). See `Cl_prior_apodization`.
        """
        if self.Cl_prior_amplitude is None:
            return np.ones(self.lmax + 1)
        sigma = np.deg2rad(self.Cl_prior_FWHM / 60.0) / np.sqrt(8.0 * np.log(2.0))
        ells = np.arange(1, self.lmax + 1)
        Dl = np.empty(self.lmax + 1)
        # `init_map_to_amplitude(1.0)` is the factor from `units` to uK_RJ at nu_ref. It enters
        # squared, because D_l is a power.
        amplitude = self.Cl_prior_amplitude * self.init_map_to_amplitude(1.0)**2
        Dl[1:] = amplitude * (ells / self.Cl_prior_l_pivot)**self.Cl_prior_beta \
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

        Below `Cl_prior_l_tighten` the factor is also multiplied by sin^4(pi l / (2 l_tighten)).
        This is 1 at l_tighten and close to 0 at l = 1, so the prior gets tighter towards the
        large scales: the variance C_l is scaled by sin^8, which is 1/16 at half of l_tighten.
        """
        ells = np.arange(self.lmax + 1)
        l_apod = min(self.Cl_prior_l_apod, self.lmax)
        f = np.exp(-np.log(1e3) * (ells - l_apod)**2 / (self.lmax - l_apod + 1)**2)
        f[:l_apod + 1] = 1.0
        # l = 0 gets the factor of l = 1, as in C3, so that C_l stays above zero.
        tightened = ells < self.Cl_prior_l_tighten
        f[tightened] *= np.sin(0.5*np.pi*np.maximum(ells[tightened], 1)/self.Cl_prior_l_tighten)**4
        return f

    @property
    def P_Cl_prior_inv(self) -> NDArray[np.floating]:
        # P_Cl_prior is strictly positive by construction (1e-10 floor), so plain inversion is safe.
        return 1.0 / self.P_Cl_prior

    def __repr__(self):
        return f"Diffuse Component {self.shortname}, with polarization: {self.pol}"\
                f" (originally defined as {self.cfg.polarization})" \
                f"\n   lmax = {self.lmax} \n   alms: {self.amps}"

    def apply_Cl_prior_sqrt(self, alms: NDArray[np.complexfloating]) \
            -> NDArray[np.complexfloating]:
        """Multiplies `alms` by the C_l prior square root S^{1/2}, in place, and returns them.

        The target is always explicit, so that scaling the component's own amplitudes and scaling
        something else that lives in its alm space read differently at the call site. To apply this
        to the component's own alms (the usual case, the CG's a = S^{1/2}x reparameterization)
        call ``comp.apply_Cl_prior_sqrt(comp.amps)``.
        """
        prior_sqrt = np.sqrt(self.P_Cl_prior)
        for ipol in range(self.npol):
            almxfl(alms[ipol], prior_sqrt, inplace=True)
        return alms

    def apply_Cl_prior_inv_sqrt(self, alms: NDArray[np.complexfloating]) \
            -> NDArray[np.complexfloating]:
        """Multiplies `alms` by S^{-1/2}, in place, and returns them; the inverse of the above.

        Same contract as `apply_Cl_prior_sqrt`, including the explicit target: use
        ``comp.apply_Cl_prior_inv_sqrt(comp.amps)`` to act on the component's own amplitudes.
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

    def load_amp_prior_mean(self) -> None:
        """Read the prior mean mu from the `amp_prior_mean_map` parameter, if it is set.

        This is C3's ``COMP_AMP_PRIOR_MAP``: a FITS sky map giving the mean of the Gaussian
        amplitude prior a ~ N(mu, S). It goes through the same conversion as an ``init_from`` map
        (unit conversion at the component's reference frequency, truncation to l = 3*nside-1, and
        the iterative LSMR inverse SHT), because mu and the amplitudes live in the same space and
        must be treated identically. C3 instead uses a single non-iterative `YtW` analysis here,
        which is the less accurate of the two on a HEALPix grid.

        Read once and cached (C3 likewise transforms its mu once, at component initialization).
        The S^{-1/2} that turns mu into a right-hand-side term is *not* applied here but per solve,
        since the prior S may change between Gibbs iterations while mu does not.
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
        """Write the shared component sample (see `Component.write_chain`), plus the alms."""
        super().write_chain(group, band_frequencies)
        group["alms"] = self.amps
        # The realized power spectrum of those alms (Commander3's `sigma_l`) and their lmax.
        group["sigma_l"] = self.sigma_l
        group["lmax"] = self.lmax
        # The C(l) prior's model parameters (Commander3's Dl_amp/Dl_beta/Dl_theta), not the
        # evaluated spectrum: the evaluated one is reconstructible, and `sigma_l` above already
        # records the realized power. A `None` amplitude means the prior is off (C3 CL_TYPE none),
        # which writes no amplitude, exactly as C3 does.
        for param_name in self.Cl_prior_param_names:
            value = getattr(self, param_name)
            if value is not None:
                group[f"Cl_prior/{param_name}"] = value

    def read_chain(self, group: h5py.Group, restore_beta: bool) -> None:
        """Restore the shared sampled state (see `Component.read_chain`), plus this view's alms.

        The stored alms are those of the joined component, so only the rows of this view's
        polarization are taken, and they are cut or zero-padded to this component's lmax. If the
        file lacks this view's polarization, the alms keep their current value.
        """
        super().read_chain(group, restore_beta)
        stored_alms = group["alms"][()]
        rows = _pol_row_indices(stored_alms, self.pol, self.shortname, group.file.filename)
        if rows is None:
            logger.debug(f"{group.file.filename!r} has no {self.pol!r} alms for component "
                         f"{self.comp_name!r}; leaving those alms unchanged.")
            return
        view_alms = project_alms(np.ascontiguousarray(stored_alms[rows]), self.lmax)
        self.amps = view_alms.astype(self.amps.dtype, copy=False)

    def get_component_map(self, nside: int, fwhm: float = 0.0, lmin: int = 0) -> NDArray:
        """The amplitude map at `nside`, smoothed by a Gaussian beam of `fwhm` radians.

        For plotting, where there is no band: the alms are cut at min(lmax, 3*nside - 1), and the
        beam is one window on every Stokes row, as in `comp2band`. Multipoles below `lmin` are left
        out. A band's sky is made by `CompList.get_sky_at_nu` instead, through `comp2band`.
        """
        if self.amps is None:
            raise ValueError("component_alms property not set.")
        lmax = min(self.lmax, 3*nside - 1)
        window = hp.gauss_beam(fwhm, lmax=lmax)
        window[:lmin] = 0.0
        alms = np.array([almxfl(alm, window) for alm in project_alms(self.amps, lmax)])
        if self.pol != "IQU":
            return alm_to_map(alms, nside, lmax, spin=self.spin)
        # A joined IQU component needs separate spin-0 and spin-2 synthesis calls.
        return np.concatenate((alm_to_map(alms[:1], nside, lmax, spin=0),
                               alm_to_map(alms[1:], nside, lmax, spin=2)), axis=0)

    def get_sky(self, nu: float, nside: int, fwhm: float = 0.0) -> NDArray:
        """The component map at frequency `nu`, seen through a beam of `fwhm` radians, for plotting.

        Amplitudes that already carry a beam (`amp_fwhm_rad`) get only the rest of it. A band's sky
        is made by `CompList.get_sky_at_nu` instead, which also cuts at the band lmax.
        """
        applied_fwhm = np.sqrt(max(fwhm**2 - self.amp_fwhm_rad**2, 0.0))
        return self.get_component_map(nside, applied_fwhm)*self.get_sed(nu)

    def get_sed(self, nu):
        raise NotImplementedError(f"{type(self).__name__}.get_sed() is not implemented.")

    def dot(self, other: "DiffuseComponent") -> float:
        """The dot product of this component's alms with those of `other`, the same view.

        The alms are complex and stored for m >= 0 only, so each m > 0 coefficient stands for two
        real numbers and is counted twice.
        """
        res = 0.0
        for ipol in range(self.npol):
            res += _dot_complex_alm_1D_arrays(self.amps[ipol], other.amps[ipol], self.lmax)
        return res

    def _band_beam(self, band: Band) -> NDArray:
        """The band's Gaussian beam window, up to the band lmax.

        Amplitudes that already carry a beam (`amp_fwhm_rad`, from the per-pixel solver) get only
        the rest of it: Gaussian beams multiply, so their widths add in quadrature.
        """
        fwhm = np.sqrt(max(np.deg2rad(band.fwhm/60.0)**2 - self.amp_fwhm_rad**2, 0.0))
        return hp.gauss_beam(fwhm, lmax=band.lmax)

    def comp2band(self, band: Band, nthreads: int = 1):
        """Add this component to the band's alms: B Y^-1 M Y a (C3's `evalBand`).

        The alms are cut at the band lmax (or zero-padded up to it), scaled to the band frequency
        by the mixing matrix M, and smoothed by the band beam B. A joined IQU component (as in the
        sky model handed to TOD processing) gives its T row to an intensity band and its E and B
        rows to a polarization band.
        """
        alms = self.amps
        if self.pol == "IQU":
            alms = alms[1:] if band.pol == "QU" else alms[:1]
        elif self.pol != band.pol:
            raise ValueError("Band and component polarization must match.")
        spin = 0 if band.pol == "I" else 2
        alms = project_alms(alms, band.lmax)
        if self.cfg.spatially_varying_MM:  # The mixing matrix is pixel-dependent: Y^-1 M Y a.
            comp_map = alm_to_map(alms, band.nside, band.lmax, spin=spin, nthreads=nthreads)
            for ipol in range(len(comp_map)):
                inplace_scale(comp_map[ipol], self.get_sed(band.nu))
            alms = map_to_alm(comp_map, band.nside, band.lmax, spin=spin, nthreads=nthreads)
        else:
            # M a. The SED is cast to the precision of the alms: a NumPy double would raise
            # single-precision alms to double precision, and everything after would run slower.
            alms = alms*np.asarray(self.get_sed(band.nu), dtype=alms.real.dtype)
        beam = self._band_beam(band)
        for ipol in range(len(alms)):
            # B M a. `alms` is a new array in both branches above, so it can be changed in place.
            band.alms[ipol] += almxfl(alms[ipol], beam, inplace=True)
        return band.alms

    def band2comp(self, band: Band, nthreads: int = 1, inplace=True):
        """The transpose of `comp2band`: Y^T M^T Y^-1^T B^T applied to the band's alms
        (C3's `projectBand`). Stores the result in this component by default and returns it.

        Each rank evaluates its own band; the master sums the contributions of all bands.
        """
        if self.pol != band.pol:
            raise ValueError("Band and component polarization must match.")
        beam = self._band_beam(band)
        tmp_alm = band.alms.copy()
        for ipol in range(self.npol):
            almxfl(tmp_alm[ipol], beam, inplace=True)  # B^T a (B^T = B: diagonal in l).

        if self.cfg.spatially_varying_MM:  # If this component's mixing matrix is pixel-dependent.
            # Y^-1^T B^T a
            pix_map = map_to_alm_adjoint(tmp_alm, band.nside, band.lmax, spin=self.spin,
                                         out=None, nthreads=nthreads)

            # M^T Y^-1 B^T a
            for ipol in range(self.npol):
                inplace_scale(pix_map[ipol], self.get_sed(band.nu))

            # Y^T M^T Y^-1^T B^T a
            tmp_alm = alm_to_map_adjoint(pix_map, band.nside, band.lmax, spin=self.spin, out=None,
                                         nthreads=nthreads)
        else:
            for ipol in range(self.npol):  # M^T B^T a
                inplace_scale(tmp_alm[ipol], self.get_sed(band.nu))

        # Project alm from band to component lmax.
        contrib_to_comp_alm = project_alms(tmp_alm, self.lmax)
        
        if inplace:
            self.amps = contrib_to_comp_alm

        return contrib_to_comp_alm
    


class CMB(DiffuseComponent):
    """The CMB, whose SED is flat in thermodynamic units."""

    # Like all diffuse components, the CMB amplitude is stored internally in uK_RJ, referenced to
    # `nu_ref`. `get_sed` is therefore the *ratio* of the thermodynamic-to-RJ conversion at `nu`
    # relative to `nu_ref`. The choice of `nu_ref` is arbitrary (the sky is invariant to it); the
    # default of 1 GHz keeps the stored amplitudes ~= uK_CMB.
    DEFAULTS = {**DiffuseComponent.DEFAULTS, "shortname": "cmb", "nu_ref": [1.0, 1.0]}
    sed_param_names = ("nu_ref",)

    def get_sed(self, nu):
        """SED for CMB emission: the thermodynamic-to-RJ conversion at `nu` relative to `nu_ref`.

        The CMB amplitude is stored in uK_RJ referenced to `nu_ref`, so multiplying by this ratio
        yields the uK_RJ brightness at `nu`. The result is dimensionless.

        Args:
            nu (float or np.ndarray): Frequency in GHz at which to evaluate the SED.
        Returns:
            The SED scaling factor (float or np.ndarray).
        """
        def cmb_to_rj(f):
            return (np.ones_like(f)*pysm3u.uK_CMB).to(
                pysm3u.uK_RJ, equivalencies=pysm3u.cmb_equivalencies(f*u.GHz)).value
        return cmb_to_rj(nu) / cmb_to_rj(self.nu_ref)
    
    def get_sky_anisotropies(self, nu, nside, fwhm=0):
        """The CMB map at `nu` without monopole, dipole and quadrupole, for plotting."""
        return self.get_component_map(nside, fwhm, lmin=3) * self.get_sed(nu)


class ThermalDust(DiffuseComponent):
    """Thermal dust, a modified blackbody with emissivity index `beta` and temperature `T`."""

    # `T` is the dust temperature in K, and `nu_ref` the reference frequency in GHz.
    DEFAULTS = {**DiffuseComponent.DEFAULTS, "shortname": "term-dust", "beta": None, "T": None,
                "nu_ref": None}
    sed_param_names = ("beta", "T", "nu_ref")

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
        x0 = (h_over_k*self.nu_ref)/(self.T)
        return (nu / self.nu_ref)**(self.beta + 1.0) * np.expm1(x0) / np.expm1(x)


class Synchrotron(DiffuseComponent):
    """Synchrotron emission, a power law in RJ brightness with spectral index `beta`."""

    DEFAULTS = {**DiffuseComponent.DEFAULTS, "shortname": "sync", "beta": None, "nu_ref": None}
    sed_param_names = ("beta", "nu_ref")

    def get_sed(self, nu):
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

    # `T` is the electron temperature in K, and `nu_ref` the reference frequency in GHz.
    DEFAULTS = {**DiffuseComponent.DEFAULTS, "shortname": "ff", "T": None, "nu_ref": None}
    sed_param_names = ("T", "nu_ref")

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
        gaunt_nu_ref = self._gaunt_factor(self.nu_ref, self.T)

        # The scaling is proportional to nu^-2 * g_ff(nu), normalized to 1 at nu_ref.
        sed = (self.nu_ref / nu)**2 * (gaunt_nu / gaunt_nu_ref)
        return sed


class SpinningDust(DiffuseComponent):
    """Spinning dust, whose spectral shape comes from a tabulated SpDust2 template.

    The template is the Cold Neutral Medium model, which peaks at 30 GHz. It is shifted in
    frequency so its peak lands at the `nu_peak` setting, which is the one shape parameter; the
    `nu_0` setting (`self.nu_ref`) only sets where the amplitude map is normalized.
    """

    DEFAULTS = {**DiffuseComponent.DEFAULTS, "shortname": "spin-dust", "template_path": None,
                "nu_peak": None, "nu_0": None}
    sed_param_names = ("nu_peak_eval", "nu_peak_ref", "nu_ref")

    def __init__(self, params: Bunch, comp_name: str, pol: str):
        """One polarization view of the component `comp_name`, with its template read."""
        super().__init__(params, comp_name, pol)
        # Two-column SpDust2 template: frequency (GHz) and emissivity (proportional to intensity).
        freqs, SED = np.loadtxt(self.cfg.template_path).T
        self.nu_peak_ref = 30.0  # Peak frequency of the template as tabulated.
        self.nu_peak_eval = self.cfg.nu_peak
        self.nu_ref = self.cfg.nu_0  # Reference frequency for the amplitude map in GHz

        # Interpolate in log-log space, where the template is smooth and spans many decades.
        log_nu = np.log(freqs)
        log_SED = np.log(SED)
        self._log_j_interp = interp1d(log_nu, log_SED, kind='cubic',
                                      bounds_error=False, fill_value=-np.inf)

    def _get_template_emissivity(self, nu):
        """Template emissivity at frequency `nu` (GHz), by log-log interpolation."""
        return np.exp(self._log_j_interp(np.log(nu)))

    def get_sed(self, nu: float|NDArray[np.floating]):
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
    
