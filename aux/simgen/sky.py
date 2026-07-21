"""Sky components (swappable) and the per-band sky-model builder.

Each ``SkyComponent`` produces a beam-smoothed, band-resolution sky map (uK_RJ, the Stokes channels
of the band's polarization) for a given band:

    band_map(band) -> ndarray (npol, npix_eval)

Frequency scaling reuses Commander4's component SED classes
(``commander4.sky_models.component``), so the simulated sky's spectral behaviour is *by
construction* the model the main code assumes. Spatial templates come from PySM3 presets (or a FITS
map) for foregrounds and a CAMB realization for the CMB -- the same pattern as
``commander4.simulations.inplace_litebird_sim``, generalized and de-hard-coded.

Add a new component by mapping its ``component_class`` to a ``SkyComponent`` subclass in
``_COMPONENT_BUILDERS``.
"""
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
import healpy as hp
from numpy.typing import NDArray
from pixell.bunch import Bunch

import commander4.sky_models.component as c4comp
from simgen.config import bget

logger = logging.getLogger(__name__)

T_CMB = 2.72548          # [K_CMB]
C_LIGHT = 299792458.0    # [m/s]

# Default PySM3 preset per foreground class (matches inplace_litebird_sim).
_DEFAULT_PRESET = {"ThermalDust": "d0", "Synchrotron": "s5", "FreeFree": "f1", "SpinningDust": "a1"}

# Stokes-row selection for a band's polarization, from an always-IQU (3, npix) template/map.
_POL_ROWS = {"I": [0], "QU": [1, 2], "IQU": [0, 1, 2]}


def _scalar_nu_ref(comp_params: Bunch) -> float:
    """Single reference frequency (the I value if ``nu_ref`` is an ``[I, QU]`` pair)."""
    nu_ref = comp_params.nu_ref
    return float(nu_ref[0] if isinstance(nu_ref, (list, tuple)) else nu_ref)


def _build_c4_component(comp_cfg: Bunch, global_params: Bunch):
    """Instantiate the matching Commander4 component (used only for its ``get_sed``).

    Missing structural keys that ``get_sed`` does not need (smoothing prior, spatially-varying MM)
    are filled with harmless defaults so a minimal sim ``components`` block also works; the SED
    parameters (beta/T/nu_ref) must be supplied by the user. ``lmax: "full"`` is resolved as in C4.
    """
    cls = getattr(c4comp, comp_cfg.component_class)
    cp = deepcopy(comp_cfg.params)
    defaults = {"polarization": "IQU", "spatially_varying_MM": False,
                "Cl_prior_amplitude": None,  # identity C_l prior; get_sed never uses it
                "lmax": (global_params.nside * 5) // 2}
    for key, val in defaults.items():
        if key not in cp:
            cp[key] = val
    if cp.lmax == "full":
        cp.lmax = (global_params.nside * 5) // 2
    return cls(cp, global_params, eval_pol="I", comp_name=comp_cfg._name)


def _select_pol(iqu_map: NDArray, polarization: str) -> NDArray:
    """Select the Stokes rows a band needs from an IQU (3, npix) map."""
    return np.ascontiguousarray(iqu_map[_POL_ROWS[polarization]])


class SkyComponent(ABC):
    def __init__(self, comp_cfg: Bunch, global_params: Bunch):
        self.comp_cfg = comp_cfg
        self.global_params = global_params
        self.name = comp_cfg._name

    @abstractmethod
    def band_map(self, band) -> NDArray[np.floating]:
        """Beam-smoothed (npol, npix_eval) map for ``band``, in the band's unit (uK_RJ)."""


class DiffuseComponent(SkyComponent):
    """Foreground with a PySM3-preset (or FITS) spatial template scaled by a C4 SED.

    The IQU template is realized once at the component's reference frequency, then for each band it
    is beam-smoothed, downgraded to the band nside and multiplied by the C4 ``get_sed(band.freq)``.
    """
    def __init__(self, comp_cfg: Bunch, global_params: Bunch):
        super().__init__(comp_cfg, global_params)
        self.c4 = _build_c4_component(comp_cfg, global_params)
        self.nu_ref = _scalar_nu_ref(comp_cfg.params)
        self.template_cfg = bget(comp_cfg.params, "template", Bunch())
        self.units = bget(global_params, "units", "uK_RJ")
        self._template_iqu: NDArray | None = None   # (3, npix_base), uK_RJ, unsmoothed

    def _template(self) -> NDArray:
        if self._template_iqu is not None:
            return self._template_iqu
        import pysm3
        import pysm3.units as u
        source = bget(self.template_cfg, "source", "pysm3")
        base_nside = min(1024, self.global_params.nside)
        if source == "pysm3":
            preset = bget(self.template_cfg, "preset", _DEFAULT_PRESET[self.comp_cfg.component_class])
            sky = pysm3.Sky(nside=base_nside, preset_strings=[preset], output_unit=u.Unit(self.units))
            iqu = np.asarray(sky.get_emission(self.nu_ref * u.GHz).value, dtype=np.float32)
        elif source == "fits":
            m = np.atleast_2d(hp.read_map(self.template_cfg.path, field=None)).astype(np.float32)
            iqu = np.zeros((3, m.shape[-1]), dtype=np.float32)
            iqu[:m.shape[0]] = m   # pad I-only templates with zero Q/U
        else:
            raise ValueError(f"Unknown template source {source!r} for component {self.name!r}.")
        self._template_iqu = iqu
        return iqu

    def band_map(self, band) -> NDArray[np.floating]:
        template = self._template()
        smoothed = hp.smoothing(template, fwhm=band.fwhm_rad)
        m = hp.ud_grade(smoothed, band.eval_nside).astype(np.float32) * self.c4.get_sed(band.freq)
        return _select_pol(m, band.polarization)


class CMBComponent(SkyComponent):
    """CMB realization from CAMB (optionally plus the solar dipole), scaled to each band.

    The realization is drawn once in uK_CMB at the configured ``seed``; per band it is beam-smoothed
    and converted from uK_CMB to the band unit at the band frequency (the CMB SED).
    """
    _DEFAULT_COSMO = dict(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0.0,
                          tau=0.06, As=2.0e-9, ns=0.965)

    def __init__(self, comp_cfg: Bunch, global_params: Bunch):
        super().__init__(comp_cfg, global_params)
        self.lmax = bget(comp_cfg.params, "lmax", 3 * global_params.nside - 1)
        if self.lmax == "full":
            self.lmax = (global_params.nside * 5) // 2
        self.seed = int(bget(global_params, "seed", 0))
        self.solar_dipole = bool(bget(comp_cfg.params, "solar_dipole", False))
        self.cosmo = {**self._DEFAULT_COSMO, **bget(comp_cfg.params, "cosmology", {})}
        self._alms: NDArray | None = None   # (3, nalm), uK_CMB

    def _build_alms(self) -> NDArray:
        if self._alms is not None:
            return self._alms
        import camb
        pars = camb.set_params(halofit_version='mead', lmax=self.lmax, **self.cosmo)
        powers = camb.get_results(pars).get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
        totCL = powers['total']
        ell = np.arange(self.lmax + 1)
        Cls = np.array([totCL[ell, 0], totCL[ell, 1], totCL[ell, 2], totCL[ell, 3]])
        np.random.seed(self.seed)
        alms = hp.synalm(Cls, lmax=self.lmax, new=True)
        if self.solar_dipole:
            alms[0] += self._solar_dipole_alm()
        self._alms = alms
        return alms

    def _solar_dipole_alm(self) -> NDArray:
        """l=1 alms for the BEYONDPLANCK solar dipole (intensity only)."""
        amp_uK, glon, glat = 3362.7, 264.11, 48.279
        theta, phi = np.deg2rad(90.0 - glat), np.deg2rad(glon)
        norm = amp_uK * np.sqrt(4 * np.pi / 3)
        dip = np.zeros(hp.Alm.getsize(self.lmax), dtype=np.complex128)
        dip[hp.Alm.getidx(self.lmax, 1, 0)] = norm * np.cos(theta)
        dip[hp.Alm.getidx(self.lmax, 1, 1)] = -norm * np.sin(theta) * np.exp(-1j * phi) / np.sqrt(2)
        return dip

    def band_map(self, band) -> NDArray[np.floating]:
        import pysm3.units as u
        alms = self._build_alms()
        smoothed = hp.smoothalm(alms, fwhm=band.fwhm_rad, inplace=False)
        cmb = hp.alm2map(smoothed, band.eval_nside, pixwin=False).astype(np.float32)  # uK_CMB
        cmb = (cmb * u.uK_CMB).to(u.Unit(band.units),
                                  equivalencies=u.cmb_equivalencies(band.freq * u.GHz)).value
        return _select_pol(np.asarray(cmb, dtype=np.float32), band.polarization)


class GriddedPointSources(SkyComponent):
    """Synthetic sky: equal-amplitude point sources on a regular (lon, lat) grid.

    Sources are placed at the nodes of a regular grid (``nlon`` x ``nlat``) spanning ``lon_range_deg``
    x ``lat_range_deg`` (Galactic), each with the same intensity ``amplitude`` (in the band unit,
    before beam smoothing). The map is intensity-only (Q/U stay zero). By default the amplitude is
    frequency-independent ("same amplitude" in every band); a power-law SED ``(nu/nu_ref)^beta`` can
    be enabled with a non-zero ``beta``.

    Params:
        amplitude: per-source intensity in the band unit (pre-smoothing pixel value).
        nlon, nlat: number of sources along longitude / latitude.
        lon_range_deg, lat_range_deg: grid extent (defaults: full longitude, [-80, 80] lat).
        beta, nu_ref: optional SED power-law (default beta=0 -> flat).
    """
    def __init__(self, comp_cfg: Bunch, global_params: Bunch):
        super().__init__(comp_cfg, global_params)
        p = comp_cfg.params
        self.amplitude = float(bget(p, "amplitude", 1.0e3))
        self.nlon = int(bget(p, "nlon", 12))
        self.nlat = int(bget(p, "nlat", 6))
        self.lon_range = bget(p, "lon_range_deg", [0.0, 360.0])
        self.lat_range = bget(p, "lat_range_deg", [-80.0, 80.0])
        self.beta = float(bget(p, "beta", 0.0))
        self.nu_ref = float(bget(p, "nu_ref", 100.0))
        self.polarization = bget(p, "polarization", "I")

    def _source_pixels(self, nside: int) -> NDArray[np.integer]:
        # Avoid placing a duplicate source at both 0 and 360 deg when the grid spans the full circle.
        full_circle = abs((self.lon_range[1] - self.lon_range[0]) - 360.0) < 1e-9
        lons = np.linspace(self.lon_range[0], self.lon_range[1], self.nlon, endpoint=not full_circle)
        lats = np.linspace(self.lat_range[0], self.lat_range[1], self.nlat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        theta = np.deg2rad(90.0 - lat_grid.ravel())
        phi = np.deg2rad(lon_grid.ravel() % 360.0)
        return np.unique(hp.ang2pix(nside, theta, phi))

    def band_map(self, band) -> NDArray[np.floating]:
        sed = (band.freq / self.nu_ref) ** self.beta if self.beta != 0.0 else 1.0
        m = np.zeros((3, 12 * band.eval_nside**2), dtype=np.float32)
        m[0, self._source_pixels(band.eval_nside)] = self.amplitude * sed
        if band.fwhm_rad > 0:
            m[0] = np.asarray(hp.smoothing(m[0], fwhm=band.fwhm_rad), dtype=np.float32)
        return _select_pol(m, band.polarization)


_COMPONENT_BUILDERS: dict[str, type[SkyComponent]] = {
    "CMB": CMBComponent,
    "ThermalDust": DiffuseComponent,
    "Synchrotron": DiffuseComponent,
    "FreeFree": DiffuseComponent,
    "SpinningDust": DiffuseComponent,
    "GriddedPointSources": GriddedPointSources,
}


def build_components(params: Bunch) -> list[SkyComponent]:
    """Instantiate enabled sky components from the (C4-style) top-level ``components`` block."""
    components = []
    for _, comp_cfg in params.components.items():
        if not bget(comp_cfg, "enabled", False):
            continue
        cls = _COMPONENT_BUILDERS.get(comp_cfg.component_class)
        if cls is None:
            raise NotImplementedError(
                f"simgen has no sky model for component_class {comp_cfg.component_class!r} "
                f"(supported: {sorted(_COMPONENT_BUILDERS)}).")
        components.append(cls(comp_cfg, params.general))
    return components


def build_band_sky_maps(params: Bunch, bands: list) -> dict[str, NDArray[np.floating]]:
    """Build the summed (npol, npix_eval) sky map for every band. Call on one rank, then broadcast."""
    components = build_components(params)
    band_maps: dict[str, NDArray] = {}
    for band in bands:
        acc = np.zeros((band.npol, 12 * band.eval_nside**2), dtype=np.float32)
        for comp in components:
            acc += comp.band_map(band)
        band_maps[band.name] = acc
        logger.info("Built sky map for band %s (%d components, nside %d).",
                    band.name, len(components), band.eval_nside)
    return band_maps


def compute_orbital_dipole(vsun: NDArray, pix: NDArray, nside: int, freq: float,
                           units: str) -> NDArray[np.floating]:
    """Relativistic orbital dipole along ``pix`` for velocity ``vsun`` (m/s, Galactic), in ``units``.

    Ported from ``inplace_litebird_sim.get_orbital_dipole``: amplitude is computed in K_CMB then
    converted to the band unit at the band frequency.
    """
    import ducc0
    import pysm3.units as u
    pointing_vec = ducc0.healpix.Healpix_Base(nside, "RING").pix2vec(pix)
    beta_vec = vsun / C_LIGHT
    gamma = 1.0 / np.sqrt(1.0 - np.dot(vsun, vsun) / C_LIGHT**2)
    dot = pointing_vec @ beta_vec
    amp_KCMB = T_CMB * (1.0 / (gamma * (1.0 - dot)) - 1.0)
    KCMB_to_units = (1.0 * u.K_CMB).to(u.Unit(units),
                                       equivalencies=u.cmb_equivalencies(freq * u.GHz)).value
    return (amp_KCMB * KCMB_to_units).astype(np.float32)
