"""`SkyModel`: a component list plus the operations that realize it as a sky map at a frequency."""
import healpy as hp
import numpy as np
from numpy.typing import NDArray
from pixell.bunch import Bunch
from commander4.data_models.band import Band
from commander4.data_models.detector_map import DetectorMap
from commander4.math_utils.sht import alm_to_map
from commander4.sky.comp_list import CompList
from commander4.sky.diffuse_components import DiffuseComponent


def build_initial_sky_model(params: Bunch) -> "SkyModel":
    """Construct the initial sky model directly from the parameter file.

    Builds the full component list, loads each component's initial state from its ``init_from``
    (else zero amplitudes), and wraps it in a SkyModel. This is rank-agnostic and performs no MPI.
    It is the fixed sky of a TOD-only run, which the TOD band masters construct locally.
    """
    comp_list = CompList.init_from_params(params.components, params)
    for comp in comp_list:
        # Nothing samples beta in a TOD-only run, so beta is that of the parameter file.
        comp.load_initial_state(None, apply_init_from=True, restore_beta=False)
    return SkyModel(comp_list)


class SkyModel:
    """A component list plus the operations that realize it as a sky map."""

    def __init__(self, components:CompList):
        self._components = components

    def get_sky(self, band):
        """ Get sky from a bandpass.
        """
        raise NotImplementedError

    @property
    def amp_fwhm_rad(self) -> float:
        """The beam this model's amplitudes carry, in radians: the coarsest of its components.

        This is normally 0.0, as the CG solver produces a deconvolved sky.
        The per-pixel solver instead returns amplitudes at `compsep.common_res_fwhm`.
        """
        return max((comp.amp_fwhm_rad for comp in self._components), default=0.0)

    def get_sky_at_band(self, band_map: DetectorMap) -> NDArray:
        """The realized sky as one band map sees it: at its frequency, resolution, beam and lmax.

        The result has the shape of `band_map.map_sky`, so `band_map.map_sky` minus it is the
        residual of the band.
        """
        return self.get_sky_at_nu(band_map.nu, band_map.nside, "QU" if band_map.pol else "I",
                                  fwhm=band_map.fwhm_rad, lmax=band_map.lmax)

    def get_sky_at_nu(self, nu, nside, pols_required, fwhm=None, lmax=None):
        """Get the realized sky of a band: its frequency, resolution, beam and lmax.

        Diffuse components go through `comp2band`, the CG solver's own operator (cut at the band
        lmax, mixing matrix, beam), and are then synthesized at `nside`, as C3's `evalBand` does:
        the sky compared with a band's data is exactly the model the amplitudes were solved for.
        Point sources are painted in pixel space and added, as in C3's `compute_residual`.

        The component list may be either the split execution list used during CompSep (`I` and
        `QU` views) or a joined logical list containing `IQU` components.

        Args:
            fwhm: Beam to realize the sky through, in radians. `None` asks for the sharpest this
                model can give, which is `amp_fwhm_rad` (normally 0.0).
            lmax: The band lmax (`DetectorMap.lmax`). `None` means `3*nside - 1`, the default band
                lmax.

        Raises:
            ValueError: If `fwhm` is finer than `amp_fwhm_rad`.
        """
        amp_fwhm = self.amp_fwhm_rad
        if fwhm is None:
            fwhm = amp_fwhm
        elif fwhm < amp_fwhm and not np.isclose(fwhm, amp_fwhm):
            raise ValueError(
                f"Asked for the sky at a beam of {np.degrees(fwhm)*60:.4g} arcmin, but these "
                f"amplitudes already carry {np.degrees(amp_fwhm)*60:.4g} arcmin of smoothing and "
                f"cannot be realized any sharper. Pass fwhm=None for the sharpest available, or "
                f"max(wanted, model.amp_fwhm_rad) if a coarser sky is acceptable.")
        if pols_required not in ("I", "QU", "IQU"):
            raise ValueError("Unrecognized polarization string")
        lmax = 3*nside - 1 if lmax is None else lmax
        nrows = len(pols_required)
        skymap = np.zeros((nrows, 12*nside**2), dtype=np.float32)
        diffuse = [comp for comp in self._components if isinstance(comp, DiffuseComponent)]
        # One band per view: intensity (map row 0, spin 0) and polarization (the last two rows,
        # spin 2). A joined IQU component contributes to both.
        for view, rows, spin in (("I", slice(0, 1), 0), ("QU", slice(nrows - 2, nrows), 2)):
            if view[0] not in pols_required:
                continue
            # Single precision, like `skymap`, which the result goes into.
            band = Band(np.zeros((len(view), hp.Alm.getsize(lmax)), dtype=np.complex64), nu,
                        np.degrees(fwhm)*60.0, nside)
            for comp in diffuse:
                if comp.eval_pol in (view, "IQU"):
                    comp.comp2band(band)
            skymap[rows] += alm_to_map(band.alms, nside, lmax, spin=spin)
        if "I" in pols_required:
            for comp in self._components:  # Point sources.
                if not isinstance(comp, DiffuseComponent) and comp.eval_pol == "I":
                    skymap[0] += comp.get_sky(nu, nside, fwhm)[0]
        return skymap
