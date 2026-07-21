import logging
import numpy as np
import healpy as hp
from copy import deepcopy
from numpy.typing import NDArray
from commander4.utils.math_operations import alm_to_map, alm_to_map_adjoint, inplace_arr_prod, almxfl

logger = logging.getLogger(__name__)


def smooth_signal_map_noiseweighted(map_signal: NDArray, map_rms: NDArray,
                                    fwhm_rad: float) -> NDArray:
    """Inverse-variance-weighted Gaussian smoothing of a signal map.

    Each pixel is weighted by 1/variance before smoothing (so noisy pixels contribute less) and the
    result is renormalized by the smoothed weights -- used to bring a band to a coarser common beam.
    """
    map_inv_var = 1.0/map_rms**2
    smoothed_weight = hp.smoothing(map_inv_var, fwhm=fwhm_rad)
    return hp.smoothing(map_signal*map_inv_var, fwhm=fwhm_rad)/smoothed_weight


def smooth_rms_map_noiseweighted(rms_map: NDArray, fwhm_rad: float) -> NDArray:
    """Per-pixel noise RMS of a map smoothed by :func:`smooth_signal_map_noiseweighted`.

    Propagates the inverse-variance-weighted Gaussian smoothing analytically, accounting for the
    beam and pixel windows, so the returned map is the correct noise RMS of the smoothed signal.
    """
    npix = rms_map.shape[0]
    nside = hp.npix2nside(npix)
    smoothed_inv_var = hp.smoothing(1.0/rms_map**2, fwhm=fwhm_rad)
    lmax = 3*nside - 1
    ell = np.arange(lmax + 1)
    b_ell = hp.gauss_beam(fwhm_rad, lmax=lmax)
    p_ell = hp.pixwin(nside, lmax=lmax)
    omega_pix = hp.nside2resol(nside)**2  # White-noise C_ell for a single pixel.
    true_empirical_norm = np.sum((2*ell + 1)/(4*np.pi)*omega_pix*(p_ell**2)*(b_ell**2))
    # Smooth the weights with the squared beam, divide by the squared smoothed weights.
    numerator = hp.smoothing(1.0/rms_map**2, fwhm=fwhm_rad/np.sqrt(2.0))*true_empirical_norm
    return np.sqrt(numerator/smoothed_inv_var**2)


class DetectorMap:
    """Holds a sky map and associated metadata for a single detector or band.

    Stores the sky signal map, inverse noise variance map, beam properties, and
    resolution parameters. Derived quantities such as ``map_rms``, ``pol``, and
    ``spin`` are computed on the fly via properties.

    Attributes:
        map_sky (NDArray): Sky signal map of shape ``(npol, npix)``.
        inv_n_map (NDArray): Inverse noise variance map, same shape as ``map_sky``.
        nu (float): Band centre frequency in GHz.
        fwhm (float): Beam full-width-at-half-maximum in arcminutes.
        nside (int): HEALPix nside of the map.
        lmax (int): Maximum multipole for harmonic transforms.
        double_precision (bool): Whether the inverse noise map is stored in float64.
    """
    def __init__(self, map_sky:NDArray, map_rms:NDArray, nu:float, fwhm:float, nside:int,
                 double_precision:bool=False, lmax:int|None = None):
        """Construct a DetectorMap.

        Args:
            map_sky: Sky signal map, shape ``(npol, npix)`` or ``(npix,)``.
            map_rms: RMS noise map (same shape as ``map_sky``).
            nu: Band centre frequency in GHz.
            fwhm: Beam FWHM in arcminutes.
            nside: HEALPix nside of the maps.
            double_precision: If True, store ``inv_n_map`` in float64.
            lmax: Maximum multipole. Defaults to ``int(2.5 * nside)``.
        """
        #cast dimensions correctly to allow constructer with 1-d array for intensity maps.
        map_sky = map_sky.reshape((1,-1)) if map_sky.ndim == 1 else map_sky
        map_rms = map_rms.reshape((1,-1)) if map_rms.ndim == 1 else map_rms
        
        if map_rms.shape != map_sky.shape:
            raise ValueError("Sky and RMS maps should have matching dimensions.")
        if map_sky.shape[0] not in [1,2]:
            raise ValueError("Trying to set sky map with wrong first axis length "
                             f"{map_sky.shape[0]} != 1 or 2")

        self.map_sky = map_sky
        self.nu = nu
        self.fwhm = fwhm #stored in arcmin
        self.nside = nside
        # Slightly higher than 2*NSIDE to avoid accumulation of numeric junk.
        self.lmax = int(2.5*nside) if lmax is None else lmax
        self._beam_Cl = hp.gauss_beam(np.deg2rad(fwhm/60.0), self.lmax)
        self.double_precision = double_precision
        self.inv_n_map = (1./map_rms**2).astype(np.float64 if double_precision else np.float32, copy=False)

    @property
    def map_rms(self):
        """RMS noise map, computed as ``1 / sqrt(inv_n_map)``."""
        return 1./np.sqrt(self.inv_n_map)

    @property
    def fwhm_rad(self):
        """Beam FWHM in radians."""
        return np.deg2rad(self.fwhm/60.0)
    
    @property
    def pol(self):
        """Whether the map is polarised (True for Q/U, False for I only)."""
        #polarization: True->Q/U, False->I
        return False if len(self.map_sky) == 1 else True
    
    @property
    def spin(self):
        """Spin weight of the map (2 for polarised, 0 for intensity)."""
        return 2 if self.pol else 0

    @property
    def npol(self):
        """Number of polarisation components (1 or 2)."""
        return len(self.map_sky) # 2 if self.pol else 1

    def smooth_to_resolution(self, target_fwhm_arcmin: float) -> None:
        """Noise-weighted in-place smoothing of this band to a target Gaussian beam.

        Degrades the band from its native beam to ``target_fwhm_arcmin``, updating the signal map,
        the propagated inverse-noise map, and the stored beam. A no-op if the band is already at the
        target beam. Smoothing can only coarsen resolution, so a target finer (smaller FWHM) than the
        current beam is skipped with a warning rather than applied.
        """
        if self.fwhm == target_fwhm_arcmin:
            return
        if target_fwhm_arcmin < self.fwhm:
            logger.warning(f"Requested smoothing to {target_fwhm_arcmin} arcmin, finer than this "
                           f"band's beam of {self.fwhm} arcmin; leaving the band unchanged.")
            return
        extra_fwhm_rad = np.deg2rad(np.sqrt(target_fwhm_arcmin**2 - self.fwhm**2)/60.0)
        map_rms = self.map_rms
        sky_dtype, inv_dtype = self.map_sky.dtype, self.inv_n_map.dtype
        smoothed_sky = np.empty(self.map_sky.shape, dtype=np.float64)
        smoothed_inv_n = np.empty(self.inv_n_map.shape, dtype=np.float64)
        for ipol in range(self.npol):
            smoothed_sky[ipol] = smooth_signal_map_noiseweighted(self.map_sky[ipol], map_rms[ipol],
                                                                 extra_fwhm_rad)
            smoothed_inv_n[ipol] = 1.0/smooth_rms_map_noiseweighted(map_rms[ipol],
                                                                    extra_fwhm_rad)**2
        self.map_sky = smoothed_sky.astype(sky_dtype, copy=False)
        self.inv_n_map = smoothed_inv_n.astype(inv_dtype, copy=False)
        self.fwhm = target_fwhm_arcmin
        self._beam_Cl = hp.gauss_beam(self.fwhm_rad, self.lmax)

    def apply_inv_N_map(self, map: NDArray, inplace = True):
        """
        Applies in-place the inverse noise variance matrix to a `map`.

        """
        map_out = map if inplace else deepcopy(map)
        for ipol in range(self.npol):
            inplace_arr_prod(map_out[ipol,:], self.inv_n_map[ipol,:])

        return map_out

    def apply_inv_N_alm(self, a: NDArray, nthreads:int = 1, inplace = True):
        """
        Applies the inverse noise variance matrix to a set of alm `a` and returns the result.

        """
        if a.shape[0] != self.npol:
            raise ValueError("Can not apply inv_N to alms with different polarization")

        a_out = a if inplace else deepcopy(a)

        # Y a
        a_out = alm_to_map(a_out, self.nside, self.lmax, spin=self.spin, nthreads=nthreads)

        # N^-1 Y a
        self.apply_inv_N_map(a_out)

        # Y^T N^-1 Y a
        a_out = alm_to_map_adjoint(a_out, self.nside, self.lmax, spin=self.spin, nthreads=nthreads)

        return a_out

    def apply_B(self, a: NDArray):
        """ Applies in-place the beam operator in harmonic space to a set of alm `a`,
            which are also returned.
        """
        for ipol in range(self.npol):
            almxfl(a[ipol], self._beam_Cl, inplace=True)
        return a

    @property
    def blm(self):
        """Returns the spherical harmonic coefficients of the beam associated
           with the detector, plus lmax and mmax. One component for
           temperature-only, three components for polarization."""
        raise NotImplementedError()

    @property
    def noiseProperties(self):
        """Returns parameters describing the noise properties of the detector. TBD"""
        raise NotImplementedError()

    @property
    def bandShape(self):
        """Returns some description of the band's frequency response. TBD."""
        raise NotImplementedError()
