import numpy as np
import healpy as hp
from numpy.typing import NDArray
from src.python.utils.math_operations import alm_to_map, alm_to_map_adjoint, inplace_arr_prod, almxfl


class DetectorMap:
    def __init__(self, map_sky, map_rms, nu, fwhm, nside, pol:bool, double_precision:bool=False, lmax:int|None = None):
        self._map_sky = map_sky
        self._nu = nu
        self._fwhm = fwhm
        self._nside = nside
        self._lmax = int(2.5*nside) if lmax is None else lmax      # Slightly higher than 2*NSIDE to avoid accumulation of numeric junk.
        self._beam_Cl = hp.gauss_beam(np.deg2rad(fwhm/60.0), self._lmax)
        self._pol = pol #polarization: True->Q/U, False->I
        self._double_precision = double_precision
        self.inv_n_map = (1./map_rms**2).astype(np.float64 if double_precision else np.float64)

    @property
    def map_sky(self):
        return self._map_sky

    @property
    def nu(self):
        return self._nu

    @property
    def fwhm(self):
        return self._fwhm
    
    @property
    def fwhm_rad(self):
        return np.deg2rad(self.fwhm/60.0)
    
    @property
    def double_precision(self):
        return self._double_precision

    @property
    def nside(self):
        return self._nside
    
    @property
    def lmax(self):
        return self._lmax
    
    @property
    def pol(self):
        return self._pol
    
    def npol(self):
        return 2 if self.pol else 1
    
    def apply_inv_N_map(self, map: NDArray):
        """
        Applies in-place the inverse noise variance matrix to a `map`.

        """
        for ipol in range(self.npol):
            inplace_arr_prod(map[ipol], self.inv_n_map[ipol])

        return map

    def apply_inv_N_alm(self, a: NDArray, nthreads:int = 1):
        """
        Applies in-place the inverse noise variance matrix to a set of alm `a`.

        """
        # Y a
        a = alm_to_map(a, self.nside, self.my_band_lmax, spin=self.spin, nthreads=nthreads)

        # N^-1 Y a
        self.inv_N_map(a)

        # Y^T N^-1 Y a
        a = alm_to_map_adjoint(a, self.nside, self.my_band_lmax, spin=self.spin, nthreads=nthreads)

        return a

    def apply_B(self, a: NDArray):
        """
        Applies in-place the beam operator in harmonic space to a set of alm `a`, which are also returned.
        """
        for ipol in range(self._npol):
            almxfl(a[ipol], self._beam_Cl, inplace=True)
        return a

    @property
    def blm(self):
        """Returns the spherical harmonic coefficients of the beam associated
           with the detector, plus lmax and mmax. One component for
           temperature-only, three components for polarization."""
        raise NotImplementedError()

    @property
    def fsamp(self) -> float:
        """Returns the sampling frequency for this detector in Hz."""
        raise NotImplementedError()

    @property
    def noiseProperties(self):
        """Returns parameters describing the noise properties of the detector. TBD"""
        raise NotImplementedError()

    @property
    def map(self) -> np.array:
        """Returns the sky map of the detector."""
        raise NotImplementedError()

    @property
    def bandShape(self):
        """Returns some description of the band's frequency response. TBD."""
        raise NotImplementedError()
