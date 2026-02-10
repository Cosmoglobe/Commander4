import numpy as np
import healpy as hp
from copy import deepcopy
from numpy.typing import NDArray
from src.python.utils.math_operations import alm_to_map, alm_to_map_adjoint, inplace_arr_prod, almxfl


class DetectorMap:
    def __init__(self, map_sky:NDArray, map_rms:NDArray, nu:float, fwhm:float, nside:int, double_precision:bool=False, lmax:int|None = None):
        #cast dimensions correctly to allow constructer with 1-d array for intensity maps.
        map_sky = map_sky.reshape((1,-1)) if map_sky.ndim == 1 else map_sky
        map_rms = map_rms.reshape((1,-1)) if map_rms.ndim == 1 else map_rms
        
        if map_rms.shape != map_sky.shape:
            raise ValueError(f"Sky and RMS maps should have matching dimensions.")
        if map_sky.shape[0] not in [1,2]:
            raise ValueError(f"Trying to set sky map with wrong first axis length {map_sky.shape[0]} != 1 or 2")

        self._map_sky = map_sky
        self._nu = nu
        self._fwhm = fwhm #stored in arcmin
        self._nside = nside
        self._lmax = int(2.5*nside) if lmax is None else lmax      # Slightly higher than 2*NSIDE to avoid accumulation of numeric junk.
        self._beam_Cl = hp.gauss_beam(np.deg2rad(fwhm/60.0), self._lmax)
        self._double_precision = double_precision
        self._inv_n_map = (1./map_rms**2).astype(np.float64 if double_precision else np.float32)

    @property
    def map_sky(self):
        return self._map_sky
    
    @property
    def inv_n_map(self):
        return self._inv_n_map
    
    @property
    def map_rms(self):
        return 1./np.sqrt(self._inv_n_map)

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
        #polarization: True->Q/U, False->I
        return False if len(self._map_sky) == 1 else True
    
    @property
    def spin(self):
        return 2 if self.pol else 0

    @property
    def npol(self):
        return len(self._map_sky) # 2 if self.pol else 1
    
    def apply_inv_N_map(self, map: NDArray, inplace = True):
        """
        Applies in-place the inverse noise variance matrix to a `map`.

        """
        map_out = map if inplace else deepcopy(map)
        for ipol in range(self.npol):
            inplace_arr_prod(map_out[ipol,:], self._inv_n_map[ipol,:])

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
        """
        Applies in-place the beam operator in harmonic space to a set of alm `a`, which are also returned.
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
