import numpy as np
from numpy.typing import NDArray
from src.python.utils.math_operations import alm_to_map, alm_to_map_adjoint, inplace_arr_prod

class DetectorMap:
    def __init__(self, map_sky, map_rms, nu, fwhm, nside, pol:bool, precision:str='single'):
        self._map_sky = map_sky
        self._nu = nu
        self._fwhm = fwhm
        self._nside = nside
        self._pol = pol #polarization: True->Q/U, False->I
        self._npol = 2 if pol else 1
        self._precision = np.float32 if precision == "single" else np.float64
        self.inv_n_map = (1./map_rms**2).astype(self._precision)

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
    def nside(self):
        return self._nside
    
    def inv_N_map(self, map: NDArray):
        """
        Applies the inverse noise variance matrix to a `map`.

        """
        for ipol in range(self._npol):
            inplace_arr_prod(map[ipol], self.inv_n_map[ipol])

        return map

    def inv_N_alm(self, a, nthreads:int = 1):
        """
        Applies the inverse noise variance matrix to a set of `alm`.

        """
        # Y a
        a = alm_to_map(a, self.my_band_nside, self.my_band_lmax, spin=self.spin, nthreads=nthreads)

        # N^-1 Y a
        self.inv_N_map(a)

        # Y^T N^-1 Y a
        a = alm_to_map_adjoint(a, self.my_band_nside, self.my_band_lmax, spin=self.spin, nthreads=nthreads)

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
