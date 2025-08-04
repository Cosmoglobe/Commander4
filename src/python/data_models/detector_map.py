import numpy as np

class DetectorMap:
    def __init__(self, map_sky, map_corr_noise, map_rms, nu, fwhm, nside):
        self._map_corr_noise = map_corr_noise
        self._map_sky = map_sky
        self._map_rms = map_rms
        self._nu = nu
        self._fwhm = fwhm
        self._nside = nside

    @property
    def map_sky(self):
        return self._map_sky

    @property
    def map_corr_noise(self):
        return self._map_corr_noise

    @property
    def map_rms(self):
        return self._map_rms

    @property
    def nu(self):
        return self._nu

    @property
    def fwhm(self):
        return self._fwhm

    @property
    def nside(self):
        return self._nside

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
