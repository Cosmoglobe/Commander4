import numpy as np
from .detector_map import DetectorMap

class SimpleDetectorMap(DetectorMap):
    def __init__(self, map_sky, map_rms):
        self._map_sky = map_sky
        self._map_rms = map_rms

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
    def map_sky(self) -> np.array:
        """Returns the sky map of the detector."""
        return self._map_sky

    @property
    def map_rms(self) -> np.array:
        """Returns the uncertainty (rms) map of the detector."""
        return self._map_rms

    @property
    def bandShape(self):
        """Returns some description of the band's frequency response. TBD."""
        raise NotImplementedError()
