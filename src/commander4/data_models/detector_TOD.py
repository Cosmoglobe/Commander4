from commander4.data_models.scan_TOD import ScanTOD
import numpy as np
from numpy.typing import NDArray

class DetectorTOD:
    def __init__(self, scanlist: list[ScanTOD], nu, fwhm, nside, data_nside,
                 experiment_name, band_name, detector_name):
        self._scanlist = scanlist
        self._nu = nu
        self._fwhm = fwhm
        self._eval_nside = nside
        self._data_nside = data_nside
        self._experiment_name = experiment_name
        self._band_name = band_name
        self._detector_name = detector_name

    @property
    def nu(self):
        return self._nu

    @property
    def fwhm(self):
        return self._fwhm

    @property
    def nside(self):
        return self._eval_nside

    @property
    def data_nside(self):
        return self._data_nside
    
    @property
    def experiment_name(self):
        return self._experiment_name

    @property
    def band_name(self):
        return self._band_name
    
    @property
    def detector_name(self):
        return self._detector_name

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
    def scans(self) -> list[ScanTOD]:
        return self._scanlist

    @property
    def bandShape(self):
        """Returns some description of the band's frequency response. TBD."""
        raise NotImplementedError()
