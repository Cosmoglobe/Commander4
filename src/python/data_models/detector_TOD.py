from src.python.data_models.scan_TOD import ScanTOD

class DetectorTOD:
    def __init__(self, scanlist: list[ScanTOD], nu, fwhm, nside, fsamp):
        self._scanlist = scanlist
        self._nu = nu
        self._fwhm = fwhm
        self._nside = nside
        self._fsamp = fsamp

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
    def fsamp(self) -> float:
        return self._fsamp

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
