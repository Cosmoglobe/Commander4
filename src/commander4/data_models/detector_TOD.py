from commander4.data_models.scan_TOD import ScanTOD

class DetectorTOD:
    def __init__(self, scanlist: list[ScanTOD], nu, fwhm, nside, data_nside,
                 experiment_name, band_name, detector_name, pols):
        self.scans = scanlist
        self.nu = nu
        self.fwhm = fwhm
        self.eval_nside = nside
        self.data_nside = data_nside
        self.experiment_name = experiment_name
        self.band_name = band_name
        self.detector_name = detector_name
        self.pols = pols

    @property
    def nside(self):
        return self.eval_nside

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
