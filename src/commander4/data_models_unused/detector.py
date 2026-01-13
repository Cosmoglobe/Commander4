from .scan import Scan

class Detector:
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
    def scans(self) -> tuple[Scan, ...]:
        """Returns a list of Scan objects associated with this detector."""
        raise NotImplementedError()

    @property
    def bandShape(self):
        """Returns some description of the band's frequency response. TBD."""
        raise NotImplementedError()
