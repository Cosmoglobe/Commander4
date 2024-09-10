class Scan:
    @property
    def nsamples(self) -> int:
        """Returns the number of individual samples in this scan."""
        raise NotImplementedError()

    @property
    def startTime(self) -> float:
        """Returns the time stamp of the first sample in this scan.
           (Reference time TBD)"""
        raise NotImplementedError()

    @property
    def data(self):
        """Returns pointing and signal data for all samples in this scan.
           Exact format TBD; could be (theta, phi, psi, value), or
           (theta, phi) could be encoded in a high-res Healpix pixel index
           etc."""
        raise NotImplementedError()


class Detector:
    @property
    def blm(self):
        """Returns the spherical harmonic coefficients of the beam associated
           with the detector, plus lmax and mmax. One component for
           temperature-only, three componens for polarization."""
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
    def scans(self) -> list[Scan]:
        """Returns a list of Scan objects associated with this detector."""
        raise NotImplementedError()

    @property
    def bandShape(self):
        """Returns some description of the band's frequency response. TBD."""
        raise NotImplementedError()


class DetectorGroup:
    @property
    def detectors(self) -> list[Detector]:
        """Returns the list of detectors in this group"""
        raise NotImplementedError()


class Band:
    @property
    def lmax(self) -> int:
        """Returns the band limit necessary to work with data in this band. TBD"""
        raise NotImplementedError()

    @property
    def detectorGroups(self) -> list[DetectorGroup]:
        """Returns a list of Detector group objects associated with this band."""
        raise NotImplementedError()
