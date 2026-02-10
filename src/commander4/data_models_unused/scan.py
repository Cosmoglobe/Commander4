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
