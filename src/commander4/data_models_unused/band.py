from .detector_group import DetectorGroup

class Band:
    @property
    def lmax(self) -> int:
        """Returns the band limit necessary to work with data in this band. TBD"""
        raise NotImplementedError()

    @property
    def detectorGroups(self) -> tuple[DetectorGroup, ...]:
        """Returns a list of Detector group objects associated with this band."""
        raise NotImplementedError()
