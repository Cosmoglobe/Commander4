from .detector_group_map import DetectorGroupMap

class BandMap:
    @property
    def lmax(self) -> int:
        """Returns the band limit necessary to work with data in this band. TBD"""
        raise NotImplementedError()

    @property
    def detectorGroups(self) -> tuple[DetectorGroupMap, ...]:
        """Returns a list of Detector group objects associated with this band."""
        raise NotImplementedError()
