from .band import BandMap
from .detector_group import DetectorGroupMap

class SimpleBandMap(BandMap):
    def __init__(self, detgrplist):
        self._detgrplist = detgrplist
    @property
    def lmax(self) -> int:
        """Returns the band limit necessary to work with data in this band. TBD"""
        raise NotImplementedError()

    @property
    def detectorGroups(self) -> list[DetectorGroupMap]:
        return self._detgrplist
