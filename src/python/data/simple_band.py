from .band import Band
from .detector_group import DetectorGroup

class SimpleBand(Band):
    def __init__(self, detgrplist):
        self._detgrplist = detgrplist
    @property
    def lmax(self) -> int:
        """Returns the band limit necessary to work with data in this band. TBD"""
        raise NotImplementedError()

    @property
    def detectorGroups(self) -> list[DetectorGroup]:
        return self._detgrplist
