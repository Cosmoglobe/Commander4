from .band import Band
from .detector_group import DetectorGroup

class SimpleBand(Band):
    def __init__(self, detgrplist, lmax):
        self._detgrplist = detgrplist
        self._lmax = lmax
    @property
    def lmax(self) -> int:
        return self._lmax

    @property
    def detectorGroups(self) -> list[DetectorGroup]:
        return self._detgrplist
