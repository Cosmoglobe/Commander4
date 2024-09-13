from .detector_group import DetectorGroupMap
from .detector import DetectorMap

class SimpleDetectorGroup(DetectorGroupMap):
    def __init__(self, detlist):
        self._detlist = detlist

    @property
    def detectors(self) -> list[DetectorMap]:
        return self._detlist

