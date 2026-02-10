from .detector_group_map import DetectorGroupMap
from .detector_map import DetectorMap

class SimpleDetectorGroupMap(DetectorGroupMap):
    def __init__(self, detlist):
        self._detlist = detlist

    @property
    def detectors(self) -> list[DetectorMap]:
        return self._detlist

