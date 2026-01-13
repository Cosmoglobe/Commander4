from .detector_group import DetectorGroup
from .detector import Detector

class SimpleDetectorGroup(DetectorGroup):
    def __init__(self, detlist):
        self._detlist = detlist

    @property
    def detectors(self) -> list[Detector]:
        return self._detlist

