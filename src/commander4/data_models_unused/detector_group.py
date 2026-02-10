from .detector import Detector

class DetectorGroup:
    @property
    def detectors(self) -> tuple[Detector, ...]:
        """Returns the list of detectors in this group"""
        raise NotImplementedError()
