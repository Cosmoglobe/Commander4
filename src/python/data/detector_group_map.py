from .detector_map import DetectorMap

class DetectorGroupMap:
    @property
    def detectors(self) -> tuple[DetectorMap, ...]:
        """Returns the list of detector maps in this group"""
        raise NotImplementedError()
