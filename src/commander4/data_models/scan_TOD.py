from commander4.data_models.detector_TOD import DetectorTOD

class ScanTOD:
    """Groups the DetectorTOD objects that belong to a single scan (pointing period).

    Attributes:
        detectors (list[DetectorTOD]): Per-detector TOD data for this scan.
        start_time (float): Start time of the scan (mission elapsed time).
        scanID (int): Unique integer identifier for this scan.
    """
    def __init__ (self, detlist: list[DetectorTOD], start_time: float, scan_id: int):
        self.detectors = detlist
        self.start_time = start_time
        self.scan_id = scan_id