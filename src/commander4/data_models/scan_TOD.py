from commander4.data_models.detector_TOD import DetectorTOD

class ScanTOD:
    """Groups the DetectorTOD objects that belong to a single scan (pointing period).

    Attributes:
        detectors (list[DetectorTOD]): Per-detector TOD data for this scan.
        start_time (float): Start time of the scan (mission elapsed time).
        scanID (int): Unique integer identifier for this scan.
        scan_idx_start (int): Global start index of this scan within the full TOD.
        scan_idx_stop (int): Global stop index (exclusive) of this scan.
    """
    def __init__ (self, detlist: list[DetectorTOD], start_time: float, scanID: int,
                  scan_idx_start: int, scan_idx_stop: int):
        self.detectors = detlist
        self.start_time = start_time
        self.scanID = scanID
        self.scan_idx_start = scan_idx_start
        self.scan_idx_stop = scan_idx_stop
