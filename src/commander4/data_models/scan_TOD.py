from commander4.data_models.detector_TOD import DetectorTOD

class ScanTOD:
    def __init__ (self, detlist: list[DetectorTOD], start_time: float, scanID: int,
                  scan_idx_start: int, scan_idx_stop: int):
        self.detectors = detlist
        self.start_time = start_time
        self.scanID = scanID
        self.scan_idx_start = scan_idx_start
        self.scan_idx_stop = scan_idx_stop