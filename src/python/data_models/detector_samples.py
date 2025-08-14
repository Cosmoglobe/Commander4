from src.python.data_models.scan_samples import ScanSamples

class DetectorSamples:
    def __init__(self, scans: list[ScanSamples]):
        self.scans = scans