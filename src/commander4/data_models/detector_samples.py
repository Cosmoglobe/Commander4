"""`DetectorSamples`: a placeholder container, used only by the deprecated Planck sim reader."""
from commander4.data_models.scan_samples import ScanSamples

class DetectorSamples:
    """Placeholder holding one detector's per-scan samples; see `ScanSamples`."""

    def __init__(self, scans: list[ScanSamples]):
        self.scans = scans