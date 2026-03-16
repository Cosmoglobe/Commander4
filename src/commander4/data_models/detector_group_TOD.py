from commander4.data_models.scan_TOD import ScanTOD

class DetGroupTOD:
    def __init__(self, scans: list[ScanTOD], experiment_name: str, band_name: str, nside: int,
                 nu: float, fwhm: float, ndet: int):
        self.scans = scans
        self.nscans = len(scans)
        self.experiment_name = experiment_name
        self.band_name = band_name
        self.nside = nside
        self.nu = nu
        self.fwhm = fwhm
        self.ndet = ndet