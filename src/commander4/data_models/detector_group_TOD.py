from commander4.data_models.scan_TOD import ScanTOD

class DetGroupTOD:
    """Container for all scan TODs belonging to one detector group (experiment + band).

    Groups together the list of ``ScanTOD`` objects with common metadata such as
    nside, frequency, beam, and polarisation configuration.

    Attributes:
        scans (list[ScanTOD]): Scans assigned to this MPI rank.
        nscans (int): Number of scans in ``scans``.
        experiment_name (str): Experiment identifier (e.g. ``'PlanckLFI'``).
        band_name (str): Band identifier (e.g. ``'30GHz'``).
        nside (int): HEALPix nside for map evaluation.
        nu (float): Band centre frequency in GHz.
        fwhm (float): Beam FWHM in arcminutes.
        ndet (int): Number of detectors per scan.
        pols (str): Polarisation configuration string (``'I'``, ``'QU'``, or ``'IQU'``).
    """
    def __init__(self, scans: list[ScanTOD], experiment_name: str, band_name: str, nside: int,
                 nu: float, fwhm: float, ndet: int, pols: str):
        self.scans = scans
        self.nscans = len(scans)
        self.experiment_name = experiment_name
        self.band_name = band_name
        self.nside = nside
        self.nu = nu
        self.fwhm = fwhm
        self.ndet = ndet
        self.pols = pols