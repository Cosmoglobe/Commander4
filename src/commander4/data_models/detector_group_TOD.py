import numpy as np
from numpy.typing import NDArray

from commander4.data_models.scan_TOD import ScanTOD
from commander4.noise_sampling.noise_psd import NoisePSD
from commander4.utils.math_operations import forward_rfft, backward_rfft

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
                 nu: float, fwhm: float, fsamp: float, ndet: int, pols: str, noise_model: NoisePSD):
        self.scans = scans
        self.nscans = len(scans)
        self.experiment_name = experiment_name
        self.band_name = band_name
        self.nside = nside
        self.nu = nu
        self.fwhm = fwhm
        self.fsamp = fsamp
        self.ndet = ndet
        self.pols = pols
        # The below values are not known until all ranks are finished reading in data, because some
        # scans might be rejected. THey will be set after-the-fact.
        self.scan_idx_start: int = 0  # Index of my first scan in a compact indexing.
        self.scan_idx_stop: int = 0  # Index of my last scan.
        self.nscans_allranks: int = 0  # Total number of scans across all ranks (on this band).
        self.noise_model = noise_model

    def apply_N_inv(self, tod: NDArray, noise_params: NDArray, samprate: float|None = None,
                    inplace=False) -> NDArray:
        """ Modulates the input TOD with the noise model of this Det-Group, using the specified
            noise parameters. If a sample rate is specified, the TOD is assumed to have been
            downsampled, and the white noise level will be adjusted accordingly.
        """
        # TODO: Some noise-PS types might require different handling, such as flat PS.
        actual_samprate = samprate if samprate is not None else self.fsamp
        tod_out = tod if inplace else np.zeros_like(tod)
        ntod = tod.shape[0]
        freqs = np.fft.rfftfreq(ntod, d=1.0/actual_samprate)
        noise_PS = self.noise_model.eval_full(freqs, noise_params)

        tod_f = forward_rfft(tod)
        if samprate is not None and samprate != self.fsamp:
            noise_PS *= samprate/self.fsamp
        tod_f /= noise_PS
        tod_out[:] = backward_rfft(tod_f, ntod)
        return tod_out