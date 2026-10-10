"""`DetectorGroupTOD`: the static TOD data one band holds on one rank.

Owns the scan list, the per-detector metadata, and the noise model. "Static" means it holds the
data as read from disk; everything the Gibbs chain samples lives in `TODSamples` instead.
"""
import numpy as np
from numpy.typing import NDArray

from commander4.data_models.pixel_domain import PixelDomain
from commander4.data_models.scan_tod import ScanTOD
from commander4.tod.noise.psd import NoisePSD
from commander4.math_utils.fft import forward_rfft_mirrored, backward_rfft_mirrored

import logging
logger = logging.getLogger(__name__)


class DetectorGroupTOD:
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
        noise_model (NoisePSD): Noise model for this detector group.
        tf_tau_ms (float|None): Transfer function time constant in milliseconds, or None if no TF.
        hfi_demodulation (bool): Whether this band contains alternating Planck HFI half-cycles.
    """
    def __init__(self, scans: list[ScanTOD], experiment_name: str, band_name: str, nside: int,
                 nu: float, fwhm: float, fsamp: float, ndet: int, pols: str, noise_model: NoisePSD, 
                 tf_tau_sec: float|None = None, instrument_filepath: str|None = None, 
                 hfi_demodulation: bool = False):
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
        self.tf_tau_sec = tf_tau_sec
        # Whether the TOD needs demodulation to be read (Planck HFI stores alternating
        # positive/negative modulation half-cycles.)
        self.hfi_demodulation = hfi_demodulation
        self.noise_model = noise_model
        # The band's PixelDomain: which pixels this rank's map buffers hold. It depends only on the
        # static pointing, so it is built once at startup (tod/processing.py) and kept for the run.
        self.pixel_domain: PixelDomain | None = None
        self.instrument_filepath = instrument_filepath

    def iter_detector_scans(self, accept: NDArray | None = None):
        """Iterate over present detector-scans, yielding ``(iscan, det)`` pairs.

        ``ScanTOD.detectors`` is sparse (each scan lists only the detectors actually present in it),
        so this nested walk is the canonical way to traverse detector-scans. The detector's
        full-band column ``det.det_idx_fullband`` is the index into the dense ``(nscans, ndet)``
        per-detector sample arrays (gain, noise params, accept, ...); the per-scan enumerate position
        must never be used for that, and this iterator deliberately never exposes one.

        Args:
            accept: Optional ``(nscans, ndet)`` boolean mask. When given, detector-scans whose entry
                is False are skipped, so callers process only accepted (good-quality) data.

        Yields:
            tuple[int, DetectorTOD]: the local scan index ``iscan`` and the present detector ``det``.
        """
        for iscan, scan in enumerate(self.scans):
            for det in scan.detectors:
                if accept is not None and not accept[iscan, det.det_idx_fullband]:
                    continue
                yield iscan, det

    def apply_N_inv(self, tod: NDArray, noise_params: NDArray, samprate: float|None = None,
                    inplace=False) -> NDArray:
        """ Applies the inverse noise covariance N^-1 of this Det-Group to the input TOD, using the
            specified noise parameters. If a sample rate is specified, the TOD is assumed to have
            been downsampled, and the noise level is scaled accordingly. The DC (mean) mode is
            projected out, matching the Commander3 ``multiply_inv_N`` convention.
        """
        actual_samprate = samprate if samprate is not None else self.fsamp
        tod_out = tod if inplace else np.zeros_like(tod)

        # White-noise fast path: P(f) = sigma0^2 (flat), so N^-1 is a scalar and the FFT is skipped.
        if self.noise_model.is_white:
            scale = float(noise_params[0])**2
            if samprate is not None and samprate != self.fsamp:
                scale *= samprate/self.fsamp
            tod_out[:] = tod/scale
            tod_out -= np.mean(tod_out)  # Project out the DC mode (mean).
            return tod_out

        # Mirrored FFT (length 2*ntod) reduces boundary/periodicity ringing.
        ntod = tod.shape[0]
        freqs = np.fft.rfftfreq(2*ntod, d=1.0/actual_samprate)
        noise_PS = self.noise_model.eval_full(freqs, noise_params)
        if samprate is not None and samprate != self.fsamp:
            noise_PS *= samprate/self.fsamp
        tod_f = forward_rfft_mirrored(tod)
        tod_f /= noise_PS
        tod_f[0] = 0.0  # Project out the DC mode (mean), matching Commander multiply_inv_N.
        tod_out[:] = backward_rfft_mirrored(tod_f, ntod)
        return tod_out
