import numpy as np
from numpy.typing import NDArray

from commander4.data_models.scan_TOD import ScanTOD
from commander4.noise_sampling.noise_psd import NoisePSD
from commander4.utils.math_operations import forward_rfft_mirrored, backward_rfft_mirrored

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
        self._pixel_domain = None  # Cached PixelDomain (built lazily; pointing is static).

    def get_pixel_domain(self, scan_view, comm, sparse: bool):
        """Return the band's map-distribution :class:`PixelDomain`, building and caching it once.

        The domain depends only on the (static) pointing, so it is built once per run and reused
        across Gibbs iterations. ``sparse=False`` gives the historical full-sky-per-rank layout;
        ``sparse=True`` restricts each rank's map buffers to its locally-observed pixels.
        """
        from commander4.utils.pixel_domain import PixelDomain
        mode = "sparse" if sparse else "full"
        if self._pixel_domain is None or self._pixel_domain.mode != mode:
            self._pixel_domain = PixelDomain.from_view(scan_view, comm, mode, self.nside)
        return self._pixel_domain

    def iter_detector_scans(self, accept: NDArray | None = None):
        """Iterate over present detector-scans, yielding ``(iscan, det)`` pairs.

        ``ScanTOD.detectors`` is sparse -- each scan lists only the detectors actually present in it
        -- so this nested walk is the canonical way to traverse detector-scans. The detector's
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