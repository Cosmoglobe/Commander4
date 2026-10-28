"""`TODDiagnostics`: per-detector-scan diagnostics written to the TOD chain, for inspection only.

The mapmaking scan loop (`tod2map`) calls `TODDiagnostics.record_scan` once per detector-scan,
with the noise residual it has built. It stores binned power spectra of the raw TOD, the
n_corr-subtracted TOD, the residual and the correlated-noise realization, and optionally the full
residual and n_corr TODs. None of this is read back on a restart.
"""
import numpy as np
import pixell
from mpi4py import MPI
from numpy.typing import NDArray
from pixell.bunch import Bunch
from scipy.fft import rfftfreq

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.math_utils.fft import forward_rfft
from commander4.tod.step import ChainArrays, TODStep
from commander4.tod.view import TODView


def _binned_tod_power_spectrum(tod: NDArray, fsamp: float, nbin: int) -> tuple[NDArray, NDArray]:
    """ Log-binned periodogram of a TOD, for low-resolution diagnostics written to the chain.

        Computes the one-sided periodogram ``|rfft(tod)|^2 / Ntod`` on the natural frequency grid,
        then averages it into exponentially spaced bins (``pixell.utils.expbin`` with ``nmin=1``).
        expbin returns at most ``nbin`` bins (fewer for short TODs), so the binned frequencies and
        power are returned padded to length ``nbin`` with NaN, giving a fixed width for the
        per-detector-scan chain arrays.
    Args:
        tod (NDArray): Time-ordered data (any units; e.g. the raw TOD or an n_corr realization).
        fsamp (float): Sampling rate (Hz).
        nbin (int): Fixed output length (the maximum number of bins).
    Returns:
        (freqs, power): Each a length-``nbin`` array, NaN-padded beyond the actual bin count.
    """
    ntod = len(tod)
    freqs = rfftfreq(ntod, 1.0 / fsamp)
    power = (1.0 / ntod) * np.abs(forward_rfft(tod)) ** 2
    bins = pixell.utils.expbin(freqs.size, nbin=nbin, nmin=1)
    nb = bins.shape[0]
    freqs_binned = np.full(nbin, np.nan, dtype=np.float64)
    power_binned = np.full(nbin, np.nan, dtype=np.float64)
    freqs_binned[:nb] = pixell.utils.bin_data(bins, freqs)
    power_binned[:nb] = pixell.utils.bin_data(bins, power)
    return freqs_binned, power_binned


class TODDiagnostics(TODStep):
    """Per-detector-scan diagnostics of one band on one rank, written to the chain file only.

    Attributes:
        tod_ps_freqs, tod_ps_raw, tod_ps_ncorr, tod_ps_ncorrsub, tod_ps_residual:
            (nscans, ndet, TOD_PS_NBIN) Low-resolution (log-binned) TOD power spectra: a shared
            binned frequency axis (the bins differ between scans of different length) plus the
            binned periodograms of four TOD views (see `record_scan`). At 2 kB per detector-scan
            they are the largest output after the maps, so
            `output.chains.include.tod_power_spectra: false` switches them off (``None``).
        ncorr_tods, residual_tods: Optional DEBUG output (`output.chains.include.ncorr_tods` /
            `residual_tods`): the full correlated-noise and residual TODs, as [iscan][idet] lists
            of float32 arrays, None where nothing was recorded; ``None`` when switched off.
        scan_runtime: (nscans,) Wall time [s] the mapmaking scan loop spent on each scan this
            iteration, measured by `tod2map`, for load balancing in later runs.
    """

    TOD_PS_NBIN = 100  # Fixed bin count for the low-resolution TOD power spectra.

    def __init__(self, experiment_data: DetectorGroupTOD, params: Bunch, my_band: Bunch,
                 band_comm: MPI.Comm, chain: int):
        """Allocate the diagnostics that `params.output.chains.include` asks for."""
        nscans, ndet = experiment_data.nscans, experiment_data.ndet
        include = params.output.chains.include
        self.tod_ps_freqs = self.tod_ps_ncorr = self.tod_ps_raw = None
        self.tod_ps_ncorrsub = self.tod_ps_residual = None
        if bool(getattr(include, "tod_power_spectra", True)):
            ps_shape = (nscans, ndet, self.TOD_PS_NBIN)
            self.tod_ps_freqs = np.full(ps_shape, np.nan, dtype=np.float32)
            self.tod_ps_ncorr = np.full(ps_shape, np.nan, dtype=np.float32)
            self.tod_ps_raw = np.full(ps_shape, np.nan, dtype=np.float32)
            self.tod_ps_ncorrsub = np.full(ps_shape, np.nan, dtype=np.float32)
            self.tod_ps_residual = np.full(ps_shape, np.nan, dtype=np.float32)

        self.ncorr_tods: list[list[NDArray | None]] | None = None
        if bool(getattr(include, "ncorr_tods", False)):
            self.ncorr_tods = [[None] * ndet for _ in range(nscans)]
        self.residual_tods: list[list[NDArray | None]] | None = None
        if bool(getattr(include, "residual_tods", False)):
            self.residual_tods = [[None] * ndet for _ in range(nscans)]

        self.scan_runtime = np.zeros(nscans, dtype=np.float64)


    def chain_arrays(self) -> ChainArrays:
        """The scan runtimes and, unless switched off, the power spectra, and the full TODs when
        asked for (as ``tods/<scan ID>/<detector>/ncorr`` and ``.../residual``). None is read
        back."""
        # The first scan each rank processes in `tod2map` also pays one-time Numba compilation
        # cost, so its runtime is replaced by the median runtime of the other scans.
        scan_runtime = self.scan_runtime.copy()
        timed = np.flatnonzero(scan_runtime)
        if timed.size > 1:
            scan_runtime[timed[0]] = np.median(scan_runtime[timed[1:]])
        arrays = ChainArrays(per_scan={"scan_runtime": scan_runtime})
        if self.tod_ps_freqs is not None:
            arrays.per_scan.update({
                "tod_ps_freqs": self.tod_ps_freqs,
                "tod_ps_raw": self.tod_ps_raw,
                "tod_ps_ncorr": self.tod_ps_ncorr,
                "tod_ps_ncorrsub": self.tod_ps_ncorrsub,
                "tod_ps_residual": self.tod_ps_residual,
            })
        if self.ncorr_tods is not None or self.residual_tods is not None:
            tods_by_detector_scan = {}
            for name, tods in (("ncorr", self.ncorr_tods), ("residual", self.residual_tods)):
                if tods is None:
                    continue
                for iscan, scan_tods in enumerate(tods):
                    for idet, tod in enumerate(scan_tods):
                        if tod is not None:
                            tods_by_detector_scan.setdefault((iscan, idet), {})[name] = tod
            arrays.per_detector_scan["tods"] = tods_by_detector_scan
        return arrays


    def start_iteration(self) -> None:
        """Forget the scan runtimes and the full TODs of the previous iteration, so the chain file
        only gets those of this one. The power spectra are not reset: a detector-scan rejected
        since keeps its last ones."""
        self.scan_runtime[:] = 0.0
        for tods in (self.residual_tods, self.ncorr_tods):
            if tods is not None:
                for scan_tods in tods:
                    scan_tods[:] = [None] * len(scan_tods)


    def record_scan(self, view: TODView, residual_tod: NDArray, n_corr: NDArray | None) -> None:
        """ Record the diagnostics of the focused detector-scan.

            Unless switched off (``tod_ps_freqs is None``), stores the low-resolution log-binned
            power spectra (sharing one binned frequency axis) of four detector-unit TOD views:
              * ``raw``:      the raw detector TOD.
              * ``ncorrsub``: the TOD with every model removed except the sky signal (sky,
                              orbital dipole and far sidelobes; `TODView.get_tod`), minus the
                              correlated noise.
              * ``residual``: the noise residual.
              * ``ncorr``:    the correlated-noise realization itself, stored only when one was
                              drawn.
            Optional full-TOD collection stores ``n_corr`` and/or the residual as float32.

            Args:
                residual_tod: The noise residual, in detector units: the TOD with every model
                    and the correlated noise removed (built by `tod2map`).
                n_corr: The correlated-noise realization, or None when none was drawn.
        """
        iscan, idet = view.iscan, view.idet
        if self.tod_ps_freqs is not None:
            nbin = self.TOD_PS_NBIN
            ncorrsub_tod = view.get_tod(keep=("sky", "orbital_dipole", "far_sidelobes"))
            if n_corr is not None:
                ncorrsub_tod -= n_corr
            freqs_binned, raw_binned = _binned_tod_power_spectrum(view.raw_tod, view.fsamp, nbin)
            _, residual_binned = _binned_tod_power_spectrum(residual_tod, view.fsamp, nbin)
            _, ncorrsub_binned = _binned_tod_power_spectrum(ncorrsub_tod, view.fsamp, nbin)
            self.tod_ps_freqs[iscan, idet] = freqs_binned
            self.tod_ps_raw[iscan, idet] = raw_binned
            self.tod_ps_residual[iscan, idet] = residual_binned
            self.tod_ps_ncorrsub[iscan, idet] = ncorrsub_binned
            if n_corr is not None:
                _, ncorr_binned = _binned_tod_power_spectrum(n_corr, view.fsamp, nbin)
                self.tod_ps_ncorr[iscan, idet] = ncorr_binned

        if self.residual_tods is not None:
            # A copy, so the stored residual stays observed minus model whatever the caller does
            # with its array.
            self.residual_tods[iscan][idet] = residual_tod.astype(np.float32, copy=True)
        if n_corr is not None and self.ncorr_tods is not None:
            self.ncorr_tods[iscan][idet] = n_corr.astype(np.float32, copy=False)
