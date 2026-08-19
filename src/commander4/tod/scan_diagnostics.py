"""Per-detector-scan diagnostics recorded into the TOD chain.

The mapmakers call `_record_tod_diagnostics` once per detector-scan, which stores binned power
spectra of the raw TOD, the residual, and the correlated-noise realization. They are written for
inspection only - nothing in the sampling reads them back.
"""
import numpy as np
import pixell
from numpy.typing import NDArray
from scipy.fft import rfftfreq

from commander4.data_models.tod_samples import TODSamples
from commander4.math_utils.fft import forward_rfft
from commander4.tod.data_selection import masked_chisq_z
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


def _record_tod_diagnostics(tod_samples: TODSamples, iscan: int, idet: int, view: TODView,
                            n_corr: NDArray | None) -> None:
    """ Record per-detector-scan TOD diagnostics into the chain arrays.

        Stores the low-resolution log-binned power spectra (sharing one binned frequency axis) of
        four detector-unit TOD views:
          * ``raw``:      the raw detector TOD.
          * ``ncorrsub``: the TOD with only the correlated noise subtracted (sky signal, orbital
                          dipole, and white noise retained); equals ``raw`` when no n_corr drawn.
          * ``residual``: the noise residual, with the sky model, orbital dipole, and correlated
                          noise all subtracted.
          * ``ncorr``:    the correlated-noise realization itself, stored only when one was drawn.
        ``ncorrsub`` and ``residual`` use the jump-corrected stream (matching mapmaking and n_corr
        sampling). When the off-by-default DEBUG full-``n_corr`` collection is enabled, also stores
        the entire ``n_corr`` TOD for this detector-scan.
    """
    nbin = tod_samples.TOD_PS_NBIN
    freqs_binned, raw_binned = _binned_tod_power_spectrum(view.tod, view.fsamp, nbin)
    tod_samples.tod_ps_freqs[iscan, idet] = freqs_binned
    tod_samples.tod_ps_raw[iscan, idet] = raw_binned

    # Sky+orbital-dipole-subtracted residual, and the TOD with only the correlated noise removed.
    # Both are fresh writable copies, so n_corr (when present) is subtracted in place from each.
    residual_tod = view.get_tod(subtract=(("sky", TODView._ALL_GAIN_TERMS),
                                          ("orbital_dipole", TODView._ALL_GAIN_TERMS)))
    ncorrsub_tod = view.get_tod()
    if n_corr is not None:
        residual_tod -= n_corr
        ncorrsub_tod -= n_corr
    _, residual_binned = _binned_tod_power_spectrum(residual_tod, view.fsamp, nbin)
    _, ncorrsub_binned = _binned_tod_power_spectrum(ncorrsub_tod, view.fsamp, nbin)
    tod_samples.tod_ps_residual[iscan, idet] = residual_binned
    tod_samples.tod_ps_ncorrsub[iscan, idet] = ncorrsub_binned

    if n_corr is not None:
        _, ncorr_binned = _binned_tod_power_spectrum(n_corr, view.fsamp, nbin)
        tod_samples.tod_ps_ncorr[iscan, idet] = ncorr_binned
        if tod_samples.ncorr_tods is not None:
            tod_samples.ncorr_tods[iscan][idet] = n_corr.astype(np.float32, copy=False)

    # White-noise-residual chi-squared z-score (uses this iteration's sigma0, sampled just above);
    # stored to the chain and consumed by the data-selection (accept) cuts.
    tod_samples.chisq_z[iscan, idet] = masked_chisq_z(
        residual_tod, view.get_mask(proc_mask=False), view.sigma0)
