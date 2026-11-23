""" White-noise level (sigma0) estimation routines.

    Top level functions:
    - `calc_sigma0_simple`: direct first-difference sigma0 estimation.
    - `calc_sigma0_robust`: same as above, but with iterative outlier-clipping.
    - `calc_sigma0_binned_psd`: "bottom of the binned PSD" estimator (Commander3-style).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from math import sqrt
from numba import njit
from scipy.fft import rfftfreq
from scipy.special import ndtri

from commander4.math_utils.fft import forward_rfft_mirrored

@njit(fastmath=True)
def calc_sigma0_simple(tod: NDArray, mask: NDArray[np.bool_]) -> float:
    """ Calculate the white noise level of `tod` using only entries where `mask` is True.
        Uses the `std(tod[1:] - tod[:-1]) / sqrt(2)` trick (first-difference noise estimation),
        which gives a fast and robust estimate of the white noise (sigma0) level of a TOD.
    Args:
        tod: The input TOD array (1D).
        mask: A boolean mask array of the same size as tod (True = valid).
    Returns:
        The calculated sigma value, or `np.inf` if fewer than two valid data points exist.
    """
    if tod.shape != mask.shape:
        raise ValueError("TOD and mask shapes must match.")
    count = 0
    mean = 0.0
    m2 = 0.0
    # Start at index 1 to safely look back at i-1
    for i in range(1, tod.size):
        # Only calculate a difference if both adjacent samples are valid
        if mask[i] and mask[i - 1]:
            diff = tod[i] - tod[i - 1]
            # Welford's online algorithm update
            count += 1
            delta = diff - mean
            mean += delta / count
            delta2 = diff - mean
            m2 += delta * delta2
    # We require at least 2 pairs to calculate an unbiased sample variance (N-1)
    if count < 2:
        return np.inf
    # Apply Bessel's correction
    var = m2 / (count - 1)
    std_dev = sqrt(var)
    return float(std_dev / sqrt(2.0))


@njit(fastmath=True)
def _decimate_for_sigma0(tod, mask, dec_wn):
    """Decimate TOD into block-averaged values, producing res0/mask0 arrays.

    Args:
        tod: 1-D float64 array.
        mask: 1-D bool array.
        dec_wn: Decimation factor (int > 1).

    Returns:
        (res0, mask0): float64 and bool arrays of length ntod0.
    """
    ntod0 = len(tod) // dec_wn - 1
    res0 = np.empty(ntod0, dtype=np.float64)
    mask0 = np.ones(ntod0, dtype=np.bool_)
    for j in range(ntod0):
        j1 = j * dec_wn
        all_valid = True
        acc = 0.0
        for k in range(dec_wn):
            if not mask[j1 + k]:
                all_valid = False
                break
            acc += tod[j1 + k]
        if not all_valid:
            mask0[j] = False
            res0[j] = 1e30
        else:
            res0[j] = acc / dec_wn
    return res0, mask0


@njit(fastmath=True)
def _sigma_clip_pairs(res0: NDArray[np.float64], mask0: NDArray[np.bool_], n_clip_iter: int,
                      threshold: float, down_factor: int):
    """Iterative sigma-clipping on pairwise differences.

    Args:
        res0: (Possibly decimated) TOD values (float64 array).
        mask0: Validity mask for *res0* (bool array).
        n_clip_iter: Number of clipping iterations.
        threshold: Rejection threshold in sigma units.
        down_factor: Decimation factor (used to scale the final estimate).

    Returns:
        sigma0_est (float).
    """
    ntod0 = len(res0)
    sqrt2 = sqrt(2.0)
    s0 = np.inf
    sigma0_est = np.inf
    for _ in range(n_clip_iter):
        s = 0.0
        nval = 0
        for j in range(0, ntod0 - 1, 2):
            if not mask0[j] or not mask0[j + 1]:
                continue
            res = (res0[j] - res0[j + 1]) / sqrt2
            if abs(res) > s0:
                continue
            s += res * res
            nval += 1
        if nval > 100:
            sigma0_est = sqrt(s / (nval - 1))
            if down_factor > 1:
                sigma0_est *= sqrt(float(down_factor))
            s0 = threshold * sigma0_est
        else:
            break
    return sigma0_est


def calc_sigma0_robust(tod: NDArray, mask: NDArray[np.bool_], down_factor: int = 1,
                             n_clip_iter: int = 3, threshold: float = 5.0) -> float:
    """ Robust white-noise estimation via pairwise differencing with outlier rejection.

    Args:
        tod: The input TOD array (1D).
        mask: A boolean mask array of the same size as tod (True = valid).
        down_factor: Decimation factor. When > 1, consecutive samples are block-averaged before
                     differencing, scaled by ``sqrt(down_factor)``.
        n_clip_iter: Number of iterative sigma-clipping passes.
        threshold: Outlier rejection threshold in units of the current sigma0 estimate.

    Returns:
        Estimated white-noise level (float), or ``np.inf`` if too few valid pairs.
    """
    tod_f64 = tod.astype(np.float64) if tod.dtype != np.float64 else tod

    if down_factor > 1:
        ntod0 = len(tod) // down_factor - 1
        if ntod0 < 2:
            return np.inf
        res0, mask0 = _decimate_for_sigma0(tod_f64, mask, down_factor)
    else:
        res0 = tod_f64
        mask0 = mask

    return float(_sigma_clip_pairs(res0, mask0, n_clip_iter, threshold, down_factor))


def _lowest_bins_bias(n_lowest: int, n_bins: int, modes_per_bin: float) -> float:
    """Expected mean of the `n_lowest` smallest of `n_bins` bin powers, in units of the true power.

    Selecting the smallest values of a noisy set always reads low, so the PSD floor underestimates
    sigma0^2 unless divided by this factor. A bin averaging `modes_per_bin` Fourier modes scatters
    about the true power as Gamma(nu, 1/nu), which for the mode counts occurring in practice (tens
    to thousands) is close to N(1, 1/sqrt(nu)); Blom's approximation then gives the expected
    standard-normal order statistics whose mean sets how far below one the lowest bins sit.
    """
    ranks = np.arange(1, n_lowest + 1)
    bias = 1.0 + ndtri((ranks - 0.375) / (n_bins + 0.25)).mean() / sqrt(modes_per_bin)
    # Only reachable with a handful of modes per bin, where the Gaussian approximation has broken
    # down anyway; clamping keeps a pathological scan from inflating sigma0 without limit.
    return max(bias, 0.25)


def calc_sigma0_binned_psd(tod: NDArray, mask: NDArray[np.bool_], fsamp: float,
                           dnu: float = 0.5, n_floor_bins: int = 10) -> float:
    """ Estimate sigma0 as the "bottom of the binned PSD": sqrt of the lowest binned periodogram.

    Follows the Commander3 steady-state white-noise estimator (comm_tod_noise_mod.f90:135-176): the
    mirrored-FFT periodogram of the (signal-subtracted) TOD is averaged into ``dnu``-Hz bins, and
    the bottom of that spectrum defines sigma0.

    ``mask`` is not applied, so glitches bias the estimate high: a glitch is short in time, and
    raises every bin of the spectrum by the same amount. Selecting the lowest bins biases the
    estimate low; ``_lowest_bins_bias`` corrects for this.

    Args:
        tod: Signal-subtracted residual TOD (1D); masked samples may still carry data.
        mask: Boolean validity mask (unused, kept for call consistency).
        fsamp: Sampling rate (Hz).
        dnu: Bin width (Hz) for averaging the periodogram.
        n_floor_bins: How many of the lowest bins to average, capped at a third of the populated
            bins. Larger values reduce the scatter but risk reaching up into the 1/f rise when the
            knee sits high in the band.
    Returns:
        Estimated white-noise level (float), or ``np.inf`` if no bin is populated.
    """
    ntod = len(tod)
    # Mirrored-FFT periodogram (length-2*ntod grid), normalized so its white floor is sigma0^2.
    power = np.abs(forward_rfft_mirrored(tod.astype(np.float64))) ** 2 / (2.0 * ntod)
    freqs = rfftfreq(2 * ntod, d=1.0 / fsamp)
    # Drop the DC and Nyquist modes (the mirrored signal's Nyquist coefficient is identically zero,
    # which would otherwise make the top bin spuriously the minimum), then bin into dnu-wide bins.
    power, freqs = power[1:-1], freqs[1:-1]
    bin_idx = np.floor(freqs / dnu).astype(np.int64)
    nbin = int(bin_idx[-1]) + 1
    bin_cnt = np.bincount(bin_idx, minlength=nbin)
    bin_sum = np.bincount(bin_idx, weights=power, minlength=nbin)
    populated = bin_cnt > 0
    if not np.any(populated):
        return np.inf
    bin_power = bin_sum[populated] / bin_cnt[populated]
    bin_modes = bin_cnt[populated]
    # Never average more than a third of the spectrum: with few bins (a low sampling rate against
    # the 0.5 Hz bin width) the requested count would otherwise reach down into the 1/f rise and
    # estimate the floor from bins that are nowhere near it.
    n_lowest = max(1, min(n_floor_bins, bin_power.size // 3))
    lowest = np.argsort(bin_power)[:n_lowest]
    floor = bin_power[lowest].mean() / _lowest_bins_bias(n_lowest, bin_power.size,
                                                         bin_modes[lowest].mean())
    return float(sqrt(floor))
