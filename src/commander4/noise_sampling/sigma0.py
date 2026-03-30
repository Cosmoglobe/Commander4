""" White-noise level (sigma0) estimation routines.

    Top level functions:
    - `calc_sigma0_simple` -- direct first-difference sigma0 estimation.
    - `calc_sigma0_robust` -- same as above, but with iterative outlier-clipping
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from math import sqrt
from numba import njit


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
    assert tod.shape == mask.shape, "Input shapes don't match"
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
        (res0, mask0) -- float64 and bool arrays of length ntod0.
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
