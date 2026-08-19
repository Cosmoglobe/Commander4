import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(fastmath=True)
def _fill_masked_region(tod: NDArray, mask: NDArray[np.bool_], i_start: int, i_end: int) -> None:
    """ Fills a contiguous masked region [i_start, i_end] (0-indexed, inclusive) in-place with a
        linear interpolation between the mean of up to 20 valid samples near each end of the gap.
        Translated from fill_masked_region in comm_tod_mod.f90.
        Args:
            tod: TOD array, modified in-place.
            mask: Boolean mask (True = valid, False = masked).
            i_start: Index of the first masked sample.
            i_end: Index of the last masked sample.
    """
    ntod = len(tod)
    n_mean = 20
    earliest = max(i_start - (n_mean + 1), 0)
    latest   = min(i_end   + (n_mean + 1), ntod - 1)

    if i_start == 0:  # gap at start of TOD
        s = 0.0; n = 0
        for i in range(i_end, latest + 1):
            if mask[i]:
                s += tod[i]; n += 1
        mu2 = s / n if n > 0 else 0.0
        for i in range(i_start, i_end + 1):
            tod[i] = mu2
    elif i_end == ntod - 1:  # gap at end of TOD
        s = 0.0; n = 0
        for i in range(earliest, i_start + 1):
            if mask[i]:
                s += tod[i]; n += 1
        mu1 = s / n if n > 0 else 0.0
        for i in range(i_start, i_end + 1):
            tod[i] = mu1
    else:  # gap in middle of TOD
        s1 = 0.0; n1 = 0
        for i in range(earliest, i_start + 1):
            if mask[i]:
                s1 += tod[i]; n1 += 1
        s2 = 0.0; n2 = 0
        for i in range(i_end, latest + 1):
            if mask[i]:
                s2 += tod[i]; n2 += 1
        mu1 = s1 / n1 if n1 > 0 else 0.0
        mu2 = s2 / n2 if n2 > 0 else 0.0
        denom = float(i_end - i_start + 2)
        for idx in range(i_end - i_start + 1):
            tod[i_start + idx] = mu1 + (mu2 - mu1) * (idx + 1) / denom


@njit(fastmath=True)
def fill_all_masked(tod: NDArray, mask: NDArray[np.bool_], sigma0: float) -> None:
    """ Fills all masked (gap) regions in the TOD with linear interpolation and then adds
        Gaussian white noise at the gap-filled positions. Translated from fill_all_masked in
        comm_tod_mod.f90.
        Args:
            tod: TOD array, modified in-place at masked positions.
            mask: Boolean mask (True = valid, False = masked).
            sigma0: White noise standard deviation used for the gap noise realizations.
    """
    ntod   = len(tod)
    in_gap = False
    j_start = 0
    for j in range(ntod):
        if mask[j]:
            if in_gap:
                j_end  = j - 1
                _fill_masked_region(tod, mask, j_start, j_end)
                for k in range(j_start, j_end + 1):
                    tod[k] += sigma0 * np.random.randn()
                in_gap = False
        else:
            if not in_gap:
                in_gap  = True
                j_start = j
    if in_gap:  # TOD ends with a masked region
        j_end = ntod - 1
        _fill_masked_region(tod, mask, j_start, j_end)
        for k in range(j_start, j_end + 1):
            tod[k] += sigma0 * np.random.randn()

