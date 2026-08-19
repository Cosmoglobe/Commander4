import numpy as np
from numba import get_num_threads, njit, prange

##########################
### SORTING ALGORITHMS ###
##########################

@njit
def bisect_search(arr: np.ndarray, val: int) -> int:
    """Leftmost insertion index of `val` in sorted `arr` (numpy side='left')."""
    lo, hi = 0, arr.shape[0]              # answer lives in [lo, hi)
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if arr[mid] < val:
            lo = mid + 1                  # arr[mid] too small: search right
        else:
            hi = mid                      # arr[mid] >= val: mid is a candidate
    return lo


@njit(parallel=True)
def bisect_search_many(arr: np.ndarray, vals: np.ndarray, out: np.ndarray) -> None:
    """Batched search over `vals`, parallelized across queries with prange."""
    for k in prange(vals.shape[0]):
        out[k] = bisect_search(arr, vals[k])


@njit
def gallop_search(arr: np.ndarray, val: int, prev: int) -> int:
    """Leftmost insertion index of `val`, galloping outward from hint `prev`.

    Identical result to bisect_one / numpy.searchsorted(side='left'); fast when
    the answer is near `prev`. Probes prev, prev+-1, +-2, +-4, ... until it
    brackets `val`, then bisects inside that bracket. Sentinel boundaries
    pred(-1)=False, pred(n)=True remove all global edge cases; any `prev`
    (even out of range) is accepted and only affects speed, never correctness.
    """
    n = arr.shape[0]
    p = min(max(prev, 0), n - 1)              # clamp hint into a probeable index
    if arr[p] < val:                          # answer is to the right; gallop up
        lo, step, hi = p, 1, p + 1
        while hi < n and arr[hi] < val:
            lo = hi
            step <<= 1
            hi = p + step
        if hi > n:
            hi = n
    else:                                     # answer is at/left of p; gallop down
        hi, step, lo = p, 1, p - 1
        while lo >= 0 and arr[lo] >= val:
            hi = lo
            step <<= 1
            lo = p - step
        if lo < 0:
            lo = -1
    # boundary bisection inside the bracket: pred(lo)=False (arr[lo]<val), pred(hi)=True
    while hi - lo > 1:
        mid = lo + ((hi - lo) >> 1)
        if arr[mid] < val:
            lo = mid
        else:
            hi = mid
    return hi


@njit(parallel=True)
def gallop_search_many(arr: np.ndarray, vals: np.ndarray, out: np.ndarray) -> None:
    """Galloping search, parallel across contiguous query chunks.

    Splits `vals` into get_num_threads() contiguous, order-preserving chunks
    (one per thread) and walks each chunk sequentially, carrying a per-chunk
    finger seeded at index 0. One long chunk per thread maximizes the run over
    which the finger amortizes and keeps each thread's accesses local. Order
    within a chunk must be preserved (do not sort), which is what makes the
    consecutive-answers-are-close assumption pay off.
    """
    m = vals.shape[0]
    nchunks = get_num_threads()
    for c in prange(nchunks):
        start = c * m // nchunks
        end = (c + 1) * m // nchunks
        pos = 0                               # cold seed; first query gallops up from 0
        for k in range(start, end):
            pos = gallop_search(arr, vals[k], pos)
            out[k] = pos
