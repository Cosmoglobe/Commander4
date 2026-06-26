import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.fft import rfftfreq

from commander4.noise_sampling.noise_psd import NoisePSD


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


def draw_local_1f(L: int, noise_model: NoisePSD, noise_params: NDArray, fsamp: float,
                  pad: int = 2) -> NDArray:
    """ Draw a short stationary realization of the *correlated* (1/f) noise component.

        Generates a length-``L`` Gaussian realization whose periodogram follows
        ``noise_model.eval_corr`` (which is zero at f=0, so the realization is mean-free up to the
        finite-grid lowest mode). To suppress the periodic-wrap artefact of a bare length-``L`` FFT
        the draw is done on a padded length ``M = pad*L`` and the central ``L`` samples are
        returned. Uses NumPy's global RNG (seeded per rank), matching ``fill_all_masked``.

        The amplitude convention matches ``_synth_corr_noise`` in the tests and the project FFT
        helpers: ``(1/M)|rfft(x)|^2`` has expectation ``eval_corr(f)``.

    Args:
        L: Number of samples to return (the gap length).
        noise_model: NoisePSD model providing ``eval_corr``.
        noise_params: Noise parameters [sigma0, fknee, alpha, ...] for the model.
        fsamp: Sampling rate (Hz).
        pad: Padding factor for the internal FFT length (>= 1).
    Returns:
        Length-``L`` float64 array: the local correlated-noise realization.
    """
    M = max(int(pad) * L, L + 2)
    freqs = rfftfreq(M, d=1.0 / fsamp)
    P = noise_model.eval_corr(freqs, noise_params)  # correlated 1/f power; 0 at f=0.
    spec = np.sqrt(M * P / 2.0) * (np.random.standard_normal(freqs.size)
                                   + 1j * np.random.standard_normal(freqs.size))
    spec[0] = 0.0  # mean-free
    if M % 2 == 0:
        spec[-1] = spec[-1].real  # real Nyquist mode
    x = np.fft.irfft(spec, n=M)
    start = (M - L) // 2
    return x[start:start + L]


def fill_gaps_local_1f(n_corr: NDArray, mask: NDArray[np.bool_], noise_model: NoisePSD,
                       noise_params: NDArray, fsamp: float, draw_1f: bool = True) -> None:
    """ Overwrite the masked regions of an n_corr realization with a *local* constrained fill.

        For each contiguous gap, the correlated noise is modeled as a local 1/f realization
        (``draw_local_1f``) to which a linear slope is added so the fill matches the surrounding
        ``n_corr`` at both gap ends -- a cheap, local alternative to the global constrained-CG
        realization. The anchor at each end is the mean of up to ``N_MEAN`` valid ``n_corr`` samples
        adjacent to the gap (matching C3's ``fill_masked_region``), which is more robust than a
        single neighbouring sample. The local realization's own endpoint trend is removed first, so
        the linear bridge alone sets the level and the fluctuation vanishes exactly at the gap
        boundaries (a Brownian-bridge-style constraint). With ``draw_1f=False`` the fill is the pure
        linear bridge (no 1/f structure).

        The gap's *white* component is not added here; downstream mapmaking adds it via
        ``fill_all_masked`` when it inpaints the residual for the operator FFT.

    Args:
        n_corr: Correlated-noise realization (1D), modified in-place at masked positions. Valid
            samples must already hold the stationary realization (they provide the bridge anchors).
        mask: Boolean validity mask (True = valid).
        noise_model: NoisePSD model providing ``eval_corr`` (used when ``draw_1f``).
        noise_params: Noise parameters [sigma0, fknee, alpha, ...].
        fsamp: Sampling rate (Hz).
        draw_1f: If True, add a local 1/f realization; if False, fill with the linear bridge only.
    """
    ntod = len(n_corr)
    n_mean = 20  # number of adjacent valid samples averaged for each bridge anchor (cf. C3).
    invalid = (~mask).astype(np.int8)
    # Vectorized run-finding: edges where validity flips, padded so leading/trailing gaps are caught.
    edges = np.flatnonzero(np.diff(np.concatenate(([0], invalid, [0]))))
    starts, ends = edges[0::2], edges[1::2] - 1  # inclusive [start, end] per gap

    def _edge_mean(lo: int, hi: int) -> float:
        """Mean of the valid n_corr samples in the half-open window [lo, hi)."""
        seg, segm = n_corr[lo:hi], mask[lo:hi]
        return float(seg[segm].mean()) if segm.any() else 0.0

    for a, b in zip(starts, ends):
        L = b - a + 1
        has_left, has_right = a > 0, b < ntod - 1
        # Gaps are separated by >=1 valid sample, so the windows always contain valid neighbours.
        left = _edge_mean(max(a - n_mean, 0), a) if has_left else None
        right = _edge_mean(b + 1, min(b + 1 + n_mean, ntod)) if has_right else None
        if has_left and has_right:
            k = np.arange(1, L + 1)
            bridge = left + (right - left) * k / (L + 1)  # linear slope anchored to both ends
        elif has_left:
            bridge = np.full(L, left)                       # trailing gap: hold the left level
        elif has_right:
            bridge = np.full(L, right)                      # leading gap: hold the right level
        else:
            bridge = np.zeros(L)                            # entire scan masked: nothing to anchor to

        if draw_1f and L > 1:
            r = draw_local_1f(L, noise_model, noise_params, fsamp)
            if has_left and has_right:
                # Remove r's own endpoint trend so the bridge sets the level and r pins to 0 at ends.
                r = r - (r[0] + (r[-1] - r[0]) * np.arange(L) / (L - 1))
            else:
                r = r - r.mean()  # one-sided/no anchor: fluctuate about the held level
            n_corr[a:b + 1] = bridge + r
        else:
            n_corr[a:b + 1] = bridge

