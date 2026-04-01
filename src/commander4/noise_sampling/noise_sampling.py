import numpy as np
from scipy.fft import rfftfreq
import pixell
from numba import njit
from numpy.typing import NDArray
from commander4.utils.math_operations import forward_rfft, backward_rfft, forward_rfft_mirrored,\
    backward_rfft_mirrored

def _inversion_sampler_1d(lnL: NDArray, grid_points: NDArray) -> float:
    """ Performs 1D inversion sampling on a grid. This involves calculating the cumulative
        log-likelihood, normalizing this to be contained in [0,1], drawing a random number in [0,1],
        and matching that to a (interpolated) grid point.
    Args:
        lnL (np.ndarray): Array of log-likelihood values at each grid point.
        grid_points (np.ndarray): The corresponding parameter values for each grid point.
    Returns:
        sample (float): A single random sample drawn from the provided distribution.
    """
    lnL -= np.max(lnL)
    L = np.exp(lnL)  # Calculate the linear likelihood.
    cdf = np.cumsum(L)  # Cumulative likelihood.
    cdf /= cdf[-1]  # Constrain it to [0,1].
    u = np.random.uniform(0, 1)
    sample = np.interp(u, cdf, grid_points)  # Find the x-value that matches the y-value we drew.
    return sample


def sample_noise_PS_params(n_corr: NDArray, sigma0: float, f_samp: float, alpha_start=-1.0,
                           freq_max=3.0, n_grid=100, n_burnin=5) -> tuple[float, float]:
    """ Function for drawing a sample of the fknee and alpha parameters for the correlated noise
        under the power spectrum data model PS = sigma0*(f/fknee)**alpha, where sigma0 is known.
        Note that this relates *only* to the correlated noise, without the "flat" white noise.
        Args:
            n_corr (np.array): 1D array of the correlated noise time series.
            sigma0 (float): White noise level of full data. Since this data model does not have a
                white noise floor, this essentially just scales the resulting fnee value.
            f_samp (float): The sampling rate of the data (n_corr), in Hertz.
            alpha_start (float): Starting guess for alpha in Gibbs sampler.
            freq_max (float): Maximum frequency to consider for the PS (all 1/f information is
                              contained at low freqs).
            n_grid (int): Number of grid points used for the inverse sampling.
            n_burnin (int): Number of burn-in samples before drawing the "actual"
                            sample of fknee and alpha.
        Returns:
            fknee_sample (float): A single sample of fknee.
            alpha_sample (float): A single sample of alpha.
    """
    Ntod = len(n_corr)
    freqs = rfftfreq(Ntod, 1.0/f_samp)[1:]  # [1:] to Exclude freq=0 mode (same on line below).
    n_corr_power = (1.0 / Ntod) * np.abs(forward_rfft(n_corr))[1:]**2
    Nrfft = freqs.size
    bins = pixell.utils.expbin(Nrfft, nbin=100, nmin=1)
    binned_freqs = pixell.utils.bin_data(bins, freqs)
    binned_n_corr_power = pixell.utils.bin_data(bins, n_corr_power)
    freq_mask = (binned_freqs <= freq_max)
    log_freqs_masked = np.log(binned_freqs[freq_mask])
    n_corr_power_masked = binned_n_corr_power[freq_mask]
    log_n_corr_power = np.log(n_corr_power_masked)
    # Set up a grid of possible fknee values: from 2 times the min frequency to the max frequency.
    fknee_grid = np.logspace(np.log10(freqs[0] * 2), np.log10(freq_max), n_grid)
    log_fknee_grid = np.log(fknee_grid)

    alpha_grid = np.linspace(-2.5, -0.25, n_grid)
    alpha_current = alpha_start

    log_sigma0_sq = np.log(sigma0**2)
    # --- Main Gibbs Loop ---
    for _ in range(n_burnin + 1):
        # 1. Sample f_knee, given a fixed alpha
        log_N_corr_ps = log_sigma0_sq + alpha_current *\
                (log_freqs_masked[:, np.newaxis] - log_fknee_grid)
        residual = log_n_corr_power[:, np.newaxis] - log_N_corr_ps
        log_L_fknee = np.sum(residual - np.exp(residual), axis=0)
        # A faster but slightly less statistically robust way of calculating the likelihood
        # (~30% speedup of code):
        # log_L_fknee = -0.5 * np.sum((log_n_corr_power[:, np.newaxis] - log_N_corr_ps)**2, axis=0)
        fknee_sample = float(_inversion_sampler_1d(log_L_fknee, fknee_grid))
        
        log_fknee_sample = np.log(fknee_sample)
        log_N_corr_ps = log_sigma0_sq + alpha_grid\
            * (log_freqs_masked[:, np.newaxis] - log_fknee_sample)
        residual = log_n_corr_power[:, np.newaxis] - log_N_corr_ps
        log_L_alpha = np.sum(residual - np.exp(residual), axis=0)
        # log_L_alpha = -0.5 * np.sum((log_n_corr_power[:, np.newaxis] - log_N_corr_ps)**2, axis=0)
        alpha_current = float(_inversion_sampler_1d(log_L_alpha, alpha_grid))
    return fknee_sample, alpha_current


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

