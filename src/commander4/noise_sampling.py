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


def corr_noise_realization_with_gaps(TOD: NDArray, mask: NDArray[np.bool_], sigma0: float,
                                     C_corr_inv: NDArray, err_tol=1e-6, max_iter=100,
                                     rnd_seed=None) -> NDArray:
    """ Draws a correlated noise realization given a TOD (with gaps/masked samples) a correlated
        noise power spectrum. Requires solving a CG, which this function solves in a very efficient
        way by splitting up the problem such that the CG only has to be performed on the missing
        data, not the full TOD (see arXiv:2011.06024).
        Gaps in the TOD should be pre-filled (e.g. with fill_all_masked) before calling this
        function: the filled values at gap positions do not affect the RHS (they are zeroed by
        C_wn=inf), but are used as a warm start for the CG, matching the Fortran get_ncorr_sm_cg.
        Args:
            TOD (np.array): The 1D time ordered data. Gap positions should be pre-filled before
                            passing (e.g. via fill_all_masked), as they seed the CG warm start.
            mask (np.array): A boolean array where False indices indicates missing or masked data.
            sigma0 (float): The stationary white noise level of the data.
            C_corr_inv (np.array): The inverse covariance of the TOD we want to sample.
            err_tol (float): The error tolerance for the CG search.
            max_iter (int): Maximum iterations used by the CG search.
            rnd_seed (int): Seed for drawing random numbers during the realization.
        Returns:
            x_final (np.array): The TOD realization of the correlated noise.
    """
    def apply_filter(vec, Fourier_filter):
        return backward_rfft_mirrored(forward_rfft_mirrored(vec) * Fourier_filter, ntod=len(vec))

    def apply_LHS_scaled(x_small):
        u_x = np.zeros(Ntod, dtype=x_small.dtype)
        u_x[~mask] = x_small
        m_inv_u_x = apply_filter(u_x, M_inv_scaled)
        return x_small - m_inv_u_x[~mask]

    out_dtype = TOD.dtype
    Ntod = TOD.shape[0]
    # All CG work is done in float64 to avoid residual drift and false convergence.
    M_inv = 1.0 / ( (1/sigma0**2) + C_corr_inv)  # The stationary LHS operator (float64).
    if rnd_seed is not None:
        np.random.seed(rnd_seed)
    omega_2 = np.random.randn(Ntod)
    omega_3 = np.random.randn(Ntod)

    C_wn_timedomain = np.ones(Ntod, dtype=np.float64)*sigma0**2
    C_wn_timedomain[~mask] = np.inf
    b_full = TOD.astype(np.float64)/C_wn_timedomain + omega_2/np.sqrt(C_wn_timedomain)\
           + apply_filter(omega_3, np.sqrt(C_corr_inv))
    m_inv_b = apply_filter(b_full, M_inv)
    # Then, apply U^T to extract the values at the flagged locations.
    b_small = m_inv_b[~mask]

    # Warm-start the CG at gap positions using the (pre-filled) TOD, matching the Fortran
    # get_ncorr_sm_cg initial vector: x(u) = (d_prime/sigma0 - mean) / 10 + mean.
    # In physical units (no sigma0 normalization) this becomes (TOD - mean) / 10 + mean.
    if (~mask).any():
        x0_full  = TOD.astype(np.float64) / sigma0**2
        x0_full  = (x0_full - np.mean(x0_full)) / 10.0 + np.mean(x0_full)
        x0_small = x0_full[~mask]
    else:
        x0_small = None

    # Normalize both RHS and LHS into sigma-units, such that the system becomes unitless.
    b_small_scaled = b_small / sigma0**2
    M_inv_scaled = M_inv / sigma0**2

    if b_small_scaled.size > 0:
        # Replicate convergence target used in Commander3: eps * sigma_bp * nmask
        nmask = b_small_scaled.size
        sigma_bp = np.std(b_small_scaled) if nmask > 1 else np.std(TOD / sigma0**2)
        target_residual = err_tol * sigma_bp * nmask

        CG_solver = pixell.utils.CG(apply_LHS_scaled, b_small_scaled, x0=x0_small)
        for i in range(1, max_iter+1):
            current_residual_norm = np.linalg.norm(CG_solver.r)
            if current_residual_norm > target_residual:
                CG_solver.step()
            else:
                break
        x_small = CG_solver.x
        CG_err = current_residual_norm
    else:
        x_small = np.zeros((0,), dtype=np.float64)
        CG_err = 0.0
        i = 0

    correction_gaps_only = np.zeros(Ntod, dtype=np.float64)
    correction_gaps_only[~mask] = x_small
    
    # Now, apply M^-1 to get the full correction term
    full_correction = apply_filter(correction_gaps_only, M_inv)
    x_final = m_inv_b + full_correction
    return x_final.astype(out_dtype), CG_err, i



def inefficient_corr_noise_realization_with_gaps(TOD: NDArray, mask: NDArray[np.bool_],
                                                 sigma0: float, C_corr_inv: NDArray, err_tol=1e-12,
                                                 max_iter=300, rnd_seed=None) -> NDArray:
    """ A simpler and less efficient implementation of the function
        'corr_noise_realization_with_gaps'. This function performs the 'full' straight-forward CG
        search. Should only be used to test the proper function. Note also that the error tolerance
        of this function has a different interpretation and should be set much lower.
    """
    def apply_filter(vec, Fourier_filter):
        return backward_rfft_mirrored(forward_rfft_mirrored(vec) * Fourier_filter, ntod=len(vec))

    def apply_LHS(x_full):
        return x_full/C_wn_timedomain + apply_filter(x_full, C_corr_inv)
    Ntod = TOD.shape[0]

    C_wn_timedomain = np.ones(Ntod)*sigma0**2
    C_wn_timedomain[~mask] = np.inf

    if rnd_seed is not None:
        np.random.seed(rnd_seed)
    omega_2 = np.random.randn(Ntod)
    omega_3 = np.random.randn(Ntod)

    b_full = TOD/C_wn_timedomain + omega_2/np.sqrt(C_wn_timedomain)\
           + apply_filter(omega_3, np.sqrt(C_corr_inv))

    CG_solver = pixell.utils.CG(apply_LHS, b_full)
    for i in range(1, max_iter+1):
        if CG_solver.err > err_tol:
            CG_solver.step()
        else:
            break
    x_full = CG_solver.x
    return x_full