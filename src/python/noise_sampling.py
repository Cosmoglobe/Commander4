import numpy as np
from scipy.fft import rfft, irfft, rfftfreq
from pixell import utils
from numpy.typing import NDArray


def _inversion_sampler_1d(lnL, grid_points):
    """ Performs 1D inversion sampling on a grid. This involves calculating the cummulative log-likelihood, normalizing
        this to be contained in [0,1], drawing a random number in [0,1], and matching that to a (interpolated) grid point.
        Args:
            lnL (np.ndarray): Array of log-likelihood values at each grid point.
            grid_points (np.ndarray): The corresponding parameter values for each grid point.
        Returns:
            sample (float): A single random sample drawn from the provided distribution.
    """
    lnL -= np.max(lnL)
    L = np.exp(lnL)  # Calculate the linear likelihood.
    cdf = np.cumsum(L)  # Cummulative likelihood.
    cdf /= cdf[-1]  # Constrain it to [0,1].
    u = np.random.uniform(0, 1)
    sample = np.interp(u, cdf, grid_points)  # Find the x-value that matches the y-value we drew.
    return sample


def sample_noise_PS_params(n_corr, sigma0, f_samp, alpha_start, freq_max=3.0, n_grid=100, n_burnin=5):
    """ Function for drawing a sample of the fknee and alpha parameters for the correlated noise under the
        power spectrum data model PS = sigma0*(f/fknee)**alpha, where sigma0 is known.
        Note that this relates *only* to the correlated noise, without the "flat" white noise.
        Args:
            n_corr (np.array): 1D array of the correlated noise time series.
            sigma0 (float): White noise level of full data. Since this data model does not have a
                white noise floor, this essentially just scales the resulting fnee value.
            f_samp (float): The sampling rate of the data (n_corr), in Hertz.
            freq_max (float): Maximum frequency to consider for the PS (all 1/f information is contained at low freqs).
            n_grid (int): Number of grid points used for the inverse sampling.
            n_burnin (int): Number of burn-in samples before drawing the "actual" sample of fknee and alpha.
        Returns:
            fknee_sample (float): A single sample of fknee.
            alpha_sample (float): A single sample of alpha.
    """
    Ntod = len(n_corr)
    freqs = rfftfreq(Ntod, 1.0/f_samp)[1:]  # [1:] to Exclude freq=0 mode (same on line below).
    n_corr_power = (1.0 / Ntod) * np.abs(rfft(n_corr))[1:]**2
    Nrfft = freqs.size
    bins = utils.expbin(Nrfft, nbin=100, nmin=1)
    binned_freqs = utils.bin_data(bins, freqs)
    binned_n_corr_power = utils.bin_data(bins, n_corr_power)
    freq_mask = (binned_freqs <= freq_max)
    log_freqs_masked = np.log(binned_freqs[freq_mask])
    n_corr_power_masked = binned_n_corr_power[freq_mask]
    log_n_corr_power = np.log(n_corr_power_masked)
    # Set up a grid of possible fknee values: from 2 times the minimum frequency to the maximum frequency.
    fknee_grid = np.logspace(np.log10(freqs[0] * 2), np.log10(freq_max), n_grid)
    log_fknee_grid = np.log(fknee_grid)

    alpha_grid = np.linspace(-2.5, -0.25, n_grid)
    alpha_current = alpha_start

    log_sigma0_sq = np.log(sigma0**2)
    # --- Main Gibbs Loop ---
    for _ in range(n_burnin + 1):
        # 1. Sample f_knee, given a fixed alpha
        log_N_corr_ps = log_sigma0_sq + alpha_current * (log_freqs_masked[:, np.newaxis] - log_fknee_grid)
        residual = log_n_corr_power[:, np.newaxis] - log_N_corr_ps
        log_L_fknee = np.sum(residual - np.exp(residual), axis=0)
        # A faster but slightly less statistically robust way of calculating the likelihood (~30% speedup of code):
        # log_L_fknee = -0.5 * np.sum((log_n_corr_power[:, np.newaxis] - log_N_corr_ps)**2, axis=0)
        fknee_sample = _inversion_sampler_1d(log_L_fknee, fknee_grid)
        
        log_fknee_sample = np.log(fknee_sample)
        log_N_corr_ps = log_sigma0_sq + alpha_grid * (log_freqs_masked[:, np.newaxis] - log_fknee_sample)
        residual = log_n_corr_power[:, np.newaxis] - log_N_corr_ps
        log_L_alpha = np.sum(residual - np.exp(residual), axis=0)
        # log_L_alpha = -0.5 * np.sum((log_n_corr_power[:, np.newaxis] - log_N_corr_ps)**2, axis=0)
        alpha_current = _inversion_sampler_1d(log_L_alpha, alpha_grid)
    return fknee_sample, alpha_current



def corr_noise_realization_with_gaps(TOD, mask, sigma0, C_corr_inv, err_tol=1e-12, max_iter=100, rnd_seed=None):
    """ Draws a correlated noise realization given a TOD (with gaps/masked samples) a correlated noise power spectrum.
        Requires solving a CG, which this function solves in a very efficient way by splitting up the problem
        such that the CG only has to be performed on the missing data, not the full TOD (see arXiv:2011.06024).
        Args:
            TOD (np.array): The 1D time ordered data potentially containing gaps.
            mask (np.array): A bollean array where False indices indicates missing or masked data.
            sigma0 (float): The stationary white noise level of the data.
            C_corr_inv (np.array): The inverse covariance of the correlated signal we want to sample.
            err_tol (float): The error tolerance for the CG search.
            max_iter (int): Maximum iterations used by the CG search.
            rnd_seed (int): Seed for drawing random numbers during the realization.
        Returns:
            x_final (np.array): The TOD realization of the correlated noise.
    """
    def apply_filter(vec, Fourier_filter):
        return irfft(rfft(vec) * Fourier_filter, n=len(vec))

    def apply_LHS(x_small):
        term1 = sigma0**2 * x_small
        u_x = np.zeros(Ntod)
        u_x[~mask] = x_small
        m_inv_u_x = apply_filter(u_x, M_inv)
        u_t_m_inv_u_x = m_inv_u_x[~mask]
        term2 = u_t_m_inv_u_x
        return term1 - term2

    Ntod = TOD.shape[0]
    M_inv = 1.0 / ( (1/sigma0**2) + C_corr_inv)  # The stationary LHS operator.
    if rnd_seed is not None:
        np.random.seed(rnd_seed)
    omega_2 = np.random.randn(Ntod)
    omega_3 = np.random.randn(Ntod)

    C_wn_timedomain = np.ones(Ntod)*sigma0**2
    C_wn_timedomain[~mask] = np.inf
    b_full = TOD/C_wn_timedomain + omega_2/np.sqrt(C_wn_timedomain) + apply_filter(omega_3, np.sqrt(C_corr_inv))

    m_inv_b = apply_filter(b_full, M_inv)
    # Then, apply U^T to extract the values at the flagged locations.
    b_small = m_inv_b[~mask]

    if b_small.size > 0:
        CG_solver = utils.CG(apply_LHS, b_small)
        for i in range(1, max_iter+1):
            if CG_solver.err > err_tol:
                CG_solver.step()
            else:
                break
        if i == max_iter:
            print(f"Corr noise CG failed to converge after {max_iter} iterations. Residual = {CG_solver.err} (err tol = {err_tol:.2e})")
        else:
            # print(f"Corr noise CG converged after {i} iterations. Residual = {CG_solver.err} (err tol = {err_tol:.2e})")
            pass
        x_small = CG_solver.x
    else:
        x_small = np.zeros((0,))

    correction_gaps_only = np.zeros(Ntod)
    correction_gaps_only[~mask] = x_small
    
    # Now, apply M^-1 to get the full correction term
    full_correction = apply_filter(correction_gaps_only, M_inv)
    x_final = m_inv_b + full_correction
    return x_final



def inefficient_corr_noise_realization_with_gaps(TOD: NDArray, mask: NDArray[np.bool_], sigma0: float, C_corr_inv: NDArray, err_tol=1e-12, max_iter=300, rnd_seed=None):
    """ A simpler and less efficient implementation of the function 'corr_noise_realization_with_gaps'.
        This function performs the 'full' straight-forward CG search. Should only be used to test the proper function.
    """
    def apply_filter(vec, Fourier_filter):
        return irfft(rfft(vec) * Fourier_filter, n=len(vec))

    def apply_LHS(x_full):
        return x_full/C_wn_timedomain + apply_filter(x_full, C_corr_inv)
    Ntod = TOD.shape[0]

    C_wn_timedomain = np.ones(Ntod)*sigma0**2
    C_wn_timedomain[~mask] = np.inf

    if rnd_seed is not None:
        np.random.seed(rnd_seed)
    omega_2 = np.random.randn(Ntod)
    omega_3 = np.random.randn(Ntod)

    b_full = TOD/C_wn_timedomain + omega_2/np.sqrt(C_wn_timedomain) + apply_filter(omega_3, np.sqrt(C_corr_inv))

    CG_solver = utils.CG(apply_LHS, b_full)
    for i in range(1, max_iter+1):
        if CG_solver.err > err_tol:
            CG_solver.step()
        else:
            break
    if i == max_iter:
        print(f"Corr noise CG failed to converge after {max_iter} iterations. Residual = {CG_solver.err} (err tol = {err_tol:.2e})")
    else:
        print(f"Corr noise CG converged after {i} iterations. Residual = {CG_solver.err} (err tol = {err_tol:.2e})")
    x_full = CG_solver.x
    return x_full