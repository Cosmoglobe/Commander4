import numpy as np
from scipy.fft import rfft, irfft
from pixell import utils
from numpy.typing import NDArray


def corr_noise_realization_with_gaps(TOD, mask, sigma0, C_corr_inv, err_tol=1e-12, max_iter=300, rnd_seed=None):
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

    CG_solver = utils.CG(apply_LHS, b_small)
    for i in range(1, max_iter+1):
        if CG_solver.err > err_tol:
            CG_solver.step()
        else:
            break
    if i == max_iter:
        print(f"Corr noise CG failed to converge after {max_iter} iterations. Residual = {CG_solver.err} (err tol = {err_tol:.2e})")
    else:
        print(f"Corr noise CG converged after {i} iterations. Residual = {CG_solver.err} (err tol = {err_tol:.2e})")
    x_small = CG_solver.x

    correction_gaps_only = np.zeros(Ntod)
    correction_gaps_only[~mask] = x_small
    
    # Now, apply M^-1 to get the full correction term
    full_correction = apply_filter(correction_gaps_only, M_inv)
    x_final = m_inv_b + full_correction
    return x_final



def inefficient_corr_noise_realization_with_gaps(TOD: NDArray, mask: NDArray[np.bool], sigma0: float, C_corr_inv: NDArray, err_tol=1e-12, max_iter=300, rnd_seed=None):
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