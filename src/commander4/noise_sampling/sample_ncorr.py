import numpy as np
import pixell
from numpy.typing import NDArray
from commander4.utils.math_operations import forward_rfft, backward_rfft, forward_rfft_mirrored,\
    backward_rfft_mirrored


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
        has_converged = False
        nmask = b_small_scaled.size
        # Replicate convergence target used in Commander3: eps * sigma_bp * nmask
        sigma_bp = np.std(b_small_scaled) if nmask > 1 else np.std(TOD / sigma0**2)
        target_residual = err_tol * sigma_bp * nmask

        CG_solver = pixell.utils.CG(apply_LHS_scaled, b_small_scaled, x0=x0_small)
        for i in range(1, max_iter+1):
            current_residual_norm = np.linalg.norm(CG_solver.r)
            if current_residual_norm > target_residual:
                CG_solver.step()
            else:
                has_converged = True
                break
        x_small = CG_solver.x
        CG_err = current_residual_norm / (sigma_bp * nmask)
    else:
        has_converged = True
        x_small = np.zeros((0,), dtype=np.float64)
        CG_err = 0.0
        i = 0

    correction_gaps_only = np.zeros(Ntod, dtype=np.float64)
    correction_gaps_only[~mask] = x_small
    
    # Now, apply M^-1 to get the full correction term
    full_correction = apply_filter(correction_gaps_only, M_inv)
    x_final = m_inv_b + full_correction
    return x_final.astype(out_dtype), CG_err, i, has_converged



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