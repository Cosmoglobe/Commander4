import logging
import numpy as np
import pixell
from scipy.fft import rfftfreq
from mpi4py import MPI
from numpy.typing import NDArray
from pixell.bunch import Bunch
from commander4.utils.math_operations import forward_rfft, backward_rfft, forward_rfft_mirrored,\
    backward_rfft_mirrored
from commander4.noise_sampling.noise_sampling import fill_all_masked, fill_gaps_local_1f
from commander4.noise_sampling.noise_psd import NoisePSD
from commander4.noise_sampling.sigma0 import calc_sigma0_robust, calc_sigma0_binned_psd

SIGMA0_METHODS = ("pairwise", "binned_psd")
GAP_FILL_METHODS = ("proper_cg", "local_1f", "linear")

from commander4.logging.performance_logger import benchmark, bench_summary, start_bench,\
                                            stop_bench, log_memory, increment_count, bench_reset

logger = logging.getLogger(__name__)


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
        start_bench("FFT")
        res = backward_rfft_mirrored(forward_rfft_mirrored(vec) * Fourier_filter, ntod=len(vec))
        stop_bench("FFT")
        return res

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


def _estimate_sigma0(residual: NDArray, n_corr: NDArray, mask: NDArray[np.bool_],
                     dec: int) -> float:
    """Robust white-noise level from the fully-cleaned residual (sky- and n_corr-subtracted)."""
    return float(calc_sigma0_robust(residual - n_corr, mask, down_factor=int(dec)))


def sample_correlated_noise(tod: NDArray, mask: NDArray[np.bool_], noise_params: NDArray,
                            noise_model: NoisePSD, fsamp: float, *, cg_err_tol: float,
                            cg_max_iter: int, sample_params: bool, sample_sigma0: bool = True,
                            sigma0_method: str = "pairwise", gap_fill_method: str = "proper_cg",
                            nomono: bool = False, onlymono: bool = False, sigma0_dec: int = 1,
                            psd_fit_nu_min: float = 0.0, psd_fit_nu_max: float = np.inf,
                            psd_bin: bool = False) -> Bunch:
    """ Draw a correlated-noise realization for one detector-scan and optionally resample sigma0 and
        the noise-model parameters. The inverse correlated-noise spectrum is supplied by
        *noise_model*, so any NoisePSD subclass (parameters of any length) can be plugged in.

        Masked gaps are inpainted in-place in *tod* (linear slope + white noise). The filled values
        seed the CG warm-start and, when the masked CG is skipped or rejected, define the stationary
        (full-mask) Wiener fallback solution.

    Args:
        tod: Sky-subtracted TOD for one detector-scan. Modified in-place by gap inpainting.
        mask: Boolean validity mask (True = valid sample).
        noise_params: Current noise parameters; ``noise_params[0]`` is sigma0 (used for the CG).
        noise_model: NoisePSD model providing ``compute_inv_corr_spectrum`` and ``sample_params``.
        fsamp: Sampling rate of the TOD (Hz).
        cg_err_tol: CG convergence tolerance (relative residual).
        cg_max_iter: Maximum CG iterations (``gap_fill_method='proper_cg'`` only). If 0, the
            masked-gap CG is skipped and the stationary (full-mask) Wiener solution is returned.
        sample_params: Whether to resample the noise-model parameters (fknee, alpha, ...).
        sample_sigma0: Whether to re-estimate sigma0 (noise_params[0]).
        sigma0_method: White-noise estimator: ``'pairwise'`` (first-difference of the
            n_corr-subtracted residual, after the draw) or ``'binned_psd'`` (bottom of the binned
            PSD of the signal-subtracted residual, *before* the draw -- Commander3 ordering, so the
            refreshed sigma0 feeds C_corr_inv and the CG).
        gap_fill_method: How the masked region of n_corr is realized: ``'proper_cg'`` (global
            constrained-CG realization -- the existing/default behavior), ``'local_1f'`` (cheap
            local 1/f realization linearly bridged to the gap ends), or ``'linear'`` (linear bridge
            only). For the latter two, valid samples get the stationary full-mask realization and
            only the gaps are overwritten locally.
        nomono: If True, project the per-scan monopole out of the residual and of n_corr (Fortran
            ``nomono``); otherwise the DC is left in n_corr.
        onlymono: If True, model the correlated noise as only the per-scan offset, skipping the CG
            and parameter sampling (Fortran ``onlymono``). Takes precedence over ``nomono``.
        sigma0_dec: Decimation (block-average) factor for the pairwise sigma0 estimator.
        psd_fit_nu_min, psd_fit_nu_max: Frequency range (Hz) for PSD-parameter fitting.
        psd_bin: Whether the PSD-parameter fit uses a (mode-count-weighted) binned periodogram.
    Returns:
        Bunch with fields ``n_corr`` (realization), ``noise_params`` (with updated sigma0 and/or
        parameters), ``residual`` (CG residual; 0 when no masked CG ran), ``niter`` (CG iterations),
        ``converged`` (bool), and ``high_var`` (variance sanity check failed).
    """
    if sigma0_method not in SIGMA0_METHODS:
        raise ValueError(f"sigma0_method must be one of {SIGMA0_METHODS}, got {sigma0_method!r}.")
    if gap_fill_method not in GAP_FILL_METHODS:
        raise ValueError(f"gap_fill_method must be one of {GAP_FILL_METHODS}, got "
                         f"{gap_fill_method!r}.")
    noise_params = np.array(noise_params, dtype=np.float64, copy=True)
    Ntod = tod.shape[0]

    # Binned-PSD sigma0 is a "before n_corr" estimate (Commander3 ordering): estimate it up front on
    # the signal-subtracted residual so C_corr_inv and the n_corr draw use the refreshed value.
    if sample_sigma0 and sigma0_method == "binned_psd":
        noise_params[0] = calc_sigma0_binned_psd(tod, mask, fsamp)
    sigma0 = float(noise_params[0])

    # "Only monopole" mode: model the correlated noise as just the per-scan offset.
    if onlymono:
        mono = float(np.mean(tod[mask])) if mask.any() else 0.0
        n_corr = np.full(Ntod, mono, dtype=tod.dtype)
        if sample_sigma0 and sigma0_method == "pairwise":
            noise_params[0] = _estimate_sigma0(tod, n_corr, mask, sigma0_dec)
        return Bunch(n_corr=n_corr, noise_params=noise_params, residual=0.0, niter=0,
                     converged=True, high_var=False)

    freq = rfftfreq(2 * Ntod, d=1.0/fsamp)  # Mirrored-FFT grid: nfft=2*Ntod -> length Ntod+1.
    C_corr_inv = noise_model.compute_inv_corr_spectrum(freq, noise_params)
    # Inpaint masked regions: seeds the CG warm-start and feeds the stationary fallback solve.
    fill_all_masked(tod, mask, sigma0)
    if nomono and mask.any():
        tod = tod - np.mean(tod[mask])  # Solve for a mean-zero correlated noise component.

    high_var = False
    if gap_fill_method == "proper_cg":
        if cg_max_iter == 0:
            # User requested no CG steps: use the stationary (full-mask) Wiener solution directly.
            n_corr, residual, niter, converged = corr_noise_realization_with_gaps(
                tod, np.ones_like(mask), sigma0, C_corr_inv)
        else:
            n_corr, residual, niter, converged = corr_noise_realization_with_gaps(
                tod, mask, sigma0, C_corr_inv, err_tol=cg_err_tol, max_iter=cg_max_iter)
            # Sanity check: the residual (data minus n_corr) should not carry more power than data.
            resid = (tod - n_corr) * mask
            high_var = bool(np.dot(resid, resid) > np.dot(tod*mask, tod*mask))
            if high_var or not converged:
                # Fall back to the stationary solution that ignores the gaps.
                n_corr, _, _, _ = corr_noise_realization_with_gaps(
                    tod, np.ones_like(mask), sigma0, C_corr_inv)
    else:
        # Middle ground: stationary realization for the valid samples, then overwrite each gap with
        # a local 1/f (or pure-linear) bridge -- no global masked CG.
        n_corr, residual, niter, converged = corr_noise_realization_with_gaps(
            tod, np.ones_like(mask), sigma0, C_corr_inv)
        if mask.any() and not mask.all():
            fill_gaps_local_1f(n_corr, mask, noise_model, noise_params, fsamp,
                               draw_1f=(gap_fill_method == "local_1f"))

    if nomono and mask.any():
        n_corr = n_corr - np.mean(n_corr[mask])

    # Re-estimate sigma0 (pairwise) from the fully-cleaned residual (after subtracting n_corr).
    if sample_sigma0 and sigma0_method == "pairwise":
        noise_params[0] = _estimate_sigma0(tod, n_corr, mask, sigma0_dec)

    # Fit the PSD parameters to the residual periodogram (full model), inpainting masked samples
    # with the n_corr realization plus white noise (Commander3 sample_noise_psd convention).
    if sample_params:
        residual_tod = tod.copy()
        ngap = int(np.count_nonzero(~mask))
        if ngap > 0:
            residual_tod[~mask] = n_corr[~mask] + float(noise_params[0]) * np.random.randn(ngap)
        noise_params = noise_model.sample_params(residual_tod, noise_params, fsamp,
                                                 nu_min=psd_fit_nu_min, nu_max=psd_fit_nu_max,
                                                 bin_psd=psd_bin)

    return Bunch(n_corr=n_corr, noise_params=noise_params, residual=float(residual),
                 niter=int(niter), converged=bool(converged), high_var=high_var)


def _log_distribution(nu: float, label: str, values: NDArray, fmt: str = ".4f") -> None:
    """Log the min / 1st-pct / mean / 99th-pct / max of *values* for one band."""
    values = np.asarray(values, dtype=np.float64)
    logger.info(f"{nu}GHz: {label} {np.nanmin(values):{fmt}} {np.nanpercentile(values, 1):{fmt}} "
                f"{np.nanmean(values):{fmt}} {np.nanpercentile(values, 99):{fmt}} "
                f"{np.nanmax(values):{fmt}}")


def log_corr_noise_stats(band_comm: MPI.Comm, nu: float, noise_model: NoisePSD,
                         sampled_params: list[NDArray], residuals: list[float],
                         niters: list[int], n_failed_conv: int, n_high_var: int,
                         worst_residual: float, n_local_scans: int) -> None:
    """ Reduce per-detector-scan correlated-noise diagnostics across the band communicator and log
        a summary on the band master. Parameter distributions are reported per model parameter name
        (skipping sigma0), so the summary adapts to any NoisePSD model.

    Args:
        band_comm: Band-level MPI communicator.
        nu: Band centre frequency (GHz), for labelling.
        noise_model: The band's NoisePSD model (used for ``param_names``).
        sampled_params: Locally sampled ``noise_params`` arrays (empty if params were not sampled).
        residuals: Local CG residuals (0 entries are excluded from the summary).
        niters: Local CG iteration counts.
        n_failed_conv: Local count of non-converged masked-CG solves.
        n_high_var: Local count of variance-sanity-check failures.
        worst_residual: Local worst CG residual.
        n_local_scans: Local number of detector-scans (for the "out of N" message).
    """
    n_failed_conv = band_comm.reduce(n_failed_conv, op=MPI.SUM)
    n_high_var = band_comm.reduce(n_high_var, op=MPI.SUM)
    worst_residual = band_comm.reduce(worst_residual, op=MPI.MAX)
    n_total = band_comm.reduce(n_local_scans, op=MPI.SUM)
    residuals = band_comm.gather(residuals)
    niters = band_comm.gather(niters)
    sampled_params = band_comm.gather(sampled_params)
    if band_comm.Get_rank() != 0:
        return

    logger.debug(f"Worst corr-noise sampling residual (band {nu}GHz) = {worst_residual:.2e}.")
    if n_failed_conv > 0:
        logger.warning(f"Band {nu}GHz failed noise CG for {n_failed_conv} out of {n_total} scans. "
                       f"Worst residual = {worst_residual:.3e}.")
    if n_high_var > 0:
        logger.warning(f"Band {nu}GHz failed variance sanity check for {n_high_var} out of "
                       f"{n_total} scans.")

    residuals = np.concatenate([np.asarray(r, dtype=np.float64) for r in residuals])
    residuals = residuals[residuals != 0]
    if residuals.size == 0:
        residuals = np.array([0.0])
    niters = np.concatenate([np.asarray(n, dtype=np.float64) for n in niters])
    if niters.size == 0:
        niters = np.array([0.0])
    _log_distribution(nu, "residuals", residuals, fmt=".2e")
    _log_distribution(nu, "iterations", niters, fmt=".4f")

    # Per-parameter distributions (model-agnostic; sigma0 at index 0 is reported elsewhere).
    flat = [p for sub in sampled_params for p in sub]
    if flat:
        arr = np.asarray(flat, dtype=np.float64)  # shape (n_sampled_scans, npar)
        for j in range(1, arr.shape[1]):
            name = noise_model.param_names[j] if j < len(noise_model.param_names) else f"p{j}"
            _log_distribution(nu, name, arr[:, j], fmt=".4f")