"""Gain sampling: the absolute, relative and temporal-variation terms.

The three terms are sampled in that order once per Gibbs iteration, each against a configurable
calibrator (orbital dipole, sky, or sky without the dipole). Absolute gain is a single number per
band, relative gain one per detector under a zero-sum constraint, and the temporal term one per
detector-scan drawn through a Wiener filter. [C3: comm_gain_mod.f90]
"""
import logging

import numpy as np
import pixell
from mpi4py import MPI
from numpy.typing import NDArray
from scipy.fft import rfftfreq

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.tod_samples import TODSamples
from commander4.diagnostics.performance import benchmark, log_memory, start_bench, stop_bench
from commander4.math_utils.fft import forward_rfft, backward_rfft
from commander4.tod.config import GainConfig
from commander4.tod.view import TODView

logger = logging.getLogger(__name__)


# Which signal a gain term is calibrated against, i.e. what the calibration residual is reduced
# to. `TODView.get_calib_tod` owns the actual signal bookkeeping; the three choices are:
#   orbital_dipole: the CMB dipole induced by the observer's motion, alone. It is known a priori
#                   from the spacecraft velocity and the CMB monopole temperature, so it does not
#                   depend on the sky model. That makes it the only *absolute* calibrator,
#                   and the natural default for the absolute gain (the average across detectors).
#   sky:            the entire modelled signal, static sky and orbital dipole. The default for
#                   the relative and temporal terms: those only have to track gain *differences*
#                   between detectors or scans, so they can use every bit of signal available.
#   sky_no_dipole:  the static sky model from component separation, with the dipole left out.

# Above this frequency the orbital dipole is faint compared with the foregrounds.
_ORBITAL_DIPOLE_MAX_FREQ_GHZ = 400.0


def _warn_about_dipole_calibration(band_comm: MPI.Comm, experiment_data: DetectorGroupTOD,
                                   gain_cfg: GainConfig, step_name: str) -> None:
    """Warn on the band master if a band above 400 GHz is calibrated against the orbital dipole."""
    if gain_cfg.calibrate_against == "orbital_dipole"\
            and experiment_data.nu > _ORBITAL_DIPOLE_MAX_FREQ_GHZ and band_comm.Get_rank() == 0:
        logger.warning(f"{step_name} for band {experiment_data.band_name} ({experiment_data.nu} "
                       "GHz) is calibrated against the orbital dipole, but above "
                       f"{_ORBITAL_DIPOLE_MAX_FREQ_GHZ:.0f} GHz the dipole is faint compared with "
                       "the foregrounds and makes a poor calibrator; consider 'sky', or an "
                       "externally determined gain for this band.")


def _solve_relative_gain_system(s_weights: NDArray, r_weights: NDArray, prev_rel_gain: NDArray,
                                rng=None) -> NDArray:
    """ Draw relative-gain deviations Delta g_i from the BP7 Sec. 3.4 constrained Gaussian.

        Solves the bordered linear system enforcing ``sum(Delta g_i) = 0`` over the *active*
        detectors only, those with nonzero calibration weight ``s_weights``. A detector rejected
        on every scan (or with a vanishing calibrator) has ``s_weights == 0``; it would contribute
        an all-zero row/column and, if two or more are present, make the matrix singular. Such
        detectors are excluded from the solve (shrinking the system to the active set) and held at
        their current relative gain.

    Args:
        s_weights (NDArray): Per-detector ``sum_scans s^T N^-1 s`` (calibration weight), shape (ndet,).
        r_weights (NDArray): Per-detector ``sum_scans r^T N^-1 s`` (residual projection), shape (ndet,).
        prev_rel_gain (NDArray): Current relative gains, kept for excluded detectors, shape (ndet,).
        rng: Optional NumPy random generator for the fluctuation term (defaults to ``np.random``).

    Returns:
        NDArray: Full-length (ndet,) float32 relative-gain vector with the active entries resampled.

    Raises:
        np.linalg.LinAlgError: If the reduced system is singular (left for the caller to handle).
    """
    rng = np.random if rng is None else rng
    out = np.array(prev_rel_gain, dtype=np.float32)
    idx = np.flatnonzero(np.asarray(s_weights) > 0.0)
    n = idx.size
    if n == 0:
        return out
    d = np.asarray(s_weights)[idx].astype(np.float64)
    r = np.asarray(r_weights)[idx].astype(np.float64)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = np.diag(d)
    A[:n, n] = 0.5     # Lagrange-multiplier column enforcing the zero-sum constraint.
    A[n, :n] = 1.0     # Constraint row: sum of active Delta g_i = 0.
    b = np.zeros(n + 1)
    b[:n] = r + np.sqrt(d) * rng.standard_normal(n)
    solution = np.linalg.solve(A, b)   # Raises LinAlgError if singular.
    out[idx] = solution[:n].astype(np.float32)
    return out


def sample_absolute_gain(band_comm: MPI.Comm, experiment_data: DetectorGroupTOD,
                         tod_samples: TODSamples, det_compsep_map: NDArray,
                         gain_cfg: GainConfig, iteration: int) -> TODSamples:
    """ Draw a realization of the absolute gain term, g0, which is constant across all
        detectors and all scans within a band, using the calibrator selected by ``gain_cfg``.
    Args:
        band_comm (MPI.Comm): The band-level MPI communicator.
        experiment_data (DetectorGroupTOD): The object holding all the scan data.
        tod_samples (TODSamples): Current sampled TOD parameters (updated in-place with g0).
        det_compsep_map (NDArray): The component-separation sky map for the detector.
        gain_cfg: Validated absolute-gain settings.
        iteration: Current Gibbs iteration for log context.
    Returns:
        tod_samples (TODSamples): Updated TOD samples with the new g0 estimate.
    """
    _warn_about_dipole_calibration(band_comm, experiment_data, gain_cfg, "abs_gain")
    start_bench("eqn-setup")
    sum_s_T_N_inv_d = 0  # Accumulators for the numerator and denominator of eqn 16.
    sum_s_T_N_inv_s = 0

    # Convert the requested averaging time [s] to a number of native samples.
    downsample_factor = max(1, round(gain_cfg.downsample_time * experiment_data.fsamp))
    scan_view = TODView(experiment_data, tod_samples, compsep_output=det_compsep_map,
                        downsample_factor=downsample_factor,
                        proc_mask_type="gain",
                        mask_threshold=gain_cfg.mask_threshold)

    # Skip detector-scans flagged as bad (accepted_only); they carry no gain info.
    for view in scan_view.iter_focused(accepted_only=True):
        calib = view.get_calib_tod("abs", gain_cfg.calibrate_against,
                                   gap_fill_method=gain_cfg.gap_fill_method)
        s_cal = calib.s_cal
        residual_tod = calib.tod

        # Calibration TODs are block-averaged, so their true rate is fsamp/downsample_factor;
        # apply_N_inv needs it to place the 1/f noise weight at the correct frequencies.
        gain_samprate = view.fsamp / view.downsample_factor
        N_inv_s = experiment_data.apply_N_inv(s_cal, view.noise_params, samprate=gain_samprate)

        # N^-1 is symmetric, so s^T N^-1 d = (N^-1 s)^T d, giving:
        sum_s_T_N_inv_d += np.dot(N_inv_s, residual_tod)
        sum_s_T_N_inv_s += np.dot(s_cal, N_inv_s)
    stop_bench("eqn-setup")

    start_bench("MPI-comm")
    # The g0 term is fully global, so we reduce across both all scans and all bands:
    sum_s_T_N_inv_d = band_comm.reduce(sum_s_T_N_inv_d, op=MPI.SUM, root=0)
    sum_s_T_N_inv_s = band_comm.reduce(sum_s_T_N_inv_s, op=MPI.SUM, root=0)
    # Default to the current value so a skipped or ill-posed solve leaves the gain unchanged.
    g_sampled = tod_samples.abs_gain
    stop_bench("MPI-comm")

    start_bench("eqn-solve")
    # Rank 0 draws a sample of g0 from eq (16) from BP6, and bcasts it to the other ranks.
    if band_comm.Get_rank() == 0:
        if not np.isfinite(sum_s_T_N_inv_s) or sum_s_T_N_inv_s <= 0.0:
            logger.error(f"Band {experiment_data.band_name} absolute gain has no calibration "
                         f"weight (all detector-scans rejected or zero calibrator): not updating.")
        else:
            eta = np.random.randn()
            g_mean = sum_s_T_N_inv_d / sum_s_T_N_inv_s
            g_std = 1.0 / np.sqrt(sum_s_T_N_inv_s)
            g_sampled = g_mean + eta * g_std
            logger.info(f"Chain {tod_samples.chain} iter{iteration} "
                        f"{experiment_data.band_name} g0: {tod_samples.abs_gain:.4e} -> "
                        f"{g_sampled:.4e} (+/- {g_std:.4e}).")
    stop_bench("eqn-solve")

    with benchmark("abs-gain-barrier"):   # reported across ranks by bench_summary
        band_comm.Barrier()
    g_sampled = band_comm.bcast(g_sampled, root=0)
    log_memory("abs-gain")

    # As of Numpy 2.0 it's good practice to explicitly cast to Python scalar types, as this would
    # otherwise have been a np.float64 type, potentially causing unexpected casting behavior later.
    tod_samples.abs_gain = float(g_sampled)

    return tod_samples


def sample_relative_gain(band_comm: MPI.Comm, experiment_data: DetectorGroupTOD,
                         tod_samples: TODSamples, det_compsep_map: NDArray,
                         gain_cfg: GainConfig, iteration: int) -> TODSamples:
    """ Samples the detector-dependent relative gain (Delta g_i). This function implements the
        logic from Sec. 3.4 of BP7.
    Args:
        band_comm (MPI.Comm): The band-level MPI communicator.
        experiment_data (DetectorGroupTOD): The object holding scan data for the band.
        tod_samples (TODSamples): Current sampled TOD parameters.
        det_compsep_map (NDArray): The component-separation sky map for the detector.
        gain_cfg: Validated relative-gain settings.
        iteration: Current Gibbs iteration for log context.
    Returns:
        tod_samples (TODSamples): Updated TOD samples with relative gain estimates.
    """
    _warn_about_dipole_calibration(band_comm, experiment_data, gain_cfg, "rel_gain")
    start_bench("eqn-setup")
    ndet = experiment_data.ndet

    #### 1. Local Calculation (on each rank) ###
    # Each rank calculates the sum of terms for its local subset of scans.
    # local_s_T_N_inv_s = 0.0
    local_s_T_N_inv_s = np.zeros(ndet, dtype=np.float32)

    # local_r_T_N_inv_s = 0.0
    local_r_T_N_inv_s = np.zeros(ndet, dtype=np.float32)
    # Convert the requested averaging time [s] to a number of native samples.
    downsample_factor = max(1, round(gain_cfg.downsample_time * experiment_data.fsamp))
    scan_view = TODView(experiment_data, tod_samples, compsep_output=det_compsep_map,
                        downsample_factor=downsample_factor,
                        proc_mask_type="gain",
                        mask_threshold=gain_cfg.mask_threshold)

    # Skip detector-scans flagged as bad (accepted_only); they carry no gain info.
    for view in scan_view.iter_focused(accepted_only=True):
        calib = view.get_calib_tod("rel", gain_cfg.calibrate_against,
                                   gap_fill_method=gain_cfg.gap_fill_method)
        s_cal = calib.s_cal
        residual_tod = calib.tod
        # Calibration TODs are block-averaged, so their true rate is fsamp/downsample_factor;
        # apply_N_inv needs it to place the 1/f noise weight at the correct frequencies.
        gain_samprate = view.fsamp / view.downsample_factor
        N_inv_s = experiment_data.apply_N_inv(s_cal, view.noise_params, samprate=gain_samprate)

        s_T_N_inv_s_scan = np.dot(s_cal, N_inv_s)
        r_T_N_inv_s_scan = np.dot(residual_tod, N_inv_s)

        # Add the contribution from this scan to the local sum (full-band detector column).
        local_s_T_N_inv_s[view.idet] += s_T_N_inv_s_scan
        local_r_T_N_inv_s[view.idet] += r_T_N_inv_s_scan
    stop_bench("eqn-setup")

    ### 2. Intra-Detector Reduction ###
    # Sum the local values across all ranks that share the same detector using det_comm.
    # After this, every rank in the det_comm will have the total sum for their detector.
    start_bench("MPI-comm")
    band_comm.Allreduce(MPI.IN_PLACE, local_s_T_N_inv_s, op=MPI.SUM)
    band_comm.Allreduce(MPI.IN_PLACE, local_r_T_N_inv_s, op=MPI.SUM)
    stop_bench("MPI-comm")

    ### 3. Solve Global System ###
    # Solve the constrained system (sum of Delta g_i = 0) over the active detectors only; detectors
    # rejected on every scan or with a vanishing calibrator carry zero weight, are held at their
    # current value, and are excluded so the bordered matrix stays non-singular.
    start_bench("eqn-solve")
    delta_g_samples = np.array(tod_samples.rel_gain, dtype=np.float32)  # default: leave unchanged
    if band_comm.Get_rank() == 0:
        n_active = int(np.count_nonzero(local_s_T_N_inv_s > 0.0))
        n_excluded = ndet - n_active
        if n_active == 0:
            logger.error(f"Band {experiment_data.band_name}: no detectors with calibration weight "
                         f"for relative gain; not updating.")
        else:
            try:
                delta_g_samples = _solve_relative_gain_system(local_s_T_N_inv_s,
                                                local_r_T_N_inv_s, tod_samples.rel_gain)
                msg = (f"Chain {tod_samples.chain} iter{iteration} "
                       f"{experiment_data.band_name}: solved relative gains for {n_active} "
                       "active detectors")
                if n_excluded:
                    msg += f" ({n_excluded} excluded: rejected on all scans or zero calibrator)"
                logger.verbose(msg + ".")
            except np.linalg.LinAlgError:
                logger.error("Failed to solve linear system for relative gain: Not updating.")
    stop_bench("eqn-solve")
    # Broadcast and apply on every rank, so all band ranks hold the identical relative-gain vector.
    prev_rel_gain = np.array(tod_samples.rel_gain)
    with benchmark("abs-gain-barrier"):   # reported across ranks by bench_summary
        band_comm.Barrier()
    band_comm.Bcast(delta_g_samples, root=0)
    tod_samples.rel_gain[:] = delta_g_samples
    log_memory("rel-gain")

    if band_comm.Get_rank() == 0:
        logger.info(f"Chain {tod_samples.chain} iter{iteration} {experiment_data.band_name} "
                    f"relative gain: min={np.min(delta_g_samples):.3e}, "
                    f"max={np.max(delta_g_samples):.3e}.")
        logger.debug(f"Rel gains for band {experiment_data.band_name}: {delta_g_samples}\n"\
                     f"Average change = {np.mean(np.abs(prev_rel_gain - delta_g_samples))}")

    return tod_samples


def sample_temporal_gain_variations(band_comm: MPI.Comm, experiment_data: DetectorGroupTOD,
                                    tod_samples: TODSamples, det_compsep_map: NDArray,
                                    gain_cfg: GainConfig, iteration: int) -> TODSamples:
    """ Samples the time-dependent relative gain variations (delta g_qi). This function implements
        the logic from Sec. 3.5 of the BP7 paper, using a Wiener filter to smooth the gain solution
        over time (PIDs). It solves a global system for all scans of a given detector, which are
        distributed across the ranks of the band_comm.

    Args:
        band_comm (MPI.Comm): The communicator for ranks sharing the same band.
        experiment_data (DetectorGroupTOD): The object holding scan data.
        tod_samples (TODSamples): The sampled TOD parameters.
        det_compsep_map (NDArray): The sky model at our band.
        gain_cfg: Validated temporal-gain settings.
        iteration: Current Gibbs iteration for log context.
    Returns:
        tod_samples (TODSamples): Updated TOD samples with per-scan gain variations.
    """
    _warn_about_dipole_calibration(band_comm, experiment_data, gain_cfg, "temporal_gain")
    band_rank = band_comm.Get_rank()
    band_size = band_comm.Get_size()
    ndet = experiment_data.ndet
    nscans_local = len(experiment_data.scans)

    # Local calculations on each rank
    A_qq_local = np.zeros((ndet, nscans_local), dtype=np.float64)
    b_q_local = np.zeros((ndet, nscans_local), dtype=np.float64)
    # (sigma0, fknee, alpha) of the Wiener prior, filled below by whichever rank solves a detector
    # and summed across the band afterwards, so the chain records the prior actually used.
    gain_prior_local = np.zeros((ndet, 3), dtype=np.float64)
    # Convert the requested averaging time [s] to a number of native samples.
    downsample_factor = max(1, round(gain_cfg.downsample_time * experiment_data.fsamp))
    scan_view = TODView(experiment_data, tod_samples, compsep_output=det_compsep_map,
                        downsample_factor=downsample_factor,
                        proc_mask_type="gain",
                        mask_threshold=gain_cfg.mask_threshold)

    # I'm still not sure what way of dealing with the masked samples are best:
    # 1. Replace masked values with 0s before FFT.
    # 2. Replace masked values with n_corr realizations before FFT.
    # 3. Remove masked values by reducing TOD size before FFTs.
    # (simply passing the full data through the FFTs seems like a bad idea because of
    # ringing from the large residual in the galactic plane).
    # Rejected detector-scans (accepted_only) contribute zero weight (A_qq = b_q = 0); the Wiener
    # prior then fills their temporal gain from neighbors.
    for view in scan_view.iter_focused(accepted_only=True):
        calib = view.get_calib_tod("temp", gain_cfg.calibrate_against,
                                   gap_fill_method=gain_cfg.gap_fill_method)
        s_cal = calib.s_cal
        residual_tod = calib.tod

        # Calibration TODs are block-averaged, so their true rate is fsamp/downsample_factor;
        # apply_N_inv needs it to place the 1/f noise weight at the correct frequencies.
        gain_samprate = view.fsamp / view.downsample_factor
        N_inv_s = experiment_data.apply_N_inv(s_cal, view.noise_params, samprate=gain_samprate)

        # N^-1 is symmetric, so s^T N^-1 d = (N^-1 s)^T d, giving:
        b_q = np.dot(N_inv_s, residual_tod)
        A_qq = np.dot(s_cal, N_inv_s)

        A_qq_local[view.idet, view.iscan] = A_qq
        b_q_local[view.idet, view.iscan] = b_q

    # Gather scan counts on all ranks (needed for gather/scatter with varying roots)
    scan_counts = np.array(band_comm.allgather(nscans_local), dtype=int)
    displacements = np.insert(np.cumsum(scan_counts), 0, 0)[:-1]

    # Distribute detector solves across ranks in round-robin fashion.
    # Each detector's equation system is gathered to, solved on, and scattered from
    # the rank given by solving_rank = idet % band_size.
    for idet in range(ndet):
        solving_rank = idet % band_size

        all_A_qq = band_comm.gather(A_qq_local[idet], root=solving_rank)
        all_b_q = band_comm.gather(b_q_local[idet], root=solving_rank)

        delta_g_sample = None
        if band_rank == solving_rank:
            # Concatenate gathered arrays into single flat arrays
            A_diag = np.concatenate(all_A_qq)
            b = np.concatenate(all_b_q)

            n_scans_total = len(A_diag)
            if n_scans_total > 1:
                # Define Prior (Wiener Filter) based on Eq. (31)
                alpha_gain = -2.5
                fknee_gain = 1.0  # Hour (which equals 1 scan)

                # However, I found the BP prior too weak, and this gives me more sensible results.
                mean_gain = tod_samples.abs_gain + tod_samples.rel_gain[idet]
                sigma0_gain = 1e-4*mean_gain
                sigma0_sq_gain = sigma0_gain**2
                # Record the prior this detector was actually sampled under, for the chain file.
                # Only the solving rank knows it, hence the Allreduce after the detector loop.
                gain_prior_local[idet] = (sigma0_gain, fknee_gain, alpha_gain)

                gain_freqs = rfftfreq(n_scans_total, d=1.0)
                prior_ps = np.zeros_like(gain_freqs)
                prior_ps[1:] = sigma0_sq_gain * (np.abs(gain_freqs[1:]) / fknee_gain)**alpha_gain

                prior_ps_inv = np.zeros_like(gain_freqs)
                prior_ps_inv[prior_ps > 0] = 1.0 / prior_ps[prior_ps > 0]
                prior_ps_inv_sqrt = np.sqrt(prior_ps_inv)

                # Define Linear Operator for Conjugate Gradient Solver
                def matvec(v, A_diag=A_diag, prior_ps_inv=prior_ps_inv,
                           n_scans_total=n_scans_total):
                    g_inv_v = backward_rfft(forward_rfft(v) * prior_ps_inv, n_scans_total).real
                    diag_v = A_diag * v
                    return g_inv_v + diag_v

                # Construct RHS of the sampling equation (Eq. 30)
                eta1 = np.random.randn(n_scans_total)
                fluctuation1 = np.sqrt(np.maximum(A_diag, 0)) * eta1

                eta2 = np.random.randn(n_scans_total)
                fluctuation2 = backward_rfft(forward_rfft(eta2) * prior_ps_inv_sqrt, n_scans_total).real

                RHS = b + fluctuation1 + fluctuation2

                ### Simpler sanity check solution  ##
                epsilon = 1e-12
                g_mean = b / (A_diag + epsilon)
                g_std = 1.0 / np.sqrt(np.maximum(A_diag, 0) + epsilon)

                CG_solver = pixell.utils.CG(matvec, RHS, x0=g_mean)
                for i in range(200):
                    CG_solver.step()
                    if CG_solver.err < 1e-10:
                        break

                delta_g_sample = CG_solver.x
                delta_g_sample -= np.mean(delta_g_sample)
                # logger.info(f"Band {experiment_data.nu}GHz det {idet} time-dependent gain: "\
                #             f"min={np.min(delta_g_sample)*1e9:14.4f} "\
                #             f"mean={np.mean(delta_g_sample)*1e9:14.4f} "\
                #             f"std={np.std(delta_g_sample)*1e9:14.4f} "\
                #             f"max={np.max(delta_g_sample)*1e9:14.4f}")

            else:
                delta_g_sample = np.zeros(n_scans_total)

        # Scatter the results back to all ranks from the solving rank
        if band_size > 1:
            delta_g_local = np.empty(nscans_local, dtype=np.float64)
            if band_rank == solving_rank:
                sendbuf = [delta_g_sample, scan_counts, displacements, MPI.DOUBLE]
            else:
                sendbuf = None
            band_comm.Scatterv(sendbuf, delta_g_local, root=solving_rank)
        else:
            delta_g_local = delta_g_sample if delta_g_sample is not None else np.array([])
        log_memory("temporal-gain")

        # Update tod_samples for this detector
        if delta_g_local.size == nscans_local:
            tod_samples.temporal_gain[:,idet] = delta_g_local.astype(np.float32, copy=False)
        else:
            logger.warning(f"Rank {band_rank} received mismatched number of gain samples "\
                           f"for det {idet}. Expected {nscans_local}, got {delta_g_local.size}.")

    # Each detector's prior was written by exactly one rank, so summing collects them all.
    band_comm.Allreduce(MPI.IN_PLACE, gain_prior_local, op=MPI.SUM)
    tod_samples.gain_prior = gain_prior_local

    if logger.isEnabledFor(logging.INFO):  # Only do all this work if logger.INFO is turned on.
        local_values_list: list[float] = []
        for iscan, det in experiment_data.iter_detector_scans():
            local_values_list.append(tod_samples.temporal_gain[iscan, det.det_idx_fullband])
        local_values = np.asarray(local_values_list, dtype=np.float64)
        local_count = local_values.size
        local_sum = np.sum(local_values)
        local_sum_sq = np.sum(local_values**2)
        local_min = np.min(local_values) if local_count else np.inf
        local_max = np.max(local_values) if local_count else -np.inf
        count = band_comm.reduce(local_count, op=MPI.SUM, root=0)
        value_sum = band_comm.reduce(local_sum, op=MPI.SUM, root=0)
        value_sum_sq = band_comm.reduce(local_sum_sq, op=MPI.SUM, root=0)
        value_min = band_comm.reduce(local_min, op=MPI.MIN, root=0)
        value_max = band_comm.reduce(local_max, op=MPI.MAX, root=0)
        if band_rank == 0 and count:
            mean = value_sum / count
            std = np.sqrt(max(0.0, value_sum_sq / count - mean**2))
            logger.info(f"Chain {tod_samples.chain} iter{iteration} "
                        f"{experiment_data.band_name} temporal gain: min={value_min:.3e}, "
                        f"mean={mean:.3e}, std={std:.3e}, max={value_max:.3e}.")

    return tod_samples
