"""`Gain`: the gain of every detector-scan, and the steps that sample it.

The gain has three terms: an absolute gain, a single number per band; a relative gain, one per
detector under a zero-sum constraint; and a temporal term, one per detector-scan, drawn through a
Wiener filter. They are sampled in that order once per Gibbs iteration, each against a
configurable calibrator (orbital dipole, sky, or sky without the dipole). BP7 is BeyondPlanck VII
(Gjerløw et al. 2023). [C3: comm_tod_driver_mod.f90::sample_calibration, comm_tod_gain_smod.f90]
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pixell
from mpi4py import MPI
from numpy.typing import NDArray
from pixell.bunch import Bunch
from scipy.fft import rfftfreq

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.diagnostics.performance import benchmark, log_memory
from commander4.math_utils.fft import forward_rfft, backward_rfft
from commander4.parameters.schema import read_settings
from commander4.tod.noise.gap_filling import fill_all_masked
from commander4.tod.step import ChainArrays, SavedBandSample, TODStep
from commander4.tod.view import TODView
from commander4.units import rj_to_band_unit_factor

if TYPE_CHECKING:
    from commander4.tod.samples import TODSamples

logger = logging.getLogger(__name__)


# Gives a warning if asked to calibrate again CMB dipole above this frequency.
ORBITAL_DIPOLE_MAX_FREQ_GHZ = 400.0
# The signals a gain term can be fitted against (`calibrate_against`; see `Gain`).
CALIBRATORS = ("orbital_dipole", "sky", "sky_no_dipole")


def _solve_relative_gain_system(s_weights: NDArray, r_weights: NDArray, prev_rel_gain: NDArray,
                                rng: np.random.Generator) -> NDArray:
    """Draw the relative gains Delta g_i from their Gaussian posterior with sum(Delta g_i) = 0
    (BP7 Sec. 3.4).

    Solves the bordered linear system over the *active* detectors only, those with calibration
    weight ``s_weights > 0``. A detector rejected on every scan (or with a vanishing calibrator)
    would add an all-zero row and column, and two of them would make the matrix singular. Such
    detectors are left out of the solve and keep their current relative gain.

    Args:
        s_weights: Per-detector ``sum_scans s^T N^-1 s`` (calibration weight), (ndet,).
        r_weights: Per-detector ``sum_scans s^T N^-1 r``, (ndet,).
        prev_rel_gain: Current relative gains, kept for the excluded detectors, (ndet,).
        rng: Random generator for the fluctuation term.

    Returns:
        The (ndet,) relative gains, with the active entries drawn anew.

    Raises:
        np.linalg.LinAlgError: If the reduced system is singular (left for the caller to handle).
    """
    out = np.array(prev_rel_gain, dtype=np.float64)
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
    out[idx] = solution[:n]
    return out


def _draw_temporal_gain(A_diag: NDArray, b: NDArray, sigma0: float, fknee: float, alpha: float,
                        rng: np.random.Generator) -> tuple[NDArray, float]:
    """Draw one detector's temporal gain for all its scans, in time order (BP7 Sec. 3.5).

    The posterior has inverse covariance ``A + S^-1``. ``A`` is diagonal, holding each scan's
    calibration weight ``s^T N^-1 s`` (`A_diag`), and ``S`` is the Wiener prior, the power spectrum
    ``sigma0^2 (f/fknee)^alpha`` over the scan index (eq. 31). The draw solves
    ``(A + S^-1) g = b + A^1/2 eta1 + S^-1/2 eta2`` (eq. 30) by CG, and has its mean removed.

    Returns:
        The draw, (nscans,), and the CG's final error (target 1e-10, at most 200 steps).
    """
    n = len(A_diag)
    freqs = rfftfreq(n, d=1.0)
    # The prior leaves the mean (f = 0) free.
    prior_ps = np.zeros_like(freqs)
    prior_ps[1:] = sigma0**2 * (freqs[1:]/fknee)**alpha
    prior_ps_inv = np.zeros_like(freqs)
    prior_ps_inv[prior_ps > 0] = 1.0/prior_ps[prior_ps > 0]

    def apply_lhs(v: NDArray) -> NDArray:
        """(A + S^-1) v."""
        return backward_rfft(forward_rfft(v)*prior_ps_inv, n).real + A_diag*v

    # The data term plus one random fluctuation each from A and from S^-1.
    eta1 = rng.standard_normal(n)
    eta2 = rng.standard_normal(n)
    rhs = (b + np.sqrt(np.maximum(A_diag, 0))*eta1
           + backward_rfft(forward_rfft(eta2)*np.sqrt(prior_ps_inv), n).real)
    # The CG starts from the solution without the prior.
    solver = pixell.utils.CG(apply_lhs, rhs, x0=b/(A_diag + 1e-12))
    for _ in range(200):
        solver.step()
        if solver.err < 1e-10:
            break
    draw = solver.x
    draw -= np.mean(draw)
    return draw, solver.err


class Gain(TODStep):
    """The gain of one band's detector-scans on one rank, and the three steps that sample it.

    The gain of detector-scan (iscan, idet) is ``abs_gain + rel_gain[idet] + temporal_gain[iscan,
    idet]``, in [detector units]/uK_RJ. The terms are kept apart because they are sampled apart,
    and the relative gains under a constraint on their sum.

    Sampled:
        abs_gain: The absolute gain g0, identical on every rank of the band.
        rel_gain: (ndet,) Relative gain of each detector, identical on every rank of the band.
        temporal_gain: (nscans, ndet) Gain variation of each detector-scan of this rank.

    Attributes:
        cfg: The settings of the three terms, as ``cfg.abs_gain``, ``cfg.rel_gain`` and
            ``cfg.temporal_gain``: whether and from which iteration the term is sampled, the
            calibrator it is sampled against, and the TOD it is sampled from (the averaging time
            in seconds, and the fraction of an averaged block that must be good).
            The calibrator (`calibrate_against`, see `_calibration_sums`) is one of:
              * ``orbital_dipole``: the CMB dipole from the observer's motion, alone. It is known
                from the spacecraft velocity and the CMB monopole temperature, without the sky
                model, so it is the only *absolute* calibrator, and the default for the absolute
                gain.
              * ``sky``: the whole modelled signal, static sky and orbital dipole. The default for
                the relative and temporal terms, which only track gain *differences* between
                detectors or scans, so they can use all the signal there is.
              * ``sky_no_dipole``: the static sky model from component separation, without the
                dipole.
        gain_prior: (ndet, 3) The temporal-gain Wiener prior each detector was sampled under, as
            (sigma0, fknee, alpha), Commander3's gain_sigma_0/gain_fknee/gain_alpha. Commander3
            samples this prior, while C4 hard-codes it in `sample_temporal`. Written to the chain,
            never read back.
        nu: Band centre frequency in GHz, for the unit conversion in `read_chain`.
        band_unit_factor: D, the value of 1 uK_RJ in the band's `band_unit`. The chain file quotes
            the gains in band_unit, and the gain has brightness in its denominator, so it is
            divided by D on writing.
    """

    def __init__(self, experiment_data: DetectorGroupTOD, params: Bunch, my_band: Bunch,
                 band_comm: MPI.Comm, chain: int):
        """Read the settings of the three terms, and start from the gains in the parameter file,
        or else in the data files.

        Each term's settings are a block of `tod_processing`, which a band can override setting by
        setting. The detector `gain` entries of the parameter file come first, then the data files'
        `init_scalars`. Either is quoted in band_unit, and is split here into the three terms.
        """
        exp, band = experiment_data.experiment_name, experiment_data.band_name
        self.cfg = Bunch()
        for term, default_calibrator in (("abs_gain", "orbital_dipole"), ("rel_gain", "sky"),
                                         ("temporal_gain", "sky")):
            cfg = read_settings(params, term, (f"experiments.{exp}.bands.{band}", "tod_processing"),
                                {"enabled": False, "from_iter": 1,
                                 "calibrate_against": default_calibrator,
                                 "downsample_time": 1.0, "mask_threshold": 0.5,
                                 "white_noise_weights": False})
            if cfg.calibrate_against not in CALIBRATORS:
                raise ValueError(f"{term}.calibrate_against must be one of {CALIBRATORS}, not "
                                 f"{cfg.calibrate_against!r}.")
            if (cfg.enabled and cfg.calibrate_against == "orbital_dipole"
                    and experiment_data.nu > ORBITAL_DIPOLE_MAX_FREQ_GHZ
                    and band_comm.Get_rank() == 0 and chain == 1):
                logger.warning(f"{term} for band {band} ({experiment_data.nu} GHz) is calibrated "
                               "against the orbital dipole, but above "
                               f"{ORBITAL_DIPOLE_MAX_FREQ_GHZ:.0f} GHz the dipole is faint "
                               "compared with the foregrounds and makes a poor calibrator; "
                               "consider 'sky', or an externally determined gain for this band.")
            self.cfg[term] = cfg

        self.nu = experiment_data.nu
        self.band_unit_factor = experiment_data.band_unit_factor
        nscans, ndet = experiment_data.nscans, experiment_data.ndet
        self.abs_gain = 0.0
        self.rel_gain = np.zeros(ndet)
        self.temporal_gain = np.zeros((nscans, ndet))
        self.gain_prior = np.full((ndet, 3), np.nan)

        # Find the first detector in the first scan. Used for checking if default values
        # are present. Default to None in case there are no scans.
        rep_det = next((det for _, det in experiment_data.iter_detector_scans()), None)

        # NaN so detector-scans with no data are excluded from the gain means below.
        all_det_gains = np.full((nscans, ndet), np.nan)
        if rep_det is not None:
            if "gain" in my_band.detectors[rep_det.name]:
                for iscan, det in experiment_data.iter_detector_scans():
                    all_det_gains[iscan, det.det_idx_fullband] = my_band.detectors[det.name].gain
            elif rep_det.init_scalars is not None:
                for iscan, det in experiment_data.iter_detector_scans():
                    all_det_gains[iscan, det.det_idx_fullband] = det.init_scalars[0]
            else:
                raise ValueError("Did not find initial gain value in input files.")

            # Initial gains are quoted in band_unit; convert to internal [det units]/uK_RJ
            # (gain multiplies by D on read) before decomposing into abs/rel/temporal.
            all_det_gains *= self.band_unit_factor

        # The absolute gain is the mean over all detector-scans of the band, and a relative gain
        # the mean of one detector minus that. Both are summed over all ranks, so every rank gets
        # the same values. The temporal gain is what is left of each detector-scan's gain (0 where
        # a detector has no data).
        observed = np.isfinite(all_det_gains)
        det_sums = band_comm.allreduce(np.where(observed, all_det_gains, 0.0).sum(axis=0))
        det_counts = band_comm.allreduce(observed.sum(axis=0))
        if det_counts.sum() > 0:
            self.abs_gain = float(det_sums.sum()/det_counts.sum())
            present = det_counts > 0
            self.rel_gain[present] = det_sums[present]/det_counts[present] - self.abs_gain
            self.temporal_gain = np.nan_to_num(all_det_gains - self.rel_gain - self.abs_gain,
                                               nan=0.0)

        if band_comm.Get_rank() == 0:
            logger.debug(f"Initial absolute gain estimate for {experiment_data.band_name}: "
                         f"{self.abs_gain:.3e}.")
            logger.debug(f"Initial rel gain estimates for {experiment_data.band_name}: "
                         f"{self.rel_gain}.")


    def chain_arrays(self) -> ChainArrays:
        """The gains, in band_unit, and the temporal-gain prior.

        All three gain terms are restart state, read back by `read_chain`. Only the sigma0 column
        of the gain prior is a gain; fknee and alpha are unit-free.
        """
        gain_prior = self.gain_prior.copy()
        gain_prior[:, 0] /= self.band_unit_factor
        return ChainArrays(per_scan={"temporal_gain": self.temporal_gain/self.band_unit_factor},
                           per_band={"abs_gain": self.abs_gain/self.band_unit_factor,
                                     "detrel_gain": self.rel_gain/self.band_unit_factor,
                                     "gain_prior": gain_prior})


    def read_chain(self, saved: SavedBandSample) -> None:
        """Continue from the gains in a band chain file, converted from its band_unit to uK_RJ."""
        # The gains are [detector units]/band_unit in the file; multiplying by D of that unit
        # brings them back to [detector units]/uK_RJ.
        gain_factor = rj_to_band_unit_factor(self.nu, saved.band_unit)
        self.abs_gain = float(saved.band("abs_gain"))*gain_factor
        self.rel_gain = saved.per_detector("detrel_gain")*gain_factor
        self.temporal_gain = saved.per_scan("temporal_gain").astype(self.temporal_gain.dtype)
        self.temporal_gain *= gain_factor


    @benchmark("eqn-setup")
    def _calibration_sums(self, tod_samples: TODSamples, band_sky: NDArray, term: str,
                          rng: np.random.Generator) -> tuple[NDArray, NDArray]:
        """The two sums the gain term `term` is drawn from, for each detector-scan of this rank.

        The model is ``d = g s_total + n``, with ``g = abs_gain + rel_gain + temporal_gain``. For
        one term, fitted against the calibrator ``s`` (a part of ``s_total``), the residual is
        ``r = g_term s + n``: the data with every signal removed at the full gain, and the term's
        own share added back. It is built from the gains as they are now, so each term is drawn
        given the latest draws of the others.

        Args:
            term: "abs_gain", "rel_gain" or "temporal_gain"; its settings are ``self.cfg[term]``.

        Returns:
            ``s^T N^-1 s`` and ``s^T N^-1 r``, both (nscans, ndet), 0 for rejected or absent
            detector-scans.
        """
        experiment_data = tod_samples.experiment_data
        gain_cfg = self.cfg[term]
        # Convert the requested averaging time [s] to a number of native samples.
        downsample_factor = max(1, round(gain_cfg.downsample_time * experiment_data.fsamp))
        scan_view = TODView(experiment_data, tod_samples, compsep_output=band_sky,
                            downsample_factor=downsample_factor, proc_mask_type="gain",
                            mask_threshold=gain_cfg.mask_threshold)
        s_N_inv_s = np.zeros((experiment_data.nscans, experiment_data.ndet))
        s_N_inv_r = np.zeros((experiment_data.nscans, experiment_data.ndet))
        # Rejected detector-scans (accepted_only) carry no gain information.
        for view in scan_view.iter_focused(accepted_only=True):
            # The calibrator. The far sidelobes see the sky model, so they belong to both sky
            # calibrators, as in C3's s_tot.
            if gain_cfg.calibrate_against == "orbital_dipole":
                s_cal = view.get_orbital_dipole_tod()
            elif gain_cfg.calibrate_against == "sky":
                s_cal = (view.get_sky_tod() + view.get_orbital_dipole_tod()
                         + view.get_far_sidelobe_tod())
            else:  # "sky_no_dipole"
                s_cal = view.get_sky_tod() + view.get_far_sidelobe_tod()
            # Python floats, so that a float32 TOD stays float32.
            if term == "abs_gain":
                g_term = self.abs_gain
            elif term == "rel_gain":
                g_term = float(self.rel_gain[view.idet])
            else:  # "temporal_gain"
                g_term = float(self.temporal_gain[view.iscan, view.idet])
            residual = view.get_tod() + g_term*s_cal
            mask = view.get_mask(proc_mask_type="gain")

            # N^-1/2 below is a Fourier filter, which would spread whatever the masked blocks hold
            # (such as a bright Galactic residual) into the good blocks nearby. Both TODs are
            # therefore filled with the same straight line between the means of the good blocks at
            # each gap edge. The fill is the same linear operation on both, so the filled residual
            # is still g_term * (filled s_cal) + noise, which keeps the gain unbiased. As in C3,
            # the residual's fill also gets white noise. [C3: sample_calibration]
            s_cal = np.array(s_cal, copy=True)  # The fill works in place; the dipole TOD is cached.
            fill_all_masked(s_cal, mask, 0.0, rng)
            fill_all_masked(residual, mask, view.sigma0/np.sqrt(view.downsample_factor), rng)

            # Whiten both with N^-1/2 and fit on the good blocks only: s^T N^-1 r becomes
            # sum(mask * s_white * r_white). [C3: accumulate_abscal] With `white_noise_weights`, N
            # is sigma0^2 alone, as in C3's HFI runs.
            noise_model, white_only = experiment_data.noise_model, gain_cfg.white_noise_weights
            s_white = noise_model.apply_N_inv(s_cal, view.noise_params, view.fsamp,
                                              view.downsample_factor, power=0.5,
                                              white_only=white_only)
            r_white = noise_model.apply_N_inv(residual, view.noise_params, view.fsamp,
                                              view.downsample_factor, power=0.5,
                                              white_only=white_only)
            s_N_inv_s[view.iscan, view.idet] = np.sum(mask * s_white**2)
            s_N_inv_r[view.iscan, view.idet] = np.sum(mask * s_white * r_white)
        return s_N_inv_s, s_N_inv_r


    @benchmark("abs-gain")
    def sample_absolute(self, tod_samples: TODSamples, det_compsep_map: NDArray,
                        iteration: int, rng: np.random.Generator) -> None:
        """Draw the absolute gain g0, one number for the whole band (BP7 eq. 16).

        The posterior is Gaussian, with mean ``sum(s^T N^-1 r) / sum(s^T N^-1 s)`` and standard
        deviation ``1/sqrt(sum(s^T N^-1 s))``, summed over every detector-scan of the band.

        Args:
            tod_samples: The band's sampled state, which the TOD views read.
            det_compsep_map: The sky model at this rank's pixels of the band.
            iteration: The Gibbs iteration, for the log.
            rng: This rank's random generator, for the gap fills and the draw.
        """
        band_comm, experiment_data = tod_samples.band_comm, tod_samples.experiment_data
        s_N_inv_s, s_N_inv_r = self._calibration_sums(tod_samples, det_compsep_map,
                                                      "abs_gain", rng)

        # g0 is common to all detector-scans of the band, so the sums run over all its ranks.
        with benchmark("MPI-comm"):
            sum_s_N_inv_r = band_comm.reduce(s_N_inv_r.sum(), op=MPI.SUM, root=0)
            sum_s_N_inv_s = band_comm.reduce(s_N_inv_s.sum(), op=MPI.SUM, root=0)
        # Default to the current value so a skipped or ill-posed solve leaves the gain unchanged.
        g_sampled = self.abs_gain

        # Rank 0 draws g0 and broadcasts it to the other ranks.
        with benchmark("eqn-solve"):
            if band_comm.Get_rank() == 0:
                if not np.isfinite(sum_s_N_inv_s) or sum_s_N_inv_s <= 0.0:
                    logger.error(f"Band {experiment_data.band_name} absolute gain has no "
                                 "calibration weight (all detector-scans rejected or zero "
                                 "calibrator): not updating.")
                else:
                    eta = rng.standard_normal()
                    g_mean = sum_s_N_inv_r / sum_s_N_inv_s
                    g_std = 1.0 / np.sqrt(sum_s_N_inv_s)
                    g_sampled = g_mean + eta * g_std
                    logger.info(f"Chain {tod_samples.chain} iter{iteration} "
                                f"{experiment_data.band_name} g0: {self.abs_gain:.4e} -> "
                                f"{g_sampled:.4e} (+/- {g_std:.4e}).")

        with benchmark("abs-gain-barrier"):   # reported across ranks by bench_summary
            band_comm.Barrier()
        g_sampled = band_comm.bcast(g_sampled, root=0)
        log_memory("abs-gain")

        # As of Numpy 2.0 it's good practice to explicitly cast to Python scalar types, as this
        # would otherwise have been a np.float64 type, potentially causing unexpected casting
        # behavior later.
        self.abs_gain = float(g_sampled)


    @benchmark("rel-gain")
    def sample_relative(self, tod_samples: TODSamples, det_compsep_map: NDArray,
                        iteration: int, rng: np.random.Generator) -> None:
        """Draw the relative gain of each detector, under the constraint that they sum to zero
        (BP7 Sec. 3.4; the solve is `_solve_relative_gain_system`).

        Args:
            tod_samples: The band's sampled state, which the TOD views read.
            det_compsep_map: The sky model at this rank's pixels of the band.
            iteration: The Gibbs iteration, for the log.
            rng: This rank's random generator, for the gap fills and the draw.
        """
        band_comm, experiment_data = tod_samples.band_comm, tod_samples.experiment_data
        ndet = experiment_data.ndet
        s_N_inv_s, s_N_inv_r = self._calibration_sums(tod_samples, det_compsep_map,
                                                      "rel_gain", rng)

        # Sum each detector's scans, then over all ranks of the band, whose scans each hold a part
        # of every detector. After this, every rank has each detector's total.
        det_s_N_inv_s = s_N_inv_s.sum(axis=0)
        det_s_N_inv_r = s_N_inv_r.sum(axis=0)
        with benchmark("MPI-comm"):
            band_comm.Allreduce(MPI.IN_PLACE, det_s_N_inv_s, op=MPI.SUM)
            band_comm.Allreduce(MPI.IN_PLACE, det_s_N_inv_r, op=MPI.SUM)

        # Solve the constrained system (sum of Delta g_i = 0) over the active detectors only;
        # detectors rejected on every scan or with a vanishing calibrator carry zero weight, are
        # held at their current value, and are excluded so the bordered matrix stays non-singular.
        # Default: leave unchanged.
        delta_g_samples = np.array(self.rel_gain, dtype=np.float64)
        with benchmark("eqn-solve"):
            if band_comm.Get_rank() == 0:
                n_active = int(np.count_nonzero(det_s_N_inv_s > 0.0))
                n_excluded = ndet - n_active
                if n_active == 0:
                    logger.error(f"Band {experiment_data.band_name}: no detectors with "
                                 "calibration weight for relative gain; not updating.")
                else:
                    try:
                        delta_g_samples = _solve_relative_gain_system(
                            det_s_N_inv_s, det_s_N_inv_r, self.rel_gain, rng)
                        msg = (f"Chain {tod_samples.chain} iter{iteration} "
                               f"{experiment_data.band_name}: solved relative gains for "
                               f"{n_active} active detectors")
                        if n_excluded:
                            msg += (f" ({n_excluded} excluded: rejected on all scans or zero "
                                    "calibrator)")
                        logger.verbose(msg + ".")
                    except np.linalg.LinAlgError:
                        logger.error("Failed to solve linear system for relative gain: Not "
                                     "updating.")
        # Broadcast and apply on every rank, so all band ranks hold the identical relative-gain
        # vector.
        prev_rel_gain = np.array(self.rel_gain)
        with benchmark("rel-gain-barrier"):   # reported across ranks by bench_summary
            band_comm.Barrier()
        band_comm.Bcast(delta_g_samples, root=0)
        self.rel_gain[:] = delta_g_samples
        log_memory("rel-gain")

        if band_comm.Get_rank() == 0:
            logger.info(f"Chain {tod_samples.chain} iter{iteration} {experiment_data.band_name} "
                        f"relative gain: min={np.min(delta_g_samples):.3e}, "
                        f"max={np.max(delta_g_samples):.3e}.")
            logger.debug(f"Rel gains for band {experiment_data.band_name}: {delta_g_samples}\n"\
                         f"Average change = {np.mean(np.abs(prev_rel_gain - delta_g_samples))}")


    @benchmark("temp-gain")
    def sample_temporal(self, tod_samples: TODSamples, det_compsep_map: NDArray,
                        iteration: int, rng: np.random.Generator) -> None:
        """Draw the temporal gain of each detector-scan, smoothed in time by a Wiener filter
        (BP7 Sec. 3.5; the draw is `_draw_temporal_gain`).

        Each detector is one system over all its scans, which lie on several ranks of the band.

        Args:
            tod_samples: The band's sampled state, which the TOD views read.
            det_compsep_map: The sky model at this rank's pixels of the band.
            iteration: The Gibbs iteration, for the log.
            rng: This rank's random generator, for the gap fills and the draw.
        """
        band_comm, experiment_data = tod_samples.band_comm, tod_samples.experiment_data
        band_rank = band_comm.Get_rank()
        band_size = band_comm.Get_size()
        ndet = experiment_data.ndet
        nscans_local = len(experiment_data.scans)
        # Rejected detector-scans have zero weight; the Wiener prior then fills their temporal
        # gain from their neighbours.
        s_N_inv_s, s_N_inv_r = self._calibration_sums(tod_samples, det_compsep_map,
                                                      "temporal_gain", rng)
        # (sigma0, fknee, alpha) of the Wiener prior, filled below by whichever rank solves a
        # detector and summed across the band afterwards, so the chain records the prior used.
        gain_prior_local = np.zeros((ndet, 3), dtype=np.float64)

        # Gather scan counts on all ranks (needed for gather/scatter with varying roots)
        scan_counts = np.array(band_comm.allgather(nscans_local), dtype=int)
        displacements = np.insert(np.cumsum(scan_counts), 0, 0)[:-1]
        # The gathers below stack the ranks' scans in rank order, but the Wiener prior couples
        # scans that are neighbours in time. `time_order` sorts the stacked scans by scan ID, i.e.
        # in time.
        time_order = np.argsort(np.concatenate(band_comm.allgather(tod_samples.scan_ids)))

        # Distribute detector solves across ranks in round-robin fashion.
        # Each detector's equation system is gathered to, solved on, and scattered from
        # the rank given by solving_rank = idet % band_size.
        for idet in range(ndet):
            solving_rank = idet % band_size

            all_A_qq = band_comm.gather(s_N_inv_s[:, idet], root=solving_rank)
            all_b_q = band_comm.gather(s_N_inv_r[:, idet], root=solving_rank)

            delta_g_sample = None
            if band_rank == solving_rank:
                # Concatenate gathered arrays into single flat arrays, in time order.
                A_diag = np.concatenate(all_A_qq)[time_order]
                b = np.concatenate(all_b_q)[time_order]

                n_scans_total = len(A_diag)
                if n_scans_total > 1:
                    # The Wiener prior, BP7 eq. 31.
                    alpha_gain = -2.5
                    fknee_gain = 1.0  # Hour (which equals 1 scan)

                    # However, I found the BP prior too weak, and this gives me more sensible
                    # results.
                    sigma0_gain = 1e-4*(self.abs_gain + self.rel_gain[idet])
                    # Record the prior this detector was actually sampled under, for the chain
                    # file. Only the solving rank knows it, hence the Allreduce after the detector
                    # loop.
                    gain_prior_local[idet] = (sigma0_gain, fknee_gain, alpha_gain)

                    delta_g_sample, cg_err = _draw_temporal_gain(A_diag, b, sigma0_gain,
                                                                 fknee_gain, alpha_gain, rng)
                    if cg_err >= 1e-10:
                        logger.warning(f"Band {experiment_data.band_name} detector {idet}: the "
                                       f"temporal-gain CG stopped after 200 steps with error "
                                       f"{cg_err:.1e}, above its target of 1e-10.")
                else:
                    delta_g_sample = np.zeros(n_scans_total)

                # Back from time order to the rank order that the scatter below expects.
                delta_g_rank_order = np.empty_like(delta_g_sample)
                delta_g_rank_order[time_order] = delta_g_sample
                delta_g_sample = delta_g_rank_order

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

            # Update the temporal gain of this detector
            if delta_g_local.size == nscans_local:
                self.temporal_gain[:, idet] = delta_g_local
            else:
                logger.warning(f"Rank {band_rank} received mismatched number of gain samples "\
                               f"for det {idet}. Expected {nscans_local}, got "
                               f"{delta_g_local.size}.")

        # Each detector's prior was written by exactly one rank, so summing collects them all.
        band_comm.Allreduce(MPI.IN_PLACE, gain_prior_local, op=MPI.SUM)
        self.gain_prior = gain_prior_local

        # Summary over the band's present detector-scans, gathered on the band master.
        if logger.isEnabledFor(logging.INFO):
            local_values = np.array([self.temporal_gain[iscan, det.det_idx_fullband]
                                     for iscan, det in experiment_data.iter_detector_scans()])
            all_values = band_comm.gather(local_values, root=0)
            if band_rank == 0:
                values = np.concatenate(all_values)
                if values.size > 0:
                    logger.info(f"Chain {tod_samples.chain} iter{iteration} "
                                f"{experiment_data.band_name} temporal gain: "
                                f"min={values.min():.3e}, mean={values.mean():.3e}, "
                                f"std={values.std():.3e}, max={values.max():.3e}.")
