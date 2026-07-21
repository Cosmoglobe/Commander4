"""Detector-scan data selection: everything around the per-detector-scan ``accept`` flags.

The cuts themselves are applied as per-scan vetoes inside the mapmaking scan loops in
tod_processing (accept/reject must be decided there, so a catastrophically bad detector-scan never
enters the current iteration's map products); the diagnostics they judge (good_fraction, chisq_z)
are recorded in the same loop and written to the chain. This module holds the rest: the chi-squared
statistic, the config resolution, and the per-band summary logging. Rejection is sticky within a
chain: rejected detector-scans are skipped by all accepted_only loops, so their samples and
diagnostics stop refreshing and they are never re-judged.

A population-relative (per-detector median/MAD) cut used to complement the absolute vetoes; it is
parked as dead code at the bottom of this file for possible re-introduction.
"""

import logging

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.data_models.TOD_samples import TODSamples, _gather_scan_distributed_array

logger = logging.getLogger(__name__)


def masked_chisq_z(residual: NDArray, mask: NDArray, sigma0: float) -> float:
    """Chi-squared z-score of a white-noise residual over the unflagged samples (C3 convention).

    z = (sum((r/sigma0)^2) - n) / sqrt(2n): the number of sigmas the masked residual chi-squared
    deviates from its expectation under N(0, sigma0^2) noise. ~N(0,1) for clean data; huge for
    scans with jumps, blobs, or a broken sigma0. Returns NaN when undefined (no good samples or
    non-positive/non-finite sigma0), which data selection treats as a rejection.
    """
    r = residual[mask].astype(np.float64, copy=False)
    n = r.size
    if n == 0 or not (np.isfinite(sigma0) and sigma0 > 0):
        return np.nan
    chisq = np.sum((r / sigma0)**2)
    return (chisq - n) / np.sqrt(2.0 * n)


def build_dataselect_cfg(params: Bunch, iter: int, do_ncorr: bool,
                         sample_corr_noise: bool) -> Bunch:
    """Resolve this iteration's data-selection config from the parameter file.

    ``enabled`` says whether the feature is on at all (diagnostics + summary logging); ``active``
    whether the vetoes fire this iteration: within [from_iter_num, until_iter_num] (inclusive;
    no upper limit when until_iter_num is absent), and -- since chisq_z needs the correlated-
    noise-subtracted residual -- not before ncorr sampling starts when that is configured.
    (from_iter_num=1, until_iter_num=1 reproduces C3's select-on-first-call-only behavior.)
    """
    dataselect = params.general.data_selection if "data_selection" in params.general else Bunch()
    enabled = (bool(getattr(dataselect, "sample_data_selection", False))
               and (do_ncorr or not sample_corr_noise))
    until = getattr(dataselect, "until_iter_num", None)
    return Bunch(
        enabled=enabled,
        active=(enabled and iter >= int(getattr(dataselect, "from_iter_num", 1))
                and (until is None or iter <= int(until))),
        chisq_abs_threshold=float(getattr(dataselect, "chisq_abs_threshold", 1.0e4)),
        min_good_fraction=float(getattr(dataselect, "min_good_fraction", 0.1)),
    )


def log_dataselect_summary(band_comm: MPI.Comm, tod_samples: TODSamples,
                           dataselect_cfg: Bunch) -> None:
    """Log a single per-band summary of this iteration's data selection (reporting only).

    Re-derives the veto counts from the diagnostics recorded this iteration (finite good_fraction
    = "entered the scan loop"; the predicates match the vetoes exactly), and reports cumulative
    acceptance plus chisq_z population quantiles -- a direct measure of noise-model quality.
    Individual detector-scans are far too numerous to log.
    """
    gf, z = tod_samples.good_fraction, tod_samples.chisq_z
    fresh = np.isfinite(gf)
    if dataselect_cfg.active:
        bad_lowfrac = fresh & (gf < dataselect_cfg.min_good_fraction)
        bad_chisq = fresh & ~bad_lowfrac & ~(np.isfinite(z)
                                             & (np.abs(z) <= dataselect_cfg.chisq_abs_threshold))
    else:
        bad_lowfrac = bad_chisq = np.zeros(fresh.shape, dtype=bool)
    good = fresh & ~bad_lowfrac & ~bad_chisq

    counts = np.array([np.count_nonzero(tod_samples.present),
                       np.count_nonzero(tod_samples.accept & tod_samples.present),
                       np.count_nonzero(bad_lowfrac), np.count_nonzero(bad_chisq)], dtype=np.int64)
    totals = np.zeros_like(counts)
    band_comm.Reduce(counts, totals, op=MPI.SUM, root=0)

    # The chisq_z quantiles need the full band population gathered on the master.
    if band_comm.Get_rank() == 0:
        scans_per_rank = np.zeros(band_comm.Get_size(), dtype=np.int64)
    else:
        scans_per_rank = None
    band_comm.Gather(np.array([tod_samples.nscans], dtype=np.int64), scans_per_rank, root=0)
    chisq_glob = _gather_scan_distributed_array(band_comm, z, scans_per_rank)
    good_glob = _gather_scan_distributed_array(band_comm, good.astype(np.int8), scans_per_rank)

    if band_comm.Get_rank() == 0:
        n_present, n_accept, n_lowfrac, n_chisq = totals
        frac = n_accept / max(n_present, 1)
        # Quantiles over the surviving bulk; worst |z| over everything fresh, so this iteration's
        # monsters stay visible even after their rejection.
        z_pop = chisq_glob[good_glob.astype(bool) & np.isfinite(chisq_glob)]
        zq = np.percentile(z_pop, [5, 50, 95]) if z_pop.size else np.full(3, np.nan)
        z_all = chisq_glob[np.isfinite(chisq_glob)]
        z_worst = np.max(np.abs(z_all)) if z_all.size else np.nan
        logger.info(f"Data selection ({tod_samples.band_name}): rejected {n_lowfrac + n_chisq} "
                    f"detector-scans this iteration (low-good-fraction: {n_lowfrac}, |chisq_z| > "
                    f"{dataselect_cfg.chisq_abs_threshold:.4g}: {n_chisq}); {n_accept}/{n_present} "
                    f"accepted ({frac:.2%}). chisq_z 5/50/95%: {zq[0]:.3g}/{zq[1]:.3g}/{zq[2]:.3g}, "
                    f"worst |chisq_z| = {z_worst:.3g}.")
        if frac < 0.9:
            logger.warning(f"Data selection ({tod_samples.band_name}): over 10% of scans rejected.")


# ============================================================================================== #
# DEAD CODE below: population-relative (per-detector median/MAD) data selection.
#
# Complemented the absolute vetoes with two-sided ``outlier_nmad`` MAD cuts on chisq_z and sigma0
# around per-detector medians measured from the full band population -- catching detector-scans
# that are consistently mildly bad rather than catastrophically so (analogous to C3's
# remove_tod_outliers, which C3 itself ships commented out at every call site). Removed from the
# live path to keep data selection simple; kept here for possible re-introduction.
# Does not match the current call paradigm: it expects cfg fields absolute_active/relative_active/
# outlier_nmad, and it owned the accept decision (re-applying the absolute cuts itself) rather
# than leaving the cuts to the in-loop vetoes.
# ============================================================================================== #


def _per_detector_median_mad(stat: NDArray, valid: NDArray,
                             min_count: int = 10) -> tuple[NDArray, NDArray]:
    """Per-detector robust location/scale of a (nscans, ndet) statistic over ``valid`` entries.

    Returns (median, mad) arrays of length ndet, with mad scaled by 1.4826 to estimate sigma for
    Gaussian data. Detectors with fewer than ``min_count`` valid entries or zero spread get
    mad=inf, disabling the outlier cut for them (too little population to define one).
    """
    ndet = stat.shape[1]
    med = np.zeros(ndet)
    mad = np.full(ndet, np.inf)
    for j in range(ndet):
        vals = stat[valid[:, j], j]
        if vals.size >= min_count:
            med[j] = np.median(vals)
            spread = 1.4826 * np.median(np.abs(vals - med[j]))
            if spread > 0:
                mad[j] = spread
    return med, mad


def sample_data_selection(band_comm: MPI.Comm, tod_samples: TODSamples, ds_cfg: Bunch) -> None:
    """Data-selection sampling step: update the ``accept`` array from this iteration's diagnostics.

    Owns all rejection decisions, re-derived here from the per-detector-scan diagnostics recorded
    in the mapmaking scan loop (good_fraction, chisq_z, sigma0). Two cut families, gated
    separately: absolute cuts (``min_good_fraction``, ``chisq_abs_threshold``; the in-loop veto in
    tod2map_* applies these same predicates eagerly so catastrophic scans never even enter the
    current iteration's maps) and population-relative cuts (two-sided ``outlier_nmad`` MAD cut on
    chisq_z and sigma0 around per-detector medians -- a dead detector-scan is as suspect as a noisy
    one). Only detector-scans with fresh diagnostics (processed this iteration) are evaluated;
    rejection is sticky within a chain.
    """
    if band_comm.Get_rank() == 0:
        scans_per_rank = np.zeros(band_comm.Get_size(), dtype=np.int64)
    else:
        scans_per_rank = None
    band_comm.Gather(np.array([tod_samples.nscans], dtype=np.int64), scans_per_rank, root=0)

    # Fresh = diagnostics recorded this iteration (NaN-reset before mapmaking): exactly the
    # detector-scans that entered the scan loop as accepted. Absolute cuts first; scans failing
    # them are excluded from the population the relative bounds are measured on.
    gf, z = tod_samples.good_fraction, tod_samples.chisq_z
    sigma0 = tod_samples.noise_params[:, :, 0]
    fresh = np.isfinite(gf)
    if ds_cfg.absolute_active:
        bad_lowfrac = fresh & (gf < ds_cfg.min_good_fraction)
        bad_chisq = fresh & ~bad_lowfrac & ~(np.isfinite(z) & (np.abs(z) <= ds_cfg.chisq_abs_threshold))
    else:
        bad_lowfrac = bad_chisq = np.zeros(fresh.shape, dtype=bool)
    eligible = fresh & ~bad_lowfrac & ~bad_chisq

    chisq_glob = _gather_scan_distributed_array(band_comm, z, scans_per_rank)
    sigma0_glob = _gather_scan_distributed_array(band_comm, np.ascontiguousarray(sigma0),
                                                 scans_per_rank)
    eligible_glob = _gather_scan_distributed_array(band_comm, eligible.astype(np.int8),
                                                   scans_per_rank)

    # Relative cuts: per-detector robust bounds from the full band population (computed on the
    # master), applied locally as a two-sided cut on both statistics.
    bad_mad = np.zeros(fresh.shape, dtype=bool)
    if ds_cfg.relative_active and ds_cfg.outlier_nmad > 0:
        if band_comm.Get_rank() == 0:
            elig_bool = eligible_glob.astype(bool)
            bounds = [_per_detector_median_mad(stat, elig_bool & np.isfinite(stat))
                      for stat in (chisq_glob, sigma0_glob)]
        else:
            bounds = None
        bounds = band_comm.bcast(bounds, root=0)
        for stat, (med, mad) in zip((z, sigma0), bounds):
            # (nscans, ndet) - (ndet,) broadcasts the per-detector bounds over scans.
            bad_mad |= eligible & np.isfinite(stat) & (np.abs(stat - med)
                                                       > ds_cfg.outlier_nmad * mad)

    tod_samples.accept[bad_lowfrac | bad_chisq | bad_mad] = False
