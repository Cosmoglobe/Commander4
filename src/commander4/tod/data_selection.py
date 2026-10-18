"""Detector-scan data selection: everything around the per-detector-scan ``accept`` flags.

`DataSelection` holds the settings, the flags and the two diagnostics the cuts judge
(good_fraction, chisq_z). The cuts are its two vetoes, which the mapmaking scan loop (`tod2map`)
applies per detector-scan, so a bad detector-scan never enters the current iteration's maps. It
also decides in which iterations the cuts apply, and logs a per-band summary. Rejection is sticky
within a chain: rejected detector-scans are skipped by all accepted_only loops and never re-judged.

A population-relative (per-detector median/MAD) cut, not in use, is parked as dead code at the
bottom of this file.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.mpi.helpers import gather_scan_distributed_array
from commander4.parameters.schema import read_settings
from commander4.tod.step import ChainArrays, SavedBandSample, TODStep

if TYPE_CHECKING:
    from commander4.data_models.tod_samples import TODSamples
    from commander4.tod.view import TODView

logger = logging.getLogger(__name__)


class DataSelection(TODStep):
    """Which detector-scans of one band on one rank are used, and the diagnostics that decide it.

    Sampled:
        accept: (nscans, ndet) Whether the detector-scan is used. Present data that is *not*
            flagged as bad; distinct from `TODSamples.present` (data exists at all). Starts all
            True, and is part of the restart state.

    Attributes:
        cfg: The `tod_processing.data_selection` settings: whether the cuts are used, the first and
            last iteration they apply in (both inclusive; `until_iter` None for no end), and the two
            thresholds.
        good_fraction: (nscans, ndet) Unflagged fraction of the samples.
        chisq_z: (nscans, ndet) White-noise-residual chi-squared z-score (see `masked_chisq_z`;
            ~N(0,1) for clean data, huge for bad scans).
        The two diagnostics are NaN-reset by `start_iteration` and refilled by the vetoes, so a
        finite entry means "evaluated this iteration"; `log_summary` reports them.
    """

    def __init__(self, experiment_data: DetectorGroupTOD, params: Bunch, my_band: Bunch,
                 band_comm: MPI.Comm, chain: int):
        self.cfg = read_settings(params, "data_selection", ("tod_processing",),
                                 {"enabled": False, "from_iter": 1, "until_iter": None,
                                  "chisq_abs_threshold": 1.0e4, "min_good_fraction": 0.1})
        # The cuts would otherwise silently never apply.
        if self.cfg.until_iter is not None and self.cfg.until_iter < self.cfg.from_iter:
            raise ValueError("data_selection.until_iter cannot be before from_iter.")

        nscans, ndet = experiment_data.nscans, experiment_data.ndet
        self.accept = np.ones((nscans, ndet), dtype=bool)
        self.chisq_z = np.full((nscans, ndet), np.nan)
        self.good_fraction = np.full((nscans, ndet), np.nan)


    def chain_arrays(self) -> ChainArrays:
        """The accept flags (restart state, as int8 for MPI and HDF5) and both diagnostics."""
        return ChainArrays(per_scan={
            "accept": self.accept.astype(np.int8),
            "chisq_z": self.chisq_z,
            "good_fraction": self.good_fraction,
        })


    def read_chain(self, saved: SavedBandSample) -> None:
        """Continue from the accept flags in a band chain file."""
        self.accept = saved.per_scan("accept").astype(bool)


    def start_iteration(self) -> None:
        """Forget the diagnostics of the previous iteration. Previously rejected or absent scans
        then remain NaN, and are not counted again by the data-selection summary."""
        self.chisq_z[:] = np.nan
        self.good_fraction[:] = np.nan


    def status(self, iteration: int, noise_cfg: Bunch) -> tuple[bool, bool]:
        """Whether this iteration reports the diagnostics, and whether it applies the cuts.

        The diagnostics wait for correlated-noise sampling if that is enabled (`noise_cfg` is
        `Noise.cfg`), since the chi-squared judges the residual after n_corr. They can be reported
        before the cuts start and after they stop.
        """
        noise_ready = not noise_cfg.enabled or iteration >= noise_cfg.from_iter
        report = self.cfg.enabled and noise_ready
        before_end = self.cfg.until_iter is None or iteration <= self.cfg.until_iter
        apply_cuts = report and iteration >= self.cfg.from_iter and before_end
        return report, apply_cuts


    def veto_low_good_fraction(self, view: TODView, good_data_mask: NDArray[np.bool_],
                               active: bool) -> bool:
        """Veto 1: record the focused detector-scan's unflagged fraction and, when the cuts are
        `active`, reject the detector-scan if too little of it is unflagged.

        Returns:
            Whether the detector-scan was rejected.
        """
        good_frac = good_data_mask.mean()
        self.good_fraction[view.iscan, view.idet] = good_frac
        if active and good_frac < self.cfg.min_good_fraction:
            self.accept[view.iscan, view.idet] = False
            return True
        return False


    def veto_high_chisq(self, view: TODView, residual_tod: NDArray, active: bool) -> bool:
        """Veto 2: record the chi-squared z-score of the focused detector-scan's noise residual
        and, when the cuts are `active`, reject the detector-scan if it is catastrophically high.

        The z-score uses this iteration's sigma0 and only the unflagged samples. Undefined (NaN)
        counts as a rejection.

        Returns:
            Whether the detector-scan was rejected.
        """
        z = masked_chisq_z(residual_tod, view.get_mask(proc_mask=False), view.sigma0)
        self.chisq_z[view.iscan, view.idet] = z
        if active and not (np.isfinite(z) and abs(z) <= self.cfg.chisq_abs_threshold):
            self.accept[view.iscan, view.idet] = False
            return True
        return False


    def log_summary(self, tod_samples: TODSamples, active: bool, iteration: int) -> None:
        """Log a single per-band summary of this iteration's data selection (collective).

        Re-derives the veto counts from the diagnostics recorded this iteration (finite
        good_fraction = "entered the scan loop"; the predicates match the vetoes exactly), and
        reports cumulative acceptance plus chisq_z population quantiles, a direct measure of
        noise-model quality. Individual detector-scans are far too numerous to log. ``iteration``
        and ``tod_samples.chain`` identify the repeated summary in interleaved two-chain runs.
        """
        band_comm = tod_samples.band_comm
        gf, z = self.good_fraction, self.chisq_z
        fresh = np.isfinite(gf)
        if active:
            bad_lowfrac = fresh & (gf < self.cfg.min_good_fraction)
            within_threshold = np.isfinite(z) & (np.abs(z) <= self.cfg.chisq_abs_threshold)
            bad_chisq = fresh & ~bad_lowfrac & ~within_threshold
        else:
            bad_lowfrac = bad_chisq = np.zeros(fresh.shape, dtype=bool)
        good = fresh & ~bad_lowfrac & ~bad_chisq

        counts = np.array([np.count_nonzero(tod_samples.present),
                           np.count_nonzero(self.accept & tod_samples.present),
                           np.count_nonzero(bad_lowfrac), np.count_nonzero(bad_chisq)],
                          dtype=np.int64)
        totals = np.zeros_like(counts)
        band_comm.Reduce(counts, totals, op=MPI.SUM, root=0)

        # The chisq_z quantiles need the full band population gathered on the master.
        if band_comm.Get_rank() == 0:
            scans_per_rank = np.zeros(band_comm.Get_size(), dtype=np.int64)
        else:
            scans_per_rank = None
        band_comm.Gather(np.array([tod_samples.experiment_data.nscans], dtype=np.int64),
                         scans_per_rank, root=0)
        chisq_glob = gather_scan_distributed_array(band_comm, z, scans_per_rank)
        good_glob = gather_scan_distributed_array(band_comm, good.astype(np.int8), scans_per_rank)

        if band_comm.Get_rank() == 0:
            n_present, n_accept, n_lowfrac, n_chisq = totals
            frac = n_accept / max(n_present, 1)
            # Quantiles over the surviving bulk; worst |z| over everything fresh, so this
            # iteration's monsters stay visible even after their rejection.
            z_pop = chisq_glob[good_glob.astype(bool) & np.isfinite(chisq_glob)]
            zq = np.percentile(z_pop, [5, 50, 95]) if z_pop.size else np.full(3, np.nan)
            z_all = chisq_glob[np.isfinite(chisq_glob)]
            z_worst = np.max(np.abs(z_all)) if z_all.size else np.nan
            context = (f"Chain {tod_samples.chain} iter{iteration} "
                       f"{tod_samples.experiment_data.band_name}")
            logger.info(f"{context} data selection: rejected {n_lowfrac + n_chisq} "
                        f"detector-scans this iteration (low-good-fraction: {n_lowfrac}, "
                        f"|chisq_z| > {self.cfg.chisq_abs_threshold:.4g}: {n_chisq}); "
                        f"{n_accept}/{n_present} accepted ({frac:.2%}). chisq_z 5/50/95%: "
                        f"{zq[0]:.3g}/{zq[1]:.3g}/{zq[2]:.3g}, "
                        f"worst |chisq_z| = {z_worst:.3g}.")
            if frac < 0.9:
                logger.warning(f"{context} data selection: over 10% of scans rejected.")


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


# ============================================================================================== #
# DEAD CODE below: population-relative (per-detector median/MAD) data selection.
#
# Complemented the absolute vetoes with two-sided ``outlier_nmad`` MAD cuts on chisq_z and sigma0
# around per-detector medians measured from the full band population, catching detector-scans
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
    """Update ``accept`` from this iteration's diagnostics. NOT CALLED at present, and stale.

    Meant as the data-selection step after mapmaking, re-deriving the rejections from the
    per-detector-scan diagnostics the scan loop records (good_fraction, chisq_z, sigma0). It has two
    cut families: the absolute cuts (``min_good_fraction``, ``chisq_abs_threshold``), and
    population-relative cuts (a two-sided ``outlier_nmad`` MAD cut on chisq_z and sigma0 around
    per-detector medians, since a dead detector-scan is as suspect as a noisy one). Only
    detector-scans with fresh diagnostics (processed this iteration) are evaluated, and rejection is
    sticky within a chain.

    Today the absolute cuts run as vetoes inside the scan loop of `tod2map`, and nothing applies the
    relative cuts. ``ds_cfg`` must provide ``absolute_active``, ``relative_active`` and
    ``outlier_nmad``, which `DataSelection.cfg` does not have, so wiring the relative cuts back in
    needs those settings first.
    """
    if band_comm.Get_rank() == 0:
        scans_per_rank = np.zeros(band_comm.Get_size(), dtype=np.int64)
    else:
        scans_per_rank = None
    band_comm.Gather(np.array([tod_samples.experiment_data.nscans], dtype=np.int64),
                     scans_per_rank, root=0)

    # Fresh = diagnostics recorded this iteration (NaN-reset before mapmaking): exactly the
    # detector-scans that entered the scan loop as accepted. Absolute cuts first; scans failing
    # them are excluded from the population the relative bounds are measured on.
    gf, z = tod_samples.selection.good_fraction, tod_samples.selection.chisq_z
    sigma0 = tod_samples.noise.noise_params[:, :, 0]
    fresh = np.isfinite(gf)
    if ds_cfg.absolute_active:
        bad_lowfrac = fresh & (gf < ds_cfg.min_good_fraction)
        bad_chisq = fresh & ~bad_lowfrac & ~(np.isfinite(z) & (np.abs(z) <= ds_cfg.chisq_abs_threshold))
    else:
        bad_lowfrac = bad_chisq = np.zeros(fresh.shape, dtype=bool)
    eligible = fresh & ~bad_lowfrac & ~bad_chisq

    chisq_glob = gather_scan_distributed_array(band_comm, z, scans_per_rank)
    sigma0_glob = gather_scan_distributed_array(band_comm, np.ascontiguousarray(sigma0),
                                                scans_per_rank)
    eligible_glob = gather_scan_distributed_array(band_comm, eligible.astype(np.int8),
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

    tod_samples.selection.accept[bad_lowfrac | bad_chisq | bad_mad] = False
