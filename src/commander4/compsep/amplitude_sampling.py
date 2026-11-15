"""Sampling the component amplitudes of one amplitude group, with the CG or per-pixel solver.

An amplitude group names the components and the bands of one solve. `sample_amplitudes` holds the
other components fixed, lets the ranks of the group's bands solve, and gives every CompSep rank
the new amplitudes.
"""
import logging
from copy import deepcopy

import numpy as np
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.compsep.cg_solver import CompSepSolver
from commander4.compsep.perpix_solver import solve_compsep_perpix
from commander4.data_models.detector_map import DetectorMap
from commander4.sky.comp_list import CompList
from commander4.sky.sky_model import SkyModel

logger = logging.getLogger(__name__)


def _build_conditional_residual(detector_data: DetectorMap, comp_list: CompList, target_pol: str,
                                active_sublist: CompList) -> DetectorMap:
    """Subtract the components held fixed by a sampling group from this band's map.

    A sampling group that solves only a subset of components must be conditioned on the rest: the
    fixed components' projected signal is removed from the data so the active components are fit
    to the residual rather than the full observed sky (C3's ``compute_residual`` convention).
    The fixed signal is realized at this band's data resolution; each fixed component removes its
    own ``amp_fwhm_rad`` (deconvolved CG vs. data-resolution per-pixel) so the subtraction matches
    the data and the solvers' data model. Returns `detector_data` unchanged when no component is
    fixed.
    """
    active_names = {comp.comp_name for comp in active_sublist}
    fixed_comps = [comp for comp in comp_list.split_for_eval_pol(target_pol)
                   if comp.comp_name not in active_names]
    if not fixed_comps:
        return detector_data
    fixed_sky = SkyModel(CompList(fixed_comps)).get_sky_at_band(detector_data)
    residual = deepcopy(detector_data)
    residual.map_sky = detector_data.map_sky - fixed_sky.astype(detector_data.map_sky.dtype,
                                                                copy=False)
    return residual


def sample_amplitudes(mpi_info: Bunch, cfg: Bunch, group: Bunch, detector_data: DetectorMap,
                      comp_list: CompList, rng: np.random.Generator) -> dict:
    """Sample the amplitudes of one amplitude group. Every CompSep rank calls it.

    Updates the components of `comp_list` in place, on every rank.

    Args:
        mpi_info: The MPI data of this rank (`compsep` and `band` are used).
        cfg: The settings of component separation, from `init_compsep_processing`.
        group: The settings of the group to solve, one of `cfg.amplitude_groups`.
        detector_data: This rank's band map.
        comp_list: The full component list of the chain.
        rng: This rank's random generator, for the fluctuation terms of the amplitude draw.

    Returns:
        On the CompSep master, the CG convergence history of each polarization that was solved,
        as ``{"I": {...}, "QU": {...}}``. Empty on every other rank, and empty for the per-pixel
        method, which is a direct solve with no history to report.
    """
    compsep, band = mpi_info.compsep, mpi_info.band
    # This rank solves if the group names its band (or its view of the band), and at least one
    # component of this rank's polarization.
    band_is_active = band.name in group.bands or band.identifier in group.bands
    active_sublist = CompList([comp for comp in comp_list if comp.eval_pol == band.polarization
                               and comp.comp_name in group.comps])
    should_solve = band_is_active and len(active_sublist) > 0

    cg_stats: dict = {}
    solver_comm = compsep.subcomm.Split(0 if should_solve else MPI.UNDEFINED, key=compsep.rank)
    if should_solve:
        residual_data = _build_conditional_residual(
            detector_data, comp_list, band.polarization, active_sublist)
        # Both solvers leave their solution in the components of `active_sublist`, which are the
        # chain's own components.
        if cfg.cg_sampling_groups:
            solver = CompSepSolver(residual_data, solver_comm, group,
                                   double_precision=cfg.double_precision,
                                   nthreads=cfg.num_threads)
            solver.solve(active_sublist, rng)
            # The CG runs its scalar recurrence on rank 0 of `solver_comm` and only broadcasts the
            # stopping decision, so that rank alone holds meaningful residuals.
            if solver_comm.Get_rank() == 0:
                cg_stats = {"cg_residuals": np.asarray(solver.CG_residuals),
                            "n_iter": len(solver.CG_residuals)}
        else:
            solve_compsep_perpix(solver_comm, residual_data, active_sublist,
                                 double_precision=cfg.double_precision,
                                 nthreads=cfg.num_threads, rng=rng)
        solver_comm.Free()

    any_active = compsep.comm.allreduce(1 if should_solve else 0, op=MPI.SUM)
    if not any_active:
        if compsep.rank == compsep.master:
            logger.verbose(f"Sampling group {group.name!r} had no active band/component overlap.")
        return {}

    # The lowest-ranked solver for each polarization owns the authoritative result.
    for eval_pol in ("I", "QU"):
        solved_here = should_solve and band.polarization == eval_pol
        source = compsep.comm.allreduce(compsep.rank if solved_here else compsep.size, op=MPI.MIN)
        if source < compsep.size:
            comp_list.broadcast_pol_views(compsep.comm, eval_pol=eval_pol, source=source)

    # `solver_comm` never spans more than one polarization, so exactly one rank per polarization
    # filled `cg_stats` above and the rest are empty. I and QU are separate solves that converge
    # differently, so they are reported separately.
    all_stats = compsep.comm.gather((band.polarization, cg_stats), root=compsep.master)
    if compsep.rank != compsep.master:
        return {}
    return {pol: stats for pol, stats in all_stats if stats}
