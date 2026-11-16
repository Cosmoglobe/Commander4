"""Sampling the component amplitudes of one amplitude group, with the CG or per-pixel solver.

An amplitude group names the components and the bands of one solve. `sample_amplitudes` holds the
other components fixed, lets the ranks of the group's bands solve, and gives every CompSep rank
the new amplitudes.
"""
import copy
import logging

import numpy as np
from pixell.bunch import Bunch

from commander4.compsep.cg_solver import CompSepSolver
from commander4.compsep.perpix_solver import solve_compsep_perpix
from commander4.data_models.detector_map import DetectorMap
from commander4.sky.comp_list import CompList

logger = logging.getLogger(__name__)


def sample_amplitudes(mpi_info: Bunch, cfg: Bunch, group: Bunch, band_map: DetectorMap,
                      comp_list: CompList, rng: np.random.Generator) -> dict:
    """Sample the amplitudes of one amplitude group. Every CompSep rank calls it.

    Updates the components of `comp_list` in place, on every rank.

    Args:
        mpi_info: The MPI data of this rank (`compsep` and `band` are used).
        cfg: The settings of component separation, from `init_compsep_processing`.
        group: The settings of the group to solve, one of `cfg.amplitude_groups`.
        band_map: This rank's band map.
        comp_list: The full component list of the chain.
        rng: This rank's random generator, for the fluctuation terms of the amplitude draw.

    Returns:
        The CG convergence history of each polarization that was solved, as
        ``{"I": {...}, "QU": {...}}``, on every rank. Empty for the per-pixel method, which is a
        direct solve with no history to report.
    """
    compsep, band = mpi_info.compsep, mpi_info.band
    if not group.solver_ranks["I"] and not group.solver_ranks["QU"]:
        if compsep.is_master:
            logger.verbose(f"Sampling group {group.name!r} had no active band/component overlap.")
        return {}

    cg_stats: dict = {}
    if compsep.rank in group.solver_ranks[band.polarization]:
        active_sublist = CompList([comp for comp in comp_list if comp.pol == band.polarization
                                   and comp.comp_name in group.comps])
        # The group is solved conditioned on the components it does not name (C3's
        # `compute_residual`): their sky at this band is subtracted from the band map, and the
        # solvers fit the group's components to what is left. The shallow copy shares the noise
        # map and the band settings with `band_map`; only its sky map is new.
        fixed_comps = CompList([comp for comp in comp_list if comp.pol == band.polarization
                                and comp.comp_name not in group.comps])
        residual_map = band_map
        if len(fixed_comps) > 0:
            fixed_sky = fixed_comps.get_sky_at_band(band_map)
            residual_map = copy.copy(band_map)
            residual_map.map_sky = band_map.map_sky - fixed_sky.astype(band_map.map_sky.dtype)
        # Both solvers leave their solution in the components of `active_sublist`, which are the
        # chain's own components.
        if cfg.cg_sampling_groups:
            solver = CompSepSolver(residual_map, group.solver_comm, group,
                                   double_precision=cfg.double_precision,
                                   nthreads=cfg.num_threads)
            residuals = solver.solve(active_sublist, rng)
            # The CG runs its scalar recurrence on rank 0 of `solver_comm` and only broadcasts the
            # stopping decision, so that rank alone holds meaningful residuals.
            if solver.is_master:
                cg_stats = {"cg_residuals": np.asarray(residuals), "n_iter": len(residuals)}
        else:
            solve_compsep_perpix(group.solver_comm, residual_map, active_sublist,
                                 double_precision=cfg.double_precision,
                                 nthreads=cfg.num_threads, rng=rng)

    # The first solver rank of each polarization (rank 0 of its `solver_comm`) gives every rank
    # the new amplitudes of that polarization, and its CG history. I and QU are separate solves
    # that converge differently, so they are reported separately.
    stats = {}
    for pol in ("I", "QU"):
        if not group.solver_ranks[pol]:
            continue
        comp_list.broadcast_pol_views(compsep.comm, pol=pol, source=group.solver_ranks[pol][0])
        if cfg.cg_sampling_groups:
            stats[pol] = compsep.comm.bcast(cg_stats, root=group.solver_ranks[pol][0])
    return stats
