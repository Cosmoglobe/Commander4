"""Config/setup stuff for component amplitude sampling, common to both the CG and per-pixel solvers.

Contains one class and one function, both called from `compsep/processing.py`:
- `AmplitudeGroup` is a dataclass holding the settings for one amplitude sampling step.
- `sample_amplitudes` sets up and executes one provided `AmplitudeGroup`.
"""
import copy
import logging
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4 import sky
from commander4.compsep.cg_solver import CGAmplitudeSolver
from commander4.compsep.perpix_solver import solve_compsep_perpix
from commander4.data_models.band_map import BandMap
from commander4.parameters.schema import (SPLIT_POLS, enabled_compsep_polbands, names_or_all,
                                          read_settings)
from commander4.mpi.setup import MPIInfo
from commander4.sky.comp_list import CompList
from commander4.sky.diffuse_components import DiffuseComponent

logger = logging.getLogger(__name__)


@dataclass
class AmplitudeGroup:
    """One amplitude sampling group of the parameter file, set up for this CompSep rank."""
    name: str
    solver: str                          # "CG" or "per_pixel", from the section the group is in.
    cfg: Bunch                           # The group's block of the parameter file, with defaults
                                         # and "all" replaced by the names.
    solver_ranks: dict[str, list[int]]   # The CompSep ranks that solve it: {"I": [..], "QU": [..]}.
    solver_comm: MPI.Comm | None         # This rank's solve; MPI.COMM_NULL if it does not solve.

    # The settings of a group in `compsep.cg_sampling_groups`.
    CG_DEFAULTS = {"enabled": True,
                   "comps": None,                            # Component names; None/"all" -> all.
                   "bands": None,                            # Band names; None/"all" -> all.
                   "optimize": False,                        # The posterior maximum, not a draw.
                   "max_iter": 200,                          # CG iterations of the I solve,
                   "max_iter_pol": 200,                      # and of the QU solve.
                   "err_tol": 1.0e-8,                        # Stop at this `DistributedCG.err`.
                   "preconditioner": "JointPreconditioner"}  # A class in preconditioners.py.
    # The settings of a group in `compsep.per_pixel_sampling_groups`.
    PER_PIXEL_DEFAULTS = {"enabled": True,
                          "comps": None,                     # Component names; None/"all" -> all.
                          "bands": None}                     # Band names; None/"all" -> all.

    @classmethod
    def init_from_params(cls, params: Bunch, section: str, name: str,
                         mpi_info: MPIInfo | None = None) -> "AmplitudeGroup":
        """Read and check the group `compsep.<section>.<name>`.

        Every CompSep rank must build every group, in the same order, because `solver_comm` is
        split from all of them. Without `mpi_info` the group is only read and checked, and
        `solver_comm` is None.
        """
        solver = "CG" if section == "cg_sampling_groups" else "per_pixel"
        cfg = read_settings(params, name, (f"compsep.{section}",),
                            cls.CG_DEFAULTS if solver == "CG" else cls.PER_PIXEL_DEFAULTS)
        comp_names = [comp for comp in params.components if params.components[comp].enabled]
        band_names = [band for band in params.compsep.bands if params.compsep.bands[band].enabled]
        cfg.comps = names_or_all(cfg.comps, comp_names, f"{section}.{name}.comps")
        cfg.bands = names_or_all(cfg.bands, band_names, f"{section}.{name}.bands")
        if solver == "per_pixel":
            # The per-pixel solver fits component maps pixel by pixel, so it has no model of point
            # sources.
            for comp_name in cfg.comps:
                component_class = getattr(sky, params.components[comp_name].component_class)
                if not issubclass(component_class, DiffuseComponent):
                    raise ValueError(f"'{section}.{name}.comps' names {comp_name!r}, a "
                                     f"{component_class.__name__}. The per-pixel solver handles "
                                     "only diffuse components.")
        # A rank solves the group if its polband is one of the group's bands, in a polarization
        # that one of the group's components has. CompSep rank i works on polband i.
        solver_ranks = {}
        polbands = enabled_compsep_polbands(params)
        for pol in ("I", "QU"):
            has_comps = any(pol in SPLIT_POLS[params.components[comp].params.polarization]
                            for comp in cfg.comps)
            solver_ranks[pol] = [rank for rank, polband in enumerate(polbands) if has_comps
                                 and polband.pol == pol and polband.band_name in cfg.bands]
        # The communicator is split from `subcomm`, which holds the ranks of one polarization, so
        # the I and QU parts of a group are two separate solves.
        solver_comm = None
        if mpi_info is not None:
            is_solver = mpi_info.compsep.rank in solver_ranks[mpi_info.compsep.pol]
            solver_comm = mpi_info.compsep.subcomm.Split(0 if is_solver else MPI.UNDEFINED,
                                                         key=mpi_info.compsep.rank)
        return cls(name=name, solver=solver, cfg=cfg, solver_ranks=solver_ranks,
                   solver_comm=solver_comm)


def sample_amplitudes(mpi_info: MPIInfo, group: AmplitudeGroup, band_map: BandMap,
                      comp_list: CompList, rng: np.random.Generator) -> dict:
    """Sample the amplitudes of one amplitude group, updating `comp_list` in place.

    Returns:
        The CG convergence history of each polarization that was solved, as
        ``{"I": {...}, "QU": {...}}``, on every rank. Empty for the per-pixel method.
    """
    compsep = mpi_info.compsep
    if not group.solver_ranks["I"] and not group.solver_ranks["QU"]:
        if compsep.rank == 0:
            logger.verbose(f"Sampling group {group.name!r} had no active band/component overlap.")
        return {}

    cg_stats: dict = {}
    if compsep.rank in group.solver_ranks[compsep.pol]:
        active_sublist = CompList([comp for comp in comp_list if comp.pol == compsep.pol
                                   and comp.comp_name in group.cfg.comps])
        # The group is solved conditioned on the components it does not name (C3's
        # `compute_residual`): their sky at this band is subtracted from the band map, and the
        # solvers fit the group's components to what is left. The shallow copy shares the noise
        # map and the band settings with `band_map`; only its sky map is new.
        fixed_comps = CompList([comp for comp in comp_list if comp.pol == compsep.pol
                                and comp.comp_name not in group.cfg.comps])
        residual_map = band_map
        if len(fixed_comps) > 0:
            fixed_sky = fixed_comps.get_sky_at_band(band_map)
            residual_map = copy.copy(band_map)
            residual_map.map_sky = band_map.map_sky - fixed_sky
        # Both solvers leave their solution in the components of `active_sublist`, which are the
        # chain's own components.
        if group.solver == "CG":
            cg_solver = CGAmplitudeSolver(residual_map, group, nthreads=compsep.num_threads)
            cg_residual_history = cg_solver.solve(active_sublist, rng)
            # The CG runs its scalar recurrence on rank 0 of `solver_comm` and only broadcasts the
            # stopping decision, so that rank alone holds a meaningful history.
            if cg_solver.is_master:
                cg_stats = {"cg_residuals": np.asarray(cg_residual_history),
                            "n_iter": len(cg_residual_history)}
        else:
            solve_compsep_perpix(group.solver_comm, residual_map, active_sublist,
                                 nthreads=compsep.num_threads, rng=rng)

    # The first solver rank of each polarization (rank 0 of its `solver_comm`) gives every rank
    # the new amplitudes of that polarization, and its CG history. I and QU are separate solves
    # that converge differently, so they are reported separately.
    stats = {}
    for pol in ("I", "QU"):
        if not group.solver_ranks[pol]:
            continue
        comp_list.broadcast_pol_views(compsep.comm, pol=pol, source=group.solver_ranks[pol][0])
        if group.solver == "CG":
            stats[pol] = compsep.comm.bcast(cg_stats, root=group.solver_ranks[pol][0])
    return stats
