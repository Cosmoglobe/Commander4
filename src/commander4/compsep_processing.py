import numpy as np
import logging
from copy import deepcopy
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.output.log import logassert
from commander4.data_models.detector_map import DetectorMap
from commander4.sky_models.component import CompList
from commander4.sky_models.sky_model import SkyModel
from commander4.solvers.CG_compsep_solver import CompSepSolver
from commander4.solvers.perpix_compsep_solver import solve_compsep_perpix
from commander4.output.write_chains_files import write_compsep_chain_to_file
from commander4.utils.execution_ids import get_execution_band_id, EXECUTION_POLS

logger = logging.getLogger(__name__)


def _sampling_group_selects_band(selected_bands: list[str] | None, band_name: str,
                                 band_identifier: str) -> bool:
    """Whether a sampling group acts on a band, matched by base name or execution-view identifier.

    `selected_bands` of None means "all bands".
    """
    if selected_bands is None:
        return True
    return band_name in selected_bands or band_identifier in selected_bands


def _filter_sampling_group_components(comp_list: CompList,
                                      selected_components: list[str] | None) -> CompList:
    """Subset of `comp_list` whose component names are selected by the sampling group.

    `selected_components` of None means "all components". The returned `CompList` shares the
    underlying `Component` objects with `comp_list` (it is a view, not a copy).
    """
    if selected_components is None:
        return CompList(list(comp_list))
    selected_names = set(selected_components)
    return CompList([comp for comp in comp_list if comp.comp_name in selected_names])


def _validate_sampling_groups(sampling_groups: Bunch, comp_list: CompList, params: Bunch) -> None:
    """Fail fast if any enabled sampling group references a non-existent component or band.

    `comps` and `bands` are expected to be lists of strings naming existing components and bands
    (bands may be given either as a base name or as an execution-view identifier).
    """
    known_comp_names = {comp.comp_name for comp in comp_list.joined()}
    known_band_names = set()
    for band_str in params.CompSep_bands:
        band = params.CompSep_bands[band_str]
        if not band.enabled:
            continue
        known_band_names.add(band_str)
        for eval_pol in EXECUTION_POLS[band.polarization]:
            known_band_names.add(get_execution_band_id(band_str, eval_pol))

    for group_name in sampling_groups:
        group = sampling_groups[group_name]
        if "enabled" in group and not group.enabled:
            continue
        if "comps" in group:
            unknown = sorted(set(group.comps) - known_comp_names)
            logassert(not unknown,
                      f"Sampling group {group_name!r} references unknown component(s) {unknown}. "
                      f"Known components: {sorted(known_comp_names)}.", logger)
        if "bands" in group:
            unknown = sorted(set(group.bands) - known_band_names)
            logassert(not unknown,
                      f"Sampling group {group_name!r} references unknown band(s) {unknown}. "
                      f"Known bands: {sorted(known_band_names)}.", logger)


def init_compsep_processing(mpi_info: Bunch, params: Bunch)\
    -> tuple[CompList, Bunch, str, Bunch]:
    """Set up the rank-local execution view for component separation.

    Each CompSep rank owns exactly one execution view of one band. The global CompSep rank space is
    split into a contiguous intensity block (ranks ``[0, QU_master)``) followed by a contiguous QU
    block (ranks ``[QU_master, size)``), and we match the current rank against those two streams.

    Args:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        params (Bunch): The parameters from the input parameter file.

    Returns:
        comp_list (CompList): The full execution-view component list (identical on all CompSep ranks).
        mpi_info (Bunch): `mpi_info`, extended with this rank's band name/identifier and the
            band-master dictionaries.
        band_identifier (str): Unique string for the band execution view this rank is working on.
        my_band (Bunch): The parameter-file subset for the band this rank is working on.
    """
    logger.info(f"CompSep: Hello from CompSep-rank {mpi_info.compsep.rank} (on machine "\
                f"{mpi_info.processor_name}), dedicated to band {mpi_info.compsep.rank}.")

    comp_list = CompList.init_from_params(params.components, params)
    comp_names = [comp.comp_name for comp in comp_list.joined()]
    logassert(len(comp_names) == len(set(comp_names)),
              f"Duplicate component names found in CompSep setup: {comp_names}", logger)

    ### Match this rank to its band execution view. Intensity views fill the I-rank block in band
    ### order, QU views fill the QU-rank block; the two cursors track those contiguous layouts. ###
    band_cursor = {"I": 0, "QU": mpi_info.compsep.QU_master}
    band_identifier = None
    band_name = None
    my_band = None
    for band_str in params.CompSep_bands:
        band = params.CompSep_bands[band_str]
        if not band.enabled:
            continue
        if band.polarization not in EXECUTION_POLS:
            raise ValueError(f"Unrecognized polarization in parameter file for band {band_str}")
        for eval_pol in EXECUTION_POLS[band.polarization]:
            if band_cursor[eval_pol] == mpi_info.compsep.rank:
                my_band = deepcopy(band)
                band_name = band_str
                band_identifier = get_execution_band_id(band_str, eval_pol)
                my_band.identifier = band_identifier
                my_band.polarization = eval_pol
                logger.info(f"Rank {mpi_info.compsep.rank} matched band {band_identifier}")
            band_cursor[eval_pol] += 1

    # Sanity checks: the I cursor must have consumed exactly the I-rank block [0, QU_master), and
    # the QU cursor exactly the QU-rank block [QU_master, size).
    n_I_ranks = mpi_info.compsep.QU_master
    n_QU_ranks = mpi_info.compsep.size - mpi_info.compsep.QU_master
    logassert(band_cursor["I"] == mpi_info.compsep.QU_master,
              f"Number of enabled Intensity band views ({band_cursor['I']}) does not match the "
              f"number of CompSep ranks assigned to Intensity ({n_I_ranks}).", logger)
    logassert(band_cursor["QU"] == mpi_info.compsep.size,
              f"Number of enabled QU band views ({band_cursor['QU'] - mpi_info.compsep.QU_master}) "
              f"does not match the number of CompSep ranks assigned to QU ({n_QU_ranks}).", logger)
    if my_band is None or band_identifier is None:
        logassert(False,
                  f"CompSep rank {mpi_info.compsep.rank} was not assigned to any enabled band. "
                  "Check that CompSep_bands matches the configured I/QU rank counts.",
                  logger)

    sampling_groups = params.sampling_groups_compsep if "sampling_groups_compsep" in params \
        else Bunch()
    _validate_sampling_groups(sampling_groups, comp_list, params)

    mpi_info.compsep.band_name = band_name
    mpi_info.compsep.band_identifier = band_identifier

    data_world = (band_identifier, mpi_info.world.rank)
    data_compsep = (band_identifier, mpi_info.compsep.rank)
    all_data_world = mpi_info.compsep.comm.allgather(data_world)
    all_data_compsep = mpi_info.compsep.comm.allgather(data_compsep)
    world_band_masters_dict = {item[0]: item[1] for item in all_data_world if item is not None}
    compsep_band_masters_dict = {item[0]: item[1] for item in all_data_compsep if item is not None}
    mpi_info.world.compsep_band_masters = world_band_masters_dict
    mpi_info.compsep.compsep_band_masters = compsep_band_masters_dict

    return comp_list, mpi_info, band_identifier, my_band


def process_compsep(mpi_info: Bunch, detector_data: DetectorMap, iter: int, chain: int,
                    params: Bunch, comp_list: CompList) -> SkyModel:
    """Perform a single component-separation iteration.

    Called by every CompSep rank, each of which owns one band execution view. Loops over the
    configured sampling groups; for each group the participating ranks (those whose band and at
    least one of whose components are selected) form a solver sub-communicator, run the requested
    sampler, and then broadcast the updated components so that `comp_list` is identical on every
    CompSep rank again before the next group.

    Args:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        detector_data (DetectorMap): The detector map for this rank's band, cleaned of all "TOD"
            components (correlated noise and orbital dipole).
        iter (int): The current Gibbs iteration (used only for printing and seeding).
        chain (int): The current chain (used only for printing and seeding).
        params (Bunch): The parameters from the input parameter file.
        comp_list (CompList): The full execution-view component list, updated in place.

    Returns:
        sky_model (SkyModel): The full sky realization, wrapping the updated `comp_list`.
    """
    compsep_comm = mpi_info.compsep.comm
    compsep_rank = mpi_info.compsep.rank
    # subcomm splits the CompSep ranks by polarization (subcolor 0 -> I, 1 -> QU).
    compsep_subcomm = mpi_info.compsep.subcomm
    target_pol = "I" if mpi_info.compsep.subcolor == 0 else "QU"
    sampling_groups = params.sampling_groups_compsep if "sampling_groups_compsep" in params \
        else Bunch()

    # SkyModel wraps `comp_list` by reference, so a single instance reflects all in-place updates
    # made below, and is what we ultimately return.
    sky_model = SkyModel(comp_list)

    for sampling_group_name in sampling_groups:
        sampling_group = sampling_groups[sampling_group_name]
        if "enabled" in sampling_group and not sampling_group.enabled:
            continue

        sampled_components = sampling_group.comps if "comps" in sampling_group else None
        sampled_bands = sampling_group.bands if "bands" in sampling_group else None
        band_is_active = _sampling_group_selects_band(sampled_bands, mpi_info.compsep.band_name,
                                                      mpi_info.compsep.band_identifier)
        # This rank's components (for its own polarization stream) that take part in this group.
        # `active_sublist` shares Component objects with `comp_list`, so copying the solver result
        # into it updates `comp_list` on this rank; the broadcast below propagates it to all ranks.
        active_sublist = _filter_sampling_group_components(
            comp_list.split_for_eval_pol(target_pol), sampled_components)
        should_solve = band_is_active and len(active_sublist) > 0

        # The solving ranks of each polarization form their own solver communicator.
        solver_comm = compsep_subcomm.Split(0 if should_solve else MPI.UNDEFINED, key=compsep_rank)
        if should_solve:
            sample_class = sampling_group.sample_class
            if sample_class == "amplitude_sampler_perpix":
                solved_sublist = solve_compsep_perpix(solver_comm, detector_data, active_sublist,
                                                      params)
            elif sample_class == "amplitude_sampler_CG":
                solved_sublist = CompSepSolver(detector_data, params, solver_comm).solve(
                    active_sublist)
            else:
                raise ValueError(
                    f"Unknown compsep sampling class {sample_class!r} for sampling group "
                    f"{sampling_group_name!r}.")
            active_sublist.copy_matching_data_from(solved_sublist)
            solver_comm.Free()

        any_active = compsep_comm.allreduce(1 if should_solve else 0, op=MPI.SUM)
        if not any_active:
            if compsep_rank == mpi_info.compsep.master:
                logger.info(
                    f"Sampling group {sampling_group_name!r} had no active band/component overlap.")
            continue

        # Restore global consistency: for each polarization that was solved this group, the
        # lowest-ranked solver (which holds the authoritative result) broadcasts its component views
        # to all ranks. A polarization that no rank solved is already identical everywhere.
        for eval_pol in ("I", "QU"):
            solved_here = should_solve and target_pol == eval_pol
            source = compsep_comm.allreduce(compsep_rank if solved_here else compsep_comm.size,
                                            op=MPI.MIN)
            if source < compsep_comm.size:
                comp_list.broadcast_pol_views(compsep_comm, eval_pol=eval_pol, source=source)

        # Print new per-band chi2s against the updated sky model.
        sky_model_at_band = sky_model.get_sky_at_nu(detector_data.nu, detector_data.nside, "IQU",
                                                    fwhm=np.deg2rad(detector_data.fwhm/60.0))
        pol_names = ["Q", "U"] if detector_data.pol else ["I"]
        pol_offset = 1 if detector_data.pol else 0
        for ipol in range(detector_data.npol):
            chi2 = np.mean(np.abs(detector_data.map_sky[ipol] -
                                sky_model_at_band[ipol + pol_offset])/detector_data.map_rms[ipol])
            logger.info(f"Reduced chi2 on rank {compsep_rank} for pol={pol_names[ipol]} "\
                        f"({detector_data.nu}GHz): {chi2:.3f}")

    if compsep_rank == mpi_info.compsep.master:
        write_compsep_chain_to_file(comp_list.joined(), params, chain, iter)

    return sky_model  # Return the full sky realization for my band.
