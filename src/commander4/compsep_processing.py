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
from commander4.solvers.spectral_index_sampler import SpectralIndexSamplingGroup
from commander4.output.write_chains_files import write_compsep_chain_to_file
from commander4.utils.execution_ids import get_execution_band_id, EXECUTION_POLS

logger = logging.getLogger(__name__)

# Sampling-group sections in the parameter file: the linear amplitude (CG) tier and the non-linear
# Metropolis-Hastings (MCMC) tier. They mirror C3's `CG_SAMPLING_GROUP*` / `MCMC_SAMPLING_GROUP*`.
CG_SAMPLING_GROUPS_KEY = "CG_sampling_groups_compsep"
MCMC_SAMPLING_GROUPS_KEY = "MCMC_sampling_groups_compsep"
# Sampler classes legal in each tier.
CG_SAMPLE_CLASSES = ("amplitude_sampler_CG", "amplitude_sampler_perpix")
MCMC_SAMPLE_CLASSES = ("sample_spectral_indices_uniform_MH",)


def _enabled_sampling_groups(params: Bunch, key: str) -> Bunch:
    """The sampling groups under `key`, excluding any with ``enabled: false`` (empty if absent)."""
    groups = params[key] if key in params else Bunch()
    return Bunch({name: groups[name] for name in groups
                  if "enabled" not in groups[name] or groups[name].enabled})


def _selected_names(sampling_group: Bunch, key: str) -> list[str] | None:
    """Normalize a sampling group's selection for `key` ('comps'/'bands'/...) to names-or-all.

    Returns None when the group selects everything, otherwise the explicit list of names. A
    selection can be spelled three ways in the parameter file: omitted, the literal string
    ``"all"``, or a list of names. The first two both mean "everything" and map to None; downstream
    consumers (`_sampling_group_selects_band`, `_filter_sampling_group_components`) treat None as
    "all", so the omitted/"all"/list distinction lives only here.
    """
    if key not in sampling_group:
        return None
    value = sampling_group[key]
    if isinstance(value, str) and value == "all":
        return None
    return value


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
    (bands may be given either as a base name or as an execution-view identifier), the string
    "all", or omitted. The latter two select everything and are not checked against names.
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
        selected_comps = _selected_names(group, "comps")
        if selected_comps is not None:
            unknown = sorted(set(selected_comps) - known_comp_names)
            logassert(not unknown,
                      f"Sampling group {group_name!r} references unknown component(s) {unknown}. "
                      f"Known components: {sorted(known_comp_names)}.", logger)
        selected_bands = _selected_names(group, "bands")
        if selected_bands is not None:
            unknown = sorted(set(selected_bands) - known_band_names)
            logassert(not unknown,
                      f"Sampling group {group_name!r} references unknown band(s) {unknown}. "
                      f"Known bands: {sorted(known_band_names)}.", logger)


def _validate_sampling_group_tiers(cg_groups: Bunch, mcmc_groups: Bunch, comp_list: CompList,
                                   params: Bunch) -> None:
    """Validate both sampling-group tiers and the MCMC->CG coupling.

    Checks (on top of `_validate_sampling_groups`' name checks for each tier): every group has a
    ``sample_class`` legal for its tier, and every MCMC group's ``update_CG_groups`` names existing
    enabled CG groups.
    """
    _validate_sampling_groups(cg_groups, comp_list, params)
    _validate_sampling_groups(mcmc_groups, comp_list, params)

    for group_name in cg_groups:
        sample_class = cg_groups[group_name].sample_class if "sample_class" in cg_groups[group_name] \
            else None
        logassert(sample_class in CG_SAMPLE_CLASSES,
                  f"CG sampling group {group_name!r} has sample_class {sample_class!r}; expected one "
                  f"of {list(CG_SAMPLE_CLASSES)}.", logger)

    for group_name in mcmc_groups:
        group = mcmc_groups[group_name]
        sample_class = group.sample_class if "sample_class" in group else None
        logassert(sample_class in MCMC_SAMPLE_CLASSES,
                  f"MCMC sampling group {group_name!r} has sample_class {sample_class!r}; expected "
                  f"one of {list(MCMC_SAMPLE_CLASSES)}.", logger)
        update_cg_groups = group.update_CG_groups if "update_CG_groups" in group else []
        unknown = sorted(set(update_cg_groups) - set(cg_groups.keys()))
        logassert(not unknown,
                  f"MCMC sampling group {group_name!r} update_CG_groups references unknown or "
                  f"disabled CG group(s) {unknown}. Known enabled CG groups: "
                  f"{sorted(cg_groups.keys())}.", logger)


def _build_conditional_residual(detector_data: DetectorMap, comp_list: CompList, target_pol: str,
                                active_sublist: CompList) -> DetectorMap:
    """Subtract the components held fixed by a sampling group from this band's map.

    A sampling group that solves only a subset of components must be conditioned on the rest: the
    fixed components' projected signal is removed from the data so the active components are fit to
    the residual rather than to the full observed sky (C3's ``compute_residual(cg_samp_group=...)``).
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
    band_pol = "QU" if detector_data.pol else "I"
    fixed_sky = SkyModel(CompList(fixed_comps)).get_sky_at_nu(
        detector_data.nu, detector_data.nside, band_pol, fwhm=detector_data.fwhm_rad)
    residual = deepcopy(detector_data)
    residual.map_sky = detector_data.map_sky - fixed_sky.astype(detector_data.map_sky.dtype,
                                                                copy=False)
    return residual


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

    _validate_sampling_group_tiers(_enabled_sampling_groups(params, CG_SAMPLING_GROUPS_KEY),
                                   _enabled_sampling_groups(params, MCMC_SAMPLING_GROUPS_KEY),
                                   comp_list, params)

    # Load the initial component alms (from each component's init_from / init_chain_path, else
    # zeros). Done identically on every CompSep rank so comp_list starts globally consistent.
    comp_list.load_initial_alms(params)

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


def get_initial_sky_model(comp_list: CompList) -> SkyModel:
    """Wrap the freshly-initialized `comp_list` as a SkyModel for the pre-loop initial send to TOD.

    `comp_list` already holds its initial alms (set in `init_compsep_processing`), so this is just
    the same `SkyModel(comp_list)` that `process_compsep` produces in later iterations.
    """
    return SkyModel(comp_list)


def process_compsep(mpi_info: Bunch, detector_data: DetectorMap, iter: int, chain: int,
                    params: Bunch, comp_list: CompList) -> SkyModel:
    """Perform a single component-separation iteration.

    Called by every CompSep rank, each of which owns one band execution view. A compsep iteration
    has two tiers: first every enabled CG (linear amplitude) sampling group is solved, then every
    enabled MCMC (non-linear Metropolis-Hastings) sampling group, each of which re-solves the
    CG groups it names via ``update_CG_groups`` between proposal and accept/reject.

    For each amplitude group the participating ranks (those whose band and at least one of whose
    components are selected) form a solver sub-communicator, the fixed components are subtracted to
    form a conditional residual, the requested sampler runs, and the updated components are
    broadcast so that `comp_list` is identical on every CompSep rank again before the next group.

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
    compsep_master = mpi_info.compsep.master
    # subcomm splits the CompSep ranks by polarization (subcolor 0 -> I, 1 -> QU).
    compsep_subcomm = mpi_info.compsep.subcomm
    target_pol = "I" if mpi_info.compsep.subcolor == 0 else "QU"
    cg_groups = _enabled_sampling_groups(params, CG_SAMPLING_GROUPS_KEY)
    mcmc_groups = _enabled_sampling_groups(params, MCMC_SAMPLING_GROUPS_KEY)

    # SkyModel wraps `comp_list` by reference, so a single instance reflects all in-place updates
    # made below, and is what we ultimately return.
    sky_model = SkyModel(comp_list)


    def run_amplitude_group(group: Bunch, group_name: str) -> None:
        """Solve one CG (linear amplitude) sampling group, updating `comp_list` in place on all ranks.

        Collective over `compsep_comm`: every rank must call it (non-solving ranks still take part
        in the communicator split, the activity allreduce, and the broadcast that restores global
        consistency).
        """
        sampled_components = _selected_names(group, "comps")
        sampled_bands = _selected_names(group, "bands")
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
            # Condition on the components this group holds fixed by subtracting them from the data.
            residual_data = _build_conditional_residual(detector_data, comp_list, target_pol,
                                                         active_sublist)
            sample_class = group.sample_class
            if sample_class == "amplitude_sampler_perpix":
                solved_sublist = solve_compsep_perpix(solver_comm, residual_data, active_sublist,
                                                      params)
            elif sample_class == "amplitude_sampler_CG":
                solved_sublist = CompSepSolver(residual_data, params, solver_comm).solve(
                    active_sublist)
            else:
                raise ValueError(
                    f"Unknown compsep amplitude sampling class {sample_class!r} for sampling group "
                    f"{group_name!r}.")
            active_sublist.copy_matching_data_from(solved_sublist)
            solver_comm.Free()

        any_active = compsep_comm.allreduce(1 if should_solve else 0, op=MPI.SUM)
        if not any_active:
            if compsep_rank == compsep_master:
                logger.info(f"Sampling group {group_name!r} had no active band/component overlap.")
            return

        # Restore global consistency: for each polarization that was solved this group, the
        # lowest-ranked solver (which holds the authoritative result) broadcasts its component views
        # to all ranks. A polarization that no rank solved is already identical everywhere.
        for eval_pol in ("I", "QU"):
            solved_here = should_solve and target_pol == eval_pol
            source = compsep_comm.allreduce(compsep_rank if solved_here else compsep_comm.size,
                                            op=MPI.MIN)
            if source < compsep_comm.size:
                comp_list.broadcast_pol_views(compsep_comm, eval_pol=eval_pol, source=source)


    def log_band_chi2(label: str) -> None:
        """Print this band's fit diagnostics against the full updated sky model.

        Reports two per-pixel whitened-residual statistics z = (d - model)/rms: the mean absolute
        deviation mean(|z|) (≈0.80 for a good fit) and the reduced chi-square mean(z^2) (≈1).
        """
        band_pol = "QU" if detector_data.pol else "I"
        sky_model_at_band = sky_model.get_sky_at_nu(detector_data.nu, detector_data.nside, band_pol,
                                                    fwhm=detector_data.fwhm_rad)
        pol_names = ["Q", "U"] if detector_data.pol else ["I"]
        for ipol in range(detector_data.npol):
            z = (detector_data.map_sky[ipol] - sky_model_at_band[ipol]) / detector_data.map_rms[ipol]
            logger.info(f"Fit after {label} on rank {compsep_rank} for pol={pol_names[ipol]} "
                        f"({detector_data.nu}GHz): mean|z|={np.mean(np.abs(z)):.3f}, "
                        f"red.chi2={np.mean(z**2):.3f}")


    # Tier 1: linear amplitude (CG) sampling groups.
    for group_name in cg_groups:
        run_amplitude_group(cg_groups[group_name], group_name)
        log_band_chi2(f"CG group {group_name!r}")

    # Tier 2: non-linear (MCMC) sampling groups, each coupled to re-solving its named CG groups.
    for group_name in mcmc_groups:
        group = mcmc_groups[group_name]
        update_cg_group_names = group.update_CG_groups if "update_CG_groups" in group else []

        # The sampler calls resolve_amplitudes between each proposal and its accept/reject, re-solving
        # the CG amplitude groups this MCMC group names in `update_CG_groups` (C3's UPDATE_CG_GROUPS)
        # so the likelihood sees amplitudes conditioned on the proposed spectral indices. The callback
        # runs synchronously inside sampler.run below, before `update_cg_group_names` is reassigned.
        def resolve_amplitudes():
            for cg_name in update_cg_group_names:
                run_amplitude_group(cg_groups[cg_name], cg_name)

        chisq_bands = _selected_names(group, "chisq_bands")
        chisq_active = _sampling_group_selects_band(chisq_bands, mpi_info.compsep.band_name,
                                                    mpi_info.compsep.band_identifier)
        # Build this rank's spectral-index MCMC group and take `numstep` coupled MH steps; proposals
        # and the accept/reject decision are made on `compsep_master` and broadcast to all ranks.
        sampler = SpectralIndexSamplingGroup(
            compsep_comm, detector_data, comp_list, target_pol=target_pol,
            selected_comps=_selected_names(group, "comps"), chisq_active=chisq_active,
            root=compsep_master)
        sampler.run(numstep=group.numstep if "numstep" in group else 1,
                    resolve_amplitudes=resolve_amplitudes)
        log_band_chi2(f"MCMC group {group_name!r}")

    if compsep_rank == compsep_master:
        write_compsep_chain_to_file(comp_list.joined(), params, chain, iter)

    return sky_model  # Return the full sky realization for my band.
