"""The component-separation side of one Gibbs iteration, and the settings that configure it.

`init_compsep_processing` builds the component list and validates the `compsep` block of the
parameter file once; `process_compsep` runs one iteration's sampling groups over the band maps
received from TOD processing. A *sampling group* names a set of components and bands to be sampled
together, by one of three solvers: CG (`cg_solver`), per-pixel (`perpix_solver`), or Metropolis-
Hastings (`mcmc`). The `*Config` dataclasses here validate each group's parameter block.
"""
import numpy as np
import healpy as hp
import logging
from copy import deepcopy
from dataclasses import dataclass, fields
from numbers import Integral, Real
from mpi4py import MPI
from numpy.typing import NDArray
from pixell.bunch import Bunch
from typing import Self

from commander4.diagnostics.log import logassert
from commander4.data_models.detector_map import DetectorMap
from commander4.sky.comp_list import CompList
from commander4.sky.diffuse_components import DiffuseComponent
from commander4.sky.sky_model import SkyModel
from commander4.compsep.cg_solver import CompSepSolver
from commander4.compsep.perpix_solver import solve_compsep_perpix
from commander4.compsep.spectral_index import SpectralIndexSamplingGroup
from commander4.file_io.chain_writer import write_compsep_chain_to_file
from commander4.polarization import get_execution_band_id, EXECUTION_POLS
from commander4.parameters.schema import resolve_band_lmax

logger = logging.getLogger(__name__)


def _normalize_names(value, field_name: str) -> tuple[str, ...] | None:
    """Normalize an omitted/``all``/list selection into an immutable names-or-all value."""
    if value is None or value == "all":
        return None
    if isinstance(value, str):
        raise ValueError(f"{field_name} must be a list of names, 'all', or omitted.")
    names = tuple(value)
    if any(not isinstance(name, str) for name in names):
        raise ValueError(f"Every entry in {field_name} must be a string.")
    return names


@dataclass(frozen=True)
class SamplingGroupConfig:
    """Common construction and component selection for one named compsep sampling group."""

    name: str
    enabled: bool = True
    comps: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(f"Sampling group {self.name!r} enabled must be true or false.")

    @classmethod
    def from_block(cls, name: str, block: Bunch | dict) -> Self:
        """Construct this group and reject parameters not declared by its config class."""
        try:
            values = dict(block)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Sampling group {name!r} must be a parameter block.") from error
        for field_name in ("comps", "bands"):
            if field_name in values:
                values[field_name] = _normalize_names(
                    values[field_name], f"{name}.{field_name}")
        constructor_fields = {item.name for item in fields(cls) if item.init}
        parameter_fields = constructor_fields - {"name"}
        unknown = sorted(set(values) - parameter_fields)
        if unknown:
            raise ValueError(f"Unknown key(s) {unknown} in sampling group {name!r}. That group "
                             f"accepts {sorted(parameter_fields)}.")
        try:
            return cls(name=name, **values)
        except TypeError as error:
            raise ValueError(f"Invalid sampling group {name!r}: {error}") from error


@dataclass(frozen=True)
class CGSamplingGroupConfig(SamplingGroupConfig):
    """One C3-style CG amplitude group, including all solver controls."""

    bands: tuple[str, ...] | None = None
    optimize: bool = False
    max_iter: int = 200
    max_iter_pol: int = 200
    err_tol: float = 1.0e-8
    preconditioner: str = "JointPreconditioner"
    dense_matrix_debug_mode: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if (not isinstance(self.max_iter, Integral) or isinstance(self.max_iter, bool)
                or not isinstance(self.max_iter_pol, Integral)
                or isinstance(self.max_iter_pol, bool)):
            raise ValueError(f"CG sampling group {self.name!r} iteration limits must be integers.")
        if self.max_iter < 0 or self.max_iter_pol < 0:
            raise ValueError(
                f"CG sampling group {self.name!r} iteration limits cannot be negative.")
        if not isinstance(self.err_tol, Real) or not np.isfinite(self.err_tol) or self.err_tol < 0:
            raise ValueError(f"CG sampling group {self.name!r} err_tol must be non-negative and "
                             "finite.")
        if (not isinstance(self.optimize, bool)
                or not isinstance(self.dense_matrix_debug_mode, bool)):
            raise ValueError(
                f"CG sampling group {self.name!r} boolean options must be true or false.")
        if not isinstance(self.preconditioner, str) or not self.preconditioner:
            raise ValueError(f"CG sampling group {self.name!r} preconditioner must be a name.")

@dataclass(frozen=True)
class PerPixelSamplingGroupConfig(SamplingGroupConfig):
    """One per-pixel amplitude sampling group."""

    bands: tuple[str, ...] | None = None

@dataclass(frozen=True)
class MCMCSamplingGroupConfig(SamplingGroupConfig):
    """One MCMC group; currently the sampled parameters are spectral indices."""

    parameters: str = "spectral_indices"
    update_amplitude_groups: tuple[str, ...] = ()
    numstep: int = 1
    chisq_bands: tuple[str, ...] | None = None
    chisq_mask: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.parameters != "spectral_indices":
            raise ValueError(f"MCMC sampling group {self.name!r} has unsupported parameters "
                             f"{self.parameters!r}; only 'spectral_indices' is implemented.")
        if not isinstance(self.numstep, int) or isinstance(self.numstep, bool) or self.numstep < 1:
            raise ValueError(f"MCMC sampling group {self.name!r} numstep must be at least 1.")
        if self.chisq_mask is not None and not isinstance(self.chisq_mask, str):
            raise ValueError(
                f"MCMC sampling group {self.name!r} chisq_mask must be a path or null.")

    @classmethod
    def from_block(cls, name: str, block: Bunch | dict) -> Self:
        values = dict(block)
        if "update_amplitude_groups" in values:
            update_groups = _normalize_names(
                values["update_amplitude_groups"], f"{name}.update_amplitude_groups")
            values["update_amplitude_groups"] = () if update_groups is None else update_groups
        if "chisq_bands" in values:
            values["chisq_bands"] = _normalize_names(
                values["chisq_bands"], f"{name}.chisq_bands")
        return super().from_block(name, values)


@dataclass(frozen=True)
class CompSepState:
    """Resolved rank-local component-separation settings and loaded mask data."""

    band_name: str
    band_identifier: str
    target_pol: str
    amplitude_method: str | None
    amplitude_groups: dict[str, CGSamplingGroupConfig | PerPixelSamplingGroupConfig]
    mcmc_groups: dict[str, MCMCSamplingGroupConfig]
    double_precision: bool
    num_threads: int
    chisq_masks: dict[str, NDArray]


def _read_sampling_groups(params: Bunch, key: str,
                          config_class: type[SamplingGroupConfig]
                          ) -> dict[str, SamplingGroupConfig]:
    """Construct and return the enabled groups from one method-specific compsep section."""
    raw_groups = params.compsep[key] if key in params.compsep else Bunch()
    groups = {}
    for name in raw_groups:
        config = config_class.from_block(name, raw_groups[name])
        if config.enabled:
            groups[name] = config
    return groups


def _resolve_sampling_groups(params: Bunch) -> tuple[
        dict[str, CGSamplingGroupConfig],
        dict[str, PerPixelSamplingGroupConfig],
        dict[str, MCMCSamplingGroupConfig]]:
    """Resolve the three method-specific group sections and enforce amplitude-solver exclusivity."""
    cg_groups = _read_sampling_groups(params, "cg_sampling_groups", CGSamplingGroupConfig)
    per_pixel_groups = _read_sampling_groups(
        params, "per_pixel_sampling_groups", PerPixelSamplingGroupConfig)
    mcmc_groups = _read_sampling_groups(params, "mcmc_sampling_groups", MCMCSamplingGroupConfig)
    if cg_groups and per_pixel_groups:
        raise ValueError("CG and per-pixel amplitude sampling groups are mutually exclusive; "
                         "configure only one method.")
    return cg_groups, per_pixel_groups, mcmc_groups


def _read_chisq_masks(compsep_comm: MPI.Comm,
                      mcmc_groups: dict[str, MCMCSamplingGroupConfig]) -> dict[str, NDArray]:
    """Read each MCMC group's optional ``chisq_mask`` HEALPix file once, and broadcast it.

    This is C3's ``MCMC_SAMPLING_GROUP_CHISQ_MASK``: one mask per sampling group, applied to every
    band in that group's chi-squared, restricting which sky the accept/reject decision sees. It is
    deliberately *not* a per-band data mask: it does not enter the amplitude solve or the noise
    model, only the MCMC likelihood.

    Read at native resolution and kept as-is; `resolve_chisq_mask` ud_grades and thresholds it to
    each band's nside when the sampling group is built. Only field 0 is read (as on the TOD side),
    so a single mask applies to both Q and U for polarization bands.

    Returns:
        Mapping from group name to its raw mask map, omitting groups that define no mask.
    """
    masks: dict[str, NDArray] = {}
    if compsep_comm.Get_rank() == 0:
        for group_name in mcmc_groups:
            group = mcmc_groups[group_name]
            if not group.chisq_mask:
                continue
            masks[group_name] = hp.read_map(group.chisq_mask, field=0, dtype=np.float64)
            logger.info(f"CompSep: read chisq mask for MCMC group {group_name!r} from "
                        f"{group.chisq_mask} (fsky = {np.mean(masks[group_name] > 0.5):.3f}).")
    return compsep_comm.bcast(masks, root=0)


def _sampling_group_selects_band(selected_bands: tuple[str, ...] | list[str] | None, band_name: str,
                                 band_identifier: str) -> bool:
    """Whether a sampling group acts on a band, matched by base name or execution-view identifier.

    `selected_bands` of None means "all bands".
    """
    if selected_bands is None:
        return True
    return band_name in selected_bands or band_identifier in selected_bands


def _filter_sampling_group_components(comp_list: CompList,
                                      selected_components: tuple[str, ...] | list[str] | None
                                      ) -> CompList:
    """Subset of `comp_list` whose component names are selected by the sampling group.

    `selected_components` of None means "all components". The returned `CompList` shares the
    underlying `Component` objects with `comp_list` (it is a view, not a copy).
    """
    if selected_components is None:
        return CompList(list(comp_list))
    selected_names = set(selected_components)
    return CompList([comp for comp in comp_list if comp.comp_name in selected_names])


def _validate_sampling_group_references(sampling_groups: dict[str, SamplingGroupConfig],
                                        comp_list: CompList,
                                        params: Bunch) -> None:
    """Fail fast if any enabled sampling group references a non-existent component or band.

    `comps` and `bands` are expected to be lists of strings naming existing components and bands
    (bands may be given either as a base name or as an execution-view identifier), the string
    "all", or omitted. The latter two select everything and are not checked against names.
    """
    known_comp_names = {comp.comp_name for comp in comp_list.joined()}
    known_band_names = set()
    for band_str in params.compsep.bands:
        band = params.compsep.bands[band_str]
        if not band.enabled:
            continue
        known_band_names.add(band_str)
        for eval_pol in EXECUTION_POLS[band.polarization]:
            known_band_names.add(get_execution_band_id(band_str, eval_pol))

    for group_name in sampling_groups:
        group = sampling_groups[group_name]
        selected_comps = group.comps
        if selected_comps is not None:
            unknown = sorted(set(selected_comps) - known_comp_names)
            logassert(not unknown,
                      f"Sampling group {group_name!r} references unknown component(s) {unknown}. "
                      f"Known components: {sorted(known_comp_names)}.", logger)
        if isinstance(group, (CGSamplingGroupConfig, PerPixelSamplingGroupConfig)):
            selected_bands = group.bands
        else:
            selected_bands = group.chisq_bands
        if selected_bands is not None:
            unknown = sorted(set(selected_bands) - known_band_names)
            logassert(not unknown,
                      f"Sampling group {group_name!r} references unknown band(s) {unknown}. "
                      f"Known bands: {sorted(known_band_names)}.", logger)


def _validate_sampling_group_dependencies(
        amplitude_groups: dict[str, CGSamplingGroupConfig | PerPixelSamplingGroupConfig],
        mcmc_groups: dict[str, MCMCSamplingGroupConfig]) -> None:
    """Ensure every MCMC amplitude update names an enabled amplitude group."""
    for group in mcmc_groups.values():
        unknown = sorted(set(group.update_amplitude_groups) - set(amplitude_groups))
        logassert(not unknown,
                  f"MCMC sampling group {group.name!r} update_amplitude_groups references unknown "
                  f"or disabled amplitude group(s) {unknown}. Known enabled amplitude groups: "
                  f"{sorted(amplitude_groups)}.", logger)


def _validate_component_lmax(comp_list: CompList, params: Bunch) -> None:
    """Report component multipoles that no band can constrain.

    `Component.project_comp_to_band` truncates a component to the band's lmax, so a component
    multipole above *every* enabled band's lmax never meets the data. In a constrained realization
    (`optimize: false`) the CG solver then draws it straight from the C(l) prior, which is usually
    set orders of magnitude above the true signal, so the amplitudes come back dominated by prior
    noise at high l and the chi^2 diagnostic blows up with them.

    C3 leaves this to the user (its BAND_LMAX and COMP_AMP_LMAX are unchecked), but its parameter
    files pair the two deliberately, and its COMP_L_APOD tapers the prior away over exactly this
    range. We check both here: apodization at or below the highest band lmax is a warning, since
    the taper is still near unity where it starts and only suppresses the far tail; no protection
    at all is an error. Both are logged rather than raised, since the run is still meaningful for
    the multipoles the data do constrain and a hard failure would be unhelpful mid-chain.
    """
    band_lmaxes = {}
    for band_str in params.compsep.bands:
        band = params.compsep.bands[band_str]
        if not band.enabled:
            continue
        experiment = None if band.get_from == "file" else band.get_from
        source = band if experiment is None else params.experiments[experiment].bands[band_str]
        band_lmaxes[band_str] = resolve_band_lmax(params, band_str, experiment, source.eval_nside)
    if not band_lmaxes:
        return
    highest_lmax = max(band_lmaxes.values())
    at_highest = sorted(name for name in band_lmaxes if band_lmaxes[name] == highest_lmax)
    highest_band = at_highest[0] if len(at_highest) == 1 else f"{len(at_highest)} bands"

    for comp in comp_list.joined():
        if not isinstance(comp, DiffuseComponent) or comp.lmax <= highest_lmax:
            continue
        unseen = f"l = {highest_lmax + 1}-{comp.lmax}"
        context = (f"Component {comp.comp_name!r} has lmax = {comp.lmax}, above the highest band "
                   f"lmax {highest_lmax} ({highest_band}), so {unseen} is constrained by the C(l) "
                   f"prior alone.")
        if comp.Cl_prior_l_apod <= highest_lmax:
            logger.warning(f"{context} Only Cl_prior_l_apod = {comp.Cl_prior_l_apod} keeps those "
                           "modes finite, and the taper barely suppresses the multipoles just "
                           "above the band lmax. Lower the component lmax if you do not need it.")
        else:
            logger.error(f"{context} Nothing suppresses them: set the component lmax to "
                         f"{highest_lmax} or below, raise the bands' lmax, or set "
                         f"Cl_prior_l_apod <= {highest_lmax} to taper the prior away.")


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
    band_pol = "QU" if detector_data.pol else "I"
    fixed_sky = SkyModel(CompList(fixed_comps)).get_sky_at_nu(
        detector_data.nu, detector_data.nside, band_pol, fwhm=detector_data.fwhm_rad)
    residual = deepcopy(detector_data)
    residual.map_sky = detector_data.map_sky - fixed_sky.astype(detector_data.map_sky.dtype,
                                                                copy=False)
    return residual


def init_compsep_processing(mpi_info: Bunch, params: Bunch)\
    -> tuple[CompList, Bunch, str, Bunch, CompSepState]:
    """Set up the rank-local execution view for component separation.

    Each CompSep rank owns exactly one execution view of one band. The global CompSep rank space is
    split into a contiguous intensity block (ranks ``[0, QU_master)``) followed by a contiguous QU
    block (ranks ``[QU_master, size)``), and we match the current rank against those two streams.

    Args:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        params (Bunch): The parameters from the input parameter file.

    Returns:
        comp_list (CompList): The full execution-view component list, identical on all CompSep
            ranks.
        mpi_info (Bunch): `mpi_info`, extended with this rank's band name/identifier and the
            band-master dictionaries.
        band_identifier (str): Unique string for the band execution view this rank is working on.
        my_band (Bunch): The parameter-file subset for the band this rank is working on.
        compsep_state: Resolved component-separation settings for this rank.
    """
    logger.debug(f"CompSep: Hello from CompSep-rank {mpi_info.compsep.rank} (on machine "
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
    for band_str in params.compsep.bands:
        band = params.compsep.bands[band_str]
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
                logger.debug(f"Rank {mpi_info.compsep.rank} matched band {band_identifier}.")
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
                  "Check that compsep.bands matches the configured I/QU rank counts.",
                  logger)

    cg_groups, per_pixel_groups, mcmc_groups = _resolve_sampling_groups(params)
    amplitude_groups = cg_groups if cg_groups else per_pixel_groups
    _validate_sampling_group_references(amplitude_groups, comp_list, params)
    _validate_sampling_group_references(mcmc_groups, comp_list, params)
    _validate_sampling_group_dependencies(amplitude_groups, mcmc_groups)
    if mpi_info.compsep.rank == mpi_info.compsep.master:  # One report, not one per band.
        _validate_component_lmax(comp_list, params)
    if cg_groups:
        amplitude_method = "cg"
    elif per_pixel_groups:
        amplitude_method = "per_pixel"
    else:
        amplitude_method = None

    double_precision = params.compsep.float_precision == "double"
    nthreads = params.resources.compsep.num_threads
    if not isinstance(nthreads, int):
        nthreads = nthreads[mpi_info.compsep.rank]
    # Read the MCMC groups' chi-squared masks once here, rather than on every Gibbs iteration when
    # the sampling groups are rebuilt.
    chisq_masks = _read_chisq_masks(mpi_info.compsep.comm, mcmc_groups)
    compsep_state = CompSepState(
        band_name=band_name,
        band_identifier=band_identifier,
        target_pol="I" if mpi_info.compsep.subcolor == 0 else "QU",
        amplitude_method=amplitude_method,
        amplitude_groups=amplitude_groups,
        mcmc_groups=mcmc_groups,
        double_precision=double_precision,
        num_threads=nthreads,
        chisq_masks=chisq_masks,
    )

    # Load the initial component alms (from each component's init_from / init_chain_path, else
    # zeros). Done identically on every CompSep rank so comp_list starts globally consistent.
    comp_list.load_initial_alms(params)
    # Likewise for the Gaussian amplitude prior's mean mu (each component's amp_prior_mean_map,
    # else a zero-mean prior). Read once here; the CG applies S^{-1/2} on every solve.
    comp_list.load_amp_prior_means()

    data_world = (band_identifier, mpi_info.world.rank)
    data_compsep = (band_identifier, mpi_info.compsep.rank)
    all_data_world = mpi_info.compsep.comm.allgather(data_world)
    all_data_compsep = mpi_info.compsep.comm.allgather(data_compsep)
    world_band_masters_dict = {item[0]: item[1] for item in all_data_world if item is not None}
    compsep_band_masters_dict = {item[0]: item[1] for item in all_data_compsep if item is not None}
    mpi_info.world.compsep_band_masters = world_band_masters_dict
    mpi_info.compsep.compsep_band_masters = compsep_band_masters_dict

    return comp_list, mpi_info, band_identifier, my_band, compsep_state


def get_initial_sky_model(comp_list: CompList) -> SkyModel:
    """Wrap the freshly-initialized `comp_list` as a SkyModel for the pre-loop initial send to TOD.

    `comp_list` already holds its initial alms (set in `init_compsep_processing`), so this is just
    the same `SkyModel(comp_list)` that `process_compsep` produces in later iterations.
    """
    return SkyModel(comp_list)


def _run_amplitude_group(mpi_info: Bunch, compsep_state: CompSepState,
                         detector_data: DetectorMap, comp_list: CompList,
                         group: CGSamplingGroupConfig | PerPixelSamplingGroupConfig) -> None:
    """Run one amplitude group and broadcast its result to every CompSep rank."""
    compsep = mpi_info.compsep
    band_is_active = _sampling_group_selects_band(
        group.bands, compsep_state.band_name, compsep_state.band_identifier)
    active_sublist = _filter_sampling_group_components(
        comp_list.split_for_eval_pol(compsep_state.target_pol), group.comps)
    should_solve = band_is_active and len(active_sublist) > 0

    solver_comm = compsep.subcomm.Split(0 if should_solve else MPI.UNDEFINED, key=compsep.rank)
    if should_solve:
        residual_data = _build_conditional_residual(
            detector_data, comp_list, compsep_state.target_pol, active_sublist)
        if compsep_state.amplitude_method == "cg":
            solved_sublist = CompSepSolver(
                residual_data, solver_comm, group,
                double_precision=compsep_state.double_precision,
                nthreads=compsep_state.num_threads,
            ).solve(active_sublist)
        elif compsep_state.amplitude_method == "per_pixel":
            solved_sublist = solve_compsep_perpix(
                solver_comm, residual_data, active_sublist,
                double_precision=compsep_state.double_precision,
            )
        else:
            raise ValueError(
                f"Unknown amplitude sampling method {compsep_state.amplitude_method!r}.")
        active_sublist.copy_matching_data_from(solved_sublist)
        solver_comm.Free()

    any_active = compsep.comm.allreduce(1 if should_solve else 0, op=MPI.SUM)
    if not any_active:
        if compsep.rank == compsep.master:
            logger.verbose(f"Sampling group {group.name!r} had no active band/component overlap.")
        return

    # The lowest-ranked solver for each polarization owns the authoritative result.
    for eval_pol in ("I", "QU"):
        solved_here = should_solve and compsep_state.target_pol == eval_pol
        source = compsep.comm.allreduce(compsep.rank if solved_here else compsep.size, op=MPI.MIN)
        if source < compsep.size:
            comp_list.broadcast_pol_views(compsep.comm, eval_pol=eval_pol, source=source)


def _evaluate_chi2(mpi_info: Bunch, detector_data: DetectorMap, sky_model: SkyModel,
                   label: str | None = None) -> tuple[float, int]:
    """Evaluate the all-band whitened residual, optionally logging detailed results."""
    compsep = mpi_info.compsep
    band_pol = "QU" if detector_data.pol else "I"
    sky_at_band = sky_model.get_sky_at_nu(
        detector_data.nu, detector_data.nside, band_pol, fwhm=detector_data.fwhm_rad)
    pol_names = ["Q", "U"] if detector_data.pol else ["I"]
    chi2_local, ndof_local = 0.0, 0
    for ipol in range(detector_data.npol):
        observed = detector_data.inv_n_map[ipol] > 0
        z = ((detector_data.map_sky[ipol] - sky_at_band[ipol])[observed]
             * np.sqrt(detector_data.inv_n_map[ipol][observed]))
        chi2_local += np.sum(z**2, dtype=np.float64)
        ndof_local += z.size
        if label is not None:
            logger.verbose(f"Fit after {label} on rank {compsep.rank} for pol={pol_names[ipol]} "
                           f"({detector_data.nu}GHz): mean|z|={np.mean(np.abs(z)):.3f}, "
                           f"red.chi2={np.mean(z**2):.3f} (ndof={z.size}).")

    chi2_total = compsep.comm.allreduce(chi2_local, op=MPI.SUM)
    ndof_total = compsep.comm.allreduce(ndof_local, op=MPI.SUM)
    if label is not None and compsep.rank == compsep.master:
        logger.info(f"Fit after {label}, all bands: chi2={chi2_total:.6e}, ndof={ndof_total}, "
                    f"red.chi2={chi2_total/ndof_total:.4f}")
    return float(chi2_total), int(ndof_total)


def process_compsep(mpi_info: Bunch, compsep_state: CompSepState,
                    detector_data: DetectorMap, iter: int, chain: int,
                    params: Bunch, comp_list: CompList) -> SkyModel:
    """Perform a single component-separation iteration.

    Runs the configured CG or per-pixel amplitude groups first, followed by MCMC groups. An MCMC
    group may re-run named amplitude groups between proposal and accept/reject.

    Args:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        compsep_state: Resolved component-separation settings and masks for this rank.
        detector_data (DetectorMap): The detector map for this rank's band, cleaned of all "TOD"
            components (correlated noise and orbital dipole).
        iter (int): The current Gibbs iteration (used only for printing and seeding).
        chain (int): The current chain (used only for printing and seeding).
        params (Bunch): The parameters from the input parameter file.
        comp_list (CompList): The full execution-view component list, updated in place.

    Returns:
        The full sky realization, wrapping the updated `comp_list`.
    """
    compsep = mpi_info.compsep
    amplitude_groups = compsep_state.amplitude_groups
    mcmc_groups = compsep_state.mcmc_groups
    sky_model = SkyModel(comp_list)
    final_chi2: float | None = None
    final_ndof: int | None = None

    for group in amplitude_groups.values():
        _run_amplitude_group(mpi_info, compsep_state, detector_data, comp_list, group)
        method_label = "CG" if compsep_state.amplitude_method == "cg" else "per-pixel"
        final_chi2, final_ndof = _evaluate_chi2(
            mpi_info, detector_data, sky_model, f"{method_label} group {group.name!r}")

    for group in mcmc_groups.values():
        def resolve_amplitudes(group_names=group.update_amplitude_groups):
            for group_name in group_names:
                _run_amplitude_group(
                    mpi_info, compsep_state, detector_data, comp_list,
                    amplitude_groups[group_name])

        chisq_active = _sampling_group_selects_band(
            group.chisq_bands, compsep_state.band_name, compsep_state.band_identifier)
        sampler = SpectralIndexSamplingGroup(
            compsep.comm, detector_data, comp_list, target_pol=compsep_state.target_pol,
            selected_comps=group.comps, chisq_active=chisq_active,
            chisq_mask=compsep_state.chisq_masks.get(group.name), root=compsep.master)
        sampler.run(numstep=group.numstep, resolve_amplitudes=resolve_amplitudes)
        final_chi2, final_ndof = _evaluate_chi2(
            mpi_info, detector_data, sky_model, f"MCMC group {group.name!r}")

    # C3 evaluates the final map-domain residual during chain output. Do the same even when no
    # sampling group ran. Otherwise the last group's diagnostic already describes the final state.
    if final_chi2 is None or final_ndof is None:
        final_chi2, final_ndof = _evaluate_chi2(mpi_info, detector_data, sky_model)

    if compsep.rank == compsep.master:
        write_compsep_chain_to_file(comp_list.joined(), params, chain, iter)
        red_chi2 = final_chi2 / final_ndof if final_ndof > 0 else float("nan")
        z_chi2 = ((final_chi2 - final_ndof) / np.sqrt(2.0 * final_ndof)
                  if final_ndof > 0 else float("nan"))
        logger.summary(f"Chain {chain}, iteration {iter} complete: chi2={final_chi2:.6e}, "
                       f"ndof={final_ndof}, red.chi2={red_chi2:.4f}, z={z_chi2:.3f}.")

    return sky_model
