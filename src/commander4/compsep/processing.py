"""The component-separation side of one Gibbs iteration, and the settings that configure it.

`init_compsep_processing` builds the component list and reads the `compsep` block of the
parameter file once; `process_compsep` runs one iteration's sampling groups over the band maps
received from TOD processing. A *sampling group* names a set of components and bands to be sampled
together. An amplitude group is solved with CG or per pixel (`amplitude_sampling`), and an MCMC
group with Metropolis-Hastings (`mcmc`). The settings of a group are a plain `Bunch`, with the
names of the parameter file (`read_sampling_groups`).
"""
import numpy as np
import healpy as hp
import logging
import os
from copy import deepcopy
from mpi4py import MPI
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.data_models.detector_map import DetectorMap
from commander4.sky.comp_list import CompList
from commander4.sky.diffuse_components import DiffuseComponent
from commander4.sky.sky_model import SkyModel
from commander4.compsep.amplitude_sampling import sample_amplitudes
from commander4.compsep.chisq import collect_fit_diagnostics, evaluate_chi2
from commander4.compsep.spectral_index import SpectralIndexSamplingGroup
from commander4.file_io import paths
from commander4.file_io.chain_restart import ChainStart
from commander4.file_io.chain_writer import write_compsep_chain_to_file
from commander4.polarization import get_execution_band_id, EXECUTION_POLS
from commander4.parameters.schema import read_settings, resolve_band_lmax, resolve_param
from commander4.diagnostics.performance import benchmark, bench_summary, bench_reset

logger = logging.getLogger(__name__)

# The settings of each kind of sampling group, with their defaults. For `comps`, `bands` and
# `chisq_bands`, None and "all" both select everything.
CG_GROUP_DEFAULTS = {"enabled": True, "comps": None, "bands": None, "optimize": False,
                     "max_iter": 200, "max_iter_pol": 200, "err_tol": 1.0e-8,
                     "preconditioner": "JointPreconditioner"}
PER_PIXEL_GROUP_DEFAULTS = {"enabled": True, "comps": None, "bands": None}
MCMC_GROUP_DEFAULTS = {"enabled": True, "parameters": "spectral_indices", "comps": None,
                       "update_amplitude_groups": [], "numstep": 1, "chisq_bands": None,
                       "chisq_mask": None}


def names_or_all(names: list[str] | str | None, known_names: list[str],
                 setting: str) -> list[str]:
    """The names a setting selects: `names` itself, or all of `known_names` for "all" or None."""
    if names is None or names == "all":
        return list(known_names)
    if isinstance(names, str):
        raise ValueError(f"'{setting}' must be a list of names or \"all\", not {names!r}.")
    unknown = sorted(set(names) - set(known_names))
    if unknown:
        raise ValueError(f"'{setting}' names unknown {unknown}. The known names are "
                         f"{known_names}.")
    return list(names)


def read_sampling_groups(params: Bunch) -> Bunch:
    """The enabled sampling groups of `compsep`, with defaults for the settings they leave out.

    The result has the names of the parameter file: `cg_sampling_groups`,
    `per_pixel_sampling_groups` and `mcmc_sampling_groups` are each a dict from group name to the
    settings of that group. A group also holds its own `name`, and its `comps`, `bands` and
    `chisq_bands` are lists of names. `amplitude_groups` is the CG groups or the per-pixel groups,
    whichever the run uses.
    """
    comp_names = [name for name in params.components if params.components[name].enabled]
    # A group can name a whole band ("Planck30GHz") or one view of it ("Planck30GHz_QU").
    band_names = []
    for band_name in params.compsep.bands:
        band = params.compsep.bands[band_name]
        if band.enabled:
            band_names.append(band_name)
            for pol in EXECUTION_POLS[band.polarization]:
                band_names.append(get_execution_band_id(band_name, pol))

    cfg = Bunch(cg_sampling_groups={}, per_pixel_sampling_groups={}, mcmc_sampling_groups={})
    for section, defaults in (("cg_sampling_groups", CG_GROUP_DEFAULTS),
                              ("per_pixel_sampling_groups", PER_PIXEL_GROUP_DEFAULTS)):
        if section not in params.compsep:
            continue
        for name in params.compsep[section]:
            group = read_settings(params, name, (f"compsep.{section}",), defaults)
            if not group.enabled:
                continue
            group.name = name
            group.comps = names_or_all(group.comps, comp_names, f"{section}.{name}.comps")
            group.bands = names_or_all(group.bands, band_names, f"{section}.{name}.bands")
            cfg[section][name] = group
    # CG and per-pixel are two solvers for the same step, and a run uses one of them.
    if cfg.cg_sampling_groups and cfg.per_pixel_sampling_groups:
        raise ValueError("CG and per-pixel amplitude sampling groups are mutually exclusive; "
                         "configure only one method.")
    cfg.amplitude_groups = cfg.cg_sampling_groups if cfg.cg_sampling_groups \
        else cfg.per_pixel_sampling_groups

    if "mcmc_sampling_groups" not in params.compsep:
        return cfg
    for name in params.compsep.mcmc_sampling_groups:
        section = f"mcmc_sampling_groups.{name}"
        group = read_settings(params, name, ("compsep.mcmc_sampling_groups",), MCMC_GROUP_DEFAULTS)
        if not group.enabled:
            continue
        if group.parameters != "spectral_indices":
            raise ValueError(f"'{section}.parameters' is {group.parameters!r}, but only "
                             "'spectral_indices' is implemented.")
        group.name = name
        # `comps` stays None for "all": the sampler then takes every component that sets
        # `sample_spectral_index`.
        if group.comps == "all":
            group.comps = None
        if group.comps is not None:
            group.comps = names_or_all(group.comps, comp_names, f"{section}.comps")
        group.chisq_bands = names_or_all(group.chisq_bands, band_names, f"{section}.chisq_bands")
        unknown = sorted(set(group.update_amplitude_groups) - set(cfg.amplitude_groups))
        if unknown:
            raise ValueError(f"'{section}.update_amplitude_groups' names unknown or disabled "
                             f"amplitude group(s) {unknown}. The enabled amplitude groups are "
                             f"{list(cfg.amplitude_groups)}.")
        cfg.mcmc_sampling_groups[name] = group
    return cfg


def _read_chisq_masks(compsep_comm: MPI.Comm, mcmc_groups: dict[str, Bunch]) -> dict[str, NDArray]:
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


def _validate_mcmc_sampled_components(mcmc_groups: dict[str, Bunch], comp_list: CompList) -> None:
    """Report components an MCMC group names that have spectral-index sampling switched off.
    """
    sampling_off = set()
    for comp in comp_list.joined():
        if "sample_spectral_index" not in comp.comp_params \
                or not bool(comp.comp_params.sample_spectral_index):
            sampling_off.add(comp.comp_name)

    for group in mcmc_groups.values():
        if group.comps is None:
            continue
        ignored = sorted(set(group.comps) & sampling_off)
        if ignored:
            logger.error(
                f"MCMC sampling group {group.name!r} names component(s) {ignored}, but they do not "
                "set 'sample_spectral_index: true' in their own params block, so their spectral "
                "indices are held fixed and the group ignores them. Set the flag on each component "
                "you want sampled, or drop it from the group's 'comps'.")
            if not set(group.comps) - sampling_off:
                logger.error(f"MCMC sampling group {group.name!r} has no component left to sample "
                             "and will do nothing.")


def _validate_component_lmax(comp_list: CompList, params: Bunch) -> None:
    """Report component multipoles that no band can constrain.

    `Component.comp2band` truncates a component to the band's lmax, so a component
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

    A component without a prior (`Cl_prior_amplitude: null`) is not reported: its unconstrained
    multipoles stay at zero.
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
        if not isinstance(comp, DiffuseComponent) or comp.lmax <= highest_lmax \
                or comp.Cl_prior_amplitude is None:
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


def init_compsep_processing(mpi_info: Bunch, params: Bunch, start: ChainStart)\
    -> tuple[dict[int, CompList], Bunch, str, Bunch, Bunch]:
    """Set up the rank-local execution view for component separation.

    Each CompSep rank owns exactly one execution view of one band. The global CompSep rank space is
    split into a contiguous intensity block (ranks ``[0, QU_master)``) followed by a contiguous QU
    block (ranks ``[QU_master, size)``), and we match the current rank against those two streams.

    Args:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        params (Bunch): The parameters from the input parameter file.
        start (ChainStart): Where the two chains start (see `file_io.chain_restart`).

    Returns:
        comp_lists_by_chain (dict[int, CompList]): Each chain's full execution-view component
            list, initialized and identical on all CompSep ranks.
        mpi_info (Bunch): `mpi_info`, extended with this rank's band name/identifier and the
            band-master dictionaries.
        band_identifier (str): Unique string for the band execution view this rank is working on.
        my_band (Bunch): The parameter-file subset for the band this rank is working on.
        cfg (Bunch): The settings of component separation: the sampling groups (see
            `read_sampling_groups`), `double_precision`, this rank's `num_threads`, and the
            chain-output settings `nside_chisq`, `include_chisq_map` and `include_residual_maps`.
    """
    logger.debug(f"CompSep: Hello from CompSep-rank {mpi_info.compsep.rank} (on machine "
                 f"{mpi_info.processor_name}), dedicated to band {mpi_info.compsep.rank}.")

    comp_list = CompList.init_from_params(params.components, params)

    # MPI setup already selected this rank's execution view from the same inventory used to count
    # tasks. Only the parameter block needs copying because its polarization is narrowed to I or QU.
    band_name = mpi_info.band.name
    band_identifier = mpi_info.band.identifier
    my_band = deepcopy(params.compsep.bands[band_name])
    my_band.identifier = band_identifier
    my_band.polarization = mpi_info.band.polarization
    logger.debug(f"Rank {mpi_info.compsep.rank} handles CompSep view {band_identifier}.")

    cfg = read_sampling_groups(params)
    if mpi_info.compsep.rank == mpi_info.compsep.master:  # One report, not one per band.
        _validate_mcmc_sampled_components(cfg.mcmc_sampling_groups, comp_list)
        _validate_component_lmax(comp_list, params)
    cfg.double_precision = resolve_param(
        params, "double_precision", ("compsep",), default=False, legal_types=bool)
    nthreads = params.resources.compsep.num_threads
    cfg.num_threads = nthreads if isinstance(nthreads, int) else nthreads[mpi_info.compsep.rank]
    # Chain-output settings. The chi-squared map resolution is C3's NSIDE_CHISQ; it should be at or
    # below the coarsest band nside, since the map only ever sums z^2 into its pixels.
    chains = params.output.chains
    cfg.nside_chisq = int(getattr(chains, "nside_chisq", 256))
    cfg.include_chisq_map = bool(getattr(chains.include, "chisq_map", False))
    cfg.include_residual_maps = bool(getattr(chains.include, "compsep_residual_maps", False))
    # Each MCMC group's chi-squared mask, as the map in its `chisq_mask` file (None without one).
    chisq_masks = _read_chisq_masks(mpi_info.compsep.comm, cfg.mcmc_sampling_groups)
    for group in cfg.mcmc_sampling_groups.values():
        group.chisq_mask_map = chisq_masks.get(group.name)

    # Read the Gaussian amplitude prior's mean mu (each component's amp_prior_mean_map, else a
    # zero-mean prior) once here; the CG applies S^{-1/2} on every solve. Then give each chain
    # its own copy of the list and its starting state: a saved sample if the run starts from one,
    # else each component's init_from (else zeros). Done identically on every CompSep rank so the
    # lists start globally consistent.
    with benchmark("fileread-compsep"):
        for comp in comp_list:
            comp.load_amp_prior_mean()
        comp_lists_by_chain = {}
        for chain in (1, 2):
            chain_file = None
            if start.source_dir is not None:
                chain_file = paths.compsep_chain_file(start.source_dir, start.source_chain[chain],
                                                      start.source_iteration)
                if not os.path.exists(chain_file):
                    chain_file = None  # The saved run has no sky (`resolve_chain_start` warned).
            # On `chain_status: append` the saved sky is this chain's own state, which must win
            # over the initial guesses in `init_from`.
            apply_init_from = chain_file is None or start.first_iteration == 1
            comp_lists_by_chain[chain] = deepcopy(comp_list)
            for comp in comp_lists_by_chain[chain]:
                comp.load_initial_state(chain_file, apply_init_from)

    data_world = (band_identifier, mpi_info.world.rank)
    data_compsep = (band_identifier, mpi_info.compsep.rank)
    all_data_world = mpi_info.compsep.comm.allgather(data_world)
    all_data_compsep = mpi_info.compsep.comm.allgather(data_compsep)
    world_band_masters_dict = {item[0]: item[1] for item in all_data_world if item is not None}
    compsep_band_masters_dict = {item[0]: item[1] for item in all_data_compsep if item is not None}
    mpi_info.world.compsep_band_masters = world_band_masters_dict
    mpi_info.compsep.compsep_band_masters = compsep_band_masters_dict

    return comp_lists_by_chain, mpi_info, band_identifier, my_band, cfg


def process_compsep(mpi_info: Bunch, cfg: Bunch, detector_data: DetectorMap, iter: int, chain: int,
                    params: Bunch, comp_list: CompList, rng: np.random.Generator) -> SkyModel:
    """One Gibbs iteration of component separation. Every CompSep rank calls it.

    Runs the amplitude groups (CG or per-pixel) first, followed by the MCMC groups. An MCMC group
    may re-run named amplitude groups between proposal and accept/reject. When there are MCMC
    groups, the amplitude groups run once more after them, as in C3.

    Args:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        cfg (Bunch): The settings of component separation, from `init_compsep_processing`.
        detector_data (DetectorMap): The detector map for this rank's band, cleaned of all "TOD"
            components (correlated noise and orbital dipole).
        iter (int): The current Gibbs iteration (used only for logging and the chain file).
        chain (int): The current chain (used only for logging and the chain file).
        params (Bunch): The parameters from the input parameter file.
        comp_list (CompList): The full execution-view component list, updated in place.
        rng: This rank's random generator for the iteration. Every draw of every group uses it.

    Returns:
        The full sky realization, wrapping the updated `comp_list`.
    """
    compsep, band = mpi_info.compsep, mpi_info.band
    sky_model = SkyModel(comp_list)
    method = "CG" if cfg.cg_sampling_groups else "per-pixel"
    # Sampler bookkeeping for the chain: C3's nonlin-samples_*.dat and its CG residual logging. A
    # tier that sampled nothing adds no key, so this mirrors the file it ends up in.
    sampler_stats: dict = {}
    # `fit` is the fit after the most recent sampling step. Each evaluation logs the fit after its
    # step, and the last one is also the final fit that the chain file stores. A run has at least
    # one amplitude or MCMC group, so `fit` is always set.

    # The Gibbs order of C3 (commander.f90): the amplitudes, the spectral parameters, and then the
    # amplitudes again, so that the chain holds amplitudes solved at the final spectral parameters.
    for group in cfg.amplitude_groups.values():
        with benchmark(f"amp-{group.name}"):
            cg_stats = sample_amplitudes(mpi_info, cfg, group, detector_data, comp_list, rng)
        if cg_stats:
            sampler_stats.setdefault("amplitude_groups", {})[group.name] = cg_stats
        fit = evaluate_chi2(mpi_info, cfg, detector_data, sky_model,
                            label=f"{method} group {group.name!r}")

    for group in cfg.mcmc_sampling_groups.values():
        # The sampler calls this between a proposal and its accept/reject decision.
        def resolve_amplitudes(group_names=group.update_amplitude_groups):
            for group_name in group_names:
                sample_amplitudes(mpi_info, cfg, cfg.amplitude_groups[group_name], detector_data,
                                  comp_list, rng)

        chisq_active = band.name in group.chisq_bands or band.identifier in group.chisq_bands
        sampler = SpectralIndexSamplingGroup(
            compsep.comm, detector_data, comp_list, target_pol=band.polarization,
            selected_comps=group.comps, chisq_active=chisq_active,
            chisq_mask=group.chisq_mask_map, root=compsep.master)
        with benchmark(f"mcmc-{group.name}"):
            mcmc_stats = sampler.run(numstep=group.numstep, resolve_amplitudes=resolve_amplitudes,
                                     rng=rng)
        if mcmc_stats:
            sampler_stats.setdefault("mcmc", {})[group.name] = mcmc_stats
        fit = evaluate_chi2(mpi_info, cfg, detector_data, sky_model,
                            label=f"MCMC group {group.name!r}")

    if cfg.mcmc_sampling_groups:
        for group in cfg.amplitude_groups.values():
            with benchmark(f"amp-{group.name}"):
                cg_stats = sample_amplitudes(mpi_info, cfg, group, detector_data, comp_list, rng)
            if cg_stats:
                sampler_stats.setdefault("amplitude_groups", {})[group.name] = cg_stats
            fit = evaluate_chi2(mpi_info, cfg, detector_data, sky_model,
                                label=f"{method} group {group.name!r}")

    with benchmark("chain-gather"):
        fit_tree, band_frequencies = collect_fit_diagnostics(compsep, fit, cfg.include_chisq_map)
    if compsep.rank == compsep.master:
        diagnostics = {**fit_tree, **sampler_stats}
        with benchmark("filewrite-compsep"):
            write_compsep_chain_to_file(comp_list, params, chain, iter, diagnostics,
                                        band_frequencies)
        chi2 = fit_tree["chi2"]
        logger.summary(f"Chain {chain}, iteration {iter} complete: chi2={chi2['total']:.6e}, "
                       f"ndof={chi2['ndof']}, red.chi2={chi2['reduced']:.4f}, z={chi2['z']:.3f}.")

    with benchmark("end-barrier"):
        compsep.comm.Barrier()

    bench_summary(compsep.comm, label="CompSep")
    bench_reset()

    return sky_model
