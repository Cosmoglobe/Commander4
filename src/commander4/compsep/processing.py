"""The component-separation side of one Gibbs iteration, and the settings that configure it.

`init_compsep_processing` builds the component list and reads the `compsep` block of the
parameter file once; `process_compsep` runs one iteration's sampling groups over the band maps
received from TOD processing. A *sampling group* names a set of components and bands to be sampled
together. An amplitude group is solved with CG or per pixel (`amplitude_sampling`), and an MCMC
group with Metropolis steps (`spectral_sampling`). The settings of a group are a plain `Bunch`,
with the names of the parameter file (`read_sampling_groups`).
"""
import numpy as np
import healpy as hp
import logging
import os
from copy import deepcopy
from mpi4py import MPI
from numpy.typing import NDArray
from pixell.bunch import Bunch

import commander4.sky as sky
from commander4.data_models.detector_map import DetectorMap
from commander4.sky.comp_list import CompList
from commander4.sky.diffuse_components import DiffuseComponent
from commander4.compsep.amplitude_sampling import sample_amplitudes
from commander4.compsep.chisq import collect_fit_diagnostics, evaluate_chi2
from commander4.compsep.spectral_sampling import sample_spectral_params
from commander4.file_io import paths
from commander4.file_io.chain_restart import ChainStart
from commander4.file_io.chain_writer import write_compsep_chain_to_file
from commander4.parameters.schema import enabled_compsep_views, read_settings, resolve_band_lmax,\
    resolve_param
from commander4.diagnostics.performance import benchmark, bench_summary, bench_reset

logger = logging.getLogger(__name__)

# The settings of each kind of sampling group, with their defaults. For `comps`, `bands` and
# `chisq_bands`, None and "all" both select everything.
CG_GROUP_DEFAULTS = {"enabled": True, "comps": None, "bands": None, "optimize": False,
                     "max_iter": 200, "max_iter_pol": 200, "err_tol": 1.0e-8,
                     "preconditioner": "JointPreconditioner"}
PER_PIXEL_GROUP_DEFAULTS = {"enabled": True, "comps": None, "bands": None}
MCMC_GROUP_DEFAULTS = {"enabled": True, "beta": None, "update_amplitude_groups": [], "numstep": 1,
                       "chisq_bands": None, "chisq_mask": None}
# The settings of one component in the `beta` block of an MCMC group: the step length of the
# proposal, the hard bounds [lowest, highest], and the mean and rms of a Gaussian prior. None is
# no bounds or no prior. `proposal_sigma` must be set.
BETA_DEFAULTS = {"proposal_sigma": None, "bounds": None, "prior_mean": None, "prior_rms": None}


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
    whichever the run uses. The `beta` of an MCMC group is a dict from the name of each component
    whose beta it samples to the settings of that sampler (`BETA_DEFAULTS`).
    """
    comp_names = [name for name in params.components if params.components[name].enabled]
    band_names = [name for name in params.compsep.bands if params.compsep.bands[name].enabled]

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
    # The per-pixel solver fits component maps pixel by pixel, so it has no model of point sources.
    for group in cfg.per_pixel_sampling_groups.values():
        for comp_name in group.comps:
            component_class = getattr(sky, params.components[comp_name].component_class)
            if not issubclass(component_class, DiffuseComponent):
                raise ValueError(f"'per_pixel_sampling_groups.{group.name}.comps' names "
                                 f"{comp_name!r}, a {component_class.__name__}. The per-pixel "
                                 "solver handles only diffuse components.")

    if "mcmc_sampling_groups" not in params.compsep:
        return cfg
    for name in params.compsep.mcmc_sampling_groups:
        section = f"mcmc_sampling_groups.{name}"
        group = read_settings(params, name, ("compsep.mcmc_sampling_groups",), MCMC_GROUP_DEFAULTS)
        if not group.enabled:
            continue
        group.name = name
        # The `beta` block names the components whose beta the group samples, each with the
        # settings of its sampler. The betas are proposed in the order of the block.
        if not group.beta:
            raise ValueError(f"'{section}.beta' must name the components whose beta to sample.")
        beta = {}
        for comp_name in names_or_all(list(group.beta), comp_names, f"{section}.beta"):
            beta[comp_name] = read_settings(params, comp_name, (f"compsep.{section}.beta",),
                                            BETA_DEFAULTS)
            if beta[comp_name].proposal_sigma is None:
                raise ValueError(f"'{section}.beta.{comp_name}.proposal_sigma' must be set.")
            # A prior with only one of its two numbers would be left out without a message.
            if (beta[comp_name].prior_mean is None) != (beta[comp_name].prior_rms is None):
                raise ValueError(f"'{section}.beta.{comp_name}' must set both 'prior_mean' and "
                                 "'prior_rms', or neither.")
        group.beta = beta
        group.chisq_bands = names_or_all(group.chisq_bands, band_names, f"{section}.chisq_bands")
        unknown = sorted(set(group.update_amplitude_groups) - set(cfg.amplitude_groups))
        if unknown:
            raise ValueError(f"'{section}.update_amplitude_groups' names unknown or disabled "
                             f"amplitude group(s) {unknown}. The enabled amplitude groups are "
                             f"{list(cfg.amplitude_groups)}.")
        cfg.mcmc_sampling_groups[name] = group
    return cfg


def _read_chisq_mask(compsep_comm: MPI.Comm, path: str, nside: int) -> NDArray[np.bool_]:
    """The pixels at `nside` that an MCMC group's chi-squared mask keeps (True = kept).

    This is C3's ``MCMC_SAMPLING_GROUP_CHISQ_MASK``: one mask per sampling group, applied to every
    band in that group's chi-squared, restricting which sky the accept/reject decision sees. It is
    deliberately *not* a per-band data mask: it does not enter the amplitude solve or the noise
    model, only the MCMC likelihood.

    The CompSep master reads the file and broadcasts it. As in C3's ``compute_chisq``, the mask is
    brought to `nside` and thresholded at 0.5, so a coarse pixel is kept only if more than half of
    it was. Only field 0 is read (as on the TOD side), so one mask applies to Q and U alike.
    """
    mask = None
    if compsep_comm.Get_rank() == 0:
        mask = hp.read_map(path, field=0, dtype=np.float64)
        logger.info(f"CompSep: read chisq mask {path} (fsky = {np.mean(mask > 0.5):.3f}).")
    mask = compsep_comm.bcast(mask, root=0)
    if hp.npix2nside(mask.size) != nside:
        mask = hp.ud_grade(mask, nside)
    return mask > 0.5


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
    -> tuple[dict[int, CompList], Bunch]:
    """Build the components and read the settings of component separation, on one CompSep rank.

    Each CompSep rank works on one view of one band (`mpi_info.band`, set by the MPI setup). This
    function also adds `mpi_info.world.compsep_band_masters`, the world rank of each CompSep view.

    Args:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        params (Bunch): The parameters from the input parameter file.
        start (ChainStart): Where the two chains start (see `file_io.chain_restart`).

    Returns:
        comp_lists_by_chain (dict[int, CompList]): Each chain's full execution-view component
            list, initialized and identical on all CompSep ranks.
        cfg (Bunch): The settings of component separation: the sampling groups (see
            `read_sampling_groups`), `double_precision`, this rank's `num_threads`, and the
            chain-output settings `nside_chisq`, `include_chisq_map` and `include_residual_maps`.
            Each amplitude group also gets `solver_ranks`, the CompSep ranks that solve it per
            polarization ({"I": [...], "QU": [...]}), and `solver_comm`, their communicator
            (`MPI.COMM_NULL` on the other ranks).
    """
    logger.debug(f"CompSep rank {mpi_info.compsep.rank} (on machine {mpi_info.processor_name}) "
                 f"handles view {mpi_info.band.identifier}.")

    comp_list = CompList.init_from_params(params)

    cfg = read_sampling_groups(params)
    if mpi_info.compsep.is_master:  # One report, not one per band.
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
    # The pixels of this rank's band that enter each MCMC group's chi-squared: those its
    # `chisq_mask` keeps, or all of them without a mask. The band map arrives at the band's
    # `eval_nside` (TOD bands: in the experiment block; map bands: in the compsep block).
    my_band = params.compsep.bands[mpi_info.band.name]
    if my_band.get_from != "file":
        my_band = params.experiments[my_band.get_from].bands[mpi_info.band.name]
    for group in cfg.mcmc_sampling_groups.values():
        group.chisq_pixels = slice(None)
        if group.chisq_mask:
            group.chisq_pixels = _read_chisq_mask(mpi_info.compsep.comm, group.chisq_mask,
                                                  my_band.eval_nside)
    # The CompSep ranks that solve each amplitude group, by polarization, and the communicator they
    # solve on. A rank solves if the group names its band and a component of its polarization. View
    # i of `enabled_compsep_views` runs on CompSep rank i, so every rank knows this for all ranks.
    views = enabled_compsep_views(params)
    for group in cfg.amplitude_groups.values():
        group.solver_ranks = {}
        for pol in ("I", "QU"):
            has_comps = any(comp.pol == pol and comp.comp_name in group.comps for comp in comp_list)
            group.solver_ranks[pol] = [view.rank for view in views if has_comps and
                                       view.polarization == pol and view.band_name in group.bands]
        # One communicator per polarization (`subcomm` is split by polarization), so the I and QU
        # views of a group are two separate solves.
        is_solver = mpi_info.compsep.rank in group.solver_ranks[mpi_info.band.polarization]
        group.solver_comm = mpi_info.compsep.subcomm.Split(0 if is_solver else MPI.UNDEFINED,
                                                           key=mpi_info.compsep.rank)
    # The components whose beta an MCMC group samples. Only they take beta from a saved chain; a
    # beta that is not sampled keeps the value of the parameter file.
    sampled_betas = set()
    for group in cfg.mcmc_sampling_groups.values():
        sampled_betas.update(group.beta)

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
                comp.load_initial_state(chain_file, apply_init_from,
                                        restore_beta=comp.comp_name in sampled_betas)

    # The world rank of each view, by view name, for the exchange with the TOD side.
    mpi_info.world.compsep_band_masters = dict(mpi_info.compsep.comm.allgather(
        (mpi_info.band.identifier, mpi_info.world.rank)))

    return comp_lists_by_chain, cfg


def process_compsep(mpi_info: Bunch, cfg: Bunch, params: Bunch, band_map: DetectorMap,
                    comp_list: CompList, chain: int, iteration: int,
                    rng: np.random.Generator) -> None:
    """One Gibbs iteration of component separation. Every CompSep rank calls it.

    Runs the amplitude groups (CG or per-pixel) first, followed by the MCMC groups. An MCMC group
    may re-run named amplitude groups between proposal and accept/reject. When there are MCMC
    groups, the amplitude groups run once more after them, as in C3.

    Args:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        cfg (Bunch): The settings of component separation, from `init_compsep_processing`.
        params (Bunch): The parameters from the input parameter file (for the chain file).
        band_map (DetectorMap): This rank's band map, cleaned of all "TOD" components
            (correlated noise and orbital dipole).
        comp_list (CompList): The full execution-view component list, updated in place.
        chain (int): The current chain (used only for logging and the chain file).
        iteration (int): The current Gibbs iteration (used only for logging and the chain file).
        rng: This rank's random generator for the iteration. Every draw of every group uses it.
    """
    compsep = mpi_info.compsep
    method = "CG" if cfg.cg_sampling_groups else "per-pixel"
    # Sampler bookkeeping for the chain: C3's nonlin-samples_*.dat and its CG residual logging. A
    # tier that sampled nothing adds no key, so this mirrors the file it ends up in.
    sampler_stats: dict = {}

    # The Gibbs order of C3 (commander.f90): the amplitudes, the spectral parameters, and then the
    # amplitudes again, so that the chain holds amplitudes solved at the final spectral parameters.
    for group in cfg.amplitude_groups.values():
        with benchmark(f"amp-{group.name}"):
            cg_stats = sample_amplitudes(mpi_info, cfg, group, band_map, comp_list, rng)
        if cg_stats:
            sampler_stats.setdefault("amplitude_groups", {})[group.name] = cg_stats
        # `fit` is the fit after the most recent sampling step. Each evaluation logs it, and the
        # last one is the final fit that the chain file stores. A run has at least one amplitude
        # or MCMC group, so `fit` is always set.
        fit = evaluate_chi2(mpi_info, cfg, band_map, comp_list,
                            label=f"{method} group {group.name!r}")

    for group in cfg.mcmc_sampling_groups.values():
        with benchmark(f"mcmc-{group.name}"):
            mcmc_stats = sample_spectral_params(mpi_info, cfg, group, band_map, comp_list, rng)
        sampler_stats.setdefault("mcmc", {})[group.name] = mcmc_stats
        fit = evaluate_chi2(mpi_info, cfg, band_map, comp_list,
                            label=f"MCMC group {group.name!r}")

    if cfg.mcmc_sampling_groups:
        for group in cfg.amplitude_groups.values():
            with benchmark(f"amp-{group.name}"):
                cg_stats = sample_amplitudes(mpi_info, cfg, group, band_map, comp_list, rng)
            if cg_stats:
                sampler_stats.setdefault("amplitude_groups", {})[group.name] = cg_stats
            fit = evaluate_chi2(mpi_info, cfg, band_map, comp_list,
                                label=f"{method} group {group.name!r}")

    with benchmark("chain-gather"):
        fit_tree, band_frequencies = collect_fit_diagnostics(compsep, fit, cfg.include_chisq_map)
    if compsep.is_master:
        diagnostics = {**fit_tree, **sampler_stats}
        with benchmark("filewrite-compsep"):
            write_compsep_chain_to_file(comp_list, params, chain, iteration, diagnostics,
                                        band_frequencies)
        chi2 = fit_tree["chi2"]
        logger.summary(f"Chain {chain}, iteration {iteration} complete: chi2={chi2['total']:.6e}, "
                       f"ndof={chi2['ndof']}, red.chi2={chi2['reduced']:.4f}, z={chi2['z']:.3f}.")

    with benchmark("end-barrier"):
        compsep.comm.Barrier()

    bench_summary(compsep.comm, label="CompSep")
    bench_reset()
