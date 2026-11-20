"""The component-separation side of one Gibbs iteration.

`init_compsep_processing` builds the component list and the sampling groups once;
`process_compsep` runs one iteration's sampling groups over the band maps received from TOD
processing. A *sampling group* names a set of components and bands to be sampled together. An
amplitude group (`AmplitudeGroup`) is solved with CG or per pixel, and an MCMC group (`MCMCGroup`)
with Metropolis steps.
"""
import numpy as np
import logging
import os
from copy import deepcopy
from pixell.bunch import Bunch

from commander4.data_models.detector_map import DetectorMap
from commander4.sky.comp_list import CompList
from commander4.sky.diffuse_components import DiffuseComponent
from commander4.compsep.amplitude_sampling import AmplitudeGroup, sample_amplitudes
from commander4.compsep.chisq import collect_fit_diagnostics, log_fit
from commander4.compsep.spectral_sampling import MCMCGroup, sample_spectral_params
from commander4.file_io import paths
from commander4.file_io.chain_restart import ChainStart
from commander4.file_io.chain_writer import write_compsep_chain_to_file
from commander4.parameters.schema import resolve_band_lmax
from commander4.diagnostics.performance import benchmark, bench_summary, bench_reset

logger = logging.getLogger(__name__)


def read_sampling_groups(params: Bunch, mpi_info: Bunch | None = None) \
        -> tuple[list[AmplitudeGroup], list[MCMCGroup]]:
    """The enabled amplitude groups and MCMC groups of `compsep`, in the order of the file.

    The amplitude groups are the CG groups or the per-pixel groups, whichever the run uses. Without
    `mpi_info` the groups are only read and checked (see `AmplitudeGroup` and `MCMCGroup`).
    """
    amplitude_groups = []
    for section in ("cg_sampling_groups", "per_pixel_sampling_groups"):
        for name in params.compsep[section] if section in params.compsep else ():
            block = params.compsep[section][name]
            if "enabled" in block and not block.enabled:
                continue
            amplitude_groups.append(AmplitudeGroup.init_from_params(params, section, name,
                                                                    mpi_info))
    # CG and per-pixel are two solvers for the same step, and a run uses one of them.
    if len({group.solver for group in amplitude_groups}) > 1:
        raise ValueError("CG and per-pixel amplitude sampling groups are mutually exclusive; "
                         "configure only one method.")

    mcmc_groups = []
    section = "mcmc_sampling_groups"
    for name in params.compsep[section] if section in params.compsep else ():
        block = params.compsep[section][name]
        if "enabled" in block and not block.enabled:
            continue
        mcmc_groups.append(MCMCGroup.init_from_params(params, name, amplitude_groups, mpi_info))
    return amplitude_groups, mcmc_groups


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
    -> tuple[dict[int, CompList], list[AmplitudeGroup], list[MCMCGroup]]:
    """Build the components and the sampling groups, on one CompSep rank.

    Each CompSep rank works on one view (I or QU) of one band (`mpi_info.band`). This function also
    adds `mpi_info.world.compsep_band_masters`, the world rank of each CompSep view.

    Returns:
        comp_lists_by_chain: Each chain's full component list, with polarized components split
            into separate I and QU components. Identical on all CompSep ranks.
        amplitude_groups, mcmc_groups: The enabled sampling groups.
    """
    logger.debug(f"CompSep rank {mpi_info.compsep.rank} (on machine {mpi_info.processor_name}) "
                 f"handles view {mpi_info.band.identifier}.")

    comp_list = CompList.init_from_params(params)
    amplitude_groups, mcmc_groups = read_sampling_groups(params, mpi_info)
    if mpi_info.compsep.is_master:  # One report, not one per band.
        _validate_component_lmax(comp_list, params)
    # The components whose beta an MCMC group samples. Only they take beta from a saved chain; a
    # beta that is not sampled keeps the value of the parameter file.
    sampled_betas = set()
    for group in mcmc_groups:
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

    return comp_lists_by_chain, amplitude_groups, mcmc_groups


def process_compsep(mpi_info: Bunch, params: Bunch, amplitude_groups: list[AmplitudeGroup],
                    mcmc_groups: list[MCMCGroup], band_map: DetectorMap, comp_list: CompList,
                    chain: int, iteration: int, rng: np.random.Generator) -> None:
    """One Gibbs iteration of component separation. Every CompSep rank calls it.

    Runs the amplitude groups (CG or per-pixel) first, followed by the MCMC groups. An MCMC group
    may re-run named amplitude groups between proposal and accept/reject. When there are MCMC
    groups, the amplitude groups run once more after them, as in C3.

    Args:
        amplitude_groups, mcmc_groups: The sampling groups, from `init_compsep_processing`.
        band_map: This rank's band map, cleaned of all "TOD" components (correlated noise and
            orbital dipole).
        comp_list: The full component list of the chain, updated in place.
        chain, iteration: The current chain and Gibbs iteration, for logging and the chain file.
        rng: This rank's random generator for the iteration. Every draw of every group uses it.
    """
    compsep = mpi_info.compsep
    # Sampler statistics for the chain file (C3's nonlin-samples_*.dat and CG residual log), by
    # group name. The second amplitude pass overwrites the first.
    amplitude_stats, mcmc_stats = {}, {}

    # The Gibbs order of C3 (commander.f90): the amplitudes, the spectral parameters, and then the
    # amplitudes again, so that the chain holds amplitudes solved at the final spectral parameters.
    for group in amplitude_groups:
        with benchmark(f"amp-{group.name}"):
            stats = sample_amplitudes(mpi_info, params, group, band_map, comp_list, rng)
        if stats:  # Empty for a per-pixel group, and for a group with nothing to solve.
            amplitude_stats[group.name] = stats
        log_fit(mpi_info, band_map, comp_list, label=f"{group.solver} group {group.name!r}")

    for group in mcmc_groups:
        with benchmark(f"mcmc-{group.name}"):
            mcmc_stats[group.name] = sample_spectral_params(mpi_info, params, group, band_map,
                                                            comp_list, rng)
        log_fit(mpi_info, band_map, comp_list, label=f"MCMC group {group.name!r}")

    if mcmc_groups:
        for group in amplitude_groups:
            with benchmark(f"amp-{group.name}"):
                stats = sample_amplitudes(mpi_info, params, group, band_map, comp_list, rng)
            if stats:
                amplitude_stats[group.name] = stats
            log_fit(mpi_info, band_map, comp_list, label=f"{group.solver} group {group.name!r}")

    # The fit of the final sky model, for the chain file.
    with benchmark("chain-gather"):
        fit_tree, band_frequencies = collect_fit_diagnostics(mpi_info, params, band_map, comp_list)
    if compsep.is_master:
        # A section is written only if some group filled it.
        diagnostics = dict(fit_tree)
        if amplitude_stats:
            diagnostics["amplitude_groups"] = amplitude_stats
        if mcmc_stats:
            diagnostics["mcmc"] = mcmc_stats
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
