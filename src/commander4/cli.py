"""Entry point for the Commander4 software, as specified in `pyproject.toml`.

After installing Commander4 with PIP, the command `commander4` runs this file. Commander4 can only
be installed as a package; this file cannot be run directly as a script.

Splits the world communicator into TOD-processing and component-separation sides, then drives the
Gibbs loop that alternates between them, with two chains always in flight.
"""
import os
import yaml
import faulthandler
from argparse import ArgumentParser
from mpi4py import MPI
import cProfile
import pstats
import logging
import time
import getpass
from copy import deepcopy
from datetime import date
from pixell.bunch import Bunch

from commander4.diagnostics import log
from commander4.file_io import paths
from commander4.mpi import setup as mpi_setup
from commander4.parameters.schema import validate_param_schema
from commander4.diagnostics.performance import benchmark, bench_summary, start_bench,\
                                               stop_bench, log_memory, increment_count, bench_reset


def parse_command_line() -> str:
    """Return the parameter-file path selected on the command line."""
    parser = ArgumentParser(prog="commander4")
    parser.add_argument(
        "-p", "--parameter-file", "--parameter_file", required=True,
        help="Path to a Commander4 YAML parameter file.",
    )
    return parser.parse_args().parameter_file


def gibbs_schedule(num_iterations: int) -> list[tuple[int, int]]:
    """The `(chain, iteration)` steps of the run, in the order both sides walk them.
    """
    return [(chain, iteration)
            for iteration in range(1, num_iterations + 1)
            for chain in (1, 2)]


def seed_iteration_rng(params: Bunch, mpi_info: Bunch, chain: int, iteration: int):
    """Sets unique seed per chain+rank+iteration for numpy's RNG.
    """
    import numpy as np

    root_seed = int(params.gibbs.seed) if "seed" in params.gibbs else 1995
    side = 0 if mpi_info.world.side == "tod" else 1
    seed = int(np.random.SeedSequence(
        [root_seed, chain, iteration, mpi_info.world.rank, side]
    ).generate_state(1)[0])
    np.random.seed(seed)


def run_tod_side(mpi_info: Bunch, params: Bunch, experiment_data, my_band_tod_id: str,
                 tod_samples_by_chain: dict, compsep_output, compsep_active: bool) -> None:
    """Walk the TOD half of the Gibbs loop.
    Args:
        tod_samples_by_chain: Per-chain `TODSamples`, updated in place. Both chains are held at
            once because a step cannot hot-swap them: the two sides send and receive independently.
        compsep_output: The initial sky model, already realized for this band.
        compsep_active: False in TOD-only mode, where there is nobody to exchange with.
    """
    from commander4.tod.processing import process_tod
    from commander4.mpi.transfer import receive_compsep, send_tod

    logger = logging.getLogger(__name__)
    for step, (chain, iteration) in enumerate(gibbs_schedule(params.gibbs.num_iterations)):
        if mpi_info.tod.rank == 0:
            logger.verbose(f"Worldrank {mpi_info.world.rank}, subrank {mpi_info.tod.rank} "
                           f"starting TOD chain {chain}, iteration {iteration}.")
        seed_iteration_rng(params, mpi_info, chain, iteration)
        t0 = time.time()
        tod_output, tod_samples_by_chain[chain] = process_tod(
            mpi_info, experiment_data, tod_samples_by_chain[chain], compsep_output, params,
            chain, iteration)
        if mpi_info.tod.is_master:
            logger.summary(f"Chain {chain}, iteration {iteration} completed TOD-proc in "
                           f"{time.time()-t0:.2f}s).")
        if not compsep_active:
            continue
        if step > 0:
            # Not on the first step: chain 2 has to run its first iteration against the same
            # initial model chain 1 did, because nothing has been sampled for it yet.
            t0 = time.time()
            compsep_output = receive_compsep(mpi_info, experiment_data, my_band_tod_id,
                                             mpi_info.world.compsep_band_masters)
            if mpi_info.band.is_master:
                logger.verbose(f"TOD: Rank {mpi_info.tod.rank} finished receiving results for "
                               f"chain {chain}, iter {iteration} (time spent waiting+receiving = "
                               f"{time.time()-t0:.1f}s).")
        send_tod(mpi_info, tod_output, my_band_tod_id, mpi_info.world.compsep_band_masters)
        if mpi_info.tod.is_master:
            logger.verbose(f"TOD: Rank {mpi_info.tod.rank} finished sending results for chain "
                           f"{chain}, iter {iteration}.")


def run_compsep_side(mpi_info: Bunch, params: Bunch, compsep_state, my_band_compsep_id: str,
                     my_band, comp_lists_by_chain: dict, tod_output) -> None:
    """Walk the component-separation half of the Gibbs loop.
    Args:
        comp_lists_by_chain: Per-chain component lists, updated in place. Each chain needs its own,
            since a `Component` carries the currently sampled amplitudes and spectral parameters.
        tod_output: The first TOD sample, already received.
    """
    from commander4.compsep.processing import process_compsep
    from commander4.mpi.transfer import receive_tod, send_compsep

    logger = logging.getLogger(__name__)
    schedule = gibbs_schedule(params.gibbs.num_iterations)
    for step, (chain, iteration) in enumerate(schedule):
        if mpi_info.compsep.rank == 0:
            logger.verbose(f"Worldrank {mpi_info.world.rank}, subrank {mpi_info.compsep.rank} "
                           f"starting CompSep chain {chain}, iteration {iteration}.")
        seed_iteration_rng(params, mpi_info, chain, iteration)
        t0 = time.time()
        compsep_output = process_compsep(mpi_info, compsep_state, tod_output, iteration, chain,
                                         params, comp_lists_by_chain[chain])
        if mpi_info.compsep.rank == 0:
            logger.verbose(f"CompSep: Rank {mpi_info.compsep.rank} finished chain {chain}, "
                           f"iteration {iteration} in {time.time()-t0:.2f}s. Sending results.")
        if step == len(schedule) - 1:
            continue
        send_compsep(mpi_info, my_band_compsep_id, compsep_output,
                     mpi_info.world.tod_band_masters)
        logger.debug(f"CompSep rank {mpi_info.compsep.rank} finished sending results for chain "
                     f"{chain}, iter {iteration}. Waiting for TOD results.")
        t0 = time.time()
        tod_output = receive_tod(mpi_info, mpi_info.world.tod_band_masters, my_band,
                                 my_band_compsep_id, tod_output, params)
        logger.debug(f"CompSep rank {mpi_info.compsep.rank} received TOD results for chain "
                     f"{chain}, iter {iteration} (waiting+receiving = {time.time()-t0:.1f}s).")


def run_commander4(params: Bunch, params_dict: dict):
    """
    Main loop function for Commander 4 Gibbs Sampling. Commander4 splits the Gibbs chain in two;
    TOD processing steps, and component separation, with dedicated hardware and separate MPI
    communicators for the two tasks. Commander4 therefore always runs two Gibbs chains in parallel,
    such that each of the two tasks are always working on one of the two chains.

    Args:
        params: The Commander4 parameter file, as a 'Bunch' object.
        params_dict: The exact same parameter file, but as a dictionary.
    """
    logger = logging.getLogger(__name__)  # Access logger, used instead of print() in Commander4.
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.summary(f"Run date: {date.today().isoformat()}.")

    # Perform initial MPI setup, assigning tasks to different MPI ranks and deciding master ranks.
    with benchmark("init-mpi"):
        mpi_info = mpi_setup.init_mpi(params)

    if mpi_info['world']['is_master']:
        import random
        # Print the entire parameter file to log.
        if logger.isEnabledFor(logging.DEBUG):  # Just to avoid the yaml.dump if debug is off.
            logger.debug("### PARAMETERS ###\n%s", yaml.dump(
                params_dict, allow_unicode=True, default_flow_style=False))
        # Print Commander4 text. Color chosen from hashed username.
        logger.summary(f"\033[{((5+sum(ord(char) for char in getpass.getuser()))%6) + 91}m" + r"""
           ______                                          __             __ __
          / ____/___  ____ ___  ____ ___  ____ _____  ____/ /__  _____   / // /
         / /   / __ \/ __ `__ \/ __ `__ \/ __ `/ __ \/ __  / _ \/ ___/  / // /_
        / /___/ /_/ / / / / / / / / / / / /_/ / / / / /_/ /  __/ /     /__  __/
        \____/\____/_/ /_/ /_/_/ /_/ /_/\__,_/_/ /_/\__,_/\___/_/        /_/""" + "\033[0m\n")
        logger.summary(f"Writing all output to {paths.resolve_output_dir(params.output)}")

    # Import the numerical pipeline after init_mpi has configured this rank's thread counts.
    from commander4.tod.processing import init_tod_processing
    from commander4.compsep.processing import init_compsep_processing, get_initial_sky_model
    from commander4.mpi.transfer import receive_tod, receive_compsep, send_compsep,\
        get_local_initial_sky

    # Give initialization its own stream. Every chain iteration is reseeded at the orchestration
    # boundary in run_tod_side or run_compsep_side.
    seed_iteration_rng(params, mpi_info, chain=0, iteration=0)

    ###### Initizatization ######
    # Setting up dictionaries mapping each experiment+band combo to the world rank of the master
    # task for that band (on both the TOD and CompSep sides).
    world_compsep_band_masters_dict = None
    world_tod_band_masters_dict = None
    compsep_state = None
    comp_lists_by_chain = None
    if mpi_info.world.side == "tod":
        mpi_info, my_band_tod_id, experiment_data, tod_samples_chain1, tod_samples_chain2\
                                                            = init_tod_processing(mpi_info, params)
        # Even though we're always only working on one of the two chains we still need two sets of
        # samples, as we can't "hot swap" them (both TOD processing and component separation would
        # have to send and receive from the same local buffer). However, perhaps it would be cleaner
        # to call these "current_chain" and "other chain" or something.
    elif mpi_info.world.side == "compsep":
        initial_comp_list, mpi_info, my_band_compsep_id, my_band, compsep_state = \
            init_compsep_processing(mpi_info, params)
        # A Component contains the current sampled amplitudes and spectral parameters, so each
        # Gibbs chain needs its own complete component list. Copy only after initialization has
        # loaded the initial alms and amplitude prior means, but before either chain is processed.
        comp_lists_by_chain = {1: initial_comp_list, 2: deepcopy(initial_comp_list)}
        # TODO: This means two copies of per-component amplitude alms are held in memory at all
        # times. Under the current design this is needed for sampling steps where some components
        # are excluded, but their amplitudes are still needed to evaluate the relevant chi2.

    if mpi_info.world.tod_master is not None:
        # All processes, both compsep and tod, need the world-specific band master dict.
        world_tod_band_masters_dict = mpi_info.world.comm.bcast(mpi_info.world.tod_band_masters,
                                                                root=mpi_info.world.tod_master)
        mpi_info['world']['tod_band_masters'] = world_tod_band_masters_dict
    if mpi_info.world.compsep_master is not None:
        world_compsep_band_masters_dict = mpi_info.world.comm.bcast(
            mpi_info.world.compsep_band_masters, root=mpi_info.world.compsep_master)
        mpi_info['world']['compsep_band_masters'] = world_compsep_band_masters_dict

    ###### Exchanging the initial sky model ######
    # Component separation is active iff CompSep ranks were allocated (compsep_master is then a
    # valid world rank). This single flag replaces the old `perform_compsep` parameter.
    tod_active = mpi_info.world.tod_master is not None
    compsep_active = mpi_info.world.compsep_master is not None
    curr_tod_output = None
    if mpi_info.world.is_master:
        if tod_active and compsep_active:
            mode = "TOD and component separation"
        elif tod_active:
            mode = "TOD-only"
        else:
            mode = "CompSep-only"
        logger.summary(f"Starting Gibbs sampling in {mode} mode with two interleaved chains.")
        component_names: list[str] = []
        for component_name in params.components:
            component = params.components[component_name]
            if "enabled" not in component or component.enabled:
                component_names.append(component_name)
        logger.summary(f"Enabled sky components: {', '.join(component_names)}.")
    if mpi_info.world.side == "tod":
        # The initial sky model is built from each component's init_from / init_chain_path (else
        # zeros). If CompSep ranks exist they build and send it (as for every later iteration);
        # otherwise we build it locally so a sensible fixed sky is available with no CompSep ranks.
        if compsep_active:
            curr_compsep_output = receive_compsep(mpi_info, experiment_data, my_band_tod_id,
                                                  mpi_info.world.compsep_band_masters)
        else:
            curr_compsep_output = get_local_initial_sky(mpi_info, experiment_data, params)
        run_tod_side(mpi_info, params, experiment_data, my_band_tod_id,
                     {1: tod_samples_chain1, 2: tod_samples_chain2}, curr_compsep_output,
                     compsep_active)

    elif mpi_info.world.side == "compsep":
        # Send the initial sky model to TOD before receiving the first TOD output, mirroring the
        # process_compsep -> send_compsep -> receive_tod order used inside the main loop.
        send_compsep(mpi_info, my_band_compsep_id, get_initial_sky_model(comp_lists_by_chain[1]),
                     mpi_info.world.tod_band_masters)
        curr_tod_output = receive_tod(mpi_info, mpi_info.world.tod_band_masters, my_band,
                                      my_band_compsep_id, curr_tod_output, params)
        run_compsep_side(mpi_info, params, compsep_state, my_band_compsep_id, my_band,
                         comp_lists_by_chain, curr_tod_output)

    # stop compsep machinery
    if mpi_info.world.is_master and compsep_active:
        logger.verbose("TOD: sending STOP signal to CompSep.")
        mpi_info.world.comm.send(True, dest=mpi_info.world.compsep_master)

    return 0

def main() -> None:
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
    logger = logging.getLogger(__name__)
    traceback_dir = None
    run_id = None
    try:
        # faulthandler is separate from normal exception handling. It asks Python's low-level C
        # signal handlers to dump thread stacks to stderr for native crashes such as SIGSEGV,
        # SIGBUS and SIGABRT, where no Python exception is available for the except blocks below.
        # It is deliberately simple: it reports a crash but does not attempt recovery.
        faulthandler.enable(all_threads=True)

        # Every rank must use the same ID so they compete for the same fatal-report filename. Rank
        # 0 creates it once; this startup broadcast happens while MPI is still healthy.
        run_id = world_comm.bcast(log.make_run_id() if world_rank == 0 else None, root=0)

        # Rank 0 reads the parameter file and broadcasts the resolved dictionary.
        from commander4.parameters.parse import load_params, params_from_dict
        parameter_file = parse_command_line()
        if world_rank == 0:
            params, params_dict, _ = load_params(parameter_file)
            validate_param_schema(params_dict)
        else:
            params_dict = None
        params_dict = world_comm.bcast(params_dict, root=0)
        params = params_from_dict(params_dict)

        output_dir = paths.resolve_output_dir(params.output)
        traceback_dir = os.path.join(output_dir, paths.LOGS)
        log_file = None
        if "file" in params.output.logging:
            log_file = paths.log_file_path(params.output, run_id)
        if world_rank == 0:
            paths.create_output_dirs(params.output)
        world_comm.Barrier()
        log.init_loggers(params.output.logging, log_file, world_rank=world_rank)
        if world_rank == 0:
            # SUMMARY is retained in normal production logs and lets users match a Slurm job's
            # output to its fatal report without relying only on directory or job names.
            logger.summary(f"Run ID: {run_id}.")

        if params.output.profiling:
            profiler = cProfile.Profile()
            profiler.enable()
        ret = run_commander4(params, params_dict)
        if params.output.profiling:
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats('tottime')
            if ret != -1:
                stats.dump_stats(os.path.join(paths.subdir(params, paths.LOGS),
                                              f"stats-{world_rank}"))
        logger.debug(f"Rank {world_rank} finished Commander 4 and is shutting down.")
        if world_rank == 0:
            logger.summary("Commander 4 completed successfully. Goodbye!")

    # mpi4py converts MPI errors that return control to Python into MPI.Exception. Catastrophic MPI
    # failures may abort internally and instead rely on faulthandler or launcher diagnostics.
    except MPI.Exception as error:
        error_code = max(1, error.Get_error_code())
        error_string = MPI.Get_error_string(error_code)
        log.report_fatal(error, logger, world_rank, error_code, traceback_dir,
                         context=f"MPI error {error_code} ({error_string})", run_id=run_id)
        # Some MPI implementations terminate a rank with SIGABRT. Disable faulthandler before our
        # intentional abort so it does not print a second, misleading native-crash traceback.
        faulthandler.disable()
        world_comm.Abort(error_code)
    except KeyboardInterrupt as error:
        # Ctrl+C normally becomes KeyboardInterrupt when Python regains control. Treat it like any
        # other fatal rank event so one report is completed before the whole MPI job is stopped.
        log.report_fatal(error, logger, world_rank, 130, traceback_dir, run_id=run_id)
        faulthandler.disable()
        world_comm.Abort(130)
    # All ordinary Commander4 validation and runtime failures arrive here as typed Python
    # exceptions. report_fatal performs no MPI communication, which is important because peers may
    # already be blocked in unrelated collectives when one rank fails.
    except Exception as error:
        log.report_fatal(error, logger, world_rank, 1, traceback_dir, run_id=run_id)
        faulthandler.disable()
        world_comm.Abort(1)

if __name__ == "__main__":
    main()
