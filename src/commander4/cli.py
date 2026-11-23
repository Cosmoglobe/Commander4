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
from datetime import date
from pixell.bunch import Bunch

from commander4.diagnostics import log
from commander4.file_io import paths
from commander4.file_io.chain_restart import ChainStart, resolve_chain_start
from commander4.mpi.setup import MPIInfo, init_mpi
from commander4.parameters.schema import validate_param_schema

logger = logging.getLogger(__name__)


def parse_command_line() -> str:
    """Return the parameter-file path selected on the command line."""
    parser = ArgumentParser(prog="commander4")
    parser.add_argument(
        "-p", "--parameter-file", "--parameter_file", required=True,
        help="Path to a Commander4 YAML parameter file.",
    )
    return parser.parse_args().parameter_file


def gibbs_schedule(first_iteration: int, num_iterations: int) -> list[tuple[int, int]]:
    """The `(chain, iteration)` steps of the run, in the order both sides walk them.
    """
    return [(chain, iteration)
            for iteration in range(first_iteration, num_iterations + 1)
            for chain in (1, 2)]


def make_iteration_rng(params: Bunch, mpi_info: MPIInfo, chain: int, iteration: int):
    """Return the random generator for one chain, iteration, and rank.
    """
    import numpy as np
    root_seed = int(params.gibbs.seed) if "seed" in params.gibbs else 1995
    return np.random.default_rng([root_seed, chain, iteration, mpi_info.world.rank])


def run_tod_side(mpi_info: MPIInfo, params: Bunch, start: ChainStart) -> None:
    """The TOD half of the run: read the band's data, then walk the TOD half of the Gibbs loop."""
    from commander4.tod.processing import init_tod_processing, process_tod
    from commander4.mpi.transfer import get_local_initial_sky, receive_compsep, send_tod

    # Both chains' samples are held at once, because the two sides send and receive independently,
    # so a step cannot swap one chain's samples for the other's in place.
    experiment_data, tod_samples_by_chain = init_tod_processing(mpi_info, params, start)

    # Setting up initial sky model, before the first TOD iteration.
    compsep_active = mpi_info.world.has_compsep
    if compsep_active:  # Receive initial sky from CompSep, if it's running.
        compsep_output = receive_compsep(mpi_info, experiment_data)
    else:  # If compsep isn't running fetch initial sky model ourselves.
        compsep_output = get_local_initial_sky(mpi_info, experiment_data, params)

    for chain, iteration in gibbs_schedule(start.first_iteration, params.gibbs.num_iterations):
        if mpi_info.tod.rank == 0:
            logger.verbose(f"Worldrank {mpi_info.world.rank}, subrank {mpi_info.tod.rank} "
                           f"starting TOD chain {chain}, iteration {iteration}.")
        rng = make_iteration_rng(params, mpi_info, chain, iteration)
        t0 = time.time()
        tod_output = process_tod(mpi_info, experiment_data, tod_samples_by_chain[chain],
                                 compsep_output, params, iteration, rng)
        if mpi_info.tod.rank == 0:
            logger.summary(f"Chain {chain}, iteration {iteration} completed TOD-proc in "
                           f"{time.time()-t0:.2f}s).")
        if not compsep_active:  # No communication happens if the compsep side isn't running.
            continue            # The sky model will simply remain at its initial state.

        # Receive the sky that the next Gibbs iteration (and opposite chain) will run with.
        t0 = time.time()
        compsep_output = receive_compsep(mpi_info, experiment_data)
        if mpi_info.band.rank == 0:
            logger.verbose(f"TOD: Rank {mpi_info.tod.rank} received the sky for the next step "
                           f"(time spent waiting+receiving = {time.time()-t0:.1f}s).")
        send_tod(mpi_info, tod_output)
        if mpi_info.tod.rank == 0:
            logger.verbose(f"TOD: Rank {mpi_info.tod.rank} finished sending results for chain "
                           f"{chain}, iter {iteration}.")


def run_compsep_side(mpi_info: MPIInfo, params: Bunch, start: ChainStart) -> None:
    """The component-separation half of the run: build the components and the sampling groups,
    then walk the CompSep half of the Gibbs loop."""
    from commander4.compsep.processing import init_compsep_processing, process_compsep
    from commander4.mpi.transfer import receive_tod, send_compsep

    # Get a dict holding the Components of both chains, and lists of both the amplitude (linear) and
    # the MCMC (nonlinear) sampling groups.
    comp_lists_by_chain, amplitude_groups, mcmc_groups = init_compsep_processing(mpi_info, params,
                                                                                 start)
    # Send each chain's initial components to the TOD side, and then get the first band map: the
    # same order (sky out, band map in) as at the end of each step of the loop below.
    for chain in (1, 2):
        send_compsep(mpi_info, comp_lists_by_chain[chain])
    band_map = receive_tod(mpi_info, None, params)

    schedule = gibbs_schedule(start.first_iteration, params.gibbs.num_iterations)
    for step, (chain, iteration) in enumerate(schedule):
        if mpi_info.compsep.rank == 0:
            logger.verbose(f"Worldrank {mpi_info.world.rank}, CompSep rank {mpi_info.compsep.rank} "
                           f"starting CompSep chain {chain}, iteration {iteration}.")
        rng = make_iteration_rng(params, mpi_info, chain, iteration)
        t0 = time.time()
        process_compsep(mpi_info, params, amplitude_groups, mcmc_groups, band_map,
                        comp_lists_by_chain[chain], chain, iteration, rng)
        if mpi_info.compsep.rank == 0:
            logger.verbose(f"CompSep: Rank {mpi_info.compsep.rank} finished chain {chain}, "
                           f"iteration {iteration} in {time.time()-t0:.2f}s. Sending results.")
        if step == len(schedule) - 1:  # Last compsep iteration.
            continue                   # TOD has already finished: Skip communication with TOD.
        send_compsep(mpi_info, comp_lists_by_chain[chain])
        logger.debug(f"CompSep rank {mpi_info.compsep.rank} finished sending results for chain "
                     f"{chain}, iter {iteration}. Waiting for TOD results.")
        t0 = time.time()
        band_map = receive_tod(mpi_info, band_map, params)
        logger.debug(f"CompSep rank {mpi_info.compsep.rank} received TOD results for chain "
                     f"{chain}, iter {iteration} (waiting+receiving = {time.time()-t0:.1f}s).")


def run_commander4(params: Bunch, params_dict: dict):
    """Set up MPI and the chain start, then run this rank's side (TOD or CompSep) of the Gibbs loop.

    Commander4 splits the Gibbs chain in two: (i) TOD processing and (ii) component separation.
    It dedicates different hardware and separate MPI communicators for the two tasks.
    Commander4 therefore always runs two Gibbs chains in parallel, such that each of the two tasks
    are always working on one of the two chains, and swapping them back and forth.

    Args:
        params: The Commander4 parameter file, as a 'Bunch' object.
        params_dict: The exact same parameter file, but as a dictionary.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.summary(f"Run date: {date.today().isoformat()}.")

    # Perform initial MPI setup, assigning each MPI rank to the TOD or CompSep side and to a band.
    mpi_info = init_mpi(params)

    if mpi_info.world.rank == 0:
        # Print the entire parameter file to log.
        if logger.isEnabledFor(logging.DEBUG):  # Avoiding the yaml.dump if debug is off.
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

    tod_active, compsep_active = mpi_info.world.has_tod, mpi_info.world.has_compsep

    # Figure out how to initialize the chain (from a previous chain or clean run).
    start = None
    if mpi_info.world.rank == 0:
        start = resolve_chain_start(params, tod_active, compsep_active)
    start = mpi_info.world.comm.bcast(start, root=0)
    if start.first_iteration > params.gibbs.num_iterations:
        if mpi_info.world.rank == 0:
            logger.summary(f"The chains already hold all {params.gibbs.num_iterations} iterations "
                           "of gibbs.num_iterations; nothing to do.")
        return 0

    if mpi_info.world.rank == 0:
        logger.summary(f"TOD side enabled: {tod_active}.")
        logger.summary(f"CompSep side enabled: {compsep_active}.")
        component_names: list[str] = []
        for component_name in params.components:
            component = params.components[component_name]
            if "enabled" not in component or component.enabled:
                component_names.append(component_name)
        logger.summary(f"Enabled sky components: {', '.join(component_names)}.")

    if mpi_info.world.side == "tod":
        run_tod_side(mpi_info, params, start)
    else:
        run_compsep_side(mpi_info, params, start)
    return 0

def main() -> None:
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
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
