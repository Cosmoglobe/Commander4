import os
import yaml
from mpi4py import MPI
import cProfile
import pstats
import logging
import io
import time
import sys
from copy import deepcopy
from pixell.bunch import Bunch
from traceback import print_exc

# Current solution to making sure the root directory is in the path. I don't like it, but it works for now (alternative seems to be running the entire thing as a module).
module_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(module_root_path)

from src.python.output import log
from src.python import mpi_management


def main(params: Bunch, params_dict: dict):
    logger = logging.getLogger(__name__)
        

    mpi_info = mpi_management.init_mpi(params)
    if mpi_info['world']['is_master']:
        import random
        logger.info(f"### PARAMETERS ###\n {yaml.dump(params_dict, allow_unicode=True, default_flow_style=False)}")
        logger.info(f"\033[{random.randint(91, 96)}m" + r"""
           ______                                          __             __ __
          / ____/___  ____ ___  ____ ___  ____ _____  ____/ /__  _____   / // /
         / /   / __ \/ __ `__ \/ __ `__ \/ __ `/ __ \/ __  / _ \/ ___/  / // /_
        / /___/ /_/ / / / / / / / / / / / /_/ / / / / /_/ /  __/ /     /__  __/
        \____/\____/_/ /_/ /_/_/ /_/ /_/\__,_/_/ /_/\__,_/\___/_/        /_/""" + "\033[0m\n")
        logger.info(f"Starting Commander 4 with {mpi_info.world.size} total MPI tasks!")
        os.makedirs(params.output_paths.plots, exist_ok=True)
        os.makedirs(params.output_paths.stats, exist_ok=True)

    import numpy as np  # Import Numpy after specifying threading, such that it respects our settings.
    import src.python.output.log as log
    from src.python.tod_processing import process_tod, init_tod_processing, get_empty_compsep_output
    from src.python.compsep_processing import process_compsep, init_compsep_processing
    from src.python.communication import receive_tod, send_tod, receive_compsep, send_compsep
    seed = hash((1995, mpi_info['world']['rank']))%(2**32)  # Unique seed per worldrank. Has it for
                                                            # slightly improved entropy. Modulus
                                                            # because seed needs to be 32 bit.
    np.random.seed(seed)  # The optimal seed solution would require carrying around an instance of a
                          # "np.random.default_rng" (see https://numpy.org/doc/2.2/reference/random/parallel.html).

    ###### Initizatization ######
    # Setting up dictionaries mapping each experiment+band combo to the world rank of the master
    # task for that band (on both the TOD and CompSep sides).
    world_compsep_band_masters_dict = None
    world_tod_band_masters_dict = None
    if mpi_info.world.color == 0:
        mpi_info, my_band_identifier, experiment_data, detector_samples = init_tod_processing(mpi_info, params)
        detector_samples_chain1 = detector_samples
        detector_samples_chain2 = deepcopy(detector_samples)
    elif mpi_info.world.color == 1:
        components, mpi_info, my_band_identifier, my_band = init_compsep_processing(mpi_info, params)

    if mpi_info.world.tod_master is not None:
        # All processes, both compsep and tod, need the world-specific band master dict
        world_tod_band_masters_dict = mpi_info.world.comm.bcast(mpi_info.world.tod_band_masters,
                                                                root=mpi_info.world.tod_master)
        mpi_info['world']['tod_band_masters'] = world_tod_band_masters_dict
    if mpi_info.world.compsep_master is not None:
        world_compsep_band_masters_dict = mpi_info.world.comm.bcast(
            mpi_info.world.compsep_band_masters, root=mpi_info.world.compsep_master)
        mpi_info['world']['compsep_band_masters'] = world_compsep_band_masters_dict
    ###### Sending empty data back and forth ######
    curr_tod_output = None
    if mpi_info.world.color == 0:
        # Chain #1 do TOD processing, resulting in maps_chain1 (we start with a fake output of
        # component separation, containing a completely empty sky).
        compsep_output_black = get_empty_compsep_output(experiment_data)

        curr_tod_output, detector_samples = process_tod(mpi_info, experiment_data,
                                                        detector_samples_chain1,
                                                        compsep_output_black, params, 1, 1)
        send_tod(mpi_info, curr_tod_output, my_band_identifier, mpi_info.world.compsep_band_masters)
        curr_compsep_output = compsep_output_black

    elif mpi_info.world.color == 1:
        curr_tod_output = receive_tod(mpi_info, mpi_info.world.tod_band_masters, my_band,
                                      my_band_identifier, curr_tod_output)

    ###### Main loop ######
    # Iteration numbers are 1-indexed, and chain 1 iter 1 TOD step is already done pre-loop.
    for i in range(1, 2 * params.niter_gibbs): # x2 because we have two chains
        # execute the appropriate part of the code (MPMD)
        if mpi_info.world.color == 0:
            t0 = time.time()
            iter_num = (i + 2) // 2  # [1, 2, 2, 3, 3,...] -  Since TOD already did iteration 1 for
                                     # chain 1, it is "half" an iteration ahead.
            chain_num = i % 2 + 1  # [2, 1, 2, 1,...] - TOD has already been done for chain 1 iter 1
                                   # pre-loop, so we start with TOD for chain 2.
            if mpi_info.tod.rank == 0:
                logger.info(f"Worldrank {mpi_info.world.rank}, subrank"
                            f"{mpi_info.tod.rank} starting TOD iteration {iter_num}.")
            if chain_num == 1:
                curr_tod_output, detector_samples_chain1 = process_tod(mpi_info, experiment_data,
                                                                       detector_samples_chain1,
                                                                       curr_compsep_output, params,
                                                                       chain_num, iter_num)
            elif chain_num == 2:
                curr_tod_output, detector_samples_chain2 = process_tod(mpi_info, experiment_data,
                                                                       detector_samples_chain2,
                                                                       curr_compsep_output, params,
                                                                       chain_num, iter_num)
            if mpi_info.tod.is_master:
                logger.info(f"TOD: Rank {mpi_info.tod.rank} finished chain {chain_num}, iter "
                            f"{iter_num} in {time.time()-t0:.2f}s. Receiving compsep results.")
            t0 = time.time()
            curr_compsep_output = receive_compsep(mpi_info, experiment_data,
                                                  my_band_identifier,
                                                  mpi_info.world.compsep_band_masters)
            if mpi_info.band.is_master:
                logger.info(f"TOD: Rank {mpi_info.tod.rank} finished receiving "
                            f"results for chain {chain_num}, iter {iter_num} "
                            f"(time spent waiting+receiving = "
                            f"{time.time()-t0:.1f}s).")
            send_tod(mpi_info, curr_tod_output, my_band_identifier,
                     mpi_info.world.compsep_band_masters)
            if mpi_info.tod.is_master:
                logger.info(f"TOD: Rank {mpi_info.tod.rank} finished sending "
                            f"results for chain {chain_num}, iter {iter_num}.")

        elif mpi_info.world.color == 1:
            iter_num = (i + 1) // 2  # [1, 1, 2, 2, 3,...] Compsep has not done iteration 1 for neither chain yet.
            chain_num = (i + 1) % 2 + 1  # [1, 2, 1, 2,...] We start as chain 1, since that's the chain that has already done a TOD step pre-loop.
            if mpi_info.compsep.rank == 0:
                logger.info(f"Worldrank {mpi_info.world.rank}, subrank {mpi_info.compsep.rank} "
                            f"going into compsep loop for chain {chain_num}, iter {iter_num}.")
            t0 = time.time()
            curr_compsep_output = process_compsep(mpi_info, curr_tod_output, iter_num, chain_num,
                                                  params, components)
            if mpi_info.compsep.rank == 0:
                logger.info(f"Compsep: Rank {mpi_info.compsep.rank} finished chain {chain_num}, "
                            f"{iter_num} in {time.time()-t0:.2f}s. Sending compsep results.")
            send_compsep(mpi_info, my_band_identifier, curr_compsep_output, mpi_info.world.tod_band_masters)
            logger.info(f"Compsep: Rank {mpi_info.compsep.rank} finished sending results for chain "
                        f"{chain_num}, iter {iter_num}. Waiting for TOD results.")
            t0 = time.time()
            curr_tod_output = receive_tod(mpi_info, mpi_info.world.tod_band_masters, my_band,
                                          my_band_identifier, curr_tod_output)
            logger.info(f"Compsep: Rank {mpi_info.compsep.rank} finished receiving TOD results for "
                        f"chain {chain_num}, iter {iter_num} (time spent waiting+receiving = "
                        f"{time.time()-t0:.1f}s).")
    # stop compsep machinery
    if mpi_info.world.is_master:
        logger.info("TOD: sending STOP signal to compsep")
        mpi_info.world.comm.send(True, dest=mpi_info.world.compsep_master)

    return 0

if __name__ == "__main__":
    # Parse parameter file
    from src.python.parse_params import params, params_dict
    log.init_loggers(params.logging)
    logger = logging.getLogger(__name__)
    try:
        if params.output_stats:
            profiler = cProfile.Profile()
            profiler.enable()
        ret = main(params, params_dict)
        logger.info(f"Rank {MPI.COMM_WORLD.Get_rank()} finished Commander 4 and is shutting down. Goodbye.")
        if params.output_stats:
            profiler.disable()
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s).sort_stats('tottime')
            if ret != -1:
                stats.print_stats(10)
                logger.info(f"Rank {MPI.COMM_WORLD.Get_rank()} cProfile stats: {s.getvalue()}")
                stats.dump_stats(f'{params.output_paths.stats}/stats-{MPI.COMM_WORLD.Get_rank()}')

    # First check for MPI-specific exceptions.
    except MPI.Exception as e:
        print_exc()
        error_code = e.Get_error_code()
        error_string = MPI.Get_error_string(error_code)
        logger.error(f">>>>>>>> MPI Error on rank {MPI.COMM_WORLD.Get_rank()}! Code: [{error_code}] - {error_string}")
        MPI.COMM_WORLD.Abort(error_code)

    # Then general exceptions.
    except Exception:
        print_exc()  # Print the full exception raise, including trace-back.
        logger.error(f">>>>>>>> Error on rank {MPI.COMM_WORLD.Get_rank()}, calling MPI abort.")
        MPI.COMM_WORLD.Abort()
