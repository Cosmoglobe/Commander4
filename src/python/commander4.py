import numpy as np
import os
import yaml
from mpi4py import MPI
import cProfile
import pstats
import logging
from output import log
import io
import time
from traceback import print_exc

from tod_processing import process_tod, receive_tod, send_tod, init_tod_processing, get_empty_compsep_output
from compsep_processing import process_compsep, receive_compsep, send_compsep, init_compsep_processing


def main(params, params_dict):
    logger = logging.getLogger(__name__)
        
    worldsize, worldrank = MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD.Get_rank()
    world_master = worldrank == 0

    if world_master:
        logger.info(f"### PARAMETERS ###\n {yaml.dump(params_dict, allow_unicode=True, default_flow_style=False)}")
        if not os.path.isdir(params.output_paths.plots):
            os.mkdir(params.output_paths.plots)
        if not os.path.isdir(params.output_paths.stats):
            os.mkdir(params.output_paths.stats)

    if worldsize != (params.MPI_config.ntask_tod + params.MPI_config.ntask_compsep + params.MPI_config.ntask_cmb):
        log.lograise(RuntimeError, f"Total number of MPI tasks ({worldsize}) must equal the sum of tasks for TOD ({params.MPI_config.ntask_tod}) + CompSep ({params.MPI_config.ntask_compsep}) + CMB realization ({params.MPI_config.ntask_cmb}).", logger)

    if (not params.MPI_config.use_MPI_for_CMB) and (params.MPI_config.ntask_cmb > 1):
        log.lograise(RuntimeError, f"Number of MPI tasks allocated to CMB realization cannot be > 1 if 'use_MPI_for_CMB' is False.", logger)

    if params.MPI_config.ntask_compsep > 1:
        log.lograise(RuntimeError, f"CompSep currently doesn't support more than 1 MPI task.")

    # check if we have at least ntask_compsep+1 MPI tasks, otherwise abort
    if params.MPI_config.ntask_compsep+1 > worldsize:
        log.lograise(RuntimeError, f"not enough MPI tasks started; need at least {params.MPI_config.ntask_compsep+1}", logger)

    doing_cmb = params.MPI_config.ntask_cmb > 0
    if doing_cmb:
        num_bands = len(params.bands)
        if num_bands > params.MPI_config.ntask_cmb:
            log.lograise(RuntimeError, "If running with concurrent CMB sampling, ntask_cmb must be greater than or equal to the number of bands", logger)

    # split the world communicator into a communicator for compsep and one for TOD
    # world rank [0; ntask_compsep[ => compsep
    # world rank [ntask_compsep; ntasks_total[ => TOD processing
    if worldrank < params.MPI_config.ntask_tod:
        color = 0  # TOD
        os.environ["OMP_NUM_THREADS"] = "1"  # Setting threading configuration depending on tasks. Important to do before Numpy is imported, as Numpy will not respect changes to these.
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
    elif worldrank < params.MPI_config.ntask_tod + params.MPI_config.ntask_compsep:
        color = 1  # Compsep
        os.environ["OMP_NUM_THREADS"] = f"{params.nthreads_compsep}"
        os.environ["OPENBLAS_NUM_THREADS"] = f"{params.nthreads_compsep}"
        os.environ["MKL_NUM_THREADS"] = f"{params.nthreads_compsep}"
        os.environ["VECLIB_MAXIMUM_THREADS"] = f"{params.nthreads_compsep}"
        os.environ["NUMEXPR_NUM_THREADS"] = f"{params.nthreads_compsep}"
    else:
        color = 2  # Constrained CMB
    proc_comm = MPI.COMM_WORLD.Split(color, key=worldrank)
    logger.info(f"MPI split performed, hi from worldrank {worldrank}, subcomrank {proc_comm.Get_rank()} from color {color} of size {proc_comm.Get_size()}.")

    # Determine the world ranks of the respective master tasks for compsep and TOD
    # We ensured that this works by the "key=worldrank" in the split command.
    tod_master = 0 
    compsep_master = params.MPI_config.ntask_tod
    if doing_cmb:
        cmb_master = params.MPI_config.ntask_tod + params.MPI_config.ntask_compsep
    else:
        cmb_master = None
    masters = {'tod': tod_master,
               'compsep': compsep_master,
               'cmb': cmb_master}

    if color == 0:
        proc_master, proc_comm, band_comm, band_master, experiment_data = init_tod_processing(proc_comm, params)
        # Initialization for all TOD processing tasks goes here

        # Chain #1
        # do TOD processing, resulting in maps_chain1
        # we start with a fake output of component separation, containing a completely empty sky
        compsep_output_black = get_empty_compsep_output(experiment_data, params)

        curr_tod_output = process_tod(band_comm, experiment_data,
                                      compsep_output_black, params)
        send_tod(band_master, curr_tod_output, 0, 1, masters['compsep'])
        curr_compsep_output = compsep_output_black

    elif color == 1:
        proc_master, proc_comm, num_bands = init_compsep_processing(proc_comm, params)
        curr_tod_output, iter_num, chain_num = receive_tod(proc_master, proc_comm, [masters['tod'] + i for i in range(num_bands)], num_bands)

    for i in range(1, 2 * params.niter_gibbs): # 2 because we have two chains
        # execute the appropriate part of the code (MPMD)
        if color == 0:
            logger.info(f"Worldrank {worldrank}, subrank {proc_comm.Get_rank()} starting TOD iteration.")
            t0 = time.time()
            chain_num = int(i % 2) + 1
            iter_num = int(i / 2) + 1
            curr_tod_output = process_tod(band_comm, experiment_data, curr_compsep_output, params)
            logger.info(f"TOD: Rank {proc_comm.Get_rank()} finished chain {chain_num}, iter {iter_num} in {time.time()-t0:.2f}s. Receiving compsep results.")
            curr_compsep_output = receive_compsep(band_master, band_comm, masters['compsep'])
            logger.info(f"TOD: Rank {proc_comm.Get_rank()} finished receiving results for chain {chain_num+1}, iter {iter_num+1}. Sending TOD results")
            send_tod(band_master, curr_tod_output, iter_num, chain_num, masters['compsep'])
            logger.info(f"TOD: Rank {proc_comm.Get_rank()} finished sending results for chain {chain_num}, iter {iter_num}. Sending TOD results")

        elif color == 1:
            logger.info(f"Worldrank {worldrank}, subrank {proc_comm.Get_rank()} going into compsep loop for chain {chain_num}, iter {iter_num}.")
            t0 = time.time()
            curr_compsep_output, curr_foreground_maps = process_compsep(
                curr_tod_output, iter_num, chain_num, params, proc_master)
            logger.info(f"Compsep: Rank {proc_comm.Get_rank()} finished chain {chain_num}, iter {iter_num} in {time.time()-t0:.2f}s. Sending results.")
            send_compsep(
                proc_master, curr_compsep_output, [masters['tod']+i for i in
                                                   range(num_bands)])
            logger.info(f"Compsep: Rank {proc_comm.Get_rank()} finished sending results for chain {chain_num}, iter {iter_num}. Receiving TOD results.")
            curr_tod_output, iter_num, chain_num = receive_tod(
                proc_master, proc_comm, [masters['tod'] + i for i in
                                         range(num_bands)], num_bands)
            logger.info(f"Compsep: Rank {proc_comm.Get_rank()} finished receiving TOD results for chain {chain_num}, iter {iter_num}.")
    # stop compsep machinery
    if world_master:
        logger.info("TOD: sending STOP signal to compsep")
        MPI.COMM_WORLD.send(True, dest=compsep_master)


if __name__ == "__main__":
    # Parse parameter file
    from parse_params import params, params_dict
    log.init_loggers(params.logging)
    logger = logging.getLogger(__name__)
    try:
        if params.output_stats:
            profiler = cProfile.Profile()
            profiler.enable()
        main(params, params_dict)
        if params.output_stats:
            profiler.disable()
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s).sort_stats('tottime')
            stats.print_stats(10)
            logger.info(f"Rank {MPI.COMM_WORLD.Get_rank()} cProfile stats: {s.getvalue()}")

            stats.dump_stats(f'{params.output_paths.stats}/stats-{MPI.COMM_WORLD.Get_rank()}')
    except Exception as error:
        print_exc()  # Print the full exception raise, including trace-back.
        logger.error(f">>>>>>>> Error encountered on rank {MPI.COMM_WORLD.Get_rank()}, calling MPI abort.")
        MPI.COMM_WORLD.Abort()
