import os
import yaml
from mpi4py import MPI
import cProfile
import pstats
import logging
from output import log
import io
import time
import sys
from traceback import print_exc

# Current solution to making sure the root directory is in the path. I don't like it, but it works for now (alternative seems to be running the entire thing as a module).
module_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(module_root_path)


def main(params, params_dict):
    logger = logging.getLogger(__name__)
        
    worldsize, worldrank = MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD.Get_rank()
    world_master = worldrank == 0

    if world_master:
        logger.info(f"### PARAMETERS ###\n {yaml.dump(params_dict, allow_unicode=True, default_flow_style=False)}")
        os.makedirs(params.output_paths.plots, exist_ok=True)
        os.makedirs(params.output_paths.stats, exist_ok=True)

    # Split the world communicator into a communicator for compsep and one for TOD (with "color" being the keyword for the split).
    if worldrank < params.MPI_config.ntask_tod:
        color = 0  # TOD
        os.environ["OMP_NUM_THREADS"] = "1"  # Setting threading configuration depending on tasks. Important to do before Numpy is imported, as Numpy will not respect changes to these.
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
    elif worldrank < params.MPI_config.ntask_tod + params.MPI_config.ntask_compsep:
        if params.betzy_mode:  # Betzy doesn't like heterogeneous MPI setups, so we oversubscribe the compsep ranks with cores we will in practice use as threads.
            if worldrank%params.nthreads_compsep == 0:  # Every nthreads_compsep rank is a real compsep rank, and gets to stay alive.
                color = 1  # Compsep
            else:
                color = 99  # Dummy rank
        else:
            color = 1  # Compsep
        os.environ["OMP_NUM_THREADS"] = f"{params.nthreads_compsep}"
        os.environ["OPENBLAS_NUM_THREADS"] = f"{params.nthreads_compsep}"
        os.environ["MKL_NUM_THREADS"] = f"{params.nthreads_compsep}"
        os.environ["VECLIB_MAXIMUM_THREADS"] = f"{params.nthreads_compsep}"
        os.environ["NUMEXPR_NUM_THREADS"] = f"{params.nthreads_compsep}"
    import numpy as np  # Import Numpy after specifying threading, such that it respects our settings.
    import src.python.output.log as log
    from src.python.tod_processing import process_tod, init_tod_processing, get_empty_compsep_output
    from src.python.compsep_processing import process_compsep, init_compsep_processing
    from src.python.communication import receive_tod, send_tod, receive_compsep, send_compsep

    tot_num_experiment_bands = np.sum([len(params.experiments[experiment].bands) for experiment in params.experiments if params.experiments[experiment].enabled])
    tot_num_compsep_bands = len(params.CompSep_bands)
    tot_num_compsep_bands_from_TOD = len([band for band in params.CompSep_bands if params.CompSep_bands[band].get_from != "file"])  # Number of the bands on CompSep side that come from the TOD side.

    if worldsize != (params.MPI_config.ntask_tod + params.MPI_config.ntask_compsep):
        log.lograise(RuntimeError, f"Total number of MPI tasks ({worldsize}) must equal the sum of tasks for TOD ({params.MPI_config.ntask_tod}) + CompSep ({params.MPI_config.ntask_compsep}).", logger)
    if not params.betzy_mode and params.MPI_config.ntask_compsep != tot_num_compsep_bands:
        log.lograise(RuntimeError, f"CompSep needs exactly as many MPI tasks {params.MPI_config.ntask_compsep} as there are bands {tot_num_compsep_bands}.", logger)
    if params.betzy_mode and params.MPI_config.ntask_compsep != params.nthreads_compsep*tot_num_experiment_bands:
        log.lograise(RuntimeError, f"For Betzy mode, CompSep currently needs exactly as many MPI tasks {params.MPI_config.ntask_compsep} as there are bands {tot_num_experiment_bands} times CompSep threads per rank ({params.nthreads_compsep}).", logger)

    proc_comm = MPI.COMM_WORLD.Split(color, key=worldrank)
    MPI.COMM_WORLD.barrier()
    time.sleep(worldrank*1e-2)  # Small sleep to get prints in nice order.
    logger.info(f"MPI split performed, hi from worldrank {worldrank} (on machine {MPI.Get_processor_name()}) subcomrank {proc_comm.Get_rank()} from color {color} of size {proc_comm.Get_size()}.")

    # Determine the world ranks of the respective master tasks for compsep and TOD
    # We ensured that this works by the "key=worldrank" in the split command.
    tod_master = 0 if params.MPI_config.ntask_tod > 0 else None
    compsep_master = params.MPI_config.ntask_tod

    MPI.COMM_WORLD.barrier()
    time.sleep(worldrank*1e-2)  # Small sleep to get prints in nice order.

    ###### Initizatization ######
    # Setting up dictionaries mapping each experiment+band combo to the world rank of the master task for that band (on both the TOD and CompSep sides).
    tod_band_masters_dict = None
    CompSep_band_masters_dict = None
    if color == 0:
        is_band_master, band_comm, my_band_identifier, tod_band_masters_dict, experiment_data = init_tod_processing(proc_comm, params)
    elif color == 1:
        components, my_band_identifier, CompSep_band_masters_dict, my_band = init_compsep_processing(proc_comm, params)
    CompSep_band_masters_dict = MPI.COMM_WORLD.bcast(CompSep_band_masters_dict, root=compsep_master)  # CompSep tells the rest which compsep ranks are band masters.
    if not tod_master is None:
        tod_band_masters_dict = MPI.COMM_WORLD.bcast(tod_band_masters_dict, root=tod_master)  # TOD tells the rest which TOD ranks are band masters.

    ###### Sending empty data back and forth ######
    curr_tod_output = None
    if color == 0:
        # Chain #1 do TOD processing, resulting in maps_chain1 (we start with a fake output of component separation, containing a completely empty sky).
        compsep_output_black = get_empty_compsep_output(experiment_data, params)

        curr_tod_output = process_tod(band_comm, experiment_data, compsep_output_black, params)
        send_tod(is_band_master, curr_tod_output, CompSep_band_masters_dict, my_band_identifier)
        curr_compsep_output = compsep_output_black

    elif color == 1:
        curr_tod_output = receive_tod(tod_band_masters_dict, proc_comm.rank, my_band, my_band_identifier, curr_tod_output)

    ###### Main loop ######
    # Iteration numbers are 1-indexed, and chain 1 iter 1 TOD step is already done pre-loop.
    for i in range(1, 2 * params.niter_gibbs + 1): # 2 because we have two chains
        # execute the appropriate part of the code (MPMD)
        if color == 0:
            logger.info(f"Worldrank {worldrank}, subrank {proc_comm.Get_rank()} starting TOD iteration.")
            t0 = time.time()
            iter_num = (i + 2) // 2  # [1, 2, 2, 3, 3,...] -  Since TOD already did iteration 1 for chain 1, it is "half" an iteration ahead.
            chain_num = i % 2 + 1  # [2, 1, 2, 1,...] - TOD has already been done for chain 1 iter 1 pre-loop, so we start with TOD for chain 2.
            curr_tod_output = process_tod(band_comm, experiment_data, curr_compsep_output, params)
            logger.info(f"TOD: Rank {proc_comm.Get_rank()} finished chain {chain_num}, iter {iter_num} in {time.time()-t0:.2f}s. Receiving compsep results.")
            curr_compsep_output = receive_compsep(band_comm, my_band_identifier, band_comm.Get_rank()==0, CompSep_band_masters_dict)
            logger.info(f"TOD: Rank {proc_comm.Get_rank()} finished receiving results for chain {chain_num+1}, iter {iter_num+1}. Sending TOD results")
            send_tod(is_band_master, curr_tod_output, CompSep_band_masters_dict, my_band_identifier)
            logger.info(f"TOD: Rank {proc_comm.Get_rank()} finished sending results for chain {chain_num}, iter {iter_num}. Sending TOD results")

        elif color == 1:
            iter_num = (i + 1) // 2  # [1, 1, 2, 2, 3,...] Compsep has not done iteration 1 for neither chain yet.
            chain_num = (i + 1) % 2 + 1  # [1, 2, 1, 2,...] We start as chain 1, since that's the chain that has already done a TOD step pre-loop.
            logger.info(f"Worldrank {worldrank}, subrank {proc_comm.Get_rank()} going into compsep loop for chain {chain_num}, iter {iter_num}.")
            t0 = time.time()
            curr_compsep_output = process_compsep(curr_tod_output, iter_num, chain_num, params, proc_comm, components)
            logger.info(f"Compsep: Rank {proc_comm.Get_rank()} finished chain {chain_num}, iter {iter_num} in {time.time()-t0:.2f}s. Sending results.")
            send_compsep(my_band_identifier, curr_compsep_output, tod_band_masters_dict)
            logger.info(f"Compsep: Rank {proc_comm.Get_rank()} finished sending results for chain {chain_num}, iter {iter_num}. Receiving TOD results.")
            curr_tod_output = receive_tod(tod_band_masters_dict, proc_comm.rank, my_band, my_band_identifier, curr_tod_output)
            logger.info(f"Compsep: Rank {proc_comm.Get_rank()} finished receiving TOD results for chain {chain_num}, iter {iter_num}.")
    # stop compsep machinery
    if world_master:
        logger.info("TOD: sending STOP signal to compsep")
        MPI.COMM_WORLD.send(True, dest=compsep_master)


if __name__ == "__main__":
    # Parse parameter file
    from src.python.parse_params import params, params_dict
    log.init_loggers(params.logging)
    logger = logging.getLogger(__name__)
    try:
        if params.output_stats:
            profiler = cProfile.Profile()
            profiler.enable()
        main(params, params_dict)
        logger.info(f"Rank {MPI.COMM_WORLD.Get_rank()} finished Commander 4 and is shutting down. Goodbye.")
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
