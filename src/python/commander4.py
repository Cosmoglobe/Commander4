import numpy as np
import os
import yaml
from mpi4py import MPI
import cProfile
import pstats
from traceback import print_exc

from tod_loop import tod_loop
from compsep_loop import compsep_loop
from constrained_cmb_loop_MPI import constrained_cmb_loop_MPI
from constrained_cmb_loop import constrained_cmb_loop


def main(params, params_dict):
        
    worldsize, worldrank = MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD.Get_rank()

    if worldrank == 0:
        print("### PARAMETERS ###\n", yaml.dump(params_dict, allow_unicode=True, default_flow_style=False))
        if not os.path.isdir(params.output_paths.plots):
            os.mkdir(params.output_paths.plots)
        if not os.path.isdir(params.output_paths.stats):
            os.mkdir(params.output_paths.stats)

    if worldsize != (params.MPI_config.ntask_tod + params.MPI_config.ntask_compsep + params.MPI_config.ntask_cmb):
        raise RuntimeError(f"Total number of MPI tasks ({worldsize}) must equal the sum of tasks for TOD ({params.MPI_config.ntask_tod}) + CompSep ({params.MPI_config.ntask_compsep}) + CMB realization ({params.MPI_config.ntask_cmb}).")

    if (not params.MPI_config.use_MPI_for_CMB) and (params.MPI_config.ntask_cmb > 1):
        raise RuntimeError(f"Number of MPI tasks allocated to CMB realization cannot be > 1 if 'use_MPI_for_CMB' is False.")

    if params.MPI_config.ntask_compsep > 1:
        raise RuntimeError(f"CompSep currently doesn't support more than 1 MPI task.")

    # check if we have at least ntask_compsep+1 MPI tasks, otherwise abort
    if params.MPI_config.ntask_compsep+1 > worldsize:
        raise RuntimeError("not enough MPI tasks started; need at least", params.MPI_config.ntask_compsep+1)

    doing_cmb = params.MPI_config.ntask_cmb > 0
    if doing_cmb:
        num_bands = len(params.bands)
        if num_bands > params.MPI_config.ntask_cmb:
            raise RuntimeError("If running with concurrent CMB sampling, ntask_cmb must be greater than or equal to the number of bands")

    # split the world communicator into a communicator for compsep and one for TOD
    # world rank [0; ntask_compsep[ => compsep
    # world rank [ntask_compsep; ntasks_total[ => TOD processing
    if worldrank < params.MPI_config.ntask_tod:
        color = 0  # TOD
    elif worldrank < params.MPI_config.ntask_tod + params.MPI_config.ntask_compsep:
        color = 1  # Compsep
    else:
        color = 2  # Constrained CMB
    mycomm = MPI.COMM_WORLD.Split(color, key=worldrank)
    print(f"MPI split performed, hi from worldrank {worldrank}, subcomrank {mycomm.Get_rank()} from color {color} of size {mycomm.Get_size()}.")

    # Determine the world ranks of the respective master tasks for compsep and TOD
    # We ensured that this works by the "key=worldrank" in the split command.
    tod_master = 0 
    compsep_master = params.MPI_config.ntask_tod
    if doing_cmb:
        cmb_master = params.MPI_config.ntask_tod + params.MPI_config.ntask_compsep
    else:
        cmb_master = None

    # execute the appropriate part of the code (MPMD)
    if color == 0:
        print(f"Worldrank {worldrank}, subrank {mycomm.Get_rank()} going into compsep loop.")
        tod_loop(mycomm, compsep_master, params.niter_gibbs, params)
    elif color == 1:
        print(f"Worldrank {worldrank}, subrank {mycomm.Get_rank()} going into TOD loop.")
        compsep_loop(mycomm, tod_master, cmb_master, params, use_MPI_for_CMB=True)
    elif color == 2:
        print(f"Worldrank {worldrank}, subrank {mycomm.Get_rank()} going into CMB loop.")
        if params.MPI_config.use_MPI_for_CMB:
            constrained_cmb_loop_MPI(mycomm, compsep_master, params)
        else:
            constrained_cmb_loop(mycomm, compsep_master, params)


if __name__ == "__main__":
    # Parse parameter file
    from parse_params import params, params_dict
    try:
        if params.output_stats:
            profiler = cProfile.Profile()
            profiler.enable()
        main(params, params_dict)
        if params.output_stats:
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats('tottime')
            print(f"Rank {MPI.COMM_WORLD.Get_rank()} cProfile stats:")
            stats.print_stats(10)
            stats.dump_stats(f'{params.output_paths.stats}/stats-{MPI.COMM_WORLD.Get_rank()}')
    except Exception as error:
        print_exc()  # Print the full exception raise, including trace-back.
        print(f">>>>>>>> Error encountered on rank {MPI.COMM_WORLD.Get_rank()}, calling MPI abort.")
        MPI.COMM_WORLD.Abort()
