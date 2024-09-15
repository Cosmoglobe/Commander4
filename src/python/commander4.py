# NOTE: I'm using simple MPI communication here (lowercase names, dealing with
# high level Python objects), to make the code shorter and easier to read.
# In the final version, communication should of course be done with the
# uppercase MPI functions.

import numpy as np
from mpi4py import MPI
from tod_loop import tod_loop
from compsep_loop import compsep_loop
from constrained_cmb_loop import constrained_cmb_loop

# PARAMETERS (will be obtained from a parameter file or similar
#             in the production version)

# the number of MPI tasks we want to work on component separation;
# the remaining tasks will do TOD processing
ntask_tod = 1
ntask_compsep = 1
ntask_cmb = 0
doing_cmb = ntask_cmb > 0

# number of iterations for the Gibbs loop
niter_gibbs=6

if __name__ == "__main__":
    # get data about world communicator
    worldsize, worldrank = MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD.Get_rank()

    # check if we have at least ntask_compsep+1 MPI tasks, otherwise abort
    if ntask_compsep+1 > worldsize:
        raise RuntimeError("not enough MPI tasks started; need at least", ntask_compsep+1)

    # split the world communicator into a communicator for compsep and one for TOD
    # world rank [0; ntask_compsep[ => compsep
    # world rank [ntask_compsep; ntasks_total[ => TOD processing
    if worldrank < ntask_tod:
        color = 0  # TOD
    elif worldrank < ntask_tod + ntask_compsep:
        color = 1  # Compsep
    else:
        color = 2  # Constrained CMB
    mycomm = MPI.COMM_WORLD.Split(color, key=worldrank)
    print(f"MPI split performed, hi from worldrank {worldrank}, subcomrank {mycomm.Get_rank()} from color {color} of size {mycomm.Get_size()}.")

    # Determine the world ranks of the respective master tasks for compsep and TOD
    # We ensured that this works by the "key=worldrank" in the split command.
    tod_master = 0 
    compsep_master = ntask_tod
    if doing_cmb:
        cmb_master = ntask_tod + ntask_compsep
    else:
        cmb_master = None

    # execute the appropriate part of the code (MPMD)
    if color == 0:
        print(f"Worldrank {worldrank}, subrank {mycomm.Get_rank()} going into compsep loop.")
        tod_loop(mycomm, compsep_master, niter_gibbs)
    elif color == 1:
        print(f"Worldrank {worldrank}, subrank {mycomm.Get_rank()} going into TOD loop.")
        compsep_loop(mycomm, tod_master, cmb_master)
    elif color == 2:
        print(f"Worldrank {worldrank}, subrank {mycomm.Get_rank()} going into CMB loop.")
        constrained_cmb_loop(mycomm, compsep_master)