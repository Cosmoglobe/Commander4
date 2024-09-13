# NOTE: I'm using simple MPI communication here (lowercase names, dealing with
# high level Python objects), to make the code shorter and easier to read.
# In the final version, communication should of course be done with the
# uppercase MPI functions.

import numpy as np
from mpi4py import MPI
from tod_loop import tod_loop
from dummy_compsep_loop import compsep_loop

# PARAMETERS (will be obtained from a parameter file or similar
#             in the production version)

# the number of MPI tasks we want to work on component separation;
# the remaining tasks will do TOD processing
ntask_compsep = 1

# number of iterations for the Gibbs loop
niter_gibbs=10

if __name__ == "__main__":
    # get data about world communicator
    worldsize, worldrank = MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD.Get_rank()

    # check if we have at least ntask_compsep+1 MPI tasks, otherwise abort
    if ntask_compsep+1 > worldsize:
        raise RuntimeError("not enough MPI tasks started; need at least", ntask_compsep+1)

    # split the world communicator into a communicator for compsep and one for TOD
    # world rank [0; ntask_compsep[ => compsep
    # world rank [ntask_compsep; ntasks_total[ => TOD processing
    mycomm = MPI.COMM_WORLD.Split(worldrank<ntask_compsep, key=worldrank)

    # Determine the world ranks of the respective master tasks for compsep and TOD
    # We ensured that this works by the "key=worldrank" in the split command.
    compsep_master = 0
    tod_master = ntask_compsep

    # execute the appropriate part of the code (MPMD)
    if worldrank < ntask_compsep:
        compsep_loop(mycomm, tod_master)
    else:
        tod_loop(mycomm, compsep_master, niter_gibbs)
