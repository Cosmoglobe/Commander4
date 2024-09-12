# NOTE: I'm using simple MPI communication here (lowercase names, dealing with
# high level Python objects), to make the code shorter and easier to read.
# In the final version, communication should of course be done with the
# uppercase MPI functions.

import numpy as np
from mpi4py import MPI
import time

# PARAMETERS (will be obtained from a parameter file or similar
#             in the production version)

# the number of MPI tasks we want to work on component separation;
# the remaining tasks will do TOD processing
ntask_compsep = 2

# number of iterations for the Gibbs loop
niter_gibbs=10

# Component separation loop
def compsep_loop(comm, tod_master):
    # am I the master of the compsep communicator?
    master = comm.Get_rank() == 0
    if master:
        print("Compsep: loop started")

    # Initialization for all component separattion tasks goes here


    # we wait for new jobs until we get a stop signal
    while True:
        # check for simulation end
        stop = MPI.COMM_WORLD.recv(source=tod_master) if master else False
        stop = comm.bcast(stop, root=0)
        if stop:
            if master:
                print("Compsep: stop requested; exiting")
            return
        if master:
            print("Compsep: new job obtained")

        # get next data set for component separation
        data = MPI.COMM_WORLD.recv(source=tod_master) if master else None 
        # Broadcast te data to all tasks, or do anything else that's appropriate
        data = comm.bcast(data, root=0)
        if master:
            print("Compsep: data obtained. Working on it ...")

        # do stuff with data
        time.sleep(1)
        result = 2*data  # dummy

        # assemble result on master, via reduce, gather, whatever ...
        # send result
        if master:
            MPI.COMM_WORLD.send(result, dest=tod_master)
            print("Compsep: results sent back")


# TOD processing loop
def tod_loop(comm, compsep_master):
    # am I the master of the TOD communicator?
    master = comm.Get_rank() == 0

    # Initialization for all TOD processing tasks goes here


    # Chain #1
    # do TOD processing, resulting in compsep_input
    # Note: compsep is still idle at this point
    time.sleep(1)
    compsep_input1 = np.zeros(10000)  # dummy

    for i in range(niter_gibbs):
        if master:
            print("TOD: sending chain1 data")
            MPI.COMM_WORLD.send(False, dest=compsep_master)  # we don't want to stop yet
            MPI.COMM_WORLD.send(compsep_input1, dest=compsep_master)

        # Chain #2
        # do TOD processing, resulting in compsep_input
        # at the same time, compsep is working on chain #1 data
        time.sleep(1)
        compsep_input2 = np.zeros(10000)  # dummy

        # get compsep results for chain #1
        if master:
            compsep_output1 = MPI.COMM_WORLD.recv(source=compsep_master)
            print("TOD: received chain1 data")

        if master:
            print("TOD: sending chain2 data")
            MPI.COMM_WORLD.send(False, dest=compsep_master)  # we don't want to stop yet
            MPI.COMM_WORLD.send(compsep_input2, dest=compsep_master)

        # Chain #1
        # do TOD processing, resulting in compsep_input
        # at the same time, compsep is working on chain #2 data
        time.sleep(1)
        compsep_input1 = np.zeros(10000)  # dummy

        # get compsep results for chain #2
        if master:
            compsep_output2 = MPI.COMM_WORLD.recv(source=compsep_master)
            print("TOD: received chain2 data")

    # stop compsep machinery
    if master:
        print("TOD: sending STOP signal to compsep")
        MPI.COMM_WORLD.send(True, dest=compsep_master)


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
        tod_loop(mycomm, compsep_master)
