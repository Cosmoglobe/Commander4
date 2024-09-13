# NOTE: I'm using simple MPI communication here (lowercase names, dealing with
# high level Python objects), to make the code shorter and easier to read.
# In the final version, communication should of course be done with the
# uppercase MPI functions.

import numpy as np
from mpi4py import MPI
import time
import h5py
import healpy as hp
from data import SimpleScan, SimpleDetector, SimpleDetectorGroup, SimpleBand, TodProcData

# PARAMETERS (will be obtained from a parameter file or similar
#             in the production version)

# the number of MPI tasks we want to work on component separation;
# the remaining tasks will do TOD processing
ntask_compsep = 1

# number of iterations for the Gibbs loop
niter_gibbs=10

class Compsep2TodprocData:
    pass

class Tod2CompsepData:
    pass

class MapMaker:
    def tod2map(staticData: TodProcData, compsepData: Compsep2TodprocData) -> Tod2CompsepData:
        pass

# adhoc TOD data reader, to be improved
def read_data() -> TodProcData:
    h5_filename = '../../../commander4_sandbox/src/python/preproc_scripts/tod_example_64_s1.0_b20_dust.h5'
    bands = ['0030', '0100', '0217', '0353']
    nside = 64
    nscan=2
    ntod=1000
    out=[]
    with h5py.File(h5_filename) as f:
        bandlist = []
        for band in bands:
            scanlist = []
            for iscan in range(nscan):
                print(iscan, band)
                tod = f[f'{iscan+1:06}/{band}/tod'][:ntod].astype(np.float64)
                pix = f[f'{iscan+1:06}/{band}/pix'][:ntod]
                psi = f[f'{iscan+1:06}/{band}/psi'][:ntod].astype(np.float64)
                theta, phi = hp.pix2ang(nside, pix)
                scanlist.append(SimpleScan(tod, theta, phi, psi, 0.))
            det = SimpleDetector(scanlist)  #, fsamp, ...)
            detGroup = SimpleDetectorGroup([det])
            bandlist.append(SimpleBand([detGroup]))
    return TodProcData(bandlist)


# Component separation loop
def compsep_loop(comm, tod_master: int):
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
# compsep result is a data structure _per detector_, describing which sky this
# detector would see. Probably best described as a set of a_lm

        if master:
            MPI.COMM_WORLD.send(result, dest=tod_master)
            print("Compsep: results sent back")


# TOD processing loop
def tod_loop(comm, compsep_master):
    # am I the master of the TOD communicator?
    master = comm.Get_rank() == 0

    # Initialization for all TOD processing tasks goes here
    experiment_data = read_data()

#    mapMaker = buildMapMaker(....)

    # Chain #1
    # do TOD processing, resulting in maps_chain1

    # we start with a fake output of component separation, containing
    # a completely empty sky(?)
    compsep_output_black = FIXME!

    todproc_output_chain1 = mapMaker.tod2map(experiment_data, compsep_output_black)

    compset_output_chain2 = compsep_output_black
 
    for i in range(niter_gibbs):
        if master:
            print("TOD: sending chain1 data")
            MPI.COMM_WORLD.send(False, dest=compsep_master)  # we don't want to stop yet
            MPI.COMM_WORLD.send(todproc_output_chain1, dest=compsep_master)
            # del todproc_output_chain1

        # Chain #2
        # do TOD processing, resulting in compsep_input
        # at the same time, compsep is working on chain #1 data
        todproc_output_chain2 = mapMaker.tod2map(experiment_data, compsep_output_chain2)
        # del compsep_output_chain2

        # get compsep results for chain #1
        if master:
            compsep_output_chain1 = MPI.COMM_WORLD.recv(source=compsep_master)
            print("TOD: received chain1 data")

        if master:
            print("TOD: sending chain2 data")
            MPI.COMM_WORLD.send(False, dest=compsep_master)  # we don't want to stop yet
            MPI.COMM_WORLD.send(todproc_output_chain2, dest=compsep_master)
            # del todproc_output_chain2

        # Chain #1
        # do TOD processing, resulting in compsep_input
        # at the same time, compsep is working on chain #2 data
        todproc_output_chain1 = mapMaker.tod2map(experiment_data, compsep_output_chain1)
        # del compsep_output_chain1

        # get compsep results for chain #2
        if master:
            compsep_output_chain2 = MPI.COMM_WORLD.recv(source=compsep_master)
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
