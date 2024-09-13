import numpy as np
from mpi4py import MPI
import time
import h5py
import healpy as hp
from data import SimpleScan, SimpleDetector, SimpleDetectorGroup, SimpleBand, TodProcData
from tod_loop import tod_loop

class Compsep2TodprocData:
    pass

class Tod2CompsepData:
    pass


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

