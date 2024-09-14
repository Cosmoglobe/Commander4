import numpy as np
import healpy as hp
from mpi4py import MPI


def compsep_loop(comm, tod_master: int):
    # am I the master of the compsep communicator?
    master = comm.Get_rank() == 0
    if master:
        print("Compsep: dummy loop started")

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
            print("Compsep: new job obtained; producing dummy output data")

        data, iter, chain = MPI.COMM_WORLD.recv(source=tod_master) if master else None 
        # Broadcast te data to all tasks, or do anything else that's appropriate
        data = comm.bcast(data, root=0)
        if master:
            print("Compsep: data obtained. Working on it ...")
            res = []
            for band in data:
                resband = []
                for detgrp in band:
                    resgrp = []
                    for _ in detgrp:
                        nside=64
                        map = np.zeros(12*nside**2,dtype=np.float64)
                        resgrp.append(map)
                    resband.append(resgrp)
                res.append(resband)
            MPI.COMM_WORLD.send(res, dest=tod_master)
            print("Compsep: results sent back")
