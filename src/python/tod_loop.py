import numpy as np
from mpi4py import MPI
import time
import h5py
import healpy as hp
from data import SimpleScan, SimpleDetector, SimpleDetectorGroup, SimpleBand, TodProcData
from compsep_loop import compsep_loop

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



# TOD processing loop
def tod_loop(comm, compsep_master, niter_gibbs):
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
