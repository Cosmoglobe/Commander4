import numpy as np
from mpi4py import MPI
import h5py
import healpy as hp
import ducc0
from data import SimpleScan, SimpleDetector, SimpleDetectorGroup, SimpleBand, TodProcData

nthreads=1

def alm2map(alm, nside, lmax):
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    return ducc0.sht.synthesis(alm=alm.reshape((1,-1)),
                               lmax=lmax,
                               spin=0,
                               nthreads=nthreads, **geom).reshape((-1,))

# this is currently a list[list[list[alm]]]
# the outermost list is over bands
# the next inner list is over detector groups
# the innermost list is over detectors
class Compsep2TodprocData:
    pass

# this is currently a list[list[list[(map, map_rms)]]]
# the outermost list is over bands
# the next inner list is over detector groups
# the innermost list is over detectors
class Tod2CompsepData:
    pass

def get_empty_compsep_output(staticData: TodProcData) -> Compsep2TodprocData:
    res = []
    for band in staticData.bands:
        resband = []
        for detgrp in band.detectorGroups:
            resgrp = []
            for det in detgrp.detectors:
                lmax = band.lmax
                resgrp.append(np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128))
            resband.append(resgrp)
        res.append(resband)
    return res

class MapMaker:
    def __init__(self):
        pass

    def tod2map(self, staticData: TodProcData, compsepData: Compsep2TodprocData) -> Tod2CompsepData:
        res = []
        for band, csband in zip(staticData.bands, compsepData):
            resband = []
            for detgrp, csdetgrp in zip(band.detectorGroups, csband):
                resgrp = []
                for det, det_cs_alm in zip(detgrp.detectors, csdetgrp):
                    lmax = hp.Alm.getlmax(det_cs_alm.shape[0])
                    nside = lmax//2
                    map_estimate = alm2map(det_cs_alm, nside, lmax)
                    detmap = np.zeros(12*nside**2)
                    detmap_inv_var = np.zeros(12*nside**2)
                    for scan in det.scans:
                        val, theta, phi, psi = scan.data
                        pix = hp.ang2pix(nside, theta, phi)
                        tod_unroll = map_estimate[pix]
                        sigma0 = np.std(tod_unroll-val)/np.sqrt(2)
                        detmap += np.bincount(pix, weights=val/sigma0**2, minlength=12*nside**2)
                        detmap_inv_var += np.bincount(pix, minlength=12*nside**2)/sigma0**2
                    detmap_rms =  1.0/np.sqrt(detmap_inv_var)
                    detmap /= detmap_inv_var
                    resgrp.append((detmap, detmap_rms))
                resband.append(resgrp)
            res.append(resband)
        return res

# adhoc TOD data reader, to be improved
def read_data() -> TodProcData:
    h5_filename = '../../../commander4_sandbox/src/python/preproc_scripts/tod_example_64_s1.0_b20_dust.h5'
    bands = ['0030', '0100', '0217', '0353']
    nside = 64
    lmax = 128
    nscan=2
    ntod=1000
    out=[]
    with h5py.File(h5_filename) as f:
        bandlist = []
        for band in bands:
            scanlist = []
            for iscan in range(nscan):
                tod = f[f'{iscan+1:06}/{band}/tod'][:ntod].astype(np.float64)
                pix = f[f'{iscan+1:06}/{band}/pix'][:ntod]
                psi = f[f'{iscan+1:06}/{band}/psi'][:ntod].astype(np.float64)
                theta, phi = hp.pix2ang(nside, pix)
                scanlist.append(SimpleScan(tod, theta, phi, psi, 0.))
            det = SimpleDetector(scanlist)  #, fsamp, ...)
            detGroup = SimpleDetectorGroup([det])
            bandlist.append(SimpleBand([detGroup], lmax))
    return TodProcData(bandlist)


# TOD processing loop
def tod_loop(comm, compsep_master, niter_gibbs):
    # am I the master of the TOD communicator?
    master = comm.Get_rank() == 0

    # Initialization for all TOD processing tasks goes here
    experiment_data = read_data()

    mapMaker = MapMaker()

    # Chain #1
    # do TOD processing, resulting in maps_chain1

    # we start with a fake output of component separation, containing
    # a completely empty sky(?)
    compsep_output_black = get_empty_compsep_output(experiment_data)

    todproc_output_chain1 = mapMaker.tod2map(experiment_data, compsep_output_black)
    todproc_output_chain2 = mapMaker.tod2map(experiment_data, compsep_output_black)

    compsep_output_chain2 = compsep_output_black
 
    for i in range(niter_gibbs):
        if master:
            print("TOD: sending chain1 data")
            MPI.COMM_WORLD.send(False, dest=compsep_master)  # we don't want to stop yet
            MPI.COMM_WORLD.send(todproc_output_chain1, dest=compsep_master)
            # del todproc_output_chain1

        # get compsep results for chain #1
        if master:
            compsep_output_chain1 = MPI.COMM_WORLD.recv(source=compsep_master)
            print("TOD: received chain1 data")

        if master:
            print("TOD: sending chain2 data")
            MPI.COMM_WORLD.send(False, dest=compsep_master)  # we don't want to stop yet
            MPI.COMM_WORLD.send(todproc_output_chain2, dest=compsep_master)
            # del todproc_output_chain2

        # get compsep results for chain #2
        if master:
            compsep_output_chain2 = MPI.COMM_WORLD.recv(source=compsep_master)
            print("TOD: received chain2 data")

    # stop compsep machinery
    if master:
        print("TOD: sending STOP signal to compsep")
        MPI.COMM_WORLD.send(True, dest=compsep_master)
