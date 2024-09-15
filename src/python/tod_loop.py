import numpy as np
from mpi4py import MPI
import h5py
import healpy as hp
import ducc0
from data_models import ScanTOD, DetectorTOD, DetectorMap

nthreads=1

def alm2map(alm, nside, lmax):
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    return ducc0.sht.synthesis(alm=alm.reshape((1,-1)),
                               lmax=lmax,
                               spin=0,
                               nthreads=nthreads, **geom).reshape((-1,))


def get_empty_compsep_output(staticData: list[DetectorTOD]) -> list[np.array]:
    res = []
    for i_det in range(len(staticData)):
        res.append(np.zeros(12*64**2,dtype=np.float64))
    return res

class MapMaker:
    def __init__(self):
        pass

    def tod2map(self, staticData: list[DetectorTOD], compsepData: list[np.array]) -> list[DetectorMap]:
        res = []
        for i_det in range(len(staticData)):
            det_cs_map = compsepData[i_det]
            det_static = staticData[i_det]
            nside = hp.npix2nside(det_cs_map.shape[0])
            detmap_signal = np.zeros(12*nside**2)
            detmap_inv_var = np.zeros(12*nside**2)
            for scan in det_static.scans:
                val, theta, phi, psi = scan.data
                pix = hp.ang2pix(nside, theta, phi)
                tod_unroll = det_cs_map[pix]
                sigma0 = np.std((tod_unroll-val)[1:] - (tod_unroll-val)[:-1])/np.sqrt(2)
                detmap_signal += np.bincount(pix, weights=val/sigma0**2, minlength=12*nside**2)
                detmap_inv_var += np.bincount(pix, minlength=12*nside**2)/sigma0**2
            detmap_rms =  1.0/np.sqrt(detmap_inv_var)
            detmap_signal /= detmap_inv_var
            detmap = DetectorMap(detmap_signal, detmap_rms, det_static.nu)
            res.append(detmap)
        return res


def read_data() -> list[ScanTOD]:
    h5_filename = '../../../commander4_sandbox/src/python/preproc_scripts/tod_example_64_s1.0_b20_dust.h5'
    bands = ['0030', '0100', '0353', '0545', '0857']
    nside = 64
    nscan=2001
    with h5py.File(h5_filename) as f:
        det_list = []
        for band in bands:
            scanlist = []
            for iscan in range(nscan):
                tod = f[f'{iscan+1:06}/{band}/tod'][()].astype(np.float64)
                pix = f[f'{iscan+1:06}/{band}/pix'][()]
                psi = f[f'{iscan+1:06}/{band}/psi'][()].astype(np.float64)
                theta, phi = hp.pix2ang(nside, pix)
                scanlist.append(ScanTOD(tod, theta, phi, psi, 0.))
            det = DetectorTOD(scanlist, float(band))
            det_list.append(det)
    return det_list


# TOD processing loop
def tod_loop(comm, compsep_master, niter_gibbs):
    # am I the master of the TOD communicator?
    master = comm.Get_rank() == 0

    # Initialization for all TOD processing tasks goes here
    experiment_data = read_data()

    mapMaker = MapMaker()

    # Chain #1
    # do TOD processing, resulting in maps_chain1
    # we start with a fake output of component separation, containing a completely empty sky
    compsep_output_black = get_empty_compsep_output(experiment_data)

    todproc_output_chain1 = mapMaker.tod2map(experiment_data, compsep_output_black)

    compsep_output_chain2 = compsep_output_black
 
    for i in range(niter_gibbs):
        if master:
            print("TOD: sending chain1 data")
            MPI.COMM_WORLD.send(False, dest=compsep_master)  # we don't want to stop yet
            MPI.COMM_WORLD.send((todproc_output_chain1, i, 1), dest=compsep_master)
            # del todproc_output_chain1

        # Chain #2
        # do TOD processing, resulting in compsep_input at the same time, compsep is working on chain #1 data
        todproc_output_chain2 = mapMaker.tod2map(experiment_data, compsep_output_chain2)

        # get compsep results for chain #1
        if master:
            compsep_output_chain1 = MPI.COMM_WORLD.recv(source=compsep_master)
            print("TOD: received chain1 data")

        if master:
            print("TOD: sending chain2 data")
            MPI.COMM_WORLD.send(False, dest=compsep_master)  # we don't want to stop yet
            MPI.COMM_WORLD.send((todproc_output_chain2, i, 2), dest=compsep_master)
            # del todproc_output_chain2

        # Chain #1
        # do TOD processing, resulting in compsep_input at the same time, compsep is working on chain #2 data
        todproc_output_chain1 = mapMaker.tod2map(experiment_data, compsep_output_chain1)

        # get compsep results for chain #2
        if master:
            compsep_output_chain2 = MPI.COMM_WORLD.recv(source=compsep_master)
            print("TOD: received chain2 data")

    # stop compsep machinery
    if master:
        print("TOD: sending STOP signal to compsep")
        MPI.COMM_WORLD.send(True, dest=compsep_master)
