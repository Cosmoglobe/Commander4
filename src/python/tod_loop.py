import numpy as np
from mpi4py import MPI
import h5py
import healpy as hp
import math
import time
from data_models import ScanTOD, DetectorTOD, DetectorMap
from utils import single_det_mapmaker, single_det_map_accumulator
from pixell import bunch

nthreads=1

def get_empty_compsep_output(staticData: list[DetectorTOD], params) -> list[np.array]:
    return np.zeros(12*params.nside**2,dtype=np.float64)


def tod2map_old(staticData: list[DetectorTOD], compsepData: list[np.array]) -> list[DetectorMap]:
    res = []
    for i_det in range(len(staticData)):
        det_cs_map = compsepData[i_det]
        det_static = staticData[i_det]

        # nside = hp.npix2nside(det_cs_map.shape[0])
        # detmap_signal = np.zeros(12*nside**2)
        # detmap_inv_var = np.zeros(12*nside**2)
        # for scan in det_static.scans:
        #     scan_map, theta, phi, psi = scan.data
        #     pix = hp.ang2pix(nside, theta, phi)
        #     sky_subtracted_tod = det_cs_map[pix] - scan_map
        #     sigma0 = np.std(sky_subtracted_tod[1:] - sky_subtracted_tod[:-1])/np.sqrt(2)
        #     detmap_signal += np.bincount(pix, weights=scan_map/sigma0**2, minlength=12*nside**2)
        #     detmap_inv_var += np.bincount(pix, minlength=12*nside**2)/sigma0**2
        # detmap_rms =  1.0/np.sqrt(detmap_inv_var)
        # detmap_signal /= detmap_inv_var

        # detmap_signal, detmap_rms = single_det_mapmaker_python(det_static, det_cs_map)
        detmap_signal, detmap_rms = single_det_mapmaker(det_static, det_cs_map)
        detmap = DetectorMap(detmap_signal, detmap_rms, det_static.nu)
        res.append(detmap)
    return res


def tod2map(comm, det_static: DetectorTOD, det_cs_map: np.array, params: bunch) -> DetectorMap:
    detmap_signal, detmap_corr_noise, detmap_inv_var = single_det_map_accumulator(det_static, det_cs_map, params.galactic_mask)
    map_signal = np.zeros_like(detmap_signal)
    map_corr_noise = np.zeros_like(detmap_corr_noise)
    map_inv_var = np.zeros_like(detmap_inv_var)
    if comm.Get_rank() == 0:
        comm.Reduce(detmap_signal, map_signal, op=MPI.SUM, root=0)
        comm.Reduce(detmap_corr_noise, map_corr_noise, op=MPI.SUM, root=0)
        comm.Reduce(detmap_inv_var, map_inv_var, op=MPI.SUM, root=0)
    else:
        comm.Reduce(detmap_signal, None, op=MPI.SUM, root=0)
        comm.Reduce(detmap_corr_noise, None, op=MPI.SUM, root=0)
        comm.Reduce(detmap_inv_var, None, op=MPI.SUM, root=0)

    if comm.Get_rank() == 0:
        map_signal[map_signal != 0] /= map_inv_var[map_signal != 0]
        map_corr_noise[map_corr_noise != 0] /= map_inv_var[map_corr_noise != 0]
        map_rms = np.zeros_like(map_inv_var) + np.inf
        map_rms[map_inv_var != 0] = 1.0/np.sqrt(map_inv_var[map_inv_var != 0])
        detmap = DetectorMap(map_signal, map_corr_noise, map_rms, det_static.nu)
        return detmap


def read_data(band_idx, scan_idx_start, scan_idx_stop, params: bunch) -> list[ScanTOD]:
    h5_filename = params.input_paths.tod_filename
    with h5py.File(h5_filename) as f:
        # for band in bands:
        band = params.bands[band_idx]
        band_formatted = f"{band:04d}"
        scanlist = []
        for iscan in range(scan_idx_start, scan_idx_stop):
            try:
                tod = f[f'{iscan+1:06}/{band_formatted}/tod'][()].astype(np.float64)
                pix = f[f'{iscan+1:06}/{band_formatted}/pix'][()]
                psi = f[f'{iscan+1:06}/{band_formatted}/psi'][()].astype(np.float64)
            except KeyError:
                print(iscan)
                print(band_formatted)
                print(list(f))
                raise KeyError
            assert np.max(pix) < 12*params.nside**2, f"Nside is {params.nside}, but found pixel index exceeding 12nside^2 ({np.max(12*params.nside**2)})"
            theta, phi = hp.pix2ang(params.nside, pix)
            scanlist.append(ScanTOD(tod, theta, phi, psi, 0., iscan))
        det = DetectorTOD(scanlist, float(band))
    return det


# TOD processing loop
def tod_loop(comm, compsep_master: int, niter_gibbs: int, params: dict):
    num_bands = len(params.bands)

    # am I the master of the TOD communicator?
    MPIsize_tod, MPIrank_tod = comm.Get_size(), comm.Get_rank()
    master = MPIrank_tod == 0
    if master:
        print(f"TOD: {MPIsize_tod} tasks allocated to TOD processing of {num_bands} bands.")
        assert MPIsize_tod >= num_bands, f"Number of MPI tasks dedicated to TOD processing ({MPIsize_tod}) must be equal to or larger than the number of bands ({num_bands})."

    MPIcolor_band = MPIrank_tod%num_bands  # Spread the MPI tasks over the different bands.
    MPIcomm_band = comm.Split(MPIcolor_band, key=MPIrank_tod)  # Create communicators for each different band.
    MPIsize_band, MPIrank_band = MPIcomm_band.Get_size(), MPIcomm_band.Get_rank()  # Get my local rank, and the total size of, the band-communicator I'm on.
    print(f"TOD: Hello from TOD-rank {MPIrank_tod}, dedicated to band {MPIcolor_band}, with local rank {MPIrank_band} (local communicator size: {MPIsize_band}).")
    
    master_band = MPIrank_band == 0  # Am I the master of my local band.

    scans_per_rank = math.ceil(params.num_scans/MPIsize_band)
    my_scans_start = scans_per_rank * MPIrank_band
    my_scans_stop = min(scans_per_rank * (MPIrank_band + 1), params.num_scans) # In case the number of scans is not divisible by the number of ranks
#    my_scans_start, my_scans_stop = scans_per_rank*MPIrank_band, scans_per_rank*(MPIrank_band + 1)
    print(f"TOD: Rank {MPIrank_tod} assigned scans {my_scans_start} - {my_scans_stop} on band{MPIcolor_band}.")

    # Initialization for all TOD processing tasks goes here
    experiment_data = read_data(MPIcolor_band, my_scans_start, my_scans_stop, params)

    # Chain #1
    # do TOD processing, resulting in maps_chain1
    # we start with a fake output of component separation, containing a completely empty sky
    compsep_output_black = get_empty_compsep_output(experiment_data, params)

    todproc_output_chain1 = tod2map(MPIcomm_band, experiment_data, compsep_output_black, params)

    compsep_output_chain1 = None
    compsep_output_chain2 = compsep_output_black
 
    for i in range(niter_gibbs):
        if master:  # If we are the master, tell the compsep-master not to stop.
            print(f"TOD: Master sending 'dont stop' signal to CompSep master.")
            MPI.COMM_WORLD.send(False, dest=compsep_master)

        if master_band:  # If we are the master of our respective band, send compsep-master our band-data.
            print(f"TOD: Rank {MPIrank_tod} sending chain1 data to CompSep master.")
            MPI.COMM_WORLD.send((todproc_output_chain1, i, 1), dest=compsep_master)
            # del todproc_output_chain1

        # Chain #2
        # do TOD processing, resulting in compsep_input at the same time, compsep is working on chain #1 data
        print(f"TOD: Rank {MPIrank_tod} starting chain 2, iter {i}.")
        t0 = time.time()
        todproc_output_chain2 = tod2map(MPIcomm_band, experiment_data, compsep_output_chain2, params)
        print(f"TOD: Rank {MPIrank_tod} finished chain 2, iter {i} in {time.time()-t0:.2f}s.")

        # get compsep results for chain #1
        if master_band:
            compsep_output_chain1 = MPI.COMM_WORLD.recv(source=compsep_master)
            print(f"TOD: Rank {MPIrank_tod} received chain1 data (iter {i}).")
        print(f"TOD: Rank {MPIrank_tod} starting chain 1, iter {i}.")
        t0 = time.time()
        compsep_output_chain1 = MPIcomm_band.bcast(compsep_output_chain1, root=0)
        print(f"TOD: Rank {MPIrank_tod} finished chain 1, iter {i} in {time.time()-t0:.2f}s.")

        if master:
            print(f"TOD: Master sending 'dont stop' signal to CompSep master.")
            MPI.COMM_WORLD.send(False, dest=compsep_master)  # we don't want to stop yet
        if master_band:
            print(f"TOD: Rank {MPIrank_tod} sending chain2 data to CompSep master.")
            MPI.COMM_WORLD.send((todproc_output_chain2, i, 2), dest=compsep_master)
            # del todproc_output_chain2

        # Chain #1
        # do TOD processing, resulting in compsep_input at the same time, compsep is working on chain #2 data
        todproc_output_chain1 = tod2map(MPIcomm_band, experiment_data, compsep_output_chain1, params)

        # get compsep results for chain #2
        if master_band:
            compsep_output_chain2 = MPI.COMM_WORLD.recv(source=compsep_master)
            print(f"TOD: Rank {MPIrank_tod} received chain2 data (iter {i}).")
        compsep_output_chain2 = MPIcomm_band.bcast(compsep_output_chain2, root=0)

    # stop compsep machinery
    if master:
        print("TOD: sending STOP signal to compsep")
        MPI.COMM_WORLD.send(True, dest=compsep_master)
