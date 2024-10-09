import numpy as np
from mpi4py import MPI
import time
import h5py
import healpy as hp
from model.component import CMB, ThermalDust, Synchrotron
from model.sky_model import SkyModel
import matplotlib.pyplot as plt
import os
import utils.math_operations as math_op

def S_inv_prior(lmax, beta) -> np.array:
    q = np.arange(lmax)**beta
    return np.diagflat(q)


def P_operator(x: np.array, comp: DiffuseComponent, M: np.array) -> np.array:
    # x - array of alm for a given component
    # comp - class for a given component model 
    # M - mixing matrix for a given component

    lmax = 3*2048-1 # should be in param file
    fwhm = 20 # [arcmin] should be in param file

    # mixing matrix application in map space
    mp = math_op.alm_to_map(x, nside=comp.nside_comp_map) # Y a_lm
    mp = M * mp # M Y a_lm
    x = math_op.map_to_alm(mp, lmax=3*2048-1) # Y^T M Y a_lm

    # beam 
    x = math_op.spherical_beam_applied_to_alm(x, fwhm)
    mp = math_op.alm_to_map(x, nside=comp.nside_comp_map)
    return mp


def A_operator(x: np.array, comp_list: list, M: np.array) -> np.array:
    # x - spherical harmonics alm
    #
    # A = (S^-1 + P^T N^-1 P)
    # P = Y B Y^T M Y

    ncomp, _ = x.shape
    for i in range(ncomp):
        mp = P_operator(x[i], comp_list[i], M[:,i])
        # mp has different size for different component, so N^-1 should too



def alm_comp_sampling_CG(map_sky: np.array, map_rms: np.array, freqs: np.array) -> np.array:
    # preparation
    ncomp = 3 # should be in parameter file
    lmax = 3*2048-1
    alm_size = hp.map2alm(np.zeros(hp.nside2npix(2048)), lmax=lmax).size
    nband, npix = map_sky.shape

    # create mixing matrix
    M = np.zeros((nband, ncomp))
    cmb = CMB()
    dust = ThermalDust()
    sync = Synchrotron()
    components = [cmb, dust, sync]
    comp_alm = np.zeros((ncomp, alm_size))
    for i in range(ncomp):
        M[:,i] = components[i].get_sed(freqs)
    
    A_operator(comp_alm, components, M)



def amplitude_sampling_per_pix(map_sky: np.array, map_rms: np.array, freqs: np.array) -> np.array:
    ncomp = 3
    nband, npix = map_sky.shape
    comp_maps = np.zeros((ncomp, npix))
    M = np.empty((nband, ncomp))
    M[:,0] = CMB().get_sed(freqs)
    M[:,1] = ThermalDust().get_sed(freqs)
    M[:,2] = Synchrotron().get_sed(freqs)
    for i in range(npix):
        xmap = 1/map_rms[:,i]
        x = M.T.dot((xmap**2*map_sky[:,i]))
        x += M.T.dot(np.random.randn(nband)*xmap)
        A = (M.T.dot(np.diag(xmap**2)).dot(M))
        try:
            comp_maps[:,i] = np.linalg.solve(A, x)
        except np.linalg.LinAlgError:
            comp_maps[:,i] = 0
    return comp_maps


# Component separation loop
def compsep_loop(comm, tod_master: int, cmb_master: int, params: dict, use_MPI_for_CMB=True):

    # am I the master of the compsep communicator?
    master = comm.Get_rank() == 0

    num_bands = len(params.bands)

    if master:
        print("Compsep: loop started")
        if not os.path.isdir(params.output_paths.plots + "maps_comps/"):
            os.mkdir(params.output_paths.plots + "maps_comps/")
        if not os.path.isdir(params.output_paths.plots + "maps_sky/"):
            os.mkdir(params.output_paths.plots + "maps_sky/")


    # we wait for new jobs until we get a stop signal
    while True:
        print("CompSep new loop iteration...")
        # check for simulation end
        stop = MPI.COMM_WORLD.recv(source=tod_master) if master else False
        stop = comm.bcast(stop, root=0)
        if stop:
            if master:
                print("Compsep: stop requested; exiting")
                if not cmb_master is None:
                    print("Compsep: Sending stop signal to CMB")
                    MPI.COMM_WORLD.send(True, dest=cmb_master)
            return
        if master:
            print("Compsep: Not asked to stop, obtaining new job...")

        # get next data set for component separation
        data, iter, chain = [], [], []
        if master:
            for i in range(num_bands):
                _data, _iter, _chain = MPI.COMM_WORLD.recv(source=tod_master+i)
                print(f"Compsep: Received data from rank {tod_master+i} for chain {_chain} iteration {_iter}.")
                data.append(_data)
                iter.append(_iter)
                chain.append(_chain)
            data = np.array(data)
            assert np.all([i == iter[0] for i in iter]), "Different CompSep tasks received different Gibbs iteration number from TOD loop!"
            assert np.all([i == chain[0] for i in chain]), "Different CompSep tasks received different Gibbs chain number from TOD loop!"
            chain = chain[0]
            iter = iter[0]
        # Broadcast te data to all tasks, or do anything else that's appropriate
        data = comm.bcast(data, root=0)
        iter = comm.bcast(iter, root=0)
        chain = comm.bcast(chain, root=0)

        if master:
            print(f"Compsep: data obtained for chain {chain}, iteration {iter}. Working on it ...")

        signal_maps = []
        rms_maps = []
        band_freqs = []
        for i_det in range(len(data)):
            detector = data[i_det]
            signal_maps.append(detector.map_sky)
            rms_maps.append(detector.map_rms)
            band_freqs.append(detector.nu)
            hp.mollview(signal_maps[-1], cmap="RdBu_r", title=f"Signal map, det {i_det}, chain {chain}, iter {iter}")
            plt.savefig(params.output_paths.plots + f"maps_sky/map_sky_det{i_det}_chain{chain}_iter{iter}.png")
            plt.close()
            hp.mollview(rms_maps[-1], title=f"RMS map, det {i_det}, chain {chain}, iter {iter}")
            plt.savefig(params.output_paths.plots + f"maps_sky/map_rms_det{i_det}_chain{chain}_iter{iter}.png")
            plt.close()


        signal_maps = np.array(signal_maps)
        rms_maps = np.array(rms_maps)
        band_freqs = np.array(band_freqs)
        comp_maps = amplitude_sampling_per_pix(signal_maps, rms_maps, band_freqs)

        component_types = [CMB, ThermalDust, Synchrotron]
        component_list = []
        for i, component_type in enumerate(component_types):
            component = component_type()
            component.component_map = comp_maps[i]
            component_list.append(component)

        sky_model = SkyModel(component_list)

        npix = signal_maps.shape[-1]
        detector_maps = []
        foreground_maps = []
        for i_det in range(len(data)):
            detector_map = sky_model.get_sky_at_nu(band_freqs[i_det], 12*64**2)
            detector_maps.append(detector_map)
            cmb_sky = component_list[0].get_sky(band_freqs[i_det])
            dust_sky = component_list[1].get_sky(band_freqs[i_det])
            sync_sky = component_list[2].get_sky(band_freqs[i_det])
            foreground_maps.append(sky_model.get_foreground_sky_at_nu(band_freqs[i_det], npix))

            hp.mollview(detector_map, title=f"Full sky realization at {band_freqs[i_det]:.2f}GHz")
            plt.savefig(params.output_paths.plots + f"maps_comps/sky_realization_det{i_det}_chain{chain}_iter{iter}.png")
            plt.close()

            hp.mollview(cmb_sky, title=f"CMB realization at {band_freqs[i_det]:.2f}GHz, det {i_det}, chain {chain}, iter {iter}")
            plt.savefig(params.output_paths.plots + f"maps_comps/CMB_realization_det{i_det}_chain{chain}_iter{iter}.png")
            plt.close()

            hp.mollview(dust_sky, title=f"Thermal dust realization at {band_freqs[i_det]:.2f}GHz, det {i_det}, chain {chain}, iter {iter}")
            plt.savefig(params.output_paths.plots + f"maps_comps/Dust_realization_det{i_det}_chain{chain}_iter{iter}.png")
            plt.close()
           
            hp.mollview(sync_sky, title=f"Synchrotron realization at {band_freqs[i_det]:.2f}GHz, det {i_det}, chain {chain}, iter {iter}")
            plt.savefig(params.output_paths.plots + f"maps_comps/Sync_realization_det{i_det}_chain{chain}_iter{iter}.png")
            plt.close()

            hp.mollview(signal_maps[i_det]-dust_sky-sync_sky, title=f"Foreground subtracted sky at {band_freqs[i_det]:.2f}GHz, det {i_det}, chain {chain}, iter {iter}")
            plt.savefig(params.output_paths.plots + f"maps_comps/foreground_subtr_det{i_det}_chain{chain}_iter{iter}.png")
            plt.close()

        foreground_maps = np.array(foreground_maps)
        foreground_subtracted_maps = signal_maps - foreground_maps

        if master:
            for i in range(len(detector_maps)):
                print(f"CompSep: Sending back data for band {i} (chain {chain} iter {iter}).")
                MPI.COMM_WORLD.send(detector_maps[i], dest=tod_master+i)
            print("Compsep: results sent back")

            if not cmb_master is None:  # cmb_master is set to None of we aren't doing CMB realizations.
                print(f"CompSep: Sending relevant data to CMB realiztaion master...")
                MPI.COMM_WORLD.send(False, dest=cmb_master)  # we don't want to stop yet
                # Sending maps to CMB loop. Not sending the last band, as it's very dust-contaminated
                if use_MPI_for_CMB:
                    for i in range(num_bands-1):
                        MPI.COMM_WORLD.send([[foreground_subtracted_maps[i], rms_maps[i]], iter, chain], dest=cmb_master+i)
                else:
                    MPI.COMM_WORLD.send([[foreground_subtracted_maps[:4], rms_maps[:4]], iter, chain], dest=cmb_master)
                print("Compsep: Sent results to CMB loop.")
