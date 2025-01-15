import numpy as np
from mpi4py import MPI
import time
import h5py
import healpy as hp
from model.component import CMB, ThermalDust, Synchrotron, DiffuseComponent
from model.sky_model import SkyModel
import matplotlib.pyplot as plt
import os
import utils.math_operations as math_op
from utils.conjugate_gradient import conjugate_gradient_alm as CG_driver

def S_inv_prior(x: np.array, comp: DiffuseComponent) -> np.array:
    # x - array of alm for a given component
    # comp - class for a given component model 
    #
    # need to figure out l^beta for each components and add it to component class

    lmax = 3*2048-1 # should be in param file
    q = np.arange(lmax, dtype=float)**comp.prior_l_power_law
    return hp.almxfl(x, q)


def P_operator(x: np.array, comp: DiffuseComponent, M: np.array) -> np.array:
    # x - [nalm] array of alm for a given component
    # comp - class for a given component model 
    # M - [nband] mixing matrix for a given component

    lmax = 3*2048-1 # should be in param file
    fwhm = 20 # [arcmin] should be in param file
    nalm = x.size
    nband = len(M)
    npix = hp.nside2npix(comp.nside_comp_map)

    # mixing matrix application in map space
    mp = math_op.alm_to_map(x, nside=comp.nside_comp_map, lmax=lmax) # Y a_lm
    mp = np.array(M.size*[mp])
    mp_new = M[:,None] * mp # M Y a_lm
    # transpose Y^T is represented by the adjoint operator
    x = np.zeros((nband, nalm), dtype=complex)
    for i in range(nband):
        x[i] = math_op.alm_to_map_adjoint(mp_new[i], nside=comp.nside_comp_map, lmax=lmax) # Y^T M Y a_lm

    # beam 
    x = math_op.spherical_beam_applied_to_alm(x, fwhm)
    mp = np.zeros((nband, npix))
    for i in range(nband):
        mp[i] = math_op.alm_to_map(x[i], nside=comp.nside_comp_map, lmax=lmax)
    return mp


def P_operator_transpose(x: np.array, comp: DiffuseComponent, M_T: np.array, nalm: int) -> np.array:
    # x - [nband, npix] array of maps for a given component
    # comp - class for a given component model 
    # M_T - [ncomp, nband] transpose of a mixing matrix
    # nalm - size of alm array

    lmax = 3*2048-1 # should be in param file
    fwhm = 20 # [arcmin] should be in param file
    ncomp, nband = M_T.shape
    _, npix = x.shape

    # beam B^T Y^T m
    alm = np.zeros((nband, nalm), dtype=complex)
    for i in range(nband):
        alm[i] = math_op.alm_to_map_adjoint(x[i], nside=comp.nside_comp_map, lmax=lmax)
    alm = math_op.spherical_beam_applied_to_alm(alm, fwhm) 

    # mixing matrix Y^T M^T Y
    mp = np.zeros((nband, npix))
    for i in range(nband):
        mp[i] = math_op.alm_to_map(alm[i], nside=comp.nside_comp_map, lmax=lmax)
    mp_new = M_T.dot(mp)
    alm = np.zeros((ncomp, nalm), dtype=complex)
    for i in range(ncomp):
        alm[i] = math_op.alm_to_map_adjoint(mp_new[i], nside=comp.nside_comp_map, lmax=lmax)

    return alm


def A_operator(x: np.array, comp_list: list, M: np.array, map_rms: np.array) -> np.array:
    # x - [ncomp, nalm] spherical harmonics alm
    # comp_list - [ncomp] list of DiffuseComponent class objects for relevant components
    # M - [nband, ncomp] array of mixing matrix
    # map_rms - [nband, npix] array of RMS maps
    #
    # A = (S^-1 + P^T N^-1 P)
    # P = Y B Y^T M Y

    ncomp, nalm = x.shape
    mp_list = [] # list because maps for different comp are of different nsize
    for i in range(ncomp):
        mp = P_operator(x[i], comp_list[i], M[:,i]) # P alm

        N = hp.ud_grade(map_rms, comp_list[i].nside_comp_map, power=2)
        mp /= N # N^-1 P a_lm

        mp_list += [mp]

    alm_new = np.zeros(x.shape, dtype=complex)
    for i in range(ncomp):
        alm = P_operator_transpose(mp_list[i], comp_list[i], M.T, nalm) # P^T N^-1 P alm
        alm_new += alm

        S_inv_alm = S_inv_prior(x[i], comp_list[i]) # S^-1 alm
        alm_new[i] += S_inv_alm


    return alm_new


def B_matrix(map_sky: np.array, map_rms: np.array, comp_list: list, M: np.array, nalm: int) -> np.array:
    # map_sky - [nband, npix] - observed sky maps in different bands
    # map_rms - [nband, npix] array of RMS maps
    # comp_list - [ncomp] list of DiffuseComponent class objects
    # M - [nband, ncomp] array of mixing matrix
    # nalm - the size of alm array
    #
    # B = P^T N^-1 m + P^T N^-0.5 eta_1 + S^-1 eta_2

    npix = map_sky.shape[1]
    ncomp = len(comp_list)

    # P^T N^-1 m
    mp = map_sky/map_rms**2 # N^-1 m
    fake_comp = DiffuseComponent()
    fake_comp.nside_comp_map = hp.npix2nside(npix)
    data_alm = P_operator_transpose(mp, fake_comp, M.T, nalm) # P^T N^-1 m

    # P^T N^-0.5 eta_1
    eta_1 = np.random.normal(size=map_sky.shape)
    eta_1 /= map_rms
    alm_eta_1 = P_operator_transpose(eta_1, fake_comp, M.T, nalm)

    # S^-1 eta_2
    eta_2 = np.random.normal(size=(ncomp, nalm))
    S_inv_eta_2 = np.zeros(eta_2.shape)
    for i in range(ncomp):
        S_inv_eta_2[i] = S_inv_prior(eta_2[i], comp_list[i])

    return data_alm + alm_eta_1 + S_inv_eta_2
    

def alm_comp_sampling_CG(map_sky: np.array, map_rms: np.array, freqs: np.array) -> np.array:
    # preparation
    ncomp = 3 # should be in parameter file
    lmax = 3*2048-1 # should be in parameter file
    alm_size = math_op.alm_to_map_adjoint(np.zeros(hp.nside2npix(2048)), nside=2048, lmax=lmax).size
    nband, npix = map_sky.shape

    # create mixing matrix
    M = np.zeros((nband, ncomp))
    cmb = CMB()
    dust = ThermalDust()
    sync = Synchrotron()
    components = [cmb, dust, sync]
    comp_alm = np.zeros((ncomp, alm_size), dtype=complex)
    for i in range(ncomp):
        M[:,i] = components[i].get_sed(freqs)
       
    B = B_matrix(map_sky, map_rms, components, M, alm_size)

    # initiliazing CG solver with 0 starting guess and preconditioner M = 1
    x0 = np.zeros(comp_alm.shape, dtype=complex)
    M_inv = np.ones(comp_alm.shape)
    A_op = lambda x: A_operator(x, components, M, map_rms) # redefining A_operator so that there is just one argument for CG solver
    x, _ = CG_driver(A_op, B, x0, M_inv, lmax)

    return x


def amplitude_sampling_per_pix(map_sky: np.array, map_rms: np.array, freqs: np.array) -> np.array:
    ncomp = 3
    nband, npix = map_sky.shape
    comp_maps = np.zeros((ncomp, npix))
    M = np.empty((nband, ncomp))
    M[:,0] = CMB().get_sed(freqs)
    M[:,1] = ThermalDust().get_sed(freqs)
    M[:,2] = Synchrotron().get_sed(freqs)
    from time import time
    t0 = time()
    rand = np.random.randn(npix,nband)
    print(f"time for random numbers: {time()-t0}s.")
    t0 = time()
    for i in range(npix):
        xmap = 1/map_rms[:,i]
        x = M.T.dot((xmap**2*map_sky[:,i]))
        x += M.T.dot(rand[i]*xmap)
        A = (M.T.dot(np.diag(xmap**2)).dot(M))
        try:
            comp_maps[:,i] = np.linalg.solve(A, x)
        except np.linalg.LinAlgError:
            comp_maps[:,i] = 0
    print(f"Time for Python solution: {time()-t0}s.")
    import cmdr4_support
    t0 = time()
    comp_maps2 = cmdr4_support.utils.amplitude_sampling_per_pix_helper(map_sky, map_rms, M, rand, nthreads=1)
    print(f"Time for native solution: {time()-t0}s.")
    import ducc0
    print(f"L2 error between solutions: {ducc0.misc.l2error(comp_maps, comp_maps2)}.")
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
            t0 = time.time()

        signal_maps = []
        rms_maps = []
        band_freqs = []
        for i_det in range(len(data)):
            detector = data[i_det]
            signal_maps.append(detector.map_sky)
            rms_maps.append(detector.map_rms)
            band_freqs.append(detector.nu)
            if params.make_plots:
                hp.mollview(signal_maps[-1], cmap="RdBu_r", title=f"Signal map, det {i_det}, chain {chain}, iter {iter}")
                plt.savefig(params.output_paths.plots + f"maps_sky/map_sky_det{i_det}_chain{chain}_iter{iter}.png")
                plt.close()
                hp.mollview(rms_maps[-1], title=f"RMS map, det {i_det}, chain {chain}, iter {iter}")
                plt.savefig(params.output_paths.plots + f"maps_sky/map_rms_det{i_det}_chain{chain}_iter{iter}.png")
                plt.close()


        signal_maps = np.array(signal_maps)
        rms_maps = np.array(rms_maps)
        band_freqs = np.array(band_freqs)
        #comp_maps = amplitude_sampling_per_pix(signal_maps, rms_maps, band_freqs)
        print(signal_maps.shape)

        comp_alm = alm_comp_sampling_CG(signal_maps, rms_maps, band_freqs)
        lmax = 3*2048-1 # should be in param file 
        print('alm_comp_sampling_CG done')

        component_types = [CMB, ThermalDust, Synchrotron]
        component_list = []
        for i, component_type in enumerate(component_types):
            component = component_type()
            #component.component_map = comp_maps[i]
            component.component_map = math_op.alm_to_map(comp_alm[i], nside=component.nside_comp_map, lmax=lmax)
            component_list.append(component)

        sky_model = SkyModel(component_list)

        npix = signal_maps.shape[-1]
        detector_maps = []
        foreground_maps = []
        for i_det in range(len(data)):
            detector_map = sky_model.get_sky_at_nu(band_freqs[i_det], 12*params.nside**2)
            detector_maps.append(detector_map)
            cmb_sky = component_list[0].get_sky(band_freqs[i_det])
            dust_sky = component_list[1].get_sky(band_freqs[i_det])
            sync_sky = component_list[2].get_sky(band_freqs[i_det])
            foreground_maps.append(sky_model.get_foreground_sky_at_nu(band_freqs[i_det], npix))

            if params.make_plots:
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
            print(f"Compsep: Finished chain {chain}, iteration {iter} in {time.time()-t0:.2f}s.")
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
