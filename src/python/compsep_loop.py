import numpy as np
from mpi4py import MPI
import time
import h5py
import healpy as hp
from model.component import CMB, ThermalDust, Synchrotron, DiffuseComponent
from model.sky_model import SkyModel
import matplotlib.pyplot as plt
import os
from solvers.comp_sep_solvers import CompSepSolver, amplitude_sampling_per_pix


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
        if not os.path.isdir(params.output_paths.plots + "CG_res/"):
            os.mkdir(params.output_paths.plots + "CG_res/")

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
                hp.mollview(detector.map_corr_noise, cmap="RdBu_r", title=f"Corr noise map, det {i_det}, chain {chain}, iter {iter}")
                plt.savefig(params.output_paths.plots + f"maps_sky/map_corr_noise_det{i_det}_chain{chain}_iter{iter}.png")
                plt.close()
                hp.mollview(rms_maps[-1], title=f"RMS map, det {i_det}, chain {chain}, iter {iter}")
                plt.savefig(params.output_paths.plots + f"maps_sky/map_rms_det{i_det}_chain{chain}_iter{iter}.png")
                plt.close()


        signal_maps = np.array(signal_maps)
        rms_maps = np.array(rms_maps)
        band_freqs = np.array(band_freqs)
        if params.pixel_compsep_sampling:
            comp_maps = amplitude_sampling_per_pix(signal_maps, rms_maps, band_freqs)
        else:
            compsep_solver = CompSepSolver(signal_maps, rms_maps, band_freqs, params.fwhm, params.CG_max_iter, params.CG_err_tol)
            comp_maps = compsep_solver.solve()
            if params.make_plots:
                plt.figure()
                plt.loglog(np.arange(compsep_solver.CG_residuals.shape[0]), compsep_solver.CG_residuals)
                plt.axhline(params.CG_err_tol, ls="--", c="k")
                plt.savefig(params.output_paths.plots + f"CG_res/CG_res_chain{chain}_iter{iter}.png")
                plt.close()

        component_types = [CMB, ThermalDust, Synchrotron]  # At the moment we always sample all components. #TODO: Move to parameter file.
        component_list = []
        for i, component_type in enumerate(component_types):
            component = component_type()
            component.component_map = comp_maps[i]
            # component.component_map = math_op.alm_to_map(comp_alm[i], nside=component.nside_comp_map, lmax=lmax)
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
