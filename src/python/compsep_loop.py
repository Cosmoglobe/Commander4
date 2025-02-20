import numpy as np
from mpi4py import MPI
import time
import logging
import h5py
import healpy as hp
from model.component import CMB, ThermalDust, Synchrotron, DiffuseComponent
from model.sky_model import SkyModel
from output import log, plotting
from solvers.comp_sep_solvers import CompSepSolver, amplitude_sampling_per_pix


# Component separation loop
def compsep_loop(comm, tod_master: int, cmb_master: int, params: dict, use_MPI_for_CMB=True):
    logger = logging.getLogger(__name__)

    # am I the master of the compsep communicator?
    master = comm.Get_rank() == 0

    num_bands = len(params.bands)

    if master:
        logger.info("Compsep: loop started")
    # we wait for new jobs until we get a stop signal
    while True:
        logger.info("CompSep new loop iteration...")
        # check for simulation end
        stop = MPI.COMM_WORLD.recv(source=tod_master) if master else False
        stop = comm.bcast(stop, root=0)
        if stop:
            if master:
                logger.warning("Compsep: stop requested; exiting")
                if not cmb_master is None:
                    logger.warning("Compsep: Sending stop signal to CMB")
                    MPI.COMM_WORLD.send(True, dest=cmb_master)
            return
        if master:
            logger.info("Compsep: Not asked to stop, obtaining new job...")

        # get next data set for component separation
        data, iter, chain = [], [], []
        if master:
            for i in range(num_bands):
                _data, _iter, _chain = MPI.COMM_WORLD.recv(source=tod_master+i)
                logger.info(f"Compsep: Received data from rank {tod_master+i} for chain {_chain} iteration {_iter}.")
                data.append(_data)
                iter.append(_iter)
                chain.append(_chain)
            data = np.array(data)
            log.logassert(np.all([i == iter[0] for i in iter]), "Different CompSep tasks received different Gibbs iteration number from TOD loop!", logger)
            log.logassert(np.all([i == chain[0] for i in chain]), "Different CompSep tasks received different Gibbs chain number from TOD loop!", logger)
            chain = chain[0]
            iter = iter[0]
        # Broadcast te data to all tasks, or do anything else that's appropriate
        data = comm.bcast(data, root=0)
        iter = comm.bcast(iter, root=0)
        chain = comm.bcast(chain, root=0)

        if master:
            logger.info(f"Compsep: data obtained for chain {chain}, iteration {iter}. Working on it ...")
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
                plotting.plot_data_maps(master, params, i_det, chain, iter,
                                        map_signal=signal_maps[-1],
                                        map_corr_noise=detector.map_corr_noise,
                                        map_rms=rms_maps[-1])
        signal_maps = np.array(signal_maps)
        rms_maps = np.array(rms_maps)
        band_freqs = np.array(band_freqs)
        if params.pixel_compsep_sampling:
            comp_maps = amplitude_sampling_per_pix(signal_maps, rms_maps, band_freqs)
        else:
            compsep_solver = CompSepSolver(signal_maps, rms_maps, band_freqs, params)
            comp_maps = compsep_solver.solve(seed=9999*chain+11*iter)
            if params.make_plots:
                plotting.plot_cg_res(master, params, chain, iter,
                                     compsep_solver.CG_residuals)

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
                plotting.plot_components(master, params, band_freqs[i_det],
                                         i_det, chain, iter, sky=detector_map,
                                         cmb=cmb_sky, dust=dust_sky,
                                         sync=sync_sky,
                                         signal=signal_maps[i_det])

        foreground_maps = np.array(foreground_maps)
        foreground_subtracted_maps = signal_maps - foreground_maps

        if master:
            logger.info(f"Compsep: Finished chain {chain}, iteration {iter} in {time.time()-t0:.2f}s.")
            for i in range(len(detector_maps)):
                logger.info(f"CompSep: Sending back data for band {i} (chain {chain} iter {iter}).")
                MPI.COMM_WORLD.send(detector_maps[i], dest=tod_master+i)
            logger.info("Compsep: results sent back")

            if not cmb_master is None:  # cmb_master is set to None of we aren't doing CMB realizations.
                logger.info(f"CompSep: Sending relevant data to CMB realiztaion master...")
                MPI.COMM_WORLD.send(False, dest=cmb_master)  # we don't want to stop yet
                # Sending maps to CMB loop. Not sending the last band, as it's very dust-contaminated
                if use_MPI_for_CMB:
                    for i in range(num_bands-1):
                        MPI.COMM_WORLD.send([[foreground_subtracted_maps[i], rms_maps[i]], iter, chain], dest=cmb_master+i)
                else:
                    MPI.COMM_WORLD.send([[foreground_subtracted_maps[:4], rms_maps[:4]], iter, chain], dest=cmb_master)
                logger.info("Compsep: Sent results to CMB loop.")
