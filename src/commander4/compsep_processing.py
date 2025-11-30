import numpy as np
from mpi4py import MPI
from mpi4py.MPI import Comm
import logging
import time
from pixell.bunch import Bunch
from numpy.typing import NDArray

from commander4.data_models.detector_map import DetectorMap
from commander4.sky_models.component import DiffuseComponent
import commander4.sky_models.component as component_lib
from commander4.sky_models.sky_model import SkyModel
import commander4.output.plotting as plotting
from commander4.solvers.comp_sep_solvers import CompSepSolver, amplitude_sampling_per_pix


def init_compsep_processing(mpi_info: Bunch, params: Bunch) -> tuple[list[DiffuseComponent], str, dict[str, int], Bunch]:
    """To be run once before starting component separation processing.

    Determines whether the process is compsep master, and the number of bands.

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        params (Bunch): The parameters from the input parameter file.

    Output:
        mpi_info (Bunch): The data structure containing all MPI relevant data,
            modified to contain also the band masters dictionary.
        band_identifier (str): Unique string identifier for the experiment+band this process is responsible for.
        my_band (Bunch): The section of the parameters describing the band this rank is responsible for.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"CompSep: Hello from CompSep-rank {mpi_info.compsep.rank} (on machine {mpi_info.processor_name}), dedicated to band {mpi_info.compsep.rank}.")

    ### Creating list of all components ###
    components = []
    for component_str in params.components:
        component = params.components[component_str]
        if component.enabled:
            # getattr loads the class specified by "component_class" from the model.component file.
            # This class is then instantiated with the "params" specified, and appended to the components list.
            if component.params.lmax == "full":
                component.params.lmax = (params.nside*5)//2
            components.append(getattr(component_lib, component.component_class)(component.params))

    ### Setting up info for each band, including where to get the data from (map from file, or receive from TOD processing) ###
    current_band_idx = 0
    for band_str in params.CompSep_bands:
        if params.CompSep_bands[band_str].enabled:
            if current_band_idx == mpi_info.compsep.rank:  # Each rank is responsible for one band, for simplicity the band matching the index of their rank.
                my_band = params.CompSep_bands[band_str]
                if my_band.get_from != "file":
                    band_identifier = f"{my_band.get_from}$$${band_str}"
                else:
                    band_identifier = band_str
            current_band_idx += 1

    data_world = (band_identifier, mpi_info.world.rank)
    data_compsep = (band_identifier, mpi_info.compsep.rank)
    all_data_world = mpi_info.compsep.comm.allgather(data_world)
    all_data_compsep = mpi_info.compsep.comm.allgather(data_compsep)
    world_band_masters_dict = {item[0]: item[1] for item in all_data_world if item is not None}
    compsep_band_masters_dict = {item[0]: item[1] for item in all_data_compsep if item is not None}
    mpi_info.world.compsep_band_masters = world_band_masters_dict
    mpi_info.compsep.compsep_band_masters = compsep_band_masters_dict

    return components, mpi_info, band_identifier, my_band


def process_compsep(mpi_info: Bunch, detector_data: DetectorMap, iter: int, chain: int,
                    params: Bunch, comp_list: list[DiffuseComponent]) -> NDArray[np.float64]:
    """ Performs a single component separation iteration.
        Called by each compsep process, which are each responsible for a single band.
    
    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        detector_data (DetectorMap): The correlated noise cleaned detector map for this MPI ranks band.
        iter (int): The current Gibbs iteration (used only for plotting and seeding)
        chain (int): The current chain (used only for plotting and seeding).
        params (Bunch): The parameters from the input parameter file.

    Output:
       detector_maps (np.array): The band-integrated total sky.
        
    """
    logger = logging.getLogger(__name__)

    compsep_comm = mpi_info.compsep.comm
    compsep_rank = mpi_info.compsep.rank
    # if params.make_plots:
        # detector_to_plot = proc_comm.Get_rank()
        # logging.info(f"Rank {proc_comm.Get_rank()} chain {chain} iter {iter} starting plotting.")
        # plotting.plot_data_maps(params, detector_to_plot, chain, iter, map_signal=signal_map,
        #                         map_corr_noise=detector_data.map_corr_noise,
        #                         map_rms=detector_data.map_rms,
        #                         map_skysub=detector_data.skysub_map,
        #                         map_orbdip=detector_data.orbdipole_map)

    if params.pixel_compsep_sampling:
        comp_list = amplitude_sampling_per_pix(compsep_comm, detector_data, comp_list, params)
    else:
        ### TOTAL INTENSITY CALCULATIONS ###
        color = 0 if detector_data.map_sky[0] is not None else MPI.UNDEFINED
        comm_local = compsep_comm.Split(color, key=compsep_rank)
        if color == 0:
            compsep_solver = CompSepSolver(comp_list, detector_data.map_sky[0].reshape((1,-1)),
                                           detector_data.map_rms[0].reshape((1,-1)),
                                           detector_data.nu, detector_data.fwhm, params, comm_local,
                                           pol=False)
            comp_list = compsep_solver.solve()
            if params.make_plots and compsep_rank:
                plotting.plot_cg_res(params, chain, iter, compsep_solver.CG_residuals)

        color = 0 if detector_data.map_sky[1] is not None else MPI.UNDEFINED
        comm_local = compsep_comm.Split(color, key=compsep_rank)
        if color == 0:
            compsep_solver = CompSepSolver(comp_list, np.array(detector_data.map_sky[1:]),
                                           np.array(detector_data.map_rms[1:]), detector_data.nu,
                                           detector_data.fwhm, params, comm_local, pol=True)
            comp_list = compsep_solver.solve()
            if params.make_plots and compsep_rank:
                plotting.plot_cg_res(params, chain, iter, compsep_solver.CG_residuals)

    comp_list = compsep_comm.bcast(comp_list, root=0)  #TODO: This needs to be handled differently.
    sky_model = SkyModel(comp_list)

    sky_model_at_band = sky_model.get_sky_at_nu(detector_data.nu, detector_data.nside,
                                                fwhm=np.deg2rad(detector_data.fwhm/60.0))
    pol_names = ["I", "Q", "U"]
    for ipol in range(3):
        if detector_data.map_sky[ipol] is not None:
            chi2 = np.mean(np.abs(detector_data.map_sky[ipol] - sky_model_at_band[ipol])/detector_data.map_rms[ipol])
            logger.info(f"Reduced chi2 for pol={pol_names[ipol]} ({detector_data.nu}GHz): {chi2:.3f}")

    if params.make_plots:
        t0 = time.time()
        detector_to_plot = compsep_rank
        plotting.plot_combo_maps(params, detector_to_plot, chain, iter, comp_list, detector_data)
        plotting.plot_components(params, detector_to_plot, chain, iter, comp_list, detector_data)
        logging.info(f"Rank {compsep_rank} chain {chain} iter {iter} Finished all plotting in {time.time()-t0:.1f}s.")

    return sky_model  # Return the full sky realization for my band.
