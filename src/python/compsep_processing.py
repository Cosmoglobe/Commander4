import numpy as np
from mpi4py import MPI
from mpi4py.MPI import Comm
import logging
from pixell.bunch import Bunch
from numpy.typing import NDArray

from src.python.data_models.detector_map import DetectorMap
from src.python.sky_models.component import DiffuseComponent
import src.python.sky_models.component as component_lib
from src.python.sky_models.sky_model import SkyModel
import src.python.output.plotting as plotting
from src.python.solvers.comp_sep_solvers import CompSepSolver, amplitude_sampling_per_pix


def init_compsep_processing(CompSep_comm: Comm, params: Bunch) -> tuple[list[DiffuseComponent], str, dict[str, int], Bunch]:
    """To be run once before starting component separation processing.

    Determines whether the process is compsep master, and the number of bands.

    Input:
        CompSep_comm (MPI.Comm): Communicator for the CompSep processes.
        params (Bunch): The parameters from the input parameter file.

    Output:
        components (list[Component]): List of Component type objects as specified by the parameters.)
        band_identifier (str): Unique string identifier for the experiment+band this process is responsible for.
        CompSep_band_masters_dict (dict[str->int]): Dictionary mapping band identifiers to the global rank of the process responsible for that band.
        my_band (Bunch): The section of the parameters describing the band this rank is responsible for.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"CompSep: Hello from CompSep-rank {CompSep_comm.rank} (on machine {MPI.Get_processor_name()}), dedicated to band {CompSep_comm.rank}.")

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
            if current_band_idx == CompSep_comm.Get_rank():  # Each rank is responsible for one band, for simplicity the band matching the index of their rank.
                my_band = params.CompSep_bands[band_str]
                if my_band.get_from != "file":
                    band_identifier = f"{my_band.get_from}$$${band_str}"
                else:
                    band_identifier = band_str
            current_band_idx += 1

    data = (band_identifier, MPI.COMM_WORLD.Get_rank())
    all_data = CompSep_comm.allgather(data)
    CompSep_band_masters_dict = {item[0]: item[1] for item in all_data if item is not None}

    return components, band_identifier, CompSep_band_masters_dict, my_band


def process_compsep(detector_data: DetectorMap, iter: int, chain: int, params: Bunch,
                    proc_comm: Comm, comp_list: list[DiffuseComponent]) -> NDArray[np.float64]:
    """ Performs a single component separation iteration.
        Called by each compsep process, which are each responsible for a single band.
    
    Input:
        detector_data (DetectorMap): The correlated noise cleaned detector map for this MPI ranks band.
        iter (int): The current Gibbs iteration (used only for plotting and seeding)
        chain (int): The current chain (used only for plotting and seeding).
        params (Bunch): The parameters from the input parameter file.
        proc_master (bool): Whether this is the master compsep process.
        proc_comm (MPI.Comm): Communicator for the compsep processes.

    Output:
       detector_maps (np.array): The band-integrated total sky.
        
    """
    signal_map = detector_data.map_sky
    band_freq = detector_data.nu
    is_CompSep_master = proc_comm.Get_rank() == 0
    if params.make_plots:
        detector_to_plot = proc_comm.Get_rank()
        logging.info(f"Rank {proc_comm.Get_rank()} chain {chain} iter {iter} starting plotting.")
        plotting.plot_data_maps(is_CompSep_master, params, detector_to_plot, chain, iter, map_signal=signal_map,
                                map_corr_noise=detector_data.map_corr_noise,
                                map_rms=detector_data.map_rms,
                                map_skysub=detector_data.skysub_map,
                                map_orbdip=detector_data.orbdipole_map)

    if params.pixel_compsep_sampling:
        comp_list = amplitude_sampling_per_pix(proc_comm, detector_data, comp_list, params)
    else:
        compsep_solver = CompSepSolver(comp_list, signal_map, detector_data.map_rms, band_freq, detector_data.fwhm, params, proc_comm)
        comp_list = compsep_solver.solve()
        if params.make_plots and is_CompSep_master:
            plotting.plot_cg_res(params, chain, iter, compsep_solver.CG_residuals)

    sky_model = SkyModel(comp_list)

    detector_maps = sky_model.get_sky_at_nu(band_freq, detector_data.nside,
                                            fwhm=detector_data.fwhm/60.0*np.pi/180.0)

    if params.make_plots:
        detector_to_plot = proc_comm.Get_rank()
        plotting.plot_combo_maps(params, detector_to_plot, chain, iter, comp_list, detector_data)
        # plotting.plot_components(params, band_freq, detector_to_plot, chain, iter, signal_map, comp_list, detector_data.nside)
        logging.info(f"Rank {proc_comm.Get_rank()} chain {chain} iter {iter} Finished all plotting.")

    return detector_maps  # Return the full sky realization for my band.
