import numpy as np
from mpi4py import MPI
from mpi4py.MPI import Comm
import logging
from pixell import bunch

from data_models import DetectorMap
from model.component import CMB, ThermalDust, Synchrotron, DiffuseComponent, Component
import model.component
from model.sky_model import SkyModel
from output import log, plotting
from solvers.comp_sep_solvers import CompSepSolver, amplitude_sampling_per_pix


def init_compsep_processing(proc_comm: Comm, params: bunch):
    """To be run once before starting component separation processing.

    Determines whether the process is compsep master, and the number of bands.

    Input:
        proc_comm (MPI.Comm): Communicator for the compsep processes.
        params (bunch): The parameters from the input parameter file.

    Output:
        proc_master (bool): Whether this process is the master of the compsep process.
        proc_comm (MPI.Comm): The same as the input communicator (just returned for clarity).
        num_bands (int): The number of data bands.
    """
    logger = logging.getLogger(__name__)
    proc_master = proc_comm.Get_rank() == 0
    logger.info(f"CompSep: Hello from TOD-rank {proc_comm.rank} (on machine {MPI.Get_processor_name()}), dedicated to band {proc_comm.rank}.")
    num_bands = len(params.bands)

    components = []
    for component_str in params.components:
        component = params.components[component_str]
        if component.enabled:
            # getattr loads the class specified by "component_class" from the model.component file.
            # This class is then instantiated with the "params" specified, and appended to the components list.
            components.append(getattr(model.component, component.component_class)(component.params))

    return proc_master, proc_comm, num_bands, components


def process_compsep(detector_data: DetectorMap, iter: int, chain: int,
                    params: bunch, proc_master: bool, proc_comm: Comm, comp_list: list[Component]):
    """ Performs a single component separation iteration.
        Called by each compsep process, which are each responsible for a single band.
    
    Input:
        detector_data (DetectorMap): The correlated noise cleaned detector map for this MPI ranks band.
        iter (int): The current Gibbs iteration (used only for plotting and seeding)
        chain (int): The current chain (used only for plotting and seeding).
        params (bunch): The parameters from the input parameter file.
        proc_master (bool): Whether this is the master compsep process.
        proc_comm (MPI.Comm): Communicator for the compsep processes.

    Output:
       detector_maps (list of np.arrays): The band-integrated total sky.
        
    """
    signal_map = detector_data.map_sky
    rms_map = detector_data.map_rms
    band_freq = detector_data.nu
    if params.make_plots:
        detector_to_plot = proc_comm.Get_rank()
        logging.info(f"Rank {proc_comm.Get_rank()} plotting detector map.")
        plotting.plot_data_maps(proc_master, params, detector_to_plot, chain, iter, map_signal=signal_map,
                                map_corr_noise=detector_data.map_corr_noise, map_rms=rms_map)
    if params.pixel_compsep_sampling:
        comp_maps = amplitude_sampling_per_pix(signal_map, rms_map, band_freq)
    else:
        compsep_solver = CompSepSolver(comp_list, signal_map, rms_map, band_freq, params, proc_comm)
        comp_list = compsep_solver.solve(seed=9999*chain+11*iter)
        if params.make_plots and proc_master:
            plotting.plot_cg_res(params, chain, iter, compsep_solver.CG_residuals)

    sky_model = SkyModel(comp_list)

    npix = signal_map.shape[-1]
    detector_map = sky_model.get_sky_at_nu(band_freq, 12*params.nside**2)
    # cmb_sky = component_list[0].get_sky(band_freq)
    # dust_sky = component_list[1].get_sky(band_freq)
    # sync_sky = component_list[2].get_sky(band_freq)

    if params.make_plots:
        detector_to_plot = proc_comm.Get_rank()
        plotting.plot_components(params, band_freq, detector_to_plot, chain, iter, signal_map, comp_list)
        # plotting.plot_components(params, band_freq,
        #                             detector_to_plot, chain, iter, sky=detector_map,
        #                             cmb=cmb_sky, dust=dust_sky,
        #                             sync=sync_sky,
        #                             signal=signal_map)

    return detector_map  # Return the full sky realization for my band.