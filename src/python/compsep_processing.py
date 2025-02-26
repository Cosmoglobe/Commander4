import numpy as np
from mpi4py import MPI
from mpi4py.MPI import Comm
import time
import logging
import h5py
import healpy as hp
from data_models import DetectorMap
from model.component import CMB, ThermalDust, Synchrotron, DiffuseComponent
from model.sky_model import SkyModel
from output import log, plotting
from pixell import bunch
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
    num_bands = len(params.bands)
    return proc_master, proc_comm, num_bands


def process_compsep(detector_data: list[DetectorMap], iter: int, chain: int,
                    params: bunch, proc_master=False):
    """ Performs a single component separation iteration.
    
    Input:
        detector_data (list of DetectorMaps): The correlated noise cleaned detector maps, one per band.
        iter (int): The current Gibbs iteration (used only for plotting and seeding)
        chain (int): The current chain (used only for plotting and seeding).
        params (bunch): The parameters from the input parameter file.
        proc_master (bool): Whether this is the master compsep process.

    Output:
       detector_maps (list of np.arrays): The band-integrated total sky.
       foreground_subtracted_maps (list of np.arrays): The input maps minus the band-integrated foregrounds (not the CMB).
        
    """
    signal_maps = []
    rms_maps = []
    band_freqs = []
    for i_det in range(len(detector_data)):
        detector = detector_data[i_det]
        signal_maps.append(detector.map_sky)
        rms_maps.append(detector.map_rms)
        band_freqs.append(detector.nu)
        if params.make_plots:
            plotting.plot_data_maps(proc_master, params, i_det, chain, iter,
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
            plotting.plot_cg_res(proc_master, params, chain, iter,
                                 compsep_solver.CG_residuals)

    component_types = [CMB, ThermalDust, Synchrotron]  # At the moment we always sample all components. #TODO: Move to parameter file.
    component_list = []
    for i, component_type in enumerate(component_types):
        component = component_type()
        component.component_map = comp_maps[i]
        component_list.append(component)

    sky_model = SkyModel(component_list)

    npix = signal_maps.shape[-1]
    detector_maps = []
    foreground_maps = []
    for i_det in range(len(detector_data)):
        detector_map = sky_model.get_sky_at_nu(band_freqs[i_det], 12*params.nside**2)
        detector_maps.append(detector_map)
        cmb_sky = component_list[0].get_sky(band_freqs[i_det])
        dust_sky = component_list[1].get_sky(band_freqs[i_det])
        sync_sky = component_list[2].get_sky(band_freqs[i_det])
        foreground_maps.append(sky_model.get_foreground_sky_at_nu(band_freqs[i_det], npix))

        if params.make_plots:
            plotting.plot_components(proc_master, params, band_freqs[i_det],
                                     i_det, chain, iter, sky=detector_map,
                                     cmb=cmb_sky, dust=dust_sky,
                                     sync=sync_sky,
                                     signal=signal_maps[i_det])

    foreground_maps = np.array(foreground_maps)
    foreground_subtracted_maps = signal_maps - foreground_maps
    return detector_maps, foreground_subtracted_maps


def send_compsep(proc_master: bool, detector_maps: list[np.array], destinations: list[int]):
    """ MPI-send the results from compsep to a set of other destinations.

    Assumes the COMM_WORLD communicator.

    Input:
        proc_master(bool): Whether this is the master compsep process.
        detector_maps (list of arrays): The compsep results (either the
            detector maps or foreground subtracted maps).
        destinations (list of ints): The world ranks of the destination
            processes, one per band (could be the same destination).
    """
        
    if proc_master:
        for i in range(len(detector_maps)):
            MPI.COMM_WORLD.send(detector_maps[i], dest=destinations[i])


def receive_compsep(band_master: bool, band_comm: Comm, sender: int):
    """ MPI-receive the results from compsep (used in conjunction with
        send_compsep).

    Input:
        band_master (bool): Whether this is the master of the inter-band
            communicator.
        band_comm (MPI.Comm): The inter-band communicator.
        sender (int): The sender world rank of the compsep information.

    Returns:
        detector_map (np.array): The detector map of a single band, distributed
            to all processes belonging to the band communicator.
    """
    detector_map = None
    if band_master:
        detector_map = MPI.COMM_WORLD.recv(source=sender)
    detector_map = band_comm.bcast(detector_map, root=0)
    return detector_map
