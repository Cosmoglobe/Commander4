import numpy as np
import logging
from pixell.bunch import Bunch

from commander4.output.log import logassert
from commander4.data_models.detector_map import DetectorMap
from commander4.sky_models.component import Component, split_complist
import commander4.sky_models.component as component_lib
from commander4.sky_models.sky_model import SkyModel
from commander4.solvers.CG_compsep_solver import CompSepSolver
from commander4.solvers.perpix_compsep_solver import solve_compsep_perpix
from commander4.output.write_chains_files import write_compsep_chain_to_file


def init_compsep_processing(mpi_info: Bunch, params: Bunch)\
    -> tuple[list[Component], str, dict[str, int], Bunch]:
    """ To be run once before starting component separation processing.
        Determines whether the process is compsep master, and the number of bands.

    Args:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        params (Bunch): The parameters from the input parameter file.

    Returns:
        mpi_info (Bunch): The data structure containing all MPI relevant data,
            modified to contain also the band masters dictionary.
        band_identifier (str): Unique string for the experiment+band this rank is working on.
        my_band (Bunch): A subset of the full parameter file for the band this rank is working on.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"CompSep: Hello from CompSep-rank {mpi_info.compsep.rank} (on machine "\
                f"{mpi_info.processor_name}), dedicated to band {mpi_info.compsep.rank}.")

    ### Creating list of all components ###
    comp_list = []
    for component_str in params.components:
        component = params.components[component_str]
        if component.enabled:
            comp_shortname = component.params.shortname
            comp_longname = component.params.longname
            if component.params.lmax == "full":
                component.params.lmax = (params.general.nside*5)//2
            if component.params.polarizations[0]: #->I
                # 'getattr' loads the class specified by "component_class" from model.component.
                # This class is then instantiated with the "params" specified, and appended to
                # the components list.
                # TODO: I don't love that we are editing these previously defined parameters,
                # maybe there is a more elegant way of doing this.
                component.params.longname = comp_longname + "_Intensity"
                component.params.shortname = comp_shortname + "_I"
                component.params.polarized = False
                # Use getattr to get and initialize current component from component_lib file.
                # Pre-allocate alm arrays so that we can seamlessly receive data from MPI comm.
                comp_list.append(getattr(component_lib, component.component_class)(component.params,
                                                        params.general, allocate_empty_alms=True))
            if component.params.polarizations[1] and component.params.polarizations[2]: #->QU
                component.params.longname = comp_longname + "_Polarization"
                component.params.shortname = comp_shortname + "_QU"
                component.params.polarized = True
                comp_list.append(getattr(component_lib, component.component_class)(component.params,
                                                        params.general, allocate_empty_alms=True))
            

    ### Setting up info for each band, including where to get the data from ###
    ### (map from file, or receive from TOD processing) ###

    current_band_idx_I = 0
    current_band_idx_QU = mpi_info.compsep.QU_master
    band_identifier = None
    for band_str in params.CompSep_bands:   # Intensity
        band = params.CompSep_bands[band_str]
        if band.enabled:
            logassert(len(band.polarizations), f"{len(band.polarizations)} stokes parameter "\
                      "definitions found in band section in param file, "\
                      "3 expected. E.g. [True, False, False]", logger)
            is_I = band.polarizations[0]
            is_QU = band.polarizations[1] and band.polarizations[2]
            if is_I:
                # Each rank is responsible for one band (the one matching the index of their rank).
                if current_band_idx_I == mpi_info.compsep.rank:
                    my_band = band
                    if my_band.get_from != "file":
                        band_identifier = f"{my_band.get_from}$$${band_str}_I"
                    else:
                        band_identifier = band_str+"_I"
                    logger.info(f"Rank {mpi_info.compsep.rank} just matched band {band_identifier}")
                    my_band.identifier = band_identifier
                current_band_idx_I += 1
            if is_QU:
                # Each rank is responsible for one band (the one matching the index of their rank).
                if current_band_idx_QU == mpi_info.compsep.rank:
                    my_band = band
                    if my_band.get_from != "file":
                        band_identifier = f"{my_band.get_from}$$${band_str}_QU"
                    else:
                        band_identifier = band_str+"_QU"
                    logger.info(f"Rank {mpi_info.compsep.rank} matched band {band_identifier}")
                    my_band.identifier = band_identifier
                current_band_idx_QU += 1
            if not (is_I or is_QU):
                raise ValueError(f"Pol of band {band_str} misconfigured in parameter file.")
    
    #sanity check:
    logassert(current_band_idx_I == mpi_info.compsep.QU_master, "Number of acquired Intensity "\
              f"bands ({current_band_idx_I}) do not match number of MPI tasks assigned to "\
              f"Intensity ({mpi_info.compsep.QU_master})", logger)
    logassert(current_band_idx_QU == mpi_info.compsep.size, "Number of acquired QU bands "\
              f"({current_band_idx_QU}) do not match number of MPI tasks assigned to QU "\
              f"({mpi_info.compsep.QU_master})", logger)
    
    data_world = (band_identifier, mpi_info.world.rank)
    data_compsep = (band_identifier, mpi_info.compsep.rank)
    all_data_world = mpi_info.compsep.comm.allgather(data_world)
    all_data_compsep = mpi_info.compsep.comm.allgather(data_compsep)
    world_band_masters_dict = {item[0]: item[1] for item in all_data_world if item is not None}
    compsep_band_masters_dict = {item[0]: item[1] for item in all_data_compsep if item is not None}
    mpi_info.world.compsep_band_masters = world_band_masters_dict
    mpi_info.compsep.compsep_band_masters = compsep_band_masters_dict

    return comp_list, mpi_info, band_identifier, my_band


def process_compsep(mpi_info: Bunch, detector_data: DetectorMap, iter: int, chain: int,
                    params: Bunch, comp_list: list[Component]) -> SkyModel:
    """ Performs a single component separation iteration.
        Called by each compsep process, which are each responsible for a single band.
    
    Args:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        detector_data (DetectorMap): The detector map for this MPI rank's band, cleaned of all
                                     "TOD" components (correlated noise and orbital dipole).
        iter (int): The current Gibbs iteration (used only for printing and seeding).
        chain (int): The current chain (used only for printing and seeding).
        params (Bunch): The parameters from the input parameter file.

    Returns:
       detector_maps (np.array): The band-integrated total sky. 
    """

    logger = logging.getLogger(__name__)

    ### 1. MPI SETUP: Split into I and QU ###
    compsep_comm = mpi_info.compsep.comm
    compsep_rank = mpi_info.compsep.rank
    subcolor = mpi_info.compsep.subcolor #Subcolor splits the compsep ranks into: Pol -> 1, Int -> 0
    compsep_subcomm = mpi_info.compsep.subcomm
    comp_sublist = split_complist(comp_list, subcolor)

    ### 2. SOLVE COMPSEP: band maps -> component alms (either by per-pixel or CG solver) ###
    if params.general.pixel_compsep_sampling:
        comp_sublist = solve_compsep_perpix(compsep_subcomm, detector_data, comp_sublist, params)
    else:
        compsep_solver = CompSepSolver(detector_data, params, compsep_subcomm)
    
        comp_sublist = compsep_solver.solve(comp_sublist)

    ### 3. CLEANUP: Gather I+QU alm solutions and make plots. ###
    # Pol master sends the portion of list to the Intensity master rank,
    # and then it will broadcast through the compsep_comm
    # FIXME: this has to change: a component can be only QU!!!  check if I master exists
    if mpi_info.compsep.is_QU_master:
        t=0
        for comp in comp_sublist:
            logger.debug(f"[MPI Comm] Sending {comp.shortname} from QU {comp._data.shape} "\
                         f"{comp._data.dtype} {t} to {mpi_info.compsep.I_master}")
            compsep_comm.Send(comp._data, dest=mpi_info.compsep.I_master, tag=t)
            t+=1
    
    if mpi_info.compsep.is_I_master:
        t_pol=0
        t_int=0
        for comp in comp_list:
            if comp.pol: #if it is a pol component receive it from the QU_master
                logger.debug(f"[MPI Comm] Receiving {comp.shortname} from QU {comp._data.shape} "\
                             f"{comp._data.dtype} {t_pol} from {mpi_info.compsep.QU_master}")
                compsep_comm.Recv(comp._data, source=mpi_info.compsep.QU_master, tag=t_pol)
                t_pol+=1
            else:  # Otherwise it copy it over from the local intensity sublist held on I_master
                logger.debug(f"[MPI Comm] Copying {comp.shortname} from I {comp._data.shape} "\
                             f"{comp._data.dtype} {t_int} from local I")
                comp._data = comp_sublist[t_int]._data
                t_int+=1
    
    # In any case the component, received from QU or computed and accumulated on I, is bcasted
    for comp in comp_list:
        comp.bcast_data_blocking(compsep_comm, root=mpi_info.compsep.master)

    #FIXME: How will we deal with this once we give the chance to the user to define different
    # parameters for polarized and non-pol detectors?
    sky_model = SkyModel(comp_list)
    sky_model_at_band = sky_model.get_sky_at_nu(detector_data.nu, detector_data.nside,
                                                fwhm=np.deg2rad(detector_data.fwhm/60.0))

    pol_names = ["Q", "U"] if detector_data.pol else ["I"]
    for ipol in range(detector_data.npol):
        chi2 = np.mean(np.abs(detector_data.map_sky[ipol] -
                              sky_model_at_band[ipol])/detector_data.map_rms[ipol])
        logger.info(f"Reduced chi2 on rank {compsep_rank} for pol={pol_names[ipol]} "\
                    f"({detector_data.nu}GHz): {chi2:.3f}")

    compsep_comm.Barrier()

    if compsep_comm.Get_rank() == 0:
        write_compsep_chain_to_file(comp_list, params, chain, iter)


    return sky_model  # Return the full sky realization for my band.

