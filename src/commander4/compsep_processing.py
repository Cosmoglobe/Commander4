import numpy as np
import logging
from copy import deepcopy
from pixell.bunch import Bunch

from commander4.output.log import logassert
from commander4.data_models.detector_map import DetectorMap
from commander4.sky_models.component import CompList
from commander4.sky_models.sky_model import SkyModel
from commander4.solvers.CG_compsep_solver import CompSepSolver
from commander4.solvers.perpix_compsep_solver import solve_compsep_perpix
from commander4.output.write_chains_files import write_compsep_chain_to_file
from commander4.utils.execution_ids import get_execution_band_id

logger = logging.getLogger(__name__)

def init_compsep_processing(mpi_info: Bunch, params: Bunch)\
    -> tuple[CompList, str, dict[str, int], Bunch]:
    """Set up the rank-local execution view for component separation.

    Each CompSep rank owns exactly one execution view of one band. The global CompSep rank space
    is split into a contiguous intensity block followed by a contiguous QU block, and we match the
    current rank against those two logical streams here.

    Args:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        params (Bunch): The parameters from the input parameter file.

    Returns:
        mpi_info (Bunch): The data structure containing all MPI relevant data,
            modified to contain also the band masters dictionary.
        band_identifier (str): Unique string for the band execution view this rank is working on.
        my_band (Bunch): A subset of the full parameter file for the band this rank is working on.
    """
    logger.info(f"CompSep: Hello from CompSep-rank {mpi_info.compsep.rank} (on machine "\
                f"{mpi_info.processor_name}), dedicated to band {mpi_info.compsep.rank}.")

    comp_list = CompList.init_from_params(params.components, params)

    ### Setting up info for each band, including where to get the data from ###
    ### (map from file, or receive from TOD processing) ###

    current_band_idx_I = 0
    current_band_idx_QU = mpi_info.compsep.QU_master
    band_identifier = None
    my_band = None
    for band_str in params.CompSep_bands:   # Intensity
        band = params.CompSep_bands[band_str]
        if band.enabled:
            if band.polarization == "I":
                if current_band_idx_I == mpi_info.compsep.rank:
                    my_band = deepcopy(band)
                    band_identifier = get_execution_band_id(band_str, "I")
                    logger.info(f"Rank {mpi_info.compsep.rank} just matched band {band_identifier}")
                    my_band.identifier = band_identifier
                    my_band.polarization = "I"
                current_band_idx_I += 1

            elif band.polarization == "QU":
                if current_band_idx_QU == mpi_info.compsep.rank:
                    my_band = deepcopy(band)
                    band_identifier = get_execution_band_id(band_str, "QU")
                    logger.info(f"Rank {mpi_info.compsep.rank} matched band {band_identifier}")
                    my_band.identifier = band_identifier
                    my_band.polarization = "QU"
                current_band_idx_QU += 1

            elif band.polarization == "IQU":
                # Each IQU band occupies one rank in the intensity block and one in the QU block.
                if current_band_idx_I == mpi_info.compsep.rank:
                    my_band = deepcopy(band)
                    band_identifier = get_execution_band_id(band_str, "I")
                    logger.info(f"Rank {mpi_info.compsep.rank} just matched band {band_identifier}")
                    my_band.identifier = band_identifier
                    my_band.polarization = "I"
                current_band_idx_I += 1
                if current_band_idx_QU == mpi_info.compsep.rank:
                    my_band = deepcopy(band)
                    band_identifier = get_execution_band_id(band_str, "QU")
                    logger.info(f"Rank {mpi_info.compsep.rank} matched band {band_identifier}")
                    my_band.identifier = band_identifier
                    my_band.polarization = "QU"
                current_band_idx_QU += 1
            else:
                raise ValueError(f"Unrecognized polarization in parameter file for band {band_str}")
    
    #sanity check:
    logassert(current_band_idx_I == mpi_info.compsep.QU_master, "Number of acquired Intensity "\
              f"bands ({current_band_idx_I}) do not match number of MPI tasks assigned to "\
              f"Intensity ({mpi_info.compsep.QU_master})", logger)
    logassert(current_band_idx_QU == mpi_info.compsep.size, "Number of acquired QU bands "\
              f"({current_band_idx_QU}) do not match number of MPI tasks assigned to QU "\
              f"({mpi_info.compsep.QU_master})", logger)
    if my_band is None or band_identifier is None:
        logassert(False,
                  f"CompSep rank {mpi_info.compsep.rank} was not assigned to any enabled band. "
                  "Check that CompSep_bands matches the configured I/QU rank counts.",
                  logger)
    
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
                    params: Bunch, comp_list: CompList) -> SkyModel:
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

    ### 1. MPI SETUP: Split into I and QU ###
    compsep_comm = mpi_info.compsep.comm
    compsep_rank = mpi_info.compsep.rank
    subcolor = mpi_info.compsep.subcolor #Subcolor splits the compsep ranks into: Pol -> 1, Int -> 0
    compsep_subcomm = mpi_info.compsep.subcomm
    target_pol = "I" if subcolor == 0 else "QU"
    comp_sublist = comp_list.split_for_eval_pol(target_pol)

    ### 2. SOLVE COMPSEP: band maps -> component alms (either by per-pixel or CG solver) ###
    if params.general.pixel_compsep_sampling:
        comp_sublist = solve_compsep_perpix(compsep_subcomm, detector_data, comp_sublist, params)
    else:
        compsep_solver = CompSepSolver(detector_data, params, compsep_subcomm)
    
        comp_sublist = compsep_solver.solve(comp_sublist)

    ### 3. CLEANUP: Gather I+QU alm solutions and make plots. ###
    comp_list.reassemble_from_split_solution(
        comp_sublist,
        compsep_comm,
        is_I_master=mpi_info.compsep.is_I_master,
        is_QU_master=mpi_info.compsep.is_QU_master,
        I_master=mpi_info.compsep.I_master,
        QU_master=mpi_info.compsep.QU_master,
        root=mpi_info.compsep.master,
    )

    sky_model = SkyModel(comp_list)
    sky_model_at_band = sky_model.get_sky_at_nu(detector_data.nu, detector_data.nside, "IQU",
                                                fwhm=np.deg2rad(detector_data.fwhm/60.0))

    pol_names = ["Q", "U"] if detector_data.pol else ["I"]
    pol_offset = 1 if detector_data.pol else 0
    for ipol in range(detector_data.npol):
        chi2 = np.mean(np.abs(detector_data.map_sky[ipol] -
                              sky_model_at_band[ipol + pol_offset])/detector_data.map_rms[ipol])
        logger.info(f"Reduced chi2 on rank {compsep_rank} for pol={pol_names[ipol]} "\
                    f"({detector_data.nu}GHz): {chi2:.3f}")

    compsep_comm.Barrier()

    if compsep_rank == mpi_info.compsep.master:
        write_compsep_chain_to_file(comp_list.joined(), params, chain, iter)

    return sky_model  # Return the full sky realization for my band.

