import os
import time
import logging
import numpy as np
import mpi4py
from mpi4py import MPI
from pixell.bunch import Bunch
from commander4.output import log

def init_mpi(params):
    """ To be run before anything else to set up the MPI environment.

    Creates a Bunch data structure, mpi_info, which contains all data relevant to the MPI layout of
    the program.  Structured as a hierarchy where the top level are the names of the MPI contexts
    that we operate in (for now, 'world', 'tod', 'compsep', and 'band'). The only exception to this
    is the 'processor_name' entry, which is independent of context and thus does not belong under
    any of the contexts.
        Below the context names are typically the following keys:
            - 'rank' : The MPI rank of the process in the context.
            - 'is_master': whether the process is a master process in the context.
            - 'master': The rank of the master process in the context.
            - 'size': The number of total processes in the context.
            - 'color': If the context is subdivided into further contexts, the color indicates which
                of the sub-contexts this process is part of.
            - 'comm': The communicator of the context.
    In addition to these, some contexts have further keys that give information relevant to that
    context. For example, the 'world' context contains the 'tod_master' and 'compsep_master' keys,
    which indicates which ranks in the world contexts are the masters in the tod and compsep
    contexts. After complete initialization (i.e. after init_tod_processing and
    init_compsep_processing is called), it will also have two dicts, 'tod_band_masters' and
    'compsep_band_masters', which contains a mapping of a band identifier to the world rank of the
    master of that band, where the bands are subcontexts of the TOD and compsep contexts,
    respectively (needed for passing data between TOD and compsep processes).

    Input:
        params (Bunch): The parameters from the input parameter file.
    Output:
        mpi_info (Bunch): The data structure containing all MPI relevant data, as explained above.
    """
    logger = logging.getLogger(__name__)
    mpi_info = Bunch()
    world_comm = MPI.COMM_WORLD
    worldsize, worldrank = world_comm.Get_size(), world_comm.Get_rank()
    is_world_master = worldrank == 0
    global_params = params.general
    tot_num_CompSep_ranks = global_params.MPI_config.ntask_compsep_I + global_params.MPI_config.ntask_compsep_QU
    if is_world_master:
        mpi4py_version = tuple(map(int, mpi4py.__version__.split('.')))
        MPI_version = MPI.Get_version()
        logger.info(f"MPI version: {MPI_version}. mpi4py version: {mpi4py_version}.")
        if MPI_version < (4,0):
            logger.warning(f"MPI version ({MPI_version}) is below (4,0)!")
        if mpi4py_version < (4,0):
            logger.warning(f"mpi4py version ({mpi4py_version}) is below (4,0)!")

    if is_world_master:  # Every rank doesn't need to throw an error.
        tot_num_Compsep_bands = len([band for band in params.CompSep_bands if   #I
                                    params.CompSep_bands[band].enabled and params.CompSep_bands[band].polarizations[0]]) +\
                                len([band for band in params.CompSep_bands if   #QU
                                    params.CompSep_bands[band].enabled and params.CompSep_bands[band].polarizations[1] and params.CompSep_bands[band].polarizations[2]])
        if worldsize != (global_params.MPI_config.ntask_tod + tot_num_CompSep_ranks):
            log.lograise(RuntimeError, f"Total number of MPI tasks ({worldsize}) must equal the sum "
                                       f"of tasks for TOD ({global_params.MPI_config.ntask_tod}) + CompSep I + QU"
                                       f"({global_params.MPI_config.ntask_compsep_I} + {global_params.MPI_config.ntask_compsep_QU}).", logger)
        if tot_num_CompSep_ranks != tot_num_Compsep_bands:
            log.lograise(RuntimeError, f"CompSep needs exactly as many MPI tasks "
                                       f"({tot_num_CompSep_ranks}) as there are bands "
                                       f"({tot_num_Compsep_bands}).", logger)

    # Split the world communicator into a communicator for compsep and one for TOD (with "color"
    # being the keyword for the split).
    if worldrank < global_params.MPI_config.ntask_tod:
        color = 0 #TOD
        # Note that Numpy will not respect these values, because Numpy has already been loaded
        # as a mpi4py dependency. Numpy does not respect changes to these values after it has been
        # imported. Ideally these variables should therefore be set before calling Python at all.
        os.environ["OMP_NUM_THREADS"] = f"{global_params.nthreads_tod}"
        os.environ["OPENBLAS_NUM_THREADS"] = f"{global_params.nthreads_tod}" 
        os.environ["MKL_NUM_THREADS"] = f"{global_params.nthreads_tod}"
        os.environ["VECLIB_MAXIMUM_THREADS"] = f"{global_params.nthreads_tod}"
        os.environ["NUMEXPR_NUM_THREADS"] = f"{global_params.nthreads_tod}"
        import numba
        numba.set_num_threads(1)
    elif worldrank < global_params.MPI_config.ntask_tod + tot_num_CompSep_ranks:
        color = 1  # Compsep

        # nthreads_compsep is either an int, or a list specifying nthreads for each rank.
        if isinstance(global_params.nthreads_compsep, int):  # If int, all ranks have same nthreads.
            nthreads_compsep = global_params.nthreads_compsep
        else:
            nthreads_compsep = global_params.nthreads_compsep[worldrank - global_params.MPI_config.ntask_tod]
        os.environ["OMP_NUM_THREADS"] = f"{nthreads_compsep}"
        os.environ["OPENBLAS_NUM_THREADS"] = f"{nthreads_compsep}"
        os.environ["MKL_NUM_THREADS"] = f"{nthreads_compsep}"
        os.environ["VECLIB_MAXIMUM_THREADS"] = f"{nthreads_compsep}"
        os.environ["NUMEXPR_NUM_THREADS"] = f"{nthreads_compsep}"
        import numba
        # Testing revealed 24 to be a good number (regardless of nside), but I tested this on the
        # new 384-core nodes, the optimal number is probably slightly lower on the older owls.
        numba.set_num_threads(min(24,nthreads_compsep))

    else:
        raise ValueError("My rank ({worldrank}) exceeds the combined number of allocated tasks to"
                         f"both TOD ({global_params.MPI_config.ntask_tod}) and compsep" \
                         f"{tot_num_CompSep_ranks}")
  
    proc_comm = world_comm.Split(color, key=worldrank)
    if color == MPI.UNDEFINED:
        return -1
    world_comm.barrier()
    time.sleep(worldrank*1e-3)  # Small sleep to get prints in nice order.
    logger.info(f"MPI split performed, hi from worldrank {worldrank} (on machine "
                f"{MPI.Get_processor_name()}) subcomrank {proc_comm.Get_rank()} from color {color} of "
                f" size {proc_comm.Get_size()}.")

    # Determine the world ranks of the respective master tasks for compsep and TOD
    # We ensured that this works by the "key=worldrank" in the split command.
    tod_master = 0 if global_params.MPI_config.ntask_tod > 0 else None
    compsep_master = global_params.MPI_config.ntask_tod

    world_comm.barrier()
    time.sleep(worldrank*1e-2)  # Small sleep to get prints in nice order.

    mpi_info['world'] = Bunch()
    mpi_info['world']['comm'] = world_comm
    mpi_info['world']['master'] = 0
    mpi_info['world']['size'] = worldsize
    mpi_info['world']['rank'] = worldrank
    mpi_info['world']['color'] = color
    mpi_info['world']['tod_master'] = tod_master
    mpi_info['world']['compsep_master'] = compsep_master
    mpi_info['world']['is_master'] = is_world_master
    mpi_info['world']['tod_band_masters'] = Bunch()
    mpi_info['world']['compsep_band_masters'] = Bunch()
    mpi_info['processor_name'] = MPI.Get_processor_name()

    if color == 0:
        mpi_info['tod'] = Bunch()
        mpi_info['tod']['comm'] = proc_comm
        mpi_info['tod']['master'] = 0
        mpi_info['tod']['size'] = proc_comm.Get_size()
        mpi_info['tod']['rank'] = proc_comm.Get_rank()
        mpi_info['tod']['is_master'] = mpi_info.tod.rank == mpi_info.tod.master
        mpi_info = init_mpi_tod(mpi_info, params)

    elif color == 1:
        proc_rank = proc_comm.Get_rank()
        mpi_info['compsep'] = Bunch()
        mpi_info['compsep']['comm'] = proc_comm
        mpi_info['compsep']['master'] = 0
        mpi_info['compsep']['size'] = proc_comm.Get_size()
        mpi_info['compsep']['rank'] = proc_rank
        mpi_info['compsep']['is_master'] = mpi_info.compsep.rank == mpi_info.compsep.master
        
        #Split between I and QU
        subcolor = 0 if proc_rank < global_params.MPI_config.ntask_compsep_I else 1
        sub_comm = proc_comm.Split(subcolor, key=proc_rank)
        mpi_info['compsep']['subcomm'] = sub_comm
        mpi_info['compsep']['subcolor'] = subcolor
        mpi_info['compsep']['subsize'] = sub_comm.Get_size()
        mpi_info['compsep']['subrank'] = sub_comm.Get_rank()
        mpi_info['compsep']['I_master'] = 0                                                             #in compsep_comm numbering
        mpi_info['compsep']['QU_master'] = mpi_info.compsep.size - global_params.MPI_config.ntask_compsep_QU   #in compsep_comm numbering
        mpi_info['compsep']['is_I_master'] = subcolor == 0 and mpi_info.compsep.subrank == 0
        mpi_info['compsep']['is_QU_master'] = subcolor == 1 and mpi_info.compsep.subrank == 0
        mpi_info = init_mpi_compsep(mpi_info, params)
    return mpi_info
    

def init_mpi_tod(mpi_info, params):
    """ Add the tod-specific information to the mpi_info structure.

    This function is called by init_mpi.

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        params (Bunch): The parameters from the input parameter file.

    Output:
        mpi_info (Bunch): The data structure containing all MPI relevant data, now including info
            for the 'tod' context.
    """

    logger = logging.getLogger(__name__)
    MPIsize_tod, MPIrank_tod = mpi_info.tod.size, mpi_info.tod.rank
    is_tod_master = mpi_info.tod.is_master
    tod_comm = mpi_info.tod.comm

    # We now loop over all bands in all experiments, and allocate them to the first ranks of the TOD MPI communicator.
    # These ranks will then become the "band masters" for those bands, handling all communication with CompSep.
    TOD_rank = 0
    current_detector_id = 0  # A unique number identifying every detector of every band.
    for exp_name in params.experiments:
        experiment = params.experiments[exp_name]
        if not experiment.enabled:
            continue
        for iband, band_name in enumerate(experiment.bands):
            band = experiment.bands[band_name]
            if not band.enabled:
                continue
            this_band_TOD_ranks = np.arange(TOD_rank, TOD_rank + band.num_MPI_tasks)
            if MPIrank_tod in this_band_TOD_ranks:
                # Splitting TOD ranks evenly among the detectors of this band.
                TOD_ranks_per_detector = np.array_split(this_band_TOD_ranks, len(band.detectors))
                my_band_name = band_name
                my_band_id = iband
                for idet, det_name in enumerate(band.detectors):
                    num_ranks_this_detector = len(TOD_ranks_per_detector[idet])
                    detector = band.detectors[det_name]
                    # Check if our rank belongs to this detector
                    if MPIrank_tod in TOD_ranks_per_detector[idet]:
                        # What is my rank number among the ranks processing this detector?
                        my_det_id = idet
                        # Setting our unique detector id. Note that this is a global, not per band.
                        my_detector_id = current_detector_id
                        my_detector_name = det_name
                    current_detector_id += 1  # Update detector counter.
            else:
                # Update detector counter for ranks not assigned to current band.
                current_detector_id += len(band.detectors)
            TOD_rank += band.num_MPI_tasks
    tot_num_bands = TOD_rank
    if tot_num_bands != MPIsize_tod:
        log.lograise(RuntimeError, f"Total number of bands dedicated to the various experiments "
                     f"({tot_num_bands}) differs from the total number of tasks dedicated to "
                     f"TOD processing ({MPIsize_tod}).", logger)

    if is_tod_master:
        logger.info(f"TOD: {MPIsize_tod} tasks allocated to TOD processing of {tot_num_bands} bands.")
        log.logassert(MPIsize_tod >= tot_num_bands, f"Number of MPI tasks dedicated to TOD "
                                                    f"processing ({MPIsize_tod}) must be equal to or "
                                                    f"larger than the number of bands "
                                                    f"({tot_num_bands}).", logger) 

    band_comm = mpi_info.tod.comm.Split(my_band_id, key=MPIrank_tod)  # Create communicators for each different band.
    # Get my local rank, and the total size of, the band-communicator IvsQU'm on.
    MPIsize_band, MPIrank_band = band_comm.Get_size(), band_comm.Get_rank()  
    det_comm = band_comm.Split(my_det_id, key=MPIrank_band)  # Create communicators for each,
                                                                 # using the local IDs
    MPIsize_det, MPIrank_det = det_comm.Get_size(), det_comm.Get_rank()  
    is_band_master = MPIrank_band == 0  # Am I the master of my local band.
    is_det_master = MPIrank_det == 0

    tod_comm.Barrier()
    time.sleep(MPIrank_tod*1e-3)  # Small sleep to get prints in nice order.
    
    logger.info(f"TOD: Hello from TOD-rank {MPIrank_tod} (on machine {MPI.Get_processor_name()}), "
                f"dedicated to band {my_band_id}, with local rank {MPIrank_band} (local "
                f"communicator size: {MPIsize_band}), and detector "
                f"{my_det_id} with local rank {MPIrank_det} and size {MPIsize_det}")

    mpi_info['tod']['band_id'] = my_band_id
    mpi_info['band'] = Bunch()
    mpi_info['band']['master'] = 0
    mpi_info['band']['comm'] = band_comm
    mpi_info['band']['size'] = MPIsize_band
    mpi_info['band']['rank'] = MPIrank_band
    mpi_info['band']['is_master'] = is_band_master
    mpi_info['band']['name'] = my_band_name
    mpi_info['band']['det_id'] = my_det_id
    mpi_info['det'] = Bunch()
    mpi_info['det']['master'] = 0
    mpi_info['det']['comm'] = det_comm
    mpi_info['det']['size'] = MPIsize_det
    mpi_info['det']['rank'] = MPIrank_det
    mpi_info['det']['is_master'] = is_det_master
    mpi_info['det']['name'] = my_detector_name

    return mpi_info


def init_mpi_compsep(mpi_info, params):
    """ Add the compsep-specific information to the mpi_info structure.

    The only thing this does currently is check whether we have enough MPI compsep processes (i.e.
    one per band), and fill in some trivial information in the mpi_info dict. This function is
    called by init_mpi.

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        params (Bunch): The parameters from the input parameter file.

    Output:
        mpi_info (Bunch): The data structure containing all MPI relevant data, now including info
            for the 'compsep' context.
    """

    logger = logging.getLogger(__name__)
    MPIsize_compsep, MPIrank_compsep = mpi_info.compsep.size, mpi_info.compsep.rank
    is_compsep_master = mpi_info.compsep.is_master

    ### Setting up info for each band, including where to get the data from (map from file, or receive from TOD processing) ###
    current_band_idx = 0
    for band_str in params.CompSep_bands:
        if params.CompSep_bands[band_str].enabled:
            current_band_idx += 1
    tot_num_bands = current_band_idx
    mpi_info['band'] = Bunch()
    mpi_info['band']['size'] = 1
    mpi_info['band']['is_master'] = True

    if tot_num_bands > MPIsize_compsep:
        log.lograise(RuntimeError, f"Total number of experiment bands {tot_num_bands} exceeds the number of Compsep MPI tasks {MPIsize_compsep}.", logger)

    return mpi_info
