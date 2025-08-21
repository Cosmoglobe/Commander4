import os
import time
import logging
from mpi4py import MPI
from pixell.bunch import Bunch
from src.python.output import log

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
    init_compsep_processing is called), it also has two dicts, 'tod_band_masters' and
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

    if worldrank < params.MPI_config.ntask_tod:
        color = 0
        os.environ["OMP_NUM_THREADS"] = "1"  # Setting threading configuration depending on tasks. Important to do before Numpy is imported, as Numpy will not respect changes to these.
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
    elif worldrank < params.MPI_config.ntask_tod + params.MPI_config.ntask_compsep:
        if params.betzy_mode:  # Betzy doesn't like heterogeneous MPI setups, so we oversubscribe the compsep ranks with cores we will in practice use as threads.
            if worldrank%params.nthreads_compsep == 0:  # Every nthreads_compsep rank is a real compsep rank, and gets to stay alive.
                color = 1  # Compsep
            else:
                color = 99  # Dummy rank
        else:
            color = 1  # Compsep
        os.environ["OMP_NUM_THREADS"] = f"{params.nthreads_compsep}"
        os.environ["OPENBLAS_NUM_THREADS"] = f"{params.nthreads_compsep}"
        os.environ["MKL_NUM_THREADS"] = f"{params.nthreads_compsep}"
        os.environ["VECLIB_MAXIMUM_THREADS"] = f"{params.nthreads_compsep}"
        os.environ["NUMEXPR_NUM_THREADS"] = f"{params.nthreads_compsep}"

    tot_num_experiment_bands = sum([len(params.experiments[experiment].bands) for experiment in params.experiments if params.experiments[experiment].enabled])
    tot_num_compsep_bands = len(params.CompSep_bands)
    tot_num_compsep_bands_from_TOD = len([band for band in params.CompSep_bands if params.CompSep_bands[band].get_from != "file"])  # Number of the bands on CompSep side that come from the TOD side.

    if worldsize != (params.MPI_config.ntask_tod + params.MPI_config.ntask_compsep):
        log.lograise(RuntimeError, f"Total number of MPI tasks ({worldsize}) must equal the sum of tasks for TOD ({params.MPI_config.ntask_tod}) + CompSep ({params.MPI_config.ntask_compsep}).", logger)
    if not params.betzy_mode and params.MPI_config.ntask_compsep != tot_num_compsep_bands:
        log.lograise(RuntimeError, f"CompSep needs exactly as many MPI tasks {params.MPI_config.ntask_compsep} as there are bands {tot_num_compsep_bands}.", logger)
    if params.betzy_mode and params.MPI_config.ntask_compsep != params.nthreads_compsep*tot_num_experiment_bands:
        log.lograise(RuntimeError, f"For Betzy mode, CompSep currently needs exactly as many MPI tasks {params.MPI_config.ntask_compsep} as there are bands {tot_num_experiment_bands} times CompSep threads per rank ({params.nthreads_compsep}).", logger)

    proc_comm = world_comm.Split(color, key=worldrank)
    world_comm.barrier()
    time.sleep(worldrank*1e-2)  # Small sleep to get prints in nice order.
    logger.info(f"MPI split performed, hi from worldrank {worldrank} (on machine {MPI.Get_processor_name()}) subcomrank {proc_comm.Get_rank()} from color {color} of size {proc_comm.Get_size()}.")

    # Determine the world ranks of the respective master tasks for compsep and TOD
    # We ensured that this works by the "key=worldrank" in the split command.
    tod_master = 0 if params.MPI_config.ntask_tod > 0 else None
    compsep_master = params.MPI_config.ntask_tod

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
        mpi_info['compsep'] = Bunch()
        mpi_info['compsep']['comm'] = proc_comm
        mpi_info['compsep']['master'] = 0
        mpi_info['compsep']['size'] = proc_comm.Get_size()
        mpi_info['compsep']['rank'] = proc_comm.Get_rank()
        mpi_info['compsep']['is_master'] = mpi_info.compsep.rank == mpi_info.compsep.master

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

    # We now loop over all bands in all experiments, and allocate them to the first ranks of the TOD MPI communicator.
    # These ranks will then become the "band masters" for those bands, handling all communication with CompSep.
    TOD_rank = 0
    for experiment in params.experiments:
        if params.experiments[experiment].enabled:
            for band in params.experiments[experiment].bands:
                if params.experiments[experiment].bands[band].enabled:
                    TOD_rank += 1
    tot_num_bands = TOD_rank
    if tot_num_bands > MPIsize_tod:
        log.lograise(RuntimeError, f"Total number of experiment bands {tot_num_bands} exceed number of TOD MPI tasks {MPIsize_tod}.", logger)

    if is_tod_master:
        logger.info(f"TOD: {MPIsize_tod} tasks allocated to TOD processing of {tot_num_bands} bands.")
        log.logassert(MPIsize_tod >= tot_num_bands, f"Number of MPI tasks dedicated to TOD processing ({MPIsize_tod}) must be equal to or larger than the number of bands ({tot_num_bands}).", logger)

    MPIcolor_band = MPIrank_tod%tot_num_bands  # Spread the MPI tasks over the different bands.
    band_comm = mpi_info.tod.comm.Split(MPIcolor_band, key=MPIrank_tod)  # Create communicators for each different band.
    MPIsize_band, MPIrank_band = band_comm.Get_size(), band_comm.Get_rank()  # Get my local rank, and the total size of, the band-communicator I'm on.
    logger.info(f"TOD: Hello from TOD-rank {MPIrank_tod} (on machine {MPI.Get_processor_name()}), dedicated to band {MPIcolor_band}, with local rank {MPIrank_band} (local communicator size: {MPIsize_band}).")
    
    is_band_master = MPIrank_band == 0  # Am I the master of my local band.

    mpi_info['tod']['band_color'] = MPIcolor_band
    mpi_info['band'] = Bunch()
    mpi_info['band']['master'] = 0
    mpi_info['band']['comm'] = band_comm
    mpi_info['band']['size'] = MPIsize_band
    mpi_info['band']['rank'] = MPIrank_band
    mpi_info['band']['is_master'] = is_band_master

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
