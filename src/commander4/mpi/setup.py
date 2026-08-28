"""Split MPI ranks between the TOD and CompSep sides and the enabled bands."""
import os
import time
import logging
import mpi4py
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.parameters.schema import (
    compsep_enabled,
    derive_task_counts,
    enabled_compsep_views,
    enabled_tod_bands,
    has_enabled_sampling_group,
    task_count_breakdown,
)

logger = logging.getLogger(__name__)

def init_mpi(params: Bunch) -> Bunch:
    """Validate the task count and build this rank's world, side, and band contexts.

    The returned ``Bunch`` contains the common ``world`` context, one named ``tod`` or ``compsep``
    context, and the selected ``band``. Every context uses the same direct fields: communicator,
    rank, size, master, and ``is_master`` where applicable.
    """
    mpi_info = Bunch()
    world_comm = MPI.COMM_WORLD
    worldsize, worldrank = world_comm.Get_size(), world_comm.Get_rank()
    is_world_master = worldrank == 0
    # The task counts are derived from the band configuration rather than stated in the parameter
    # file: the TOD total is the sum of the per-band `num_tasks`, and CompSep takes one task per
    # enabled band view (one for I, one for QU).
    ntasks = derive_task_counts(params)
    tot_num_CompSep_ranks = ntasks.compsep_I + ntasks.compsep_QU
    if is_world_master:
        mpi4py_version = tuple(map(int, mpi4py.__version__.split('.')))
        MPI_version = MPI.Get_version()
        logger.info(f"MPI version: {MPI_version}. mpi4py version: {mpi4py_version}.")
        logger.summary(f"MPI task layout: {task_count_breakdown(ntasks)}.")
        if MPI_version < (4,0):
            logger.warning(f"MPI version ({MPI_version}) is below (4,0)!")
        if mpi4py_version < (4,0):
            logger.warning(f"mpi4py version ({mpi4py_version}) is below (4,0)!")

    if is_world_master:  # Every rank doesn't need to throw an error.
        # Check if compsep.enabled is False (if the compsep group exist at all).
        turned_off = "compsep" in params and "enabled" in params.compsep \
            and not params.compsep.enabled
        # If compsep is turned off, there can't be sampling groups enabled, that's be an error.
        if has_enabled_sampling_group(params) and turned_off:
            raise RuntimeError("Enabled compsep sampling groups are configured, but "
                               "'compsep.enabled' is false, so no CompSep MPI ranks are "
                               "allocated to run them. Enable compsep, or disable the groups.")
        if worldsize != ntasks.total:
            raise RuntimeError(f"This run needs {task_count_breakdown(ntasks)} MPI tasks, "
                               f"but was started with {worldsize}. The counts follow from the "
                               "parameter file (the per-band 'num_tasks' of every enabled band, "
                               "and one CompSep task per enabled 'compsep.bands' view); run with "
                               f"'mpirun -n {ntasks.total}'.")
        # Otherwise CompSep is simply off (TOD-only) and compsep.bands is ignored.
        if not compsep_enabled(params):
            reason = ("compsep.enabled is false" if turned_off
                      else "no compsep sampling group is enabled, so compsep ranks would have "
                           "nothing to sample")
            logger.info(f"Running TOD-only: {reason}. Any 'compsep.bands' are ignored, and TOD "
                        "ranks use the initial sky model built from the components.")

    # Split the world communicator into a communicator for compsep and one for TOD (with "color"
    # being the keyword for the split).
    nthreads_compsep = params.resources.compsep.num_threads
    if worldrank < ntasks.tod:
        color = 0
        side = "tod"
        my_num_threads = params.resources.tod.num_threads
        my_num_threads_numba = my_num_threads

    elif worldrank < ntasks.tod + tot_num_CompSep_ranks:
        color = 1  # Compsep
        side = "compsep"
        # num_threads is either an int, or a list specifying nthreads for each rank.
        if isinstance(nthreads_compsep, int):  # If int, all ranks have same nthreads.
            my_num_threads = nthreads_compsep
        else:
            if len(nthreads_compsep) != tot_num_CompSep_ranks:
                raise ValueError(
                    f"Length of resources.compsep.num_threads ({len(nthreads_compsep)}) does not "
                    f"match the number of CompSep ranks ({tot_num_CompSep_ranks}).")
            my_num_threads = nthreads_compsep[worldrank - ntasks.tod]
        # Testing revealed 24 to be a good number (regardless of nside), but I tested this on the
        # new 384-core nodes, the optimal number is probably slightly lower on the older owls.
        my_num_threads_numba = min(24,my_num_threads)
    else:
        raise ValueError(f"My rank ({worldrank}) exceeds the combined number of allocated tasks to"
                         f"both TOD ({ntasks.tod}) and compsep ({tot_num_CompSep_ranks})")

    # It's important to set these environment variables before importing any package that might
    # use them, such as Numpy or Scipy, as they will not apply retroactively!
    os.environ["OMP_NUM_THREADS"] = f"{my_num_threads}"
    os.environ["OPENBLAS_NUM_THREADS"] = f"{my_num_threads}" 
    os.environ["MKL_NUM_THREADS"] = f"{my_num_threads}"
    os.environ["VECLIB_MAXIMUM_THREADS"] = f"{my_num_threads}"
    os.environ["NUMEXPR_NUM_THREADS"] = f"{my_num_threads}"
    os.environ["NUMBA_NUM_THREADS"] = f"{my_num_threads_numba}"
    # I tried using numba.set_num_threads(x) here instead (or as well) but that
    # resulted in some weirdeties, like many duplicate open file handles even when x=1.

    if False: # This code should enter production, but threadpoolctl is not yet a dependency.
        import numba
        from threadpoolctl import threadpool_info

        pool_info = threadpool_info()
        for pool in pool_info:
            if pool["num_threads"] != my_num_threads:
                raise RuntimeError(f"Loaded library {pool} has {pool['num_threads']} threads, "
                                   f"expected {my_num_threads}.")
        if numba.get_num_threads() != my_num_threads_numba:
            raise RuntimeError(f"Numba has {numba.get_num_threads()} threads, expected "
                               f"{my_num_threads_numba}.")


    proc_comm = world_comm.Split(color, key=worldrank)
    if color == MPI.UNDEFINED:
        return -1
    world_comm.barrier()
    time.sleep(worldrank*1e-5)  # Small sleep to get prints in nice order.
    logger.debug(f"MPI split performed, hi from worldrank {worldrank} (on machine "\
                f"{MPI.Get_processor_name()}) subcomrank {proc_comm.Get_rank()} from color "\
                f"{color} of size {proc_comm.Get_size()}. Threads = {my_num_threads}.")

    # Determine the world ranks of the respective master tasks for compsep and TOD
    # We ensured that this works by the "key=worldrank" in the split command.
    tod_master = 0 if ntasks.tod > 0 else None
    compsep_master = ntasks.tod if tot_num_CompSep_ranks > 0 else None

    world_comm.barrier()
    time.sleep(worldrank*1e-5)  # Small sleep to get prints in nice order.

    mpi_info['world'] = Bunch()
    mpi_info['world']['comm'] = world_comm
    mpi_info['world']['master'] = 0
    mpi_info['world']['size'] = worldsize
    mpi_info['world']['rank'] = worldrank
    mpi_info['world']['side'] = side
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
        
        # Split between I and QU. Numeric colors remain private to MPI setup; scientific code sees
        # the named polarization below.
        polarization = enabled_compsep_views(params)[proc_rank].polarization
        subcolor = 0 if polarization == "I" else 1
        sub_comm = proc_comm.Split(subcolor, key=proc_rank)
        mpi_info['compsep']['subcomm'] = sub_comm
        mpi_info['compsep']['polarization'] = polarization
        mpi_info['compsep']['subsize'] = sub_comm.Get_size()
        mpi_info['compsep']['subrank'] = sub_comm.Get_rank()
        mpi_info['compsep']['I_master'] = 0  # in compsep_comm numbering
        mpi_info['compsep']['QU_master'] = mpi_info.compsep.size \
            - ntasks.compsep_QU   # in compsep_comm numbering
        mpi_info['compsep']['is_I_master'] = subcolor == 0 and mpi_info.compsep.subrank == 0
        mpi_info['compsep']['is_QU_master'] = subcolor == 1 and mpi_info.compsep.subrank == 0
        mpi_info = init_mpi_compsep(mpi_info, params)
    return mpi_info
    

def init_mpi_tod(mpi_info: Bunch, params: Bunch) -> Bunch:
    """Assign this TOD rank from the shared band inventory and create its band communicator."""
    tod_rank = mpi_info.tod.rank
    tod_bands = enabled_tod_bands(params)
    my_band = next(
        (band for band in tod_bands if band.rank_start <= tod_rank < band.rank_stop),
        None,
    )
    if my_band is None:
        raise RuntimeError(f"TOD rank {tod_rank} was not assigned to an enabled band.")
    if tod_bands[-1].rank_stop != mpi_info.tod.size:
        raise RuntimeError(
            f"Enabled TOD bands require {tod_bands[-1].rank_stop} ranks, but the TOD side has "
            f"{mpi_info.tod.size}."
        )

    # The global inventory index is the communicator color. Unlike the old per-experiment index,
    # it cannot merge bands from different experiments.
    band_comm = mpi_info.tod.comm.Split(my_band.index, key=tod_rank)
    band_rank = band_comm.Get_rank()
    band_size = band_comm.Get_size()

    if mpi_info.tod.is_master:
        logger.verbose(f"TOD: {mpi_info.tod.size} tasks allocated across {len(tod_bands)} bands.")
    logger.debug(
        f"TOD rank {tod_rank} (on {MPI.Get_processor_name()}) handles "
        f"{my_band.experiment_name}/{my_band.band_name}, band rank {band_rank}/{band_size}."
    )

    mpi_info.experiment = Bunch(name=my_band.experiment_name)
    mpi_info.band = Bunch(
        index=my_band.index,
        name=my_band.band_name,
        master=0,
        comm=band_comm,
        size=band_size,
        rank=band_rank,
        is_master=band_rank == 0,
    )
    return mpi_info


def init_mpi_compsep(mpi_info: Bunch, params: Bunch) -> Bunch:
    """Attach the CompSep view assigned to this rank from the shared ordered inventory."""
    views = enabled_compsep_views(params)
    if len(views) != mpi_info.compsep.size:
        raise RuntimeError(
            f"Enabled CompSep views require {len(views)} ranks, but the CompSep side has "
            f"{mpi_info.compsep.size}."
        )
    view = views[mpi_info.compsep.rank]
    mpi_info.band = Bunch(
        name=view.band_name,
        identifier=view.identifier,
        polarization=view.polarization,
        comm=MPI.COMM_SELF,
        size=1,
        rank=0,
        master=0,
        is_master=True,
    )
    return mpi_info
