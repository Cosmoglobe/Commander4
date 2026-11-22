"""Split MPI ranks between the TOD and CompSep sides and the enabled bands."""
import itertools
import os
import logging
from dataclasses import dataclass

import mpi4py
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.parameters.schema import (
    compsep_enabled,
    derive_task_counts,
    enabled_compsep_views,
    enabled_tod_bands,
    has_enabled_sampling_group,
    split_tod_ranks,
    task_count_breakdown,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorldInfo:
    """The world communicator: every rank of the run."""
    comm: MPI.Comm
    rank: int
    size: int
    side: str                            # "tod" or "compsep".
    has_tod: bool                        # Whether the run has a TOD side,
    has_compsep: bool                    # and a CompSep side.


@dataclass(frozen=True)
class TODInfo:
    """The TOD communicator: every TOD rank. Also the band this rank works on, and its partners."""
    comm: MPI.Comm
    rank: int
    size: int
    num_threads: int
    experiment_name: str                 # The band this rank works on.
    band_name: str
    compsep_sender: int | None           # World rank that sends the band its sky.
    compsep_receivers: dict[str, int]    # World rank of each CompSep view of the band, by pol.


@dataclass(frozen=True)
class BandInfo:
    """The band communicator: the TOD ranks of this rank's band."""
    comm: MPI.Comm
    rank: int
    size: int
    node_comm: MPI.Comm                  # The band's ranks on this physical node.


@dataclass(frozen=True)
class CompSepInfo:
    """The CompSep communicator: every CompSep rank. Also the band this rank works on, and its
    partners."""
    comm: MPI.Comm
    rank: int
    size: int
    num_threads: int
    subcomm: MPI.Comm                    # The CompSep ranks of this rank's polarization.
    band_name: str                       # The band this rank works on,
    pol: str                             # and its polarization, "I" or "QU".
    tod_sender: int | None               # World rank of the band's TOD master; None if none.
    tod_receiver: int | None             # Where this rank sends the sky; None if it does not.


@dataclass(frozen=True)
class MPIInfo:
    """Where this rank sits in the run: one block per communicator it belongs to."""
    world: WorldInfo
    tod: TODInfo | None = None           # On a TOD rank.
    band: BandInfo | None = None         # On a TOD rank.
    compsep: CompSepInfo | None = None   # On a CompSep rank.


def init_mpi(params: Bunch) -> MPIInfo:
    """Split the ranks between the TOD and CompSep sides and the bands, and build `MPIInfo`.

    Also checks the task count, and sets this rank's thread counts in the environment, which must
    happen before NumPy is imported. Rank 0 of a communicator is its master.
    """
    world_comm = MPI.COMM_WORLD
    world_size, world_rank = world_comm.Get_size(), world_comm.Get_rank()
    # The task counts are derived from the band configuration rather than stated in the parameter
    # file: CompSep takes one task per enabled band view (one for I, one for QU), and every other
    # task is a TOD task. The TOD tasks are split between the bands below.
    ntasks = derive_task_counts(params)
    ntask_compsep = ntasks.compsep_I + ntasks.compsep_QU
    ntask_tod = world_size - ntask_compsep
    if world_rank == 0:  # One report and one error, not one per rank.
        mpi4py_version = tuple(map(int, mpi4py.__version__.split('.')))
        MPI_version = MPI.Get_version()
        logger.info(f"MPI version: {MPI_version}. mpi4py version: {mpi4py_version}.")
        logger.summary(f"MPI task layout: {ntask_tod} TOD + {ntasks.compsep_I} CompSep-I + "
                       f"{ntasks.compsep_QU} CompSep-QU.")
        if MPI_version < (4,0):
            logger.warning(f"MPI version ({MPI_version}) is below (4,0)!")
        if mpi4py_version < (4,0):
            logger.warning(f"mpi4py version ({mpi4py_version}) is below (4,0)!")

        # Check if compsep.enabled is False (if the compsep group exist at all).
        turned_off = "compsep" in params and "enabled" in params.compsep \
            and not params.compsep.enabled
        # If compsep is turned off, there can't be sampling groups enabled, that's be an error.
        if has_enabled_sampling_group(params) and turned_off:
            raise RuntimeError("Enabled compsep sampling groups are configured, but "
                               "'compsep.enabled' is false, so no CompSep MPI ranks are "
                               "allocated to run them. Enable compsep, or disable the groups.")
        if world_size < ntasks.min_total or (ntasks.tod_bands == 0 and ntask_tod > 0):
            raise RuntimeError(f"This run needs {task_count_breakdown(ntasks)} MPI tasks, "
                               f"but was started with {world_size}. The counts follow from the "
                               "parameter file: one CompSep task per enabled 'compsep.bands' view, "
                               "and at least one TOD task per enabled band.")
        # Otherwise CompSep is simply off (TOD-only) and compsep.bands is ignored.
        if not compsep_enabled(params):
            reason = ("compsep.enabled is false" if turned_off
                      else "no compsep sampling group is enabled, so compsep ranks would have "
                           "nothing to sample")
            logger.info(f"Running TOD-only: {reason}. Any 'compsep.bands' are ignored, and TOD "
                        "ranks use the initial sky model built from the components.")

    # The first ntask_tod world ranks are TOD ranks, the rest CompSep ranks.
    nthreads_compsep = params.resources.compsep.num_threads
    if world_rank < ntask_tod:
        side = "tod"
        num_threads = params.resources.tod.num_threads
        num_threads_numba = num_threads
    else:
        side = "compsep"
        # num_threads is either an int, or a list specifying nthreads for each rank.
        if isinstance(nthreads_compsep, int):  # If int, all ranks have same nthreads.
            num_threads = nthreads_compsep
        else:
            if len(nthreads_compsep) != ntask_compsep:
                raise ValueError(
                    f"Length of resources.compsep.num_threads ({len(nthreads_compsep)}) does not "
                    f"match the number of CompSep ranks ({ntask_compsep}).")
            num_threads = nthreads_compsep[world_rank - ntask_tod]
        # Testing revealed 24 to be a good number (regardless of nside), but I tested this on the
        # new 384-core nodes, the optimal number is probably slightly lower on the older owls.
        num_threads_numba = min(24, num_threads)

    # It's important to set these environment variables before importing any package that might
    # use them, such as Numpy or Scipy, as they will not apply retroactively!
    os.environ["OMP_NUM_THREADS"] = f"{num_threads}"
    os.environ["OPENBLAS_NUM_THREADS"] = f"{num_threads}"
    os.environ["MKL_NUM_THREADS"] = f"{num_threads}"
    os.environ["VECLIB_MAXIMUM_THREADS"] = f"{num_threads}"
    os.environ["NUMEXPR_NUM_THREADS"] = f"{num_threads}"
    os.environ["NUMBA_NUM_THREADS"] = f"{num_threads_numba}"
    # I tried using numba.set_num_threads(x) here instead (or as well) but that
    # resulted in some weirdeties, like many duplicate open file handles even when x=1.

    if False: # This code should enter production, but threadpoolctl is not yet a dependency.
        import numba
        from threadpoolctl import threadpool_info

        pool_info = threadpool_info()
        for pool in pool_info:
            if pool["num_threads"] != num_threads:
                raise RuntimeError(f"Loaded library {pool} has {pool['num_threads']} threads, "
                                   f"expected {num_threads}.")
        if numba.get_num_threads() != num_threads_numba:
            raise RuntimeError(f"Numba has {numba.get_num_threads()} threads, expected "
                               f"{num_threads_numba}.")


    # The side communicator. The key keeps the world order, so TOD rank i is world rank i.
    side_comm = world_comm.Split(0 if side == "tod" else 1, key=world_rank)
    logger.debug(f"MPI split performed, hi from world rank {world_rank} (on machine "
                 f"{MPI.Get_processor_name()}), {side} rank {side_comm.Get_rank()} of "
                 f"{side_comm.Get_size()}. Threads = {num_threads}.")

    # Where each band runs. The TOD ranks are world ranks 0 to ntask_tod-1, and each TOD band has a
    # contiguous block of them, in the order of `enabled_tod_bands`. World rank 0 weighs the bands
    # (see `tod_band_weights`) and splits the ranks by these weights (see `split_tod_ranks`).
    # CompSep view i is world rank ntask_tod + i.
    tod_bands = enabled_tod_bands(params)
    rank_counts = []
    if world_rank == 0 and tod_bands:
        # Imported here: the readers import numba, which has to come after the thread counts are
        # set above.
        from commander4.file_io.tod_reader import tod_band_weights
        weights = tod_band_weights(params, tod_bands)
        rank_counts = split_tod_ranks(weights, ntask_tod)
        mean_weight_per_rank = sum(weights)/ntask_tod
        logger.info(f"TOD: split {ntask_tod} ranks between {len(tod_bands)} bands: "
                    + ", ".join(f"{band.band_name} {count}"
                                for band, count in zip(tod_bands, rank_counts))
                    + ". The band with the most weight per rank has "
                    f"{max(w/n for w, n in zip(weights, rank_counts))/mean_weight_per_rank:.2f} x "
                    "the mean.")
    rank_counts = world_comm.bcast(rank_counts, root=0)
    # The world rank of each TOD band's master, by band name.
    band_starts = [0, *itertools.accumulate(rank_counts)]
    tod_masters = {band.band_name: start for band, start in zip(tod_bands, band_starts)}
    # The world rank of each CompSep view, by band name and then polarization.
    compsep_ranks = {}
    for view in enabled_compsep_views(params):
        if view.band_name not in compsep_ranks:
            compsep_ranks[view.band_name] = {}
        compsep_ranks[view.band_name][view.polarization] = ntask_tod + view.rank

    world = WorldInfo(comm=world_comm, rank=world_rank, size=world_size, side=side,
                      has_tod=ntask_tod > 0, has_compsep=ntask_compsep > 0)
    if side == "tod":
        tod, band = init_mpi_tod(params, side_comm, num_threads, rank_counts, compsep_ranks)
        return MPIInfo(world=world, tod=tod, band=band)
    compsep = init_mpi_compsep(params, side_comm, num_threads, tod_masters, compsep_ranks)
    return MPIInfo(world=world, compsep=compsep)


def init_mpi_tod(params: Bunch, tod_comm: MPI.Comm, num_threads: int, rank_counts: list[int],
                 compsep_ranks: dict[str, dict[str, int]]) -> tuple[TODInfo, BandInfo]:
    """The TOD and band communicators of a TOD rank (see `init_mpi`).

    Args:
        rank_counts: The number of ranks of each band of `enabled_tod_bands`.
        compsep_ranks: The world rank of each CompSep view, by band name and polarization.
    """
    tod_rank = tod_comm.Get_rank()
    tod_bands = enabled_tod_bands(params)
    rank_stops = list(itertools.accumulate(rank_counts))
    my_band = tod_bands[next(iband for iband, stop in enumerate(rank_stops) if tod_rank < stop)]

    # The global inventory index is the communicator color. Unlike the old per-experiment index,
    # it cannot merge bands from different experiments.
    band_comm = tod_comm.Split(my_band.index, key=tod_rank)

    # The band's ranks that sit on one physical node, for data identical across the band that is
    # too large to hold per rank (the far-sidelobe cubes). Splitting the band rather than the world
    # matters: a node can host ranks of two different bands, and they need different data.
    node_comm = band_comm.Split_type(MPI.COMM_TYPE_SHARED)

    logger.debug(
        f"TOD rank {tod_rank} (on {MPI.Get_processor_name()}) handles "
        f"{my_band.experiment_name}/{my_band.band_name}, band rank "
        f"{band_comm.Get_rank()}/{band_comm.Get_size()}, node rank "
        f"{node_comm.Get_rank()}/{node_comm.Get_size()}."
    )

    # Each CompSep view of the band takes the band's map in its polarization. The I view sends the
    # band its sky, or the QU view if the band has no I view.
    compsep_receivers = compsep_ranks.get(my_band.band_name, {})
    if compsep_ranks and not compsep_receivers:
        raise RuntimeError(f"TOD band {my_band.band_name!r} has no enabled entry in "
                           "'compsep.bands', so no CompSep rank would send it a sky model.")
    compsep_sender = compsep_receivers["I"] if "I" in compsep_receivers \
        else compsep_receivers.get("QU")

    tod = TODInfo(comm=tod_comm, rank=tod_rank, size=tod_comm.Get_size(), num_threads=num_threads,
                  experiment_name=my_band.experiment_name, band_name=my_band.band_name,
                  compsep_sender=compsep_sender, compsep_receivers=compsep_receivers)
    band = BandInfo(comm=band_comm, rank=band_comm.Get_rank(), size=band_comm.Get_size(),
                    node_comm=node_comm)
    return tod, band


def init_mpi_compsep(params: Bunch, compsep_comm: MPI.Comm, num_threads: int,
                     tod_masters: dict[str, int],
                     compsep_ranks: dict[str, dict[str, int]]) -> CompSepInfo:
    """The CompSep communicator of a CompSep rank (see `init_mpi`).

    CompSep rank i works on view i of `enabled_compsep_views`: one band in one polarization.

    Args:
        tod_masters: The world rank of each TOD band's master, by band name.
        compsep_ranks: The world rank of each CompSep view, by band name and polarization.
    """
    views = enabled_compsep_views(params)
    rank = compsep_comm.Get_rank()
    if len(views) != compsep_comm.Get_size():
        raise RuntimeError(f"Enabled CompSep views require {len(views)} ranks, but the CompSep "
                           f"side has {compsep_comm.Get_size()}.")
    view = views[rank]
    # A band read from file has no TOD side. A band's I view sends its sky to the TOD side, or its
    # QU view if the band has no I view.
    get_from = params.compsep.bands[view.band_name].get_from
    if get_from != "file" and view.band_name not in tod_masters:
        raise RuntimeError(f"'compsep.bands.{view.band_name}' takes its map from {get_from!r}, "
                           f"but that experiment has no enabled TOD band {view.band_name!r}.")
    tod_sender = None if get_from == "file" else tod_masters[view.band_name]
    sends_sky = view.polarization == "I" or "I" not in compsep_ranks[view.band_name]
    # `subcomm` holds all I views, or all QU views; the solver communicators of the amplitude
    # groups are split from it.
    return CompSepInfo(comm=compsep_comm, rank=rank, size=compsep_comm.Get_size(),
                       num_threads=num_threads,
                       subcomm=compsep_comm.Split(0 if view.polarization == "I" else 1, key=rank),
                       band_name=view.band_name, pol=view.polarization, tod_sender=tod_sender,
                       tod_receiver=tod_sender if sends_sky else None)
