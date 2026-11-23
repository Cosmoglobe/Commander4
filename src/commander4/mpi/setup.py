"""Split MPI ranks between the TOD and CompSep sides and the enabled bands."""
import itertools
import os
import logging
from dataclasses import dataclass

import mpi4py
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.diagnostics.performance import benchmark
from commander4.parameters.schema import (
    compsep_enabled,
    enabled_compsep_polbands,
    enabled_tod_bands,
    has_enabled_sampling_group,
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
    compsep_receivers: dict[str, int]    # World rank of each CompSep polband of the band, by pol.


@dataclass(frozen=True)
class BandInfo:
    """The band communicator: the TOD ranks of this rank's band."""
    comm: MPI.Comm
    rank: int
    size: int
    node_comm: MPI.Comm                  # The band's ranks on this physical node.


@dataclass(frozen=True)
class CompSepInfo:
    """The CompSep communicator: every CompSep rank. Also the polband this rank works on, and its
    partners."""
    comm: MPI.Comm
    rank: int
    size: int
    num_threads: int
    subcomm: MPI.Comm                    # The CompSep ranks of this rank's polarization.
    band_name: str                       # The polband this rank works on: its band,
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


@benchmark("init-mpi")
def init_mpi(params: Bunch) -> MPIInfo:
    """Split the ranks between the TOD and CompSep sides and the bands, and build `MPIInfo`.

    Also checks the task count, and sets this rank's thread counts in the environment, which must
    happen before NumPy is imported. Rank 0 of a communicator is its master.
    """
    world_comm = MPI.COMM_WORLD
    world_size, world_rank = world_comm.Get_size(), world_comm.Get_rank()
    # The task counts follow from the band configuration: CompSep takes one task per polband (a
    # band in I or in QU; an IQU band gives two), and every other task is a TOD task. The TOD tasks
    # are split between the bands below.
    tod_bands = enabled_tod_bands(params)
    compsep_polbands = enabled_compsep_polbands(params)
    ntask_compsep = len(compsep_polbands)
    ntask_tod = world_size - ntask_compsep  # The remaining MPI ranks are all TOD ranks.
    if world_rank == 0:
        mpi4py_version = tuple(map(int, mpi4py.__version__.split('.')))
        MPI_version = MPI.Get_version()
        logger.info(f"MPI version: {MPI_version}. mpi4py version: {mpi4py_version}.")
        logger.summary(f"MPI task layout: {ntask_tod} TOD + {ntask_compsep} CompSep.")
        if MPI_version < (4,0):
            logger.warning(f"MPI version ({MPI_version}) is below (4,0)!")
        if mpi4py_version < (4,0):
            logger.warning(f"mpi4py version ({mpi4py_version}) is below (4,0)!")

        # Check if compsep.enabled is False (if the compsep group exist at all).
        turned_off = "compsep" in params and "enabled" in params.compsep \
            and not params.compsep.enabled
        # If compsep is turned off, there can't be sampling groups enabled.
        if has_enabled_sampling_group(params) and turned_off:
            raise RuntimeError("Enabled compsep sampling groups are configured, but "
                               "'compsep.enabled' is false, so no CompSep MPI ranks are "
                               "allocated to run them. Enable compsep, or disable the groups.")
        if world_size < len(tod_bands) + ntask_compsep or (not tod_bands and ntask_tod > 0):
            raise RuntimeError(f"This run needs {'at least ' if tod_bands else ''}"
                               f"{len(tod_bands) + ntask_compsep} = {len(tod_bands)} TOD + "
                               f"{ntask_compsep} CompSep MPI tasks, but was started with "
                               f"{world_size}. The counts follow from the parameter file: one "
                               "CompSep task per enabled 'compsep.bands' entry and polarization "
                               "(I, QU), and at least one TOD task per enabled band.")
        # Otherwise CompSep is simply off (TOD-only) and compsep.bands is ignored.
        if not compsep_enabled(params):
            reason = ("compsep.enabled is false" if turned_off
                      else "no compsep sampling group is enabled, so compsep ranks would have "
                           "nothing to sample")
            logger.info(f"Running TOD-only: {reason}. Any 'compsep.bands' are ignored, and TOD "
                        "ranks use the initial sky model built from the components.")

    # The first ntask_tod world ranks are TOD ranks, the rest CompSep ranks.
    if world_rank < ntask_tod:
        side = "tod"
        num_threads = params.resources.tod.num_threads
        num_threads_numba = num_threads
    else:
        side = "compsep"
        # num_threads is either an int, or a list specifying nthreads for each rank.
        nthreads_compsep = params.resources.compsep.num_threads
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

    # Figure out where each TOD-band runs. The TOD ranks are world ranks 0 to ntask_tod-1, and each
    # TOD band has a contiguous block of them, in the order of `enabled_tod_bands`. Number of ranks
    # allocated to each band is distributed accoring to their "weight".
    # CompSep polband i is always world rank ntask_tod + i.
    rank_counts = []
    if world_rank == 0 and tod_bands:
        # This import must come after we set Numba threads above, because it imports Numba.
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
    # A dict mapping bands to their world masters (e.g. {"LFI_30GHZ" : 0, "LFI_40GHz" : 63})
    tod_masters = {band.band_name: start for band, start in zip(tod_bands, band_starts)}

    # Set up a dict giving the MPI world-rank of each compsep polband. It could look like:
    # compsep_ranks = {"LFI_30GHz": {"I": 65, "QU": 67}, "LFI_40GHz": {"I": 66, "QU": 68}}
    compsep_ranks = {polband.band_name: {} for polband in compsep_polbands}
    for rank, polband in enumerate(compsep_polbands):
        compsep_ranks[polband.band_name][polband.pol] = ntask_tod + rank

    # Construct the MPIInfo dataclass, which holds everything we need to know about the MPI setup.
    world = WorldInfo(comm=world_comm, rank=world_rank, size=world_size, side=side,
                      has_tod=ntask_tod > 0, has_compsep=ntask_compsep > 0)
    if side == "tod":  # Run the TOD-specific MPI setup.
        tod, band = init_mpi_tod(params, side_comm, num_threads, rank_counts, compsep_ranks)
        return MPIInfo(world=world, tod=tod, band=band)
    else:  # Run the CompSep-specific MPI setup.
        compsep = init_mpi_compsep(params, side_comm, num_threads, tod_masters, compsep_ranks)
        return MPIInfo(world=world, compsep=compsep)


def init_mpi_tod(params: Bunch, tod_comm: MPI.Comm, num_threads: int, rank_counts: list[int],
                 compsep_ranks: dict[str, dict[str, int]]) -> tuple[TODInfo, BandInfo]:
    """Setting up the TOD and band communicators."""
    tod_rank = tod_comm.Get_rank()
    tod_bands = enabled_tod_bands(params)
    # The band of each TOD rank, e.g. [0, 0, 0, 1, 1] for rank_counts [3, 2].
    band_of_rank = []
    for iband, count in enumerate(rank_counts):
        band_of_rank += [iband]*count
    iband = band_of_rank[tod_rank]
    my_band = tod_bands[iband]

    # Create MPI communicator for all ranks on the same TOD band.
    band_comm = tod_comm.Split(iband, key=tod_rank)

    # Create communicator for all ranks on the same TOD band that are on the same physical node.
    node_comm = band_comm.Split_type(MPI.COMM_TYPE_SHARED)

    logger.debug(
        f"TOD rank {tod_rank} (on {MPI.Get_processor_name()}) handles "
        f"{my_band.experiment_name}/{my_band.band_name}, band rank "
        f"{band_comm.Get_rank()}/{band_comm.Get_size()}, node rank "
        f"{node_comm.Get_rank()}/{node_comm.Get_size()}."
    )

    # Each CompSep polband of the band takes the band's map in its polarization. The I polband sends
    # the band its sky, or the QU polband if the band has no I polband.
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
    """Setting up the CompSep communicator. One rank per band+polarization combo."""
    polbands = enabled_compsep_polbands(params)
    rank = compsep_comm.Get_rank()
    if len(polbands) != compsep_comm.Get_size():
        raise RuntimeError(f"The enabled CompSep polbands require {len(polbands)} ranks, but the "
                           f"CompSep side has {compsep_comm.Get_size()}.")
    polband = polbands[rank]
    # A band read from file has no TOD side. A band's I polband sends its sky to the TOD side, or
    # its QU polband if the band has no I polband.
    get_from = params.compsep.bands[polband.band_name].get_from
    if get_from != "file" and polband.band_name not in tod_masters:
        raise RuntimeError(f"'compsep.bands.{polband.band_name}' takes its map from {get_from!r}, "
                           f"but that experiment has no enabled TOD band {polband.band_name!r}.")
    tod_sender = None if get_from == "file" else tod_masters[polband.band_name]
    sends_sky = polband.pol == "I" or "I" not in compsep_ranks[polband.band_name]
    # `subcomm` holds all I polbands, or all QU polbands; the solver communicators of the
    # amplitude groups are split from it.
    return CompSepInfo(comm=compsep_comm, rank=rank, size=compsep_comm.Get_size(),
                       num_threads=num_threads,
                       subcomm=compsep_comm.Split(0 if polband.pol == "I" else 1, key=rank),
                       band_name=polband.band_name, pol=polband.pol, tod_sender=tod_sender,
                       tod_receiver=tod_sender if sends_sky else None)


def split_tod_ranks(weights: list[float], ntask_tod: int) -> list[int]:
    """The number of TOD ranks of each band, in proportion to the bands' weights.

    Every band gets one rank, and each further rank goes to the band that has the most weight per
    rank at that point. This gives the smallest possible largest weight per rank.
    Whole-number weights that sum to ``ntask_tod`` give exactly those numbers.
    """
    counts = [1]*len(weights)
    for _ in range(ntask_tod - len(weights)):
        busiest = max(range(len(weights)), key=lambda iband: weights[iband]/counts[iband])
        counts[busiest] += 1
    return counts
