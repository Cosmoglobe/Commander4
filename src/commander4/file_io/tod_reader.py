"""Read one band's TOD: choose this rank's scans, run the experiment's reader, finish the band.

Every experiment reader is `TODReader` (``file_io/experiments/base_reader.py``) or a subclass of it,
in the module of ``file_io/experiments/`` named after the ``experiment_id`` it is registered under
below. `TODReader` reads the standard Commander scan-file format. A subclass passes its experiment's
fixed values to ``TODReader.__init__``, and overrides one of the reader's steps only if its files
need logic that the base class lacks. They all return the same `DetectorGroupTOD`, so nothing
downstream needs to know which one ran.

Add a new experiment by writing such a subclass and registering it in ``experiment_tod_readers``.
"""
from mpi4py import MPI
from numpy.typing import NDArray
from pixell.bunch import Bunch
import healpy as hp
import logging
import numpy as np

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.parameters.schema import resolve_param
from commander4.simulations.inplace_litebird_sim import replace_tod_with_sim
from commander4.file_io.experiments.base_reader import TODReader
from commander4.file_io.experiments.akari import AkariReader
from commander4.file_io.experiments.litebird_sim_spawndetectors import SpawnDetectorsReader
from commander4.file_io.experiments.planck_lfi import PlanckLFIReader
from commander4.file_io.experiments.planck_hfi import PlanckHFIReader
from commander4.file_io.experiments.SO_LAT import SOLATReader
from commander4.file_io.experiments.SO_SAT import SOSATReader

logger = logging.getLogger(__name__)

# Known experiments and the reader class each one uses. The parameter file's `experiment_id`
# selects the entry, and every key matches the module name its class is imported from.
experiment_tod_readers = {
    "akari" : AkariReader,
    # The standard format with no instrument-specific values. Everything `simgen` writes reads
    # through this, as does any other dataset already in that layout.
    "general" : TODReader,
    "litebird_sim_spawndetectors" : SpawnDetectorsReader,
    "planck_lfi" : PlanckLFIReader,
    "planck_hfi" : PlanckHFIReader,
    "SO_LAT" : SOLATReader,
    "SO_SAT" : SOSATReader,
}


def split_by_sky_position(vectors: NDArray, weights: NDArray, nparts: int) -> NDArray:
    """Assign each scan to one of ``nparts`` ranks: equal total weight per rank, with each rank's
    scans close together on the sky, so that they overlap and the rank's maps hold few pixels.

    Recursive bisection: sort the scans along the direction in which their sky positions spread
    most, and cut where the cumulative weight reaches the left side's share of the ranks. Then split
    each side the same way, until there is one part per rank. The sort keeps the order of scans at
    the same position, so a filelist without positions gives contiguous blocks of scans in time.

    Args:
        vectors: (nscans, 3) unit vectors of the scans' sky positions, in time order.
        weights: (nscans,) cost of each scan, e.g. its measured run time or number of samples.
        nparts: The number of ranks.

    Returns:
        (nscans,) the rank of each scan. With more ranks than scans, some ranks get none.
    """
    rank = np.zeros(len(weights), dtype=np.int64)

    def split(indices: NDArray, first_rank: int, nranks: int) -> None:
        """Give the scans ``indices`` to the ``nranks`` ranks from ``first_rank`` on."""
        if nranks == 1 or len(indices) < 2:
            rank[indices] = first_rank
            return
        # The direction of largest spread: the eigenvector of the positions' 3x3 covariance with the
        # largest eigenvalue (eigh sorts them in ascending order).
        direction = np.linalg.eigh(np.cov(vectors[indices].T))[1][:, -1]
        indices = indices[np.argsort(vectors[indices] @ direction, kind="stable")]
        nleft = nranks // 2
        # A scan goes left if more than half of its weight lies before the cut.
        cut = np.searchsorted(np.cumsum(weights[indices]) - weights[indices]/2,
                              weights[indices].sum()*nleft/nranks)
        split(indices[:cut], first_rank, nleft)
        split(indices[cut:], first_rank + nleft, nranks - nleft)

    split(np.arange(len(weights)), 0, nparts)
    return rank


def read_tods_from_file(band_comm: MPI.Comm, params: Bunch, my_experiment: Bunch,
                        my_band: Bunch) -> DetectorGroupTOD:
    """Read this rank's share of one band's scans.

    The band master reads the filelist and sorts its rows by scan ID, which is time order. It keeps
    the scans with IDs from ``scan_id_start`` (included) to ``scan_id_stop`` (excluded), and drops
    the scans listed in ``bad_PIDs_path`` and those the filelist marks as not to be read (0 in its
    7th column). It splits the remaining scans between the ranks by sky position and weight (see
    `split_by_sky_position`), where ``scan_weights`` picks the weight: the filelist's measured run
    time ("runtime", the default) or number of samples ("ntod"). Each rank reads its share with the
    experiment's reader. If the experiment sets ``replace_tod_with_sim``, the TOD is then replaced
    by a simulation.

    Args:
        band_comm: The band's MPI communicator; each rank reads a disjoint set of scans.
        params, my_experiment, my_band: The full parameter file, and this band's experiment and
            band blocks.

    Returns:
        This rank's `DetectorGroupTOD`. The reader may drop scans (empty or bad data), so it can
        hold fewer scans than the rank was given.
    """
    if my_experiment.experiment_id not in experiment_tod_readers:
        raise ValueError("An experiment in the parameter file has experiment_id = "\
                f"{my_experiment.experiment_id}, which is not in {experiment_tod_readers.keys()}. "\
                "You either misspelled the experiment ID, or your experiment does not yet have a "\
                "specified TOD reader. See this file for how to add it.")
    band_name = my_band._name
    rank = band_comm.Get_rank()
    scopes = (f"experiments.{my_experiment._name}.bands.{band_name}",
              f"experiments.{my_experiment._name}")
    scan_id_start = resolve_param(params, "scan_id_start", scopes, default=None,
                                  legal_types=(int, type(None)))
    scan_id_stop = resolve_param(params, "scan_id_stop", scopes, default=None,
                                 legal_types=(int, type(None)))
    scan_weights = resolve_param(params, "scan_weights", scopes, default="runtime",
                                 legal_values=("runtime", "ntod"))

    # The band master reads the filelist and selects the scans; the other ranks get them from it,
    # each as (scan ID, file path). Scan IDs increase with time, but the rows of a filelist need
    # not (the LFI and HFI filelists are not), so the scans are selected and ordered by scan ID.
    scans: list[tuple[int, str]] | None = None
    if rank == 0:
        with open(my_band.filelist) as infile:
            infile.readline()  # The first line holds the number of scans.
            # Each row is a scan ID, a quoted file path, and optional columns (see c4-scan-weights).
            rows = [line.split() for line in infile if line.strip()]
        rows.sort(key=lambda columns: int(columns[0]))
        in_range = [columns for columns in rows
                    if (scan_id_start is None or int(columns[0]) >= scan_id_start)
                    and (scan_id_stop is None or int(columns[0]) < scan_id_stop)]
        skipped_scan_ids = set()
        if "bad_PIDs_path" in my_experiment:
            skipped_scan_ids = {int(scan_id) for scan_id in np.load(my_experiment.bad_PIDs_path)}
        # Column 7 is 0 for scans that an earlier run found to hold no usable data: nothing in the
        # file, or every detector flagged or cut at read time.
        skipped_scan_ids |= {int(columns[0]) for columns in rows
                             if len(columns) > 6 and columns[6] == "0"}
        selected = [columns for columns in in_range if int(columns[0]) not in skipped_scan_ids]
        scans = [(int(columns[0]), columns[1].strip('"')) for columns in selected]
        logger.info(f"Band {band_name}: {len(in_range)} of the filelist's {len(rows)} scans have "
                    f"IDs in [{scan_id_start}, {scan_id_stop}), of which "
                    f"{len(in_range) - len(scans)} are listed as bad or marked in the filelist as "
                    "holding no usable data.")
    scans = band_comm.bcast(scans, root=0)
    if len(scans) == 0:
        raise ValueError(f"Band {band_name}: scan IDs [{scan_id_start}, {scan_id_stop}) select no "
                         "usable scans.")

    # The band master splits the scans between the ranks. A rank's scans need not be contiguous in
    # time; every step that needs time order sorts by scan ID.
    scan_ranks: NDArray | None = None
    if rank == 0:
        # A scan's weight is its measured run time (column 3) or its number of samples (column 6),
        # and columns 4-5 hold its sky position (theta, phi). Without the run time and position
        # columns, every scan has weight 1 and the same position, which gives equal blocks of
        # scans in time. c4-scan-weights writes all of them, and -1 for an ntod it does not know.
        if scan_weights == "ntod":
            weights = np.array([float(columns[5]) if len(columns) > 5 else -1.0
                                for columns in selected])
            if (weights <= 0).any():
                raise ValueError(f"Band {band_name}: scan_weights is 'ntod', but the filelist has no "
                                 f"ntod (column 6) for {(weights <= 0).sum()} of the selected scans.")
        else:
            weights = np.array([float(columns[2]) if len(columns) > 2 else 1.0
                                for columns in selected])
        theta = np.array([float(columns[3]) if len(columns) > 4 else 0.0 for columns in selected])
        phi = np.array([float(columns[4]) if len(columns) > 4 else 0.0 for columns in selected])
        nranks = band_comm.Get_size()
        scan_ranks = split_by_sky_position(np.array(hp.ang2vec(theta, phi)), weights, nranks)
        scans_per_rank = np.bincount(scan_ranks, minlength=nranks)
        weight_per_rank = np.bincount(scan_ranks, weights, minlength=nranks)
        logger.info(f"Band {band_name}: split {len(scans)} scans between {nranks} ranks by sky "
                    f"position and {scan_weights}: {scans_per_rank.min()} to "
                    f"{scans_per_rank.max()} scans per rank, largest total {scan_weights} "
                    f"{weight_per_rank.max()/weight_per_rank.mean():.2f} x the mean.")
    scan_ranks = band_comm.bcast(scan_ranks, root=0)
    my_scans = [scan for scan, scan_rank in zip(scans, scan_ranks) if scan_rank == rank]
    reader = experiment_tod_readers[my_experiment.experiment_id](
        band_comm, params, my_experiment, my_band)
    experiment_data = reader.read(my_scans)
    # The scans the reader kept nothing from, for the chain (see `DetectorGroupTOD`).
    kept = {scan.scan_id for scan in experiment_data.scans}
    experiment_data.dropped_scan_ids = np.array(
        [scan_id for scan_id, _ in my_scans if scan_id not in kept], dtype=np.int64)

    if reader.tod_is_simulated:
        replace_tod_with_sim(band_comm, experiment_data, my_band, params, my_experiment.sim_params)

    # Summarize what survived the reader's cuts and the Fourier cut, over the whole band.
    detectors = [det for scan in experiment_data.scans for det in scan.detectors]
    local_stats = np.array([experiment_data.nscans, len(detectors),
                            sum(det.ntod for det in detectors),
                            sum(det.ntod_original for det in detectors)], dtype=np.int64)
    global_stats = np.zeros_like(local_stats)
    band_comm.Reduce(local_stats, global_stats, op=MPI.SUM, root=0)
    if rank == 0:
        nscans_kept, ndetscans_kept, ntod_kept, ntod_file = global_stats
        logger.info(f"Band {band_name}: read {nscans_kept} of {len(scans)} scans and "
                    f"{ndetscans_kept} of {len(scans)*len(reader.det_names)} detector-scans, with "
                    f"{100*ntod_kept/max(ntod_file, 1):.1f}% of their samples retained after the "
                    "Fourier cut.")
    return experiment_data
