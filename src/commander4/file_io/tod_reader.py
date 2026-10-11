"""Read one band's TOD: choose this rank's scans, run the experiment's reader, finish the band.

Every experiment reader is `TODReader` (``file_io/experiments/base_reader.py``) or a subclass of it,
in the module of ``file_io/experiments/`` named after the ``experiment_id`` it is registered under
below. `TODReader` reads the standard Commander scan-file format. A subclass passes its experiment's
fixed values to ``TODReader.__init__``, and overrides one of the reader's steps only if its files
need logic that the base class lacks. They all return the same `DetectorGroupTOD`, so nothing
downstream needs to know which one ran.

Add a new experiment by writing such a subclass and registering it in ``experiment_tod_readers``.
"""
from collections.abc import Sequence
from mpi4py import MPI
from numpy.typing import NDArray
from pixell.bunch import Bunch
import healpy as hp
import logging
import numpy as np

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.parameters.schema import TODBandConfig, resolve_param
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

    Recursive bisection: sort the scans along the direction in which their pointing-plane normals
    spread most, and cut where the cumulative weight reaches the left side's share of the ranks.
    Then split each side the same way, until there is one part per rank. The sort keeps the order
    of scans with the same normal, so scans without known normals, all given the same one, end up
    in contiguous blocks in time.

    Args:
        vectors: (nscans, 3) unit vectors of the scans' pointing-plane normals, the normals of the
            planes that best fit their pointing (see `read_tods_from_file`), in time order.
        weights: (nscans,) cost of each scan, e.g. its number of samples.
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
        # The direction of largest spread: the eigenvector of the normals' 3x3 covariance with the
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


def select_scans(params: Bunch, my_experiment: Bunch, my_band: Bunch) -> list[list[str]]:
    """The filelist rows of the scans one band reads, sorted by scan ID.

    Keeps the scans with IDs from ``scan_id_start`` (included) to ``scan_id_stop`` (excluded), and
    drops the scans listed in ``bad_PIDs_path`` and those whose filelist row gives ntod 0, which
    means the file holds no data for them. Scan IDs increase with time, but the rows of a filelist
    need not (the LFI and HFI filelists are not), so the scans are selected and ordered by scan ID.

    Returns:
        Each selected row, split into its columns. Commander3's filelists have five: scan ID, quoted
        file path, weight, and the scan's pointing-plane normal (theta, phi). Its newer converters
        add a sixth, the scan's ntod.
    """
    band_name = my_band._name
    scopes = (f"experiments.{my_experiment._name}.bands.{band_name}",
              f"experiments.{my_experiment._name}")
    scan_id_start = resolve_param(params, "scan_id_start", scopes, default=None,
                                  legal_types=(int, type(None)))
    scan_id_stop = resolve_param(params, "scan_id_stop", scopes, default=None,
                                 legal_types=(int, type(None)))
    with open(my_band.filelist) as infile:
        infile.readline()  # The first line holds the number of scans.
        rows = [line.split() for line in infile if line.strip()]
    rows.sort(key=lambda columns: int(columns[0]))
    in_range = [columns for columns in rows
                if (scan_id_start is None or int(columns[0]) >= scan_id_start)
                and (scan_id_stop is None or int(columns[0]) < scan_id_stop)]
    bad_scan_ids = set()
    if "bad_PIDs_path" in my_experiment:
        bad_scan_ids = {int(scan_id) for scan_id in np.load(my_experiment.bad_PIDs_path)}
    # Commander3's newer converters write ntod 0 for a scan without data in its file, such as the
    # HFI converter's placeholder scans.
    selected = [columns for columns in in_range if int(columns[0]) not in bad_scan_ids
                and not (len(columns) > 5 and float(columns[5]) == 0)]
    logger.debug(f"Band {band_name}: {len(in_range)} of the filelist's {len(rows)} scans have IDs "
                 f"in [{scan_id_start}, {scan_id_stop}), of which {len(in_range) - len(selected)} "
                 "are listed as bad or hold no data.")
    if len(selected) == 0:
        raise ValueError(f"Band {band_name}: scan IDs [{scan_id_start}, {scan_id_stop}) select no "
                         "usable scans.")
    return selected


def tod_band_weights(params: Bunch, tod_bands: Sequence[TODBandConfig]) -> list[float]:
    """The weights by which the TOD ranks are split between the bands (see `split_tod_ranks`).

    A band's weight is its ``band_weight`` if the parameter file sets one. Otherwise ("auto") it is
    the band's number of detector-samples: the sum of ntod (the filelist's 6th column) over the
    scans it reads (see `select_scans`), times its number of detectors. If any of these scans, in
    any band, has no ntod, every band counts its scans instead (number of scans times detectors),
    so that all bands are weighed on the same scale.
    """
    if all(band.weight is not None for band in tod_bands):
        return [band.weight for band in tod_bands]
    ntods, ndets = [], []
    for band in tod_bands:
        experiment = params.experiments[band.experiment_name]
        rows = select_scans(params, experiment, experiment.bands[band.band_name])
        # Rows of Commander3's older, 5-column filelists have no ntod.
        ntods.append([float(columns[5]) if len(columns) > 5 else 0.0 for columns in rows])
        ndets.append(len(experiment.bands[band.band_name].detectors))
    if all(ntod > 0 for band_ntods in ntods for ntod in band_ntods):
        return [ndet*sum(band_ntods) for ndet, band_ntods in zip(ndets, ntods)]
    logger.info("Some selected scans have no ntod in their filelist (column 6), so the bands are "
                "weighed by number of scans times detectors.")
    return [ndet*len(band_ntods) for ndet, band_ntods in zip(ndets, ntods)]


def read_tods_from_file(band_comm: MPI.Comm, params: Bunch, my_experiment: Bunch,
                        my_band: Bunch) -> DetectorGroupTOD:
    """Read this rank's share of one band's scans.

    The band master selects the scans from the filelist (see `select_scans`), and splits them
    between the ranks by ntod and pointing-plane normal (see `split_by_sky_position`), both taken
    from the filelist (columns 4-6, as Commander3's newer converters write them). The split uses
    only what every selected scan has: if a scan lacks ntod, every scan gets weight 1, and if one
    lacks a normal, every scan gets the same one. Without both, the scans are split into equal
    blocks in time. Each rank then reads its share with the experiment's reader. If the experiment
    sets ``replace_tod_with_sim``, the TOD is then replaced by a simulation.

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
    rank, nranks = band_comm.Get_rank(), band_comm.Get_size()

    # The band master selects the scans, each as (scan ID, path), and splits them between the
    # ranks. A rank's scans need not be contiguous in time; every step that needs time order sorts
    # by scan ID.
    scans: list[tuple[int, str]] = []
    scans_per_rank = None
    if rank == 0:
        selected = select_scans(params, my_experiment, my_band)
        scans = [(int(columns[0]), columns[1].strip('"')) for columns in selected]
        # Commander3's newer converters write ntod as the 6th column, and theta and phi (the
        # pointing-plane normal) as 0 for a scan without one. The theta and phi of older, 5-column
        # filelists are not used: the old converters' method can put them far off.
        ntod, theta, phi = np.array([[float(columns[5]), float(columns[3]), float(columns[4])]
                                     if len(columns) > 5 else [0.0, 0.0, 0.0]
                                     for columns in selected]).T
        ntod_known = (ntod > 0).all()
        normal_known = ((theta != 0) | (phi != 0)).all()
        if not (ntod_known and normal_known):
            logger.info(f"Band {band_name}: the filelist lacks some scans' ntod or pointing-plane "
                        "normal (columns 4-6), so the split uses only what every scan has.")
        weights = ntod if ntod_known else np.ones(len(scans))
        if not normal_known:
            theta, phi = np.zeros(len(scans)), np.zeros(len(scans))
        scan_ranks = split_by_sky_position(np.array(hp.ang2vec(theta, phi)), weights, nranks)
        scans_per_rank = [[] for _ in range(nranks)]
        for scan, scan_rank in zip(scans, scan_ranks):
            scans_per_rank[scan_rank].append(scan)
        counts = np.bincount(scan_ranks, minlength=nranks)
        weight_per_rank = np.bincount(scan_ranks, weights, minlength=nranks)
        logger.info(f"Band {band_name}: split {len(scans)} scans between {nranks} ranks: "
                    f"{counts.min()} to {counts.max()} scans per rank, largest total weight "
                    f"{weight_per_rank.max()/weight_per_rank.mean():.2f} x the mean.")
    my_scans = band_comm.scatter(scans_per_rank, root=0)
    reader = experiment_tod_readers[my_experiment.experiment_id](
        band_comm, params, my_experiment, my_band)
    experiment_data = reader.read(my_scans)

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
