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
from pixell.bunch import Bunch
import logging
import numpy as np

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.parameters.schema import resolve_param, split_integer_range
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


def read_tods_from_file(band_comm: MPI.Comm, params: Bunch, my_experiment: Bunch,
                        my_band: Bunch) -> DetectorGroupTOD:
    """Read this rank's share of one band's scans.

    The band master reads the filelist, keeps the rows selected by ``filelist_idx_start`` and
    ``filelist_idx_stop``, and drops the scans listed in ``bad_PIDs_path``. The remaining scans are
    split into contiguous blocks, one per rank, and each rank reads its block with the experiment's
    reader. If the experiment sets ``replace_tod_with_sim``, the TOD is then replaced by a
    simulation.

    Args:
        band_comm: The band's MPI communicator; each rank reads a disjoint block of scans.
        params, my_experiment, my_band: The full parameter file, and this band's experiment and
            band blocks.

    Returns:
        The band's `DetectorGroupTOD`, with its scan-index bookkeeping filled in. The reader may
        drop scans (empty or bad data), so the actual per-rank ranges are only known afterwards.
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
    filelist_idx_start = resolve_param(params, "filelist_idx_start", scopes, default=None,
                                       legal_types=(int, type(None)))
    filelist_idx_stop = resolve_param(params, "filelist_idx_stop", scopes, default=None,
                                      legal_types=(int, type(None)))

    # The band master reads the filelist; the other ranks get the selected scans from it.
    # TODO(pre-pass): read each scan's ntod here too (and a pointing summary), so the split below
    # can balance the ranks and the whole band can share a few FFT sizes.
    # Each selected scan as (filelist row, scan ID, file path). The rows run in time order, so a
    # scan's row is its place in time.
    scans: list[tuple[int, int, str]] | None = None
    if rank == 0:
        with open(my_band.filelist) as infile:
            infile.readline()  # The first line holds the number of scans.
            # Each row is a scan ID, a quoted file path, and three columns that are not used.
            rows = [line.split() for line in infile if line.strip()]
        start, stop, _ = slice(filelist_idx_start, filelist_idx_stop).indices(len(rows))
        bad_scan_ids = set()
        if "bad_PIDs_path" in my_experiment:
            bad_scan_ids = {int(scan_id) for scan_id in np.load(my_experiment.bad_PIDs_path)}
        scans = []
        for row in range(start, stop):
            scan_id = int(rows[row][0])
            if scan_id not in bad_scan_ids:
                scans.append((row, scan_id, rows[row][1].strip('"')))
        logger.info(f"Band {band_name}: selected filelist rows [{start}:{max(start, stop)}], "
                    f"{max(0, stop - start)} of {len(rows)} scans, of which "
                    f"{max(0, stop - start) - len(scans)} are listed as bad.")
    scans = band_comm.bcast(scans, root=0)
    if len(scans) == 0:
        raise ValueError(f"Band {band_name}: filelist slice "
                         f"[{filelist_idx_start}:{filelist_idx_stop}] selects no usable scans.")

    # TODO(distribution): split by sky position and weight by scan cost. A rank's scans need not
    # be contiguous in time, since every time-ordered step sorts by `scan_time_index`.
    my_start, my_stop = split_integer_range(len(scans), band_comm.Get_size(), rank)
    my_scans = scans[my_start:my_stop]
    reader = experiment_tod_readers[my_experiment.experiment_id](
        band_comm, params, my_experiment, my_band)
    experiment_data = reader.read([scan_id for _, scan_id, _ in my_scans],
                                  [path for _, _, path in my_scans])

    # The reader may drop scans, so the time index and the total are only known now.
    row_of_scan = {scan_id: row for row, scan_id, _ in my_scans}
    experiment_data.scan_time_index = np.array([row_of_scan[scan.scan_id]
                                                for scan in experiment_data.scans], dtype=np.int64)
    experiment_data.nscans_allranks = band_comm.allreduce(experiment_data.nscans, op=MPI.SUM)

    if getattr(my_experiment, "replace_tod_with_sim", False):
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
