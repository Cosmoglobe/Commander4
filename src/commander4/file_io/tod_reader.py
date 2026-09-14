"""Dispatch to the per-experiment TOD reader named by a band's ``experiment_id``.

Each experiment needs its own reader because the file layouts differ, but they all return the same
`DetectorGroupTOD`, so nothing downstream needs to know which one ran. Every reader lives in
``file_io/experiments/`` in a module named after the ``experiment_id`` it registers, so a parameter
file's ``experiment_id: "SO_LAT"`` is served by ``file_io/experiments/SO_LAT.py``.

Add a new experiment by writing a reader with the signature below, in a module named after its id,
and registering it in ``experiment_tod_readers``.
"""
from mpi4py import MPI
from pixell.bunch import Bunch
import logging
import numpy as np

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.parameters.schema import resolve_param, split_integer_range
from commander4.file_io.experiments.akari import tod_reader as tod_reader_akari
from commander4.file_io.experiments.litebird_sim import tod_reader as tod_reader_litebird_sim
from commander4.file_io.experiments.litebird_sim_spawndetectors import tod_reader\
    as tod_reader_litebird_sim_spawndetectors
from commander4.file_io.experiments.planck_lfi import tod_reader as tod_reader_planck_lfi
from commander4.file_io.experiments.planck_hfi import tod_reader as tod_reader_planck_hfi
from commander4.file_io.experiments.general import tod_reader as tod_reader_general
from commander4.file_io.experiments.SO_LAT import tod_reader as tod_reader_SO_LAT
from commander4.file_io.experiments.SO_SAT import tod_reader as tod_reader_SO_SAT

logger = logging.getLogger(__name__)

# Known experiments and the reader each one uses. The parameter file's `experiment_id` selects the
# entry, and every key matches the module name it is imported from.
experiment_tod_readers = {
    "akari" : tod_reader_akari,
    "litebird_sim" : tod_reader_litebird_sim,
    "litebird_sim_spawndetectors" : tod_reader_litebird_sim_spawndetectors,
    "planck_lfi" : tod_reader_planck_lfi,
    "planck_hfi" : tod_reader_planck_hfi,
    # The plain reader for the standard format, with no instrument-specific behaviour. Everything
    # `simgen` output reads through this, as does any other dataset already in that layout.
    "general" : tod_reader_general,
    "SO_LAT" : tod_reader_SO_LAT,
    "SO_SAT" : tod_reader_SO_SAT,
}


def read_tods_from_file(band_comm: MPI.Comm, my_experiment: Bunch, my_band: Bunch,
                        det_names: list[str], params: Bunch) -> DetectorGroupTOD:
    """Slice the band's filelist, distribute its rows, and read this rank's scans.

    Figures out which scans this rank should read by checking  `filelist_idx_start` and
    `filelist_idx_stop` from the parameter file. Then calls the TOD-reader of our experiment.

    Args:
        band_comm: The band's MPI communicator; each rank reads a disjoint range of scans.
        my_experiment, my_band: The experiment and band parameter blocks.
        det_names: Detector names in full-band index order.
        params: The full parameter file. Used for finding the optional filelist slice bounds.

    Returns:
        The band's `DetectorGroupTOD`, with its scan-index bookkeeping filled in. The reader may
        discard scans (bad PIDs, empty data), so the actual per-rank ranges are only known
        afterwards and are recomputed here rather than taken from the requested ones.
    """
    # Confirm that the specified experiment type (e.g. "planck") is in dictionary.
    if my_experiment.experiment_id not in experiment_tod_readers.keys():
        raise ValueError("An experiment in the parameter file has experiment_id = "\
                f"{my_experiment.experiment_id}, which is not in {experiment_tod_readers.keys()}. "\
                "You either misspelled the experiment ID, or your experiment does not yet have a "\
                "specified TOD reader. See this file for how to add it.")

    scopes = (f"experiments.{my_experiment._name}.bands.{my_band._name}",
              f"experiments.{my_experiment._name}")
    filelist_idx_start = resolve_param(params, "filelist_idx_start", scopes, default=None,
                                       legal_types=(int, type(None)))
    filelist_idx_stop = resolve_param(params, "filelist_idx_stop", scopes, default=None,
                                      legal_types=(int, type(None)))
    # Count the actual rows once per band; no TOD files are opened to choose the slice.
    total_scans: int | None = None
    if band_comm.Get_rank() == 0:
        with open(my_band.filelist) as infile:
            infile.readline()
            total_scans = len(infile.readlines())
    total_scans = band_comm.bcast(total_scans, root=0)
    scan_start, scan_stop, _ = slice(filelist_idx_start, filelist_idx_stop).indices(total_scans)
    if scan_stop <= scan_start:
        raise ValueError(f"Band {my_band._name}: filelist slice "
                         f"[{filelist_idx_start}:{filelist_idx_stop}] selects no scans.")
    my_scans_start, my_scans_stop = split_integer_range(
        scan_stop - scan_start, band_comm.Get_size(), band_comm.Get_rank())
    my_scans_start += scan_start
    my_scans_stop += scan_start
    if band_comm.Get_rank() == 0:
        logger.info(f"Band {my_band._name}: selected filelist rows [{scan_start}:{scan_stop}], "
                    f"{scan_stop - scan_start} of {total_scans} scans.")

    # Load and execute TOD loader script for this specific experiment.
    my_tod_reader = experiment_tod_readers[my_experiment.experiment_id]
    experiment_data: DetectorGroupTOD = my_tod_reader(
        band_comm, my_experiment, my_band, det_names, params, my_scans_start, my_scans_stop)

    # Because some scans might have been discarded during read-in, we can only now figure out what
    # the scan start and stop index each rank holds.
    scans_per_rank = np.zeros(band_comm.Get_size(), dtype=np.int32)
    band_comm.Allgather(np.array([experiment_data.nscans], dtype=np.int32), scans_per_rank)
    rank = band_comm.Get_rank()
    my_scans_start = int(np.sum(scans_per_rank[:rank]))
    my_scans_stop = int(np.sum(scans_per_rank[:rank+1]))
    # Overwrite start and stop entries to reflect correct values.
    experiment_data.scan_idx_start = my_scans_start
    experiment_data.scan_idx_stop = my_scans_stop
    experiment_data.nscans_allranks = int(np.sum(scans_per_rank))

    return experiment_data
