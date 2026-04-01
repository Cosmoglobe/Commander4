from mpi4py import MPI
from pixell.bunch import Bunch
import numpy as np

from commander4.data_models.detector_group_TOD import DetGroupTOD
from commander4.experiments.litebird.tod_reader_litebird_sim import tod_reader\
    as tod_reader_litebird_sim
from commander4.experiments.litebird.tod_reader_litebird_sim_spawndetectors import tod_reader\
    as tod_reader_litebird_sim_spawndetectors
from commander4.experiments.planck.tod_reader_planck import tod_reader as tod_reader_planck
from commander4.experiments.planck.tod_reader_planck_sim import tod_reader as tod_reader_planck_sim
from commander4.experiments.akari.tod_reader_akari import tod_reader as tod_reader_akari

# Dictionary containing known experiments and the location of their TOD reading scripts.
# The `experiment_id`` field in the parameter file decides what TOD reader is used in this dict.
experiment_tod_readers = {
    "planck" : tod_reader_planck,
    "planck_sim" : tod_reader_planck_sim,
    # "litebird" : tod_reader_litebird,
    "litebird_sim" : tod_reader_litebird_sim,
    "litebird_sim_spawndetectors" : tod_reader_litebird_sim_spawndetectors,
    "akari" : tod_reader_akari,
}

def read_tods_from_file(band_comm: MPI.Comm, my_experiment: Bunch, my_band: Bunch, my_det: Bunch,
                        params: Bunch, my_scans_start: int,
                        my_scans_stop: int) -> DetGroupTOD:
    
    # Confirm that the specified experiment type (e.g. "planck") is in dictionary.
    if my_experiment.experiment_id not in experiment_tod_readers.keys():
        raise ValueError("An experiment in the parameter file has experiment_id = "\
                f"{my_experiment.experiment_id}, which is not in {experiment_tod_readers.keys()}. "\
                "You either misspelled the experiment ID, or your experiment does not yet have a "\
                "specified TOD reader. See this file for how to add it.")

    # Load and execute TOD loader script for this specific experiment.
    my_tod_reader = experiment_tod_readers[my_experiment.experiment_id]
    experiment_data: DetGroupTOD = my_tod_reader(band_comm, my_experiment, my_band, my_det, params,
                                                 my_scans_start, my_scans_stop)

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