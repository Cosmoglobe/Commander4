from pixell.bunch import Bunch
from mpi4py import MPI

# from commander4.experiments.litebird.tod_reader_litebird import tod_reader as tod_reader_litebird
from commander4.experiments.litebird.tod_reader_litebird_sim import tod_reader as tod_reader_litebird_sim
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
    "akari": tod_reader_akari
}

def read_tods_from_file(det_comm: MPI.Comm, my_experiment: Bunch, my_band: Bunch, my_det: Bunch,
                        params: Bunch, my_detector_id: int, my_scans_start: int, my_scans_stop: int):
    
    # Confirm that the specified experiment type (e.g. "planck") is in dictionary.
    if my_experiment.experiment_id not in experiment_tod_readers.keys():
        raise ValueError("An experiment in the parameter file has experiment_id = "\
                f"{my_experiment.experiment_id}, which is not in {experiment_tod_readers.keys()}. "\
                "You either misspelled the experiment ID, or your experiment does not yet have a "\
                "specified TOD reader. See this file for how to add it.")

    # Load and execute TOD loader script for this specific experiment.
    my_tod_reader = experiment_tod_readers[my_experiment.experiment_id]
    experiment_data = my_tod_reader(det_comm, my_experiment, my_band, my_det, params,
                                    my_detector_id, my_scans_start, my_scans_stop)
    return experiment_data