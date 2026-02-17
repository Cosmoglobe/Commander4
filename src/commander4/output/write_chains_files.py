import os
import h5py
import numpy as np
import datetime

from mpi4py import MPI

from commander4.data_models.detector_samples import DetectorSamples
from commander4.utils.params import Params
from commander4.sky_models.component import Component


def write_map_chain_to_file(params: Params, chain: int, iter: int, exp_name:str,
                            band_name: str, maps_to_file: dict) -> None:
    chain_dir = os.path.join(params.general.output_paths.chains, "datamaps")
    chain_file = os.path.join(chain_dir, f"{exp_name}_{band_name}_chain{chain:02d}_iter{iter:04d}.h5")

    with h5py.File(chain_file, "w") as file:
        file["metadata/datetime"] = datetime.datetime.now().isoformat()
        file["metadata/parameter_file_as_string"] = params.parameter_file_as_string
        file["metadata/parameter_file_as_binary_yaml"] = params.parameter_file_binary_yaml
        for key, value, in maps_to_file.items():
            file[key] = value


def write_tod_chain_to_file(det_comm: MPI.Comm, detector_samples: DetectorSamples,
                            params: Params, chain: int, iter: int) -> None:
    detector_samples_batches = det_comm.gather(detector_samples, root=0)
    if det_comm.Get_rank() == 0:
        # TODO: Make DetectorSamples arrays. Currently this gather takes minutes.

        exp_name = detector_samples.experiment_name
        det_name = detector_samples.detector_name
        chain_dir = os.path.join(params.general.output_paths.chains, "tod")
        chain_file = os.path.join(chain_dir, f"{exp_name}_{det_name}_chain{chain:02d}_iter{iter:04d}.h5")

        scanIDs = []
        for detector_samples_batch in detector_samples_batches:
            for scan_samples in detector_samples_batch.scans:
                scanIDs.append(scan_samples.scanID)
        scanIDs = np.array(scanIDs, dtype=int)
        scans_sort_indices = np.argsort(scanIDs)

        write_dict = {}
        for key, value in vars(scan_samples).items():
            write_dict[key] = []

        for detector_samples_batch in detector_samples_batches:
            for scan_samples in detector_samples_batch.scans:
                for key, value in vars(scan_samples).items():
                    write_dict[key].append(value)
        with h5py.File(chain_file, "w") as file:
            file["metadata/datetime"] = datetime.datetime.now().isoformat()
            file["metadata/parameter_file_as_string"] = params.parameter_file_as_string
            file["metadata/parameter_file_as_binary_yaml"] = params.parameter_file_binary_yaml
            for key in write_dict.keys():
                arr = np.array(write_dict[key])[scans_sort_indices]
                file[key] = arr


def write_compsep_chain_to_file(comp_list: list[Component], params: Params, chain: int, iter: int):
    chain_dir = os.path.join(params.general.output_paths.chains, "compsep")
    chain_file = os.path.join(chain_dir, f"chain{chain:02d}_iter{iter:04d}.h5")
    with h5py.File(chain_file, "w") as file:
        file["metadata/datetime"] = datetime.datetime.now().isoformat()
        file["metadata/parameter_file_as_string"] = params.parameter_file_as_string
        file["metadata/parameter_file_as_binary_yaml"] = params.parameter_file_binary_yaml
        for comp in comp_list:
            file[f"comps/{comp.shortname}/alms"] = comp.alms
            file[f"comps/{comp.shortname}/longname"] = comp.longname
            file[f"comps/{comp.shortname}/shortname"] = comp.shortname