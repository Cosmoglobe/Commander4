import os
import h5py
import numpy as np
import yaml

from mpi4py import MPI

from commander4.data_models.detector_samples import DetectorSamples
from commander4.utils.params import Params
from commander4.sky_models.component import Component

def write_tod_chain_to_file(det_comm: MPI.Comm, detector_samples: DetectorSamples,
                            params: Params, chain: int, iter: int) -> None:
    # TODO: Make DetectorSamples arrays. Currently this gather takes minutes.
    detector_samples_batches = det_comm.gather(detector_samples, root=0)

    expname = detector_samples.experiment_name
    detname = detector_samples.detector_name
    chain_outpath = os.path.join(params.general.output_paths.chains, f"{expname}_{detname}_chain{chain:02d}_iter{iter:04d}.h5")

    if det_comm.Get_rank() == 0:
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
        with h5py.File(chain_outpath, "w") as file:
            for key in write_dict.keys():
                arr = np.array(write_dict[key])[scans_sort_indices]
                file[key] = arr


def write_compsep_chain_to_file(comp_list: list[Component], params: Params, chain: int, iter: int):
    print(params.parameter_file_as_string, flush=True)
    print("asdf:", params.parameter_file_binary_yaml, flush=True)
    chain_outpath = os.path.join(params.general.output_paths.chains, f"compsep_chain{chain:02d}_iter{iter:04d}.h5")
    with h5py.File(chain_outpath, "w") as f:
        f["metadata/parameter_file_as_string"] = params.parameter_file_as_string
        f["metadata/parameter_file_as_binary_yaml"] = params.parameter_file_binary_yaml
        for comp in comp_list:
            f[f"{comp.shortname}/alms"] = comp.alms
            f[f"{comp.shortname}/longname"] = comp.longname
            f[f"{comp.shortname}/shortname"] = comp.shortname