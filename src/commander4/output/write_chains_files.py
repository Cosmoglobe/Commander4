import os
import h5py
import numpy as np

from mpi4py import MPI

from commander4.data_models.detector_samples import DetectorSamples
from commander4.utils.params import Params

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

