import os
import numpy as np
import h5py
import datetime
import glob
from numpy.typing import NDArray
from mpi4py import MPI
from pixell.bunch import Bunch
import logging

from commander4.data_models.detector_group_TOD import DetGroupTOD

logger = logging.getLogger(__name__)


def _gather_scan_distributed_array(band_comm: MPI.Comm, local_array: NDArray,
                                   scans_per_rank: NDArray | None) -> NDArray | None:
        """ Gathers an N-dimensional array distributed along the 0th axis (scans) to the root rank.
            Automatically handles MPI Datatypes, count multipliers, and memory layouts.
            Should be called from multiple ranks on the same `band_comm`, with each`local_array`
            having shape (nscans, ...), where `nscans` can vary between ranks. The returned array
            is concatenated along the `nscans` axis.
        """
        rank = band_comm.Get_rank()
        
        # 1. Determine the number of elements per scan
        # For shape (scans, det, params), trailing_shape is (det, params). 
        # np.prod calculates the total elements that make up one complete 'scan' slice.
        trailing_shape = local_array.shape[1:]
        elements_per_scan = int(np.prod(trailing_shape)) if trailing_shape else 1

        recvbuf = None
        global_array = None

        if rank == 0:
            # 2. Calculate explicit routing arrays for this specific array shape
            elements_per_rank = scans_per_rank * elements_per_scan
            displacements = (np.cumsum(scans_per_rank) - scans_per_rank) * elements_per_scan

            # 3. Allocate the global structure
            nscans_allranks = np.sum(scans_per_rank)
            global_shape = (nscans_allranks, *trailing_shape)
            global_array = np.zeros(global_shape, dtype=local_array.dtype)

            # 4. Automatically map the NumPy dtype to the explicit MPI Datatype
            mpi_type = MPI._typedict[local_array.dtype.char]

            recvbuf = (global_array, elements_per_rank, displacements, mpi_type)

        # 5. Execute Gather ensuring contiguous memory
        band_comm.Gatherv(
            sendbuf=np.ascontiguousarray(local_array),
            recvbuf=recvbuf,
            root=0
        )

        return global_array


class TODSamples:
    """ A class for holding all sampled TOD-quantities, such as gains and correlated noise
        parameters, for one MPI rank. Quantities that vary with detectors and/or scans are stored as
        arrays, and the iscan and idet indices match those of the DetGroupTOD class.
        The DetGroupTOD class holds all static data, while this class holds all sampled data.
        Quantities that are the same across a band (like absolute gain) have identical copies for
        all ranks on the same band.
    """
    def __init__(self,
                 experiment_data: DetGroupTOD,
                 params: Bunch,
                 band_comm: MPI.Comm,
                 chain: int,
                 noise_psd_class: str = "oof",
                 ):
        # Meta-information
        self.params = params
        self.band_comm = band_comm
        self.chain = chain
        self.experiment_name = experiment_data.experiment_name
        self.band_name = experiment_data.band_name
        self.ndet = experiment_data.ndet
        self.nscans = experiment_data.nscans
        self.scan_idx_start = experiment_data.scan_idx_start
        self.scan_idx_stop = experiment_data.scan_idx_stop
        self.scan_ids = np.array([scan.scan_id for scan in experiment_data.scans])

        # Gibbs-sampled quantities
        if not params.general.init_from_chain:
            # ---------------------------------------------------------
            # Standard Initialization (No file provided)
            # ---------------------------------------------------------
            if self.band_comm.Get_rank() == 0:
                logger.info(f"Band {self.band_name} initializing TOD samples from default values.")

            all_det_gains = []
            myband_noise_params = None
            
            for exp_name in params.experiments:
                experiment = params.experiments[exp_name]
                for iband, band_name in enumerate(experiment.bands):
                    band = experiment.bands[band_name]
                    # Fixed typo here: self.band.name -> self.band_name
                    if self.experiment_name == exp_name and self.band_name == band_name:
                        myband_noise_params = band.initial_noise_params
                        for idet, det_name in enumerate(band.detectors):
                            detector = band.detectors[det_name]
                            all_det_gains.append(detector.gain_est)
                            
            all_det_gains = np.array(all_det_gains)
            abs_gain = float(np.mean(all_det_gains))
            self.abs_gain = abs_gain
            self.rel_gain = all_det_gains - abs_gain
            self.temporal_gain = np.zeros((self.nscans, self.ndet))
            self.noise_params = np.full((self.nscans, self.ndet, 3), myband_noise_params)

        else:
            # ---------------------------------------------------------
            # Disk Initialization (Read from previous chain)
            # ---------------------------------------------------------
            # 1. Find the latest iteration for chain 01
            init_dir = params.general.init_chain_dir
            pattern = f"tod/{self.experiment_name}_{self.band_name}_chain{self.chain:02d}_iter*.h5"
            search_path = os.path.join(init_dir, pattern)
            files = glob.glob(search_path)
            
            if not files:
                raise FileNotFoundError(f"No chain files found matching: {search_path}")
            
            # Sorting alphabetically naturally sorts by the zero-padded iteration number
            files.sort()
            latest_file = files[-1]

            if self.band_comm.Get_rank() == 0:
                logger.info(f"Band {self.band_name} initializing TOD samples from existing chain: "\
                            f"{latest_file}.")

            # 2. Extract data mapping
            with h5py.File(latest_file, "r") as f:
                # Read the global scan_ids saved by the Gatherv operation
                global_scan_ids = f["scan_ids"][:]
                
                # Create an O(N) lookup dictionary mapping the ID to its row index in the HDF5 array
                global_id_to_index = {sid: idx for idx, sid in enumerate(global_scan_ids)}
                
                # Verify and map the local scans to the global indices
                try:
                    local_indices = [global_id_to_index[sid] for sid in self.scan_ids]
                except KeyError as e:
                    raise ValueError(f"Local scan ID {e} not found in the global chain file {latest_file}.")

                # 3. Load Per-Band and Per-Detector arrays (Identical across ranks)
                self.abs_gain = float(f["abs_gain"][...]) if "abs_gain" in f else None
                self.rel_gain = f["detrel_gain"][:] if "detrel_gain" in f else None

                # 4. Load and slice Per-Scan arrays (Distributed across ranks)
                self.temporal_gain = f["temporal_gain"][local_indices, :] if "temporal_gain" in f else None
                self.noise_params = f["noise_params"][local_indices, ...] if "noise_params" in f else None

        if self.band_comm.Get_rank() == 0:
            logger.debug(f"Initial absolute gain estimate for {self.band_name}: {self.abs_gain:.3e}.")
            logger.debug(f"Initial rel gain estimates for {self.band_name}: {self.rel_gain}.")


    def gain(self, iscan: int, idet: int) -> float:
        """ Returns the scalar gain for a single detector+scan combination.
        """
        gain = self.abs_gain
        if self.rel_gain is not None:
            gain += self.rel_gain[idet]
        if self.temporal_gain is not None:
            gain += self.temporal_gain[iscan,idet]
        return gain


    def gain_all(self) -> NDArray[np.floating]:
        """ Returns the full gain for a all detectors and scans as an array.
        """
        gain = np.zeros((self.nscans, self.ndet), dtype=np.float64)
        gain[:] += self.abs_gain
        if self.rel_gain is not None:
            gain[:] += self.rel_gain
        if self.temporal_gain is not None:
            gain[:] += self.temporal_gain
        return gain


    def write_chain_to_file(self, itr: int):
        band_comm = self.band_comm
        params = self.params

        ####################################################################
        # Gather nscan info.
        ####################################################################
        if band_comm.Get_rank() == 0:
            scans_per_rank = np.zeros(band_comm.Get_size(), dtype=np.int64)
        else:
            scans_per_rank = None
        band_comm.Gather(sendbuf = np.array([self.nscans], dtype=np.int64),
                         recvbuf = scans_per_rank,
                         root=0)

        ####################################################################
        # Gather the various TOD samples.
        ####################################################################
        # 0. Unique scan-IDs (per-scan quantity)
        scan_ids_global = _gather_scan_distributed_array(band_comm, self.scan_ids, scans_per_rank)

        # 1. Absolute gain (per-band quantity)
        abs_gain_global = self.abs_gain  # Copies held on each rank, no communication required.

        # 2. Relative gain (per-detector quantity)
        rel_gain_global = self.rel_gain  # Copies held on each rank, no communication required.

        # 3. Temporal gain (per-scan per-detector quantity)
        temporal_gain_global = None
        if self.temporal_gain is not None:
            temporal_gain_global = _gather_scan_distributed_array(band_comm, self.temporal_gain,
                                                                  scans_per_rank)

        # 4. Noise params (per-scan per-detector per-parameter quantity)
        noise_params_global = None
        if self.noise_params is not None:
            noise_params_global = _gather_scan_distributed_array(band_comm, self.noise_params,
                                                                 scans_per_rank)
        ####################################################################
        # Write results to file.
        ####################################################################
        if band_comm.Get_rank() == 0:
            exp_name = self.experiment_name
            band_name = self.band_name
            chain_dir = os.path.join(params.general.output_paths.chains, "tod")
            filename = f"{exp_name}_{band_name}_chain{self.chain:02d}_iter{itr:04d}.h5"
            chain_file = os.path.join(chain_dir, filename)

            with h5py.File(chain_file, "w") as file:
                file["metadata/datetime"] = datetime.datetime.now().isoformat()
                file["metadata/parameter_file_as_string"] = params.parameter_file_as_string
                file["scan_ids"] = scan_ids_global
                if abs_gain_global is not None:
                    file["abs_gain"] = abs_gain_global
                if rel_gain_global is not None:
                    file["detrel_gain"] = rel_gain_global
                if temporal_gain_global is not None:
                    file["temporal_gain"] = temporal_gain_global
                if noise_params_global is not None:
                    file["noise_params"] = noise_params_global