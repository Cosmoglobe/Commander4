from __future__ import annotations

import os
import numpy as np
import h5py
import datetime
import glob
from numpy.typing import NDArray
from mpi4py import MPI
from pixell.bunch import Bunch
import logging
import typing

from commander4.data_models.jump_corrections import JumpCatalog
if typing.TYPE_CHECKING:
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


def _gather_variable_length_1d_array(band_comm: MPI.Comm, local_array: NDArray) -> NDArray | None:
        """Gather a 1-D array with rank-dependent length onto the root rank."""
        local_array = np.ascontiguousarray(local_array)
        local_count = np.array([local_array.size], dtype=np.int64)

        if band_comm.Get_rank() == 0:
            counts = np.zeros(band_comm.Get_size(), dtype=np.int64)
        else:
            counts = None
        band_comm.Gather(local_count, counts, root=0)

        recvbuf = None
        global_array = None
        if band_comm.Get_rank() == 0:
            displacements = np.cumsum(counts) - counts
            global_array = np.empty(np.sum(counts), dtype=local_array.dtype)
            mpi_type = MPI._typedict[local_array.dtype.char]
            recvbuf = (global_array, counts, displacements, mpi_type)

        band_comm.Gatherv(local_array, recvbuf=recvbuf, root=0)
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
                 my_band: Bunch,
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
        self.jumps = JumpCatalog.empty(self.nscans, self.ndet)

        init_chain_path = getattr(params.general, "init_chain_path", False)
        init_from_chain = bool(init_chain_path)
        # Gibbs-sampled quantities
        if not init_from_chain:
            # ---------------------------------------------------------
            # Standard Initialization (No file provided)
            # ---------------------------------------------------------
            if self.band_comm.Get_rank() == 0:
                logger.info("No previous chain provided. Starting fresh Gibbs chain.")

            self.noise_params = np.zeros((self.nscans, self.ndet, 3)) + np.nan
            self.abs_gain = 0.0
            self.rel_gain = np.zeros((self.ndet))
            self.temporal_gain = np.zeros((self.nscans, self.ndet))

            all_det_gains = np.zeros((self.nscans, self.ndet))
            # all_det_gains = []
            # myband_noise_params = None

            if "initial_noise_params" in my_band:
                # Option 1: They are specified in the parameter file.
                self.noise_params[:] = np.array(my_band.initial_noise_params)
            elif experiment_data.scans[0].detectors[0].init_scalars is not None:
                # Option 2: There were entries in the read-in files.
                for iscan, scan in enumerate(experiment_data.scans):
                    for idet, det in enumerate(scan.detectors):
                        self.noise_params[iscan,idet] = det.init_scalars[1:]
            else:
                # Option 3: Fallback to sensible defaults.
                logger.warning("Did not find initial noise parameters, falling back to sensible defaults.")
                self.noise_params[:] = np.array([1e-3, 0.1, -1.0])

            if "gain" in my_band.detectors[experiment_data.scans[0].detectors[0].name]:
                for iscan, scan in enumerate(experiment_data.scans):
                    for idet, det in enumerate(scan.detectors):
                        all_det_gains[iscan,idet] = my_band[det.name].gain
            elif experiment_data.scans[0].detectors[0].init_scalars is not None:
                for iscan, scan in enumerate(experiment_data.scans):
                    for idet, det in enumerate(scan.detectors):
                        all_det_gains[iscan,idet] = det.init_scalars[0]
            else:
                raise ValueError("Did not find initial gain value in input files.")

            all_det_gains = np.array(all_det_gains)
            self.abs_gain = float(np.nanmean(all_det_gains))
            self.rel_gain = np.nanmean(all_det_gains, axis=0) - self.abs_gain
            self.temporal_gain = all_det_gains - self.rel_gain - self.abs_gain

        else:
            # ---------------------------------------------------------
            # Disk Initialization (Read from previous chain)
            # ---------------------------------------------------------
            # 1. Find the latest iteration for chain 01
            # init_dir = params.general.init_chain_dir
            # pattern = f"tod/{self.experiment_name}_{self.band_name}_chain{self.chain:02d}_iter*.h5"
            # search_path = os.path.join(init_dir, pattern)
            # files = glob.glob(search_path)
            
            # if not files:
            #     raise FileNotFoundError(f"No chain files found matching: {search_path}")
            
            # # Sorting alphabetically naturally sorts by the zero-padded iteration number
            # files.sort()
            # latest_file = files[-1]

            if self.band_comm.Get_rank() == 0:
                logger.info(f"Band {self.band_name} initializing TOD samples from existing chain: "\
                            f"{init_chain_path}.")

            # 2. Extract data mapping
            with h5py.File(init_chain_path, "r") as f:
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
                self.jumps = JumpCatalog.from_hdf5(f, local_indices, self.ndet)

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

        # 5. Jump corrections (per-scan per-detector ragged quantity)
        jump_counts_local, jump_locations_local, jump_offsets_local = self.jumps.pack()
        jump_counts_global = _gather_scan_distributed_array(band_comm, jump_counts_local,
                                                            scans_per_rank)
        jump_locations_global = _gather_variable_length_1d_array(band_comm, jump_locations_local)
        jump_offsets_global = _gather_variable_length_1d_array(band_comm, jump_offsets_local)
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
                if jump_counts_global is not None:
                    file["jump_counts"] = jump_counts_global
                    file["jump_locations"] = jump_locations_global
                    file["jump_offsets"] = jump_offsets_global