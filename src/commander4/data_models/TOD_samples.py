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
from commander4.utils.unit_conversions import rj_to_band_unit_factor
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

    TOD_PS_NBIN = 100  # Fixed bin count for the optional low-resolution TOD power spectra.

    def __init__(self,
                 experiment_data: DetGroupTOD,
                 params: Bunch,
                 my_band: Bunch,
                 band_comm: MPI.Comm,
                 chain: int,
                 ):
        # Meta-information
        self.params = params
        self.band_comm = band_comm
        self.chain = chain
        self.experiment_name = experiment_data.experiment_name
        self.band_name = experiment_data.band_name
        self.ndet = experiment_data.ndet
        self.nscans = experiment_data.nscans
        # The noise model defines how many parameters per detector-scan (first entry is sigma0).
        self.noise_model = experiment_data.noise_model
        self.npar = self.noise_model.npar
        self.scan_idx_start = experiment_data.scan_idx_start
        self.scan_idx_stop = experiment_data.scan_idx_stop
        # C4 works internally in uK_RJ; a band may quote its gain and maps in another unit via
        # `band_unit`. `band_unit_factor` D (= value of 1 uK_RJ in band_unit) converts at the file
        # boundary: brightness maps multiply by D, the gain (brightness in its denominator) divides by
        # D on write and multiplies on read. Defaults to uK_RJ (D=1, a no-op). See [[unit_conversions]].
        if "band_unit" in my_band:
            band_unit = my_band.band_unit
        else:
            band_unit = "uK_RJ"
            if self.band_comm.Get_rank() == 0:
                logger.warning(f"Band {self.band_name} has no `band_unit`; assuming uK_RJ. Set it "
                               f"explicitly (e.g. uK_CMB for CMB-calibrated gains) to silence this.")
        self.band_unit = band_unit
        self.band_unit_factor = rj_to_band_unit_factor(experiment_data.nu, band_unit)
        # Explicit int64 so a rank that holds no scans still yields an int64 array.
        self.scan_ids = np.array([scan.scan_id for scan in experiment_data.scans], dtype=np.int64)
        # Ordered per-band detector names. Their position is the ``idet`` axis shared by every
        # per-detector array (rel_gain, noise_params, temporal_gain, accept, tod_ps_*, jumps), so
        # writing them to the chain lets a reader map each array column back to a physical detector.
        # Identical across all ranks of a band (taken from the band's detector list).
        self.det_names = list(my_band.detectors)
        self.jumps = JumpCatalog.empty(self.nscans, self.ndet)
        # Two distinct per-detector-scan boolean masks over the dense (nscans, ndet) grid:
        #   * present: whether this detector actually has data in this scan. Scans hold only the
        #     detectors present in them (DetGroupTOD/ScanTOD are sparse), so a detector missing from
        #     a scan leaves a `present=False` hole in the dense arrays. Derived from the data, not
        #     sampled, so it is rebuilt here on every construction (chain init included).
        #   * accept: data-quality flag for present data that is *not* flagged as bad. Defaults to
        #     all True; a chain-tracked quantity so bad-data rejection can become a sampled step.
        self.present = np.zeros((self.nscans, self.ndet), dtype=bool)
        for iscan, det in experiment_data.iter_detector_scans():
            self.present[iscan, det.det_idx_fullband] = True
        self.accept = np.ones((self.nscans, self.ndet), dtype=bool)

        # Low-resolution (log-binned) TOD power spectra, written to the chain by default: a shared
        # binned frequency axis plus the binned periodograms of several per-detector-scan TOD views
        # (all in detector units): the raw TOD, the correlated-noise realization, the TOD with only
        # the correlated noise removed (sky + white noise retained), and the residual (sky model,
        # orbital dipole, and correlated noise all subtracted). Filled during TOD processing. The
        # binned frequency edges differ per scan (scans have different lengths), so freqs are stored.
        ps_shape = (self.nscans, self.ndet, self.TOD_PS_NBIN)
        self.tod_ps_freqs = np.full(ps_shape, np.nan, dtype=np.float32)
        self.tod_ps_ncorr = np.full(ps_shape, np.nan, dtype=np.float32)
        self.tod_ps_raw = np.full(ps_shape, np.nan, dtype=np.float32)
        self.tod_ps_ncorrsub = np.full(ps_shape, np.nan, dtype=np.float32)
        self.tod_ps_residual = np.full(ps_shape, np.nan, dtype=np.float32)

        # Optional DEBUG: the entire per-sample correlated-noise (n_corr) TODs, written to the chain
        # only when explicitly requested (the data is very large). Collected ragged as one float32
        # array per detector-scan; ``None`` disables collection.
        if bool(getattr(params.general, "write_ncorr_tods_to_chain", False)):
            self.ncorr_tods: list[list[NDArray | None]] | None = \
                [[None] * self.ndet for _ in range(self.nscans)]
        else:
            self.ncorr_tods = None

        init_chain_path = getattr(params.general, "init_chain_path", False)
        init_from_chain = bool(init_chain_path)
        # Gibbs-sampled quantities
        if not init_from_chain:
            # ---------------------------------------------------------
            # Standard Initialization (No file provided)
            # ---------------------------------------------------------
            if self.band_comm.Get_rank() == 0:
                logger.info("No previous chain provided. Starting fresh Gibbs chain.")

            self.noise_params = np.zeros((self.nscans, self.ndet, self.npar)) + np.nan
            self.abs_gain = 0.0
            self.rel_gain = np.zeros((self.ndet))
            self.temporal_gain = np.zeros((self.nscans, self.ndet))

            # Find the first detector in the first scan. Used for checking if default values
            # are present. Default to None in case there are no scans.
            rep_det = next((det for _, det in experiment_data.iter_detector_scans()), None)

            if rep_det is not None:
                # NaN so detector-scans with no data are excluded from the gain means below.
                all_det_gains = np.full((self.nscans, self.ndet), np.nan)

                if "initial_noise_params" in my_band:
                    # Option 1: They are specified in the parameter file.
                    self.noise_params[:] = np.array(my_band.initial_noise_params)
                elif rep_det.init_scalars is not None:
                    # Option 2: There were entries in the read-in files. Index by the detector's
                    # full-band column; absent detector-scans stay NaN.
                    for iscan, det in experiment_data.iter_detector_scans():
                        self.noise_params[iscan, det.det_idx_fullband] = det.init_scalars[1:]
                else:
                    # Option 3: Fall back to the noise model's default parameters (ensuring a finite
                    # sigma0, which the model leaves as NaN to be estimated from the data).
                    logger.warning("Did not find initial noise parameters, falling back to the "
                                   "noise model's default parameters.")
                    default_params = np.array(self.noise_model.params, dtype=np.float64)
                    if not np.isfinite(default_params[0]):
                        default_params[0] = 1.0
                    self.noise_params[:] = default_params

                if "gain" in my_band.detectors[rep_det.name]:
                    for iscan, det in experiment_data.iter_detector_scans():
                        all_det_gains[iscan, det.det_idx_fullband] = my_band[det.name].gain
                elif rep_det.init_scalars is not None:
                    for iscan, det in experiment_data.iter_detector_scans():
                        all_det_gains[iscan, det.det_idx_fullband] = det.init_scalars[0]
                else:
                    raise ValueError("Did not find initial gain value in input files.")

                # Initial gains are quoted in band_unit; convert to internal [det units]/uK_RJ
                # (gain multiplies by D on read) before decomposing into abs/rel/temporal.
                all_det_gains *= self.band_unit_factor

                self.abs_gain = float(np.nanmean(all_det_gains))
                # Relative gain only for detectors with data in >=1 local scan; detectors absent
                # from every local scan get 0 (never used downstream) and are kept out of the
                # empty-slice mean. temporal_gain holes (absent detector-scans) collapse to 0.
                present_any = np.isfinite(all_det_gains).any(axis=0)
                self.rel_gain[present_any] = (np.nanmean(all_det_gains[:, present_any], axis=0)
                                              - self.abs_gain)
                self.temporal_gain = np.nan_to_num(all_det_gains - self.rel_gain - self.abs_gain,
                                                   nan=0.0)

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
                    raise ValueError(f"Local scan ID {e} not found in the global chain file "
                                     f"{init_chain_path}.") from e

                # 3. Load Per-Band and Per-Detector arrays (Identical across ranks)
                self.abs_gain = float(f["abs_gain"][...]) if "abs_gain" in f else None
                self.rel_gain = f["detrel_gain"][:] if "detrel_gain" in f else None

                # 4. Load and slice Per-Scan arrays (Distributed across ranks)
                self.temporal_gain = f["temporal_gain"][local_indices, :] if "temporal_gain" in f else None
                self.noise_params = f["noise_params"][local_indices, ...] if "noise_params" in f else None
                self.accept = f["accept"][local_indices, ...].astype(bool)
                self.jumps = JumpCatalog.from_hdf5(f, local_indices, self.ndet)

            # Chain gains are stored in band_unit; convert back to internal [det units]/uK_RJ.
            if self.band_unit_factor != 1.0:
                if self.abs_gain is not None:
                    self.abs_gain *= self.band_unit_factor
                if self.rel_gain is not None:
                    self.rel_gain = self.rel_gain * self.band_unit_factor
                if self.temporal_gain is not None:
                    self.temporal_gain = self.temporal_gain * self.band_unit_factor

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


    def _pack_ncorr_tods(self) -> tuple[NDArray, NDArray]:
        """Pack the optional per-(scan, det) correlated-noise TODs for ragged chain storage.

        Returns a ``(nscans, ndet)`` int64 array of per-detector-scan lengths and a 1-D float32
        concatenation of all segments in scan-major, detector-minor order (matching how the
        gather routines concatenate). The reader reconstructs each TOD by walking the lengths.
        """
        lengths = np.zeros((self.nscans, self.ndet), dtype=np.int64)
        segments = []
        for iscan in range(self.nscans):
            for idet in range(self.ndet):
                seg = self.ncorr_tods[iscan][idet]
                if seg is not None:
                    seg = np.asarray(seg, dtype=np.float32).ravel()
                    lengths[iscan, idet] = seg.size
                    segments.append(seg)
        flat = np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)
        return lengths, flat


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

        # 4b. Presence and acceptance flags (per-scan per-detector; int8 for MPI/HDF compatibility).
        # `present` marks real vs. absent (dummy) detector-scans; `accept` marks good vs. bad data.
        present_global = _gather_scan_distributed_array(band_comm, self.present.astype(np.int8),
                                                        scans_per_rank)
        accept_global = _gather_scan_distributed_array(band_comm, self.accept.astype(np.int8),
                                                       scans_per_rank)

        # 4c. Low-resolution TOD power spectra (per-scan per-detector per-bin).
        tod_ps_freqs_global = _gather_scan_distributed_array(band_comm, self.tod_ps_freqs,
                                                            scans_per_rank)
        tod_ps_ncorr_global = _gather_scan_distributed_array(band_comm, self.tod_ps_ncorr,
                                                            scans_per_rank)
        tod_ps_raw_global = _gather_scan_distributed_array(band_comm, self.tod_ps_raw,
                                                          scans_per_rank)
        tod_ps_ncorrsub_global = _gather_scan_distributed_array(band_comm, self.tod_ps_ncorrsub,
                                                               scans_per_rank)
        tod_ps_residual_global = _gather_scan_distributed_array(band_comm, self.tod_ps_residual,
                                                               scans_per_rank)

        # 4d. Optional DEBUG: full per-sample correlated-noise TODs (ragged per-scan per-detector).
        ncorr_lengths_global = ncorr_flat_global = None
        if self.ncorr_tods is not None:
            ncorr_lengths_local, ncorr_flat_local = self._pack_ncorr_tods()
            ncorr_lengths_global = _gather_scan_distributed_array(band_comm, ncorr_lengths_local,
                                                                 scans_per_rank)
            ncorr_flat_global = _gather_variable_length_1d_array(band_comm, ncorr_flat_local)

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

            # Write gains in band_unit: gain has brightness in its denominator, so divide by D
            # (output maps instead multiply by D). Division makes copies, so self.* stays uK_RJ.
            if abs_gain_global is not None:
                abs_gain_global = abs_gain_global / self.band_unit_factor
            if rel_gain_global is not None:
                rel_gain_global = rel_gain_global / self.band_unit_factor
            if temporal_gain_global is not None:
                temporal_gain_global = temporal_gain_global / self.band_unit_factor

            with h5py.File(chain_file, "w") as file:
                file["metadata/datetime"] = datetime.datetime.now().isoformat()
                file["metadata/parameter_file_as_string"] = params.parameter_file_as_string
                # Thermodynamic unit the written gains are expressed in (gain is [det units]/band_unit).
                file["metadata/band_unit"] = self.band_unit
                file["scan_ids"] = scan_ids_global
                # Detector names (per-band, identical across ranks): the `idet` axis of every
                # per-detector array below. Variable-length UTF-8 for a clean string round-trip.
                file.create_dataset("det_names",
                                    data=np.array(self.det_names, dtype=h5py.string_dtype()))
                if abs_gain_global is not None:
                    file["abs_gain"] = abs_gain_global
                if rel_gain_global is not None:
                    file["detrel_gain"] = rel_gain_global
                if temporal_gain_global is not None:
                    file["temporal_gain"] = temporal_gain_global
                if noise_params_global is not None:
                    file["noise_params"] = noise_params_global
                file["present"] = present_global
                file["accept"] = accept_global
                file["tod_ps_freqs"] = tod_ps_freqs_global
                file["tod_ps_ncorr"] = tod_ps_ncorr_global
                file["tod_ps_raw"] = tod_ps_raw_global
                file["tod_ps_ncorrsub"] = tod_ps_ncorrsub_global
                file["tod_ps_residual"] = tod_ps_residual_global
                if ncorr_lengths_global is not None:
                    file["ncorr_tod_lengths"] = ncorr_lengths_global
                    file["ncorr_tod_flat"] = ncorr_flat_global
                if jump_counts_global is not None:
                    file["jump_counts"] = jump_counts_global
                    file["jump_locations"] = jump_locations_global
                    file["jump_offsets"] = jump_offsets_global