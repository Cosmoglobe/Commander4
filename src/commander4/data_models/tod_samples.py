"""`TODSamples`: everything the Gibbs chain samples on the TOD side, for one band on one rank.

The sampled state lives in one object per sampling step (see `tod/step.py`). `TODSamples` holds
those objects next to the band's identity (scan IDs, detector names), and does the MPI part of the
chain I/O for all of them alike: `gather_chain_arrays` collects every step's arrays onto the band
master for `chain_writer.write_band_chain_to_file`, and `read_chain` hands every step this rank's
part of a saved sample.
"""
from __future__ import annotations

import numpy as np
import h5py
from numpy.typing import NDArray
from mpi4py import MPI
from pixell.bunch import Bunch
import logging
import typing

from commander4.file_io.chain_writer import should_write_chain
from commander4.mpi.helpers import gather_scan_distributed_array
from commander4.tod.data_selection import DataSelection
from commander4.tod.gain import Gain
from commander4.tod.glitches.sampling import Glitches
from commander4.tod.hfi_demodulation import HFIModulation
from commander4.tod.jumps.sampling import Jumps
from commander4.tod.noise.sampling import Noise
from commander4.tod.scan_diagnostics import TODDiagnostics
from commander4.tod.sidelobe_deconvolve import FarSidelobes
from commander4.tod.step import SavedBandSample, TODStep
if typing.TYPE_CHECKING:
    from commander4.data_models.detector_group_tod import DetectorGroupTOD

logger = logging.getLogger(__name__)


class TODSamples:
    """ Everything sampled on the TOD side for one band on one MPI rank, and its chain I/O.

        The sampled state is held by one object per sampling step. Their per-detector-scan arrays
        are dense (nscans, ndet) grids, and the iscan and idet indices match those of the
        DetectorGroupTOD class, which holds all static data. Quantities that are the same across a
        band (like absolute gain) have identical copies on all ranks of the band.
    """

    def __init__(self,
                 experiment_data: DetectorGroupTOD,
                 params: Bunch,
                 my_band: Bunch,
                 band_comm: MPI.Comm,
                 chain: int,
                 ):
        # Meta-information. The sampling steps reach the band's static data and communicator here.
        self.experiment_data = experiment_data
        self.band_comm = band_comm
        self.chain = chain
        nscans, ndet = experiment_data.nscans, experiment_data.ndet
        # Explicit int64 so a rank that holds no scans still yields an int64 array.
        self.scan_ids = np.array([scan.scan_id for scan in experiment_data.scans], dtype=np.int64)
        # Ordered per-band detector names. Their position is the ``idet`` axis shared by every
        # per-detector array (rel_gain, noise_params, temporal_gain, accept, tod_ps_*, jumps), so
        # writing them to the chain lets a reader map each array column back to a physical
        # detector. Identical across all ranks of a band (taken from the band's detector list).
        self.det_names = list(my_band.detectors)
        # Whether this detector actually has data in this scan. Scans hold only the detectors
        # present in them (DetectorGroupTOD/ScanTOD are sparse), so a detector missing from a scan
        # leaves a `present=False` hole in the dense arrays. Derived from the data, not sampled.
        # Not to be confused with `selection.accept`, the data-quality flag of present data.
        self.present = np.zeros((nscans, ndet), dtype=bool)
        # Spacecraft velocity the orbital dipole is built from, in m/s. Static (it comes from the
        # data files, not the sampler) but written to the chain because it is what anchors the
        # absolute calibration, so a gain surprise can be checked against it.
        self.orbital_velocity = np.zeros((nscans, ndet, 3), dtype=np.float32)
        for iscan, det in experiment_data.iter_detector_scans():
            self.present[iscan, det.det_idx_fullband] = True
            self.orbital_velocity[iscan, det.det_idx_fullband] = det.orbital_velocity_m_per_s
        # Per-scan start time, Commander3's `MJD`. The reader takes it from the scan file's
        # `<pid>/common/time`, and passes 0.0, the "no absolute time available" value here, for
        # files without one (today the simgen and Akari files).
        self.scan_start_time = np.array([scan.start_time for scan in experiment_data.scans],
                                        dtype=np.float64)

        # The sampling steps, each reading its settings and starting from the parameter file and
        # the data files; `read_chain` overwrites their state when a chain continues a saved one.
        self.gain =             Gain(experiment_data, params, my_band, band_comm, chain)
        self.noise =            Noise(experiment_data, params, my_band, band_comm, chain)
        self.selection =        DataSelection(experiment_data, params, my_band, band_comm, chain)
        self.jumps =            Jumps(experiment_data, params, my_band, band_comm, chain)
        self.hfi =              HFIModulation(experiment_data, params, my_band, band_comm, chain)
        self.glitches =         Glitches(experiment_data, params, my_band, band_comm, chain)
        self.far_sidelobes =    FarSidelobes(experiment_data, params, my_band, band_comm, chain)
        self.diagnostics =      TODDiagnostics(experiment_data, params, my_band, band_comm, chain)
        # Every step above, in creation order, which is also the order of the collective gathers.
        self.steps: list[TODStep] = [value for value in vars(self).values()
                                     if isinstance(value, TODStep)]


    def gather_chain_arrays(self, params: Bunch, itr: int) -> dict[str, NDArray] | None:
        """Collect every per-band chain array onto the band master, keyed by dataset path.

        Collective over `band_comm`. Returns the datasets the band master writes to the band chain
        file, or ``None`` on the other ranks and on iterations whose chain output is skipped
        (`params.output.chains`). The write gate is applied here, before the gathers, so that all
        ranks leave together; `write_band_chain_to_file` only decides about the maps.
        """
        band_comm = self.band_comm

        # Every rank evaluates this on the same params and itr, so they all leave together.
        writes_chain = self.chain in params.output.chains.write
        if not writes_chain or not should_write_chain(params, "bands", itr):
            return None

        ####################################################################
        # Gather nscan info.
        ####################################################################
        if band_comm.Get_rank() == 0:
            scans_per_rank = np.zeros(band_comm.Get_size(), dtype=np.int64)
        else:
            scans_per_rank = None
        band_comm.Gather(sendbuf = np.array([self.experiment_data.nscans], dtype=np.int64),
                         recvbuf = scans_per_rank,
                         root=0)

        ####################################################################
        # Collect this rank's arrays.
        ####################################################################
        # First the band's identity: the unit of the gains and maps, and the detector names, whose
        # order is the `idet` axis of every per-detector array. Booleans go as int8, for MPI and
        # HDF5. Then each step's own arrays (see `ChainArrays`).
        per_scan: dict[str, NDArray] = {
            "scan_ids": self.scan_ids,
            "present": self.present.astype(np.int8),
            "scan_start_time": self.scan_start_time,
            "orbital_velocity": self.orbital_velocity,
        }
        per_band: dict[str, NDArray] = {
            "metadata/band_unit": self.experiment_data.band_unit,
            "det_names": np.array(self.det_names, dtype=h5py.string_dtype()),
        }
        per_detector_scan: dict[str, dict] = {}
        for step in self.steps:
            arrays = step.chain_arrays()
            per_scan.update(arrays.per_scan)
            per_band.update(arrays.per_band)
            per_detector_scan.update(arrays.per_detector_scan)

        ####################################################################
        # Gather them onto the band master.
        ####################################################################
        # The gathers are collective, so every rank must make them in the same order. It does: the
        # dicts above hold the same names, in the same order, on every rank of the band.
        global_per_scan: dict[str, NDArray | None] = {}
        for name, values in per_scan.items():
            global_per_scan[name] = gather_scan_distributed_array(band_comm, values,
                                                                  scans_per_rank)
        # Arrays whose length differs between detector-scans go as one object gather per group,
        # of (scan ID, detector name, arrays) entries.
        global_per_detector_scan: dict[str, list | None] = {}
        for group, arrays_by_detector_scan in per_detector_scan.items():
            entries = []
            for (iscan, idet), arrays in arrays_by_detector_scan.items():
                entries.append((self.scan_ids[iscan], self.det_names[idet], arrays))
            global_per_detector_scan[group] = band_comm.gather(entries, root=0)

        ####################################################################
        # Assemble the datasets, on the band master only.
        ####################################################################
        if band_comm.Get_rank() != 0:
            return None

        # The top level of the band chain file, dataset name by dataset name.
        datasets: dict[str, NDArray] = dict(per_band)
        # The ranks hold their scans in no particular order, so the per-scan arrays are sorted
        # back into time order, which is scan-ID order.
        time_order = np.argsort(global_per_scan["scan_ids"])
        for name, values in global_per_scan.items():
            datasets[name] = values[time_order]
        for group, entries_by_rank in global_per_detector_scan.items():
            for entries in entries_by_rank:
                for scan_id, det_name, arrays in entries:
                    for name, values in arrays.items():
                        datasets[f"{group}/{scan_id}/{det_name}/{name}"] = values
        return datasets


    def read_chain(self, path: str) -> None:
        """Continue from a saved sample: overwrite the sampled state with the one in `path`.

        `path` is a band chain file holding what `gather_chain_arrays` wrote. Every rank reads only
        its own scans (see `SavedBandSample`), and each step reads back its own restart state.
        """
        band_name = self.experiment_data.band_name
        if self.band_comm.Get_rank() == 0:
            logger.info(f"Band {band_name}, chain {self.chain}: continuing from {path}.")
        with h5py.File(path, "r") as f:
            saved = SavedBandSample(f, self.scan_ids, self.det_names, band_name)
            for step in self.steps:
                step.read_chain(saved)
