from dataclasses import dataclass

import h5py
import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class JumpCorrection:
    """Additive offsets that are applied after one or more detected jump locations."""

    locations: NDArray[np.int64]
    offsets: NDArray[np.float32]

    def __post_init__(self):
        """Normalize storage and validate that the jump metadata is well-formed."""
        self.locations = np.asarray(self.locations, dtype=np.int64)
        self.offsets = np.asarray(self.offsets, dtype=np.float32)
        if self.locations.ndim != 1 or self.offsets.ndim != 1:
            raise ValueError("JumpCorrection expects 1-D locations and offsets arrays.")
        if self.locations.size != self.offsets.size:
            raise ValueError("jump locations and offsets must have the same length.")

    @classmethod
    def empty(cls) -> "JumpCorrection":
        """Return an empty correction object for detectors without any detected jumps."""
        return cls(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))

    @property
    def size(self) -> int:
        """Number of jump discontinuities represented by this correction."""
        return int(self.locations.size)

    def is_empty(self) -> bool:
        """Return whether this correction contains any offsets."""
        return self.size == 0

    def apply(
        self,
        tod: NDArray[np.floating],
        *,
        inplace: bool = False,
    ) -> NDArray[np.floating]:
        """Apply the stored offsets to a TOD array in detector units."""
        corrected_tod = tod if inplace else np.array(tod, copy=True)
        for jump_location, jump_offset in zip(self.locations, self.offsets):
            corrected_tod[jump_location:] += jump_offset
        return corrected_tod

    @classmethod
    def detect(
        cls,
        tod: NDArray[np.floating],
        flag: NDArray[np.integer],
        valid_mask: NDArray[np.bool_],
        n_window: int,
        *,
        jump_bitmask: int,
    ) -> tuple["JumpCorrection", int]:
        """Estimate jump offsets from flagged regions and neighboring valid samples.

        Args:
            tod: Raw detector TOD in detector units.
            flag: Per-sample flag stream. Contiguous regions with a non-zero
                ``flag & jump_bitmask`` mark jumps.
            valid_mask: Boolean mask defining which samples are allowed in the pre/post windows.
            n_window: Number of valid samples to average on each side of a jump.
            jump_bitmask: Integer bitmask used to tag jumps in the flag stream.

        Returns:
            A ``JumpCorrection`` object plus the number of flagged jump regions that were skipped
            because either side lacked enough valid samples.
        """
        if n_window < 1:
            raise ValueError("n_window must be >= 1.")
        if jump_bitmask < 1:
            raise ValueError("jump_bitmask must be >= 1.")

        jump_indices = np.flatnonzero((flag & jump_bitmask) != 0)
        if jump_indices.size == 0:
            return cls.empty(), 0

        breaks = np.flatnonzero(np.diff(jump_indices) > 1)
        jump_starts = np.concatenate(([jump_indices[0]], jump_indices[breaks + 1]))
        jump_stops = np.concatenate((jump_indices[breaks] + 1, [jump_indices[-1] + 1]))
        valid_indices = np.flatnonzero(valid_mask)
        corrected_tod = np.array(tod, copy=True)
        jump_locations = []
        jump_offsets = []
        num_skipped = 0

        for jump_start, jump_stop in zip(jump_starts, jump_stops):
            before_stop = np.searchsorted(valid_indices, jump_start, side="left")
            after_start = np.searchsorted(valid_indices, jump_stop, side="left")
            before_indices = valid_indices[max(0, before_stop - n_window):before_stop]
            after_indices = valid_indices[after_start:after_start + n_window]
            if before_indices.size < n_window or after_indices.size < n_window:
                num_skipped += 1
                continue

            mean_before = np.mean(corrected_tod[before_indices], dtype=np.float64)
            mean_after = np.mean(corrected_tod[after_indices], dtype=np.float64)
            jump_offset = float(mean_before - mean_after)

            # Later jumps should be estimated relative to the already corrected baseline.
            corrected_tod[jump_stop:] += jump_offset
            jump_locations.append(int(jump_stop))
            jump_offsets.append(jump_offset)

        return cls(jump_locations, jump_offsets), num_skipped


class JumpCatalog:
    """Per-scan and per-detector container for jump corrections."""

    def __init__(self, entries: NDArray):
        """Wrap a 2-D object array of ``JumpCorrection`` instances."""
        if entries.ndim != 2:
            raise ValueError("JumpCatalog expects a 2-D object array.")
        self._entries = entries
        self.nscans, self.ndet = entries.shape

    @classmethod
    def empty(cls, nscans: int, ndet: int) -> "JumpCatalog":
        """Allocate an empty jump catalog for one MPI rank's local scans."""
        entries = np.empty((nscans, ndet), dtype=object)
        for iscan in range(nscans):
            for idet in range(ndet):
                entries[iscan, idet] = JumpCorrection.empty()
        return cls(entries)

    @classmethod
    def from_hdf5(
        cls,
        file: h5py.File,
        local_indices: list[int],
        ndet: int,
    ) -> "JumpCatalog":
        """Reconstruct a local jump catalog from packed HDF5 datasets when present."""
        catalog = cls.empty(len(local_indices), ndet)
        dataset_names = {"jump_counts", "jump_locations", "jump_offsets"}
        if not dataset_names.issubset(file.keys()):
            return catalog

        counts_global = file["jump_counts"][:]
        locations_global = file["jump_locations"][:]
        offsets_global = file["jump_offsets"][:]
        counts_flat = counts_global.reshape(-1)
        starts_flat = np.cumsum(counts_flat, dtype=np.int64) - counts_flat

        for iscan_local, iscan_global in enumerate(local_indices):
            row_start = iscan_global * ndet
            for idet in range(ndet):
                flat_index = row_start + idet
                count = int(counts_flat[flat_index])
                start = int(starts_flat[flat_index])
                stop = start + count
                catalog.set(
                    iscan_local,
                    idet,
                    JumpCorrection(locations_global[start:stop], offsets_global[start:stop]),
                )
        return catalog

    def get(self, iscan: int, idet: int) -> JumpCorrection:
        """Return the correction object for one detector in one scan."""
        jump = self._entries[iscan, idet]
        return JumpCorrection.empty() if jump is None else jump

    def set(self, iscan: int, idet: int, jump: JumpCorrection | None):
        """Store one detector-local correction, replacing ``None`` by an empty correction."""
        self._entries[iscan, idet] = JumpCorrection.empty() if jump is None else jump

    def apply(
        self,
        tod: NDArray[np.floating],
        iscan: int,
        idet: int,
        *,
        inplace: bool = False,
    ) -> NDArray[np.floating]:
        """Apply the stored correction for one detector and scan to a TOD array."""
        return self.get(iscan, idet).apply(tod, inplace=inplace)

    def pack(self) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]:
        """Pack ragged corrections into counts plus flat arrays for MPI/HDF5 output."""
        counts = np.zeros((self.nscans, self.ndet), dtype=np.int64)
        flat_locations = []
        flat_offsets = []

        for iscan in range(self.nscans):
            for idet in range(self.ndet):
                jump = self.get(iscan, idet)
                counts[iscan, idet] = jump.size
                if not jump.is_empty():
                    flat_locations.append(jump.locations)
                    flat_offsets.append(jump.offsets)

        packed_locations = (
            np.concatenate(flat_locations) if flat_locations else np.empty(0, dtype=np.int64)
        )
        packed_offsets = (
            np.concatenate(flat_offsets) if flat_offsets else np.empty(0, dtype=np.float32)
        )
        return counts, packed_locations, packed_offsets