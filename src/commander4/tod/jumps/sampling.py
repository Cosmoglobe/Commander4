"""Jump detection: finding and correcting sudden baseline offsets in a detector-scan.

C4 samples this every Gibbs iteration (C3 leaves the equivalent commented out). The jumps are
stored per detector-scan in `TODSamples.jumps` (see `events.py` next to this file), and
`TODView.pre_demodulation_tod` adds their offsets to every later TOD request, so the data
themselves are never modified.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.diagnostics.performance import log_memory
from commander4.tod.config import JumpDetectionConfig
from commander4.tod.jumps.events import JumpEvents
from commander4.tod.view import TODView

if TYPE_CHECKING:
    from commander4.data_models.tod_samples import TODSamples

logger = logging.getLogger(__name__)


def sample_jump_detection(band_comm: MPI.Comm, experiment_data: DetectorGroupTOD,
                          tod_samples: TODSamples, compsep_output: NDArray,
                          jump_detection_cfg: JumpDetectionConfig, iteration: int) -> TODSamples:
    """Detect jump discontinuities from the flag stream and store additive post-jump offsets.

    A jump is identified by a contiguous region with a non-zero
    ``flag & experiments.[experiment_name].jump_bitmask``. For each region, the offset is
    estimated from the last ``window`` valid samples before the jump and the first ``window``
    valid samples after it, where validity is defined by the "jump" processing mask. The means
    are taken of the residual, the raw TOD minus the gain-scaled sky model and orbital dipole, so a
    change of the sky across the jump is not taken as part of it. The offset is then added to all
    later samples whenever a TOD is requested through ``TODView``.
    """
    if experiment_data.hfi_demodulation:
        raise ValueError(f"Band {experiment_data.band_name}: jump detection does not support HFI "
                         "demodulation, since the sky model would have to be modulated like the "
                         "raw TOD.")
    scan_view = TODView(experiment_data, tod_samples, compsep_output=compsep_output)
    num_applied_local = 0
    num_skipped_local = 0
    offsets_local = []
    jump_counts_local = []

    for view in scan_view.iter_focused():
        jumps, num_skipped = JumpEvents.empty(), 0
        # The sky model is only built where jumps are flagged, which is a small part of the data.
        if np.any(view.flag & jump_detection_cfg.jump_bitmask):
            # The sky model may hold a little of the jump itself, but that error is smaller than
            # taking the sky change across the jump into the offset.
            sky_tod = view.get_static_sky_tod() + view.get_orbital_dipole_tod()
            residual = view.raw_tod - view.get_gain() * sky_tod
            jumps, num_skipped = _detect_jumps(residual, view.flag,
                                               view.get_mask(proc_mask_type="jump"),
                                               jump_detection_cfg.window,
                                               jump_detection_cfg.jump_bitmask)
        tod_samples.jumps[view.iscan, view.idet] = jumps
        jump_counts_local.append(jumps.num_events)
        num_skipped_local += num_skipped
        if jumps.num_events > 0:
            offsets_local.extend(jumps.offsets.astype(np.float64, copy=False))
            num_applied_local += jumps.num_events

    num_applied = band_comm.reduce(num_applied_local, op=MPI.SUM, root=0)
    num_skipped = band_comm.reduce(num_skipped_local, op=MPI.SUM, root=0)
    gathered_offsets = band_comm.gather(np.asarray(offsets_local, dtype=np.float64), root=0)
    gathered_jump_counts = band_comm.gather(np.asarray(jump_counts_local, dtype=np.int32), root=0)

    if band_comm.Get_rank() == 0:
        all_jump_counts = np.concatenate(gathered_jump_counts)
        if all_jump_counts.size > 0:
            logger.debug(
                f"Band {experiment_data.band_name} jump counts per detector-scan: "
                f"min={np.min(all_jump_counts)}, avg={np.mean(all_jump_counts):.2f}, "
                f"max={np.max(all_jump_counts)} over {all_jump_counts.size} samples."
            )
        if num_applied > 0:
            all_offsets = np.concatenate([arr for arr in gathered_offsets if arr.size > 0])
            logger.info(f"Chain {tod_samples.chain} iter{iteration} "
                        f"{experiment_data.band_name} jump detection: applied {num_applied} "
                        f"offsets, skipped {num_skipped}, median |offset| = "
                        f"{np.median(np.abs(all_offsets)):.3e}.")
        elif num_skipped > 0:
            logger.info(f"Chain {tod_samples.chain} iter{iteration} "
                        f"{experiment_data.band_name} jump detection skipped {num_skipped} flagged "
                        "regions because there were not enough valid samples around them.")

    log_memory("jump-detect")
    return tod_samples


def _detect_jumps(tod: NDArray[np.floating], flag: NDArray[np.integer],
                  valid_mask: NDArray[np.bool_], n_window: int,
                  jump_bitmask: int) -> tuple[JumpEvents, int]:
    """Estimate jump offsets from flagged regions and neighboring valid samples.

    Args:
        tod: TOD in detector units to measure the offsets on, normally the residual.
        flag: Per-sample flag stream. Contiguous regions with a non-zero
            ``flag & jump_bitmask`` mark jumps.
        valid_mask: Boolean mask defining which samples are allowed in the pre/post windows.
        n_window: Number of valid samples to average on each side of a jump.
        jump_bitmask: Integer bitmask used to tag jumps in the flag stream.

    Returns:
        The jumps found, plus the number of flagged jump regions that were skipped because either
        side lacked enough valid samples.
    """
    jump_indices = np.flatnonzero((flag & jump_bitmask) != 0)
    if jump_indices.size == 0:
        return JumpEvents.empty(), 0

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

    return JumpEvents(jump_locations, jump_offsets), num_skipped
