"""Jump detection: finding and correcting sudden baseline offsets in a detector-scan.

C4 samples this every Gibbs iteration (C3 leaves the equivalent commented out). The correction
itself is stored as a `JumpCatalog` on `TODSamples`, so it is applied to every later TOD request
rather than modifying the data in place.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar, TYPE_CHECKING

import numpy as np
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.jump_corrections import JumpCorrection
from commander4.diagnostics.performance import log_memory
from commander4.tod.step_config import StepConfig
from commander4.tod.view import TODView

if TYPE_CHECKING:
    from commander4.data_models.tod_samples import TODSamples

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JumpDetectionConfig(StepConfig):
    """Validated jump-detection parameters and experiment-specific flag bitmask."""

    PARAMETER_NAME: ClassVar[str] = "jump_detection"

    window: int = 10
    jump_bitmask: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.window, int) or isinstance(self.window, bool) or self.window < 1:
            raise ValueError("jump_detection.window must be an integer of at least 1.")
        if self.enabled and self.jump_bitmask is None:
            raise ValueError("Jump detection is enabled, but the experiment has no jump_bitmask.")
        if self.jump_bitmask is not None and not isinstance(self.jump_bitmask, int):
            raise ValueError("The experiment jump_bitmask must be an integer.")

    @classmethod
    def from_params(cls, params: Bunch, experiment_data: DetectorGroupTOD):
        """Build jump settings from their step block and the experiment flag bitmask.

        The parameter block owns scheduling and detection settings. The experiment owns the flag
        meaning, so ``jump_bitmask`` is injected as a resolved value rather than read from the
        parameter block.
        """
        experiment = params.experiments[experiment_data.experiment_name]
        jump_bitmask = experiment.jump_bitmask if "jump_bitmask" in experiment else None

        # An absent block is valid and produces the disabled StepConfig defaults.
        block = (params.tod_processing[cls.PARAMETER_NAME]
                 if cls.PARAMETER_NAME in params.tod_processing else Bunch())
        return cls._from_block(f"tod_processing.{cls.PARAMETER_NAME}", block,
                               jump_bitmask=jump_bitmask)


def sample_jump_detection(band_comm: MPI.Comm, experiment_data: DetectorGroupTOD,
                          tod_samples: TODSamples,
                          config: JumpDetectionConfig, iteration: int) -> TODSamples:
    """Detect jump discontinuities from the flag stream and store additive post-jump offsets.

    A jump is identified by a contiguous region with a non-zero
    ``flag & experiments.[experiment_name].jump_bitmask``. For each region, the offset is
    estimated from the last ``window`` valid samples before the jump and the first ``window``
    valid samples after it, where validity is defined by ``full_mask``. The correction is then
    applied to all later samples when a TOD is requested through ``TODView.get_tod()``.
    """
    scan_view = TODView(experiment_data, tod_samples)
    num_applied_local = 0
    num_skipped_local = 0
    offsets_local = []
    jump_counts_local = []

    for view in scan_view.iter_focused():
        # Jump detection needs the flag stream both to locate jumps (via jump_bitmask) and to
        # define valid pre/post-jump samples; skip detector-scans without it.
        if getattr(view.detector, "_flag_encoded", None) is None:
            tod_samples.jumps.set(view.iscan, view.idet, None)
            jump_counts_local.append(0)
            continue
        jump, num_skipped = JumpCorrection.detect(
            view.raw_tod,
            view.flag,
            view.get_mask(proc_mask_type="jump"),
            config.window,
            jump_bitmask=config.jump_bitmask,
        )
        tod_samples.jumps.set(view.iscan, view.idet, jump)
        jump_counts_local.append(jump.size)
        num_skipped_local += num_skipped
        if not jump.is_empty():
            offsets_local.extend(jump.offsets.astype(np.float64, copy=False))
            num_applied_local += jump.size

    num_applied = band_comm.reduce(num_applied_local, op=MPI.SUM, root=0)
    num_skipped = band_comm.reduce(num_skipped_local, op=MPI.SUM, root=0)
    gathered_offsets = band_comm.gather(np.asarray(offsets_local, dtype=np.float64), root=0)
    gathered_jump_counts = band_comm.gather(np.asarray(jump_counts_local, dtype=np.int32), root=0)

    if band_comm.Get_rank() == 0:
        all_jump_counts = np.concatenate(gathered_jump_counts) if gathered_jump_counts else np.empty(0)
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
