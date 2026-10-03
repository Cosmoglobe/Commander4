"""Container for the jumps found in the TOD.

`JumpEvents` holds the jumps of one detector-scan: sudden offsets of the baseline.
`TODSamples.jumps` holds one per detector-scan, in an (nscans, ndet) object array made by
`empty_jump_grid`. The step
that *finds* the jumps is `sampling.py` next to this file. The number of jumps per detector-scan is
always written to the chain file; the jumps themselves only as optional debug output. They are not
read back on restart, since detection finds them again from the data.
"""
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# The arrays of a `JumpEvents`, which are also its dataset names in the debug chain output.
JUMP_EVENT_FIELDS = ("locations", "offsets")


@dataclass(slots=True)
class JumpEvents:
    """The jumps of one detector-scan, with one entry per jump in both arrays.

    Attributes:
        locations: (njump,) First sample after each jump, from which its offset applies.
        offsets: (njump,) Offset in detector units that undoes each jump when added to the TOD.
    """

    locations: NDArray[np.int64]
    offsets: NDArray[np.float32]

    def __post_init__(self):
        """Normalize storage and check that both arrays have one entry per jump."""
        self.locations = np.asarray(self.locations, dtype=np.int64)
        self.offsets = np.asarray(self.offsets, dtype=np.float32)
        if self.locations.ndim != 1 or self.locations.shape != self.offsets.shape:
            raise ValueError("JumpEvents expects 1-D locations and offsets of the same length.")

    @classmethod
    def empty(cls) -> "JumpEvents":
        """Return a container without jumps, for detector-scans with no jumps found."""
        return cls(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))

    @property
    def num_events(self) -> int:
        """Number of jumps in this detector-scan."""
        return int(self.locations.size)

    def offset_tod(self, ntod: int) -> NDArray[np.float32]:
        """Return the (ntod,) offsets that undo the jumps when added to the raw TOD.

        Each jump's offset applies from its location to the end of the scan, so later jumps add
        on top of earlier ones.
        """
        offsets = np.zeros(ntod, dtype=np.float32)
        for location, offset in zip(self.locations, self.offsets):
            offsets[location:] += offset
        return offsets


def empty_jump_grid(nscans: int, ndet: int) -> NDArray[np.object_]:
    """Return an (nscans, ndet) object array with its own empty `JumpEvents` in every cell.

    The cells are filled one by one: `np.full` would put one shared object in every cell, so
    changing the arrays of one cell in place would change all of them.
    """
    grid = np.empty((nscans, ndet), dtype=object)
    for iscan in range(nscans):
        for idet in range(ndet):
            grid[iscan, idet] = JumpEvents.empty()
    return grid
