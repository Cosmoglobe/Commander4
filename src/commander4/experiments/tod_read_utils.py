from mpi4py import MPI
from pixell.bunch import Bunch
from numpy.typing import NDArray
import healpy as hp
import numpy as np


def find_good_Fourier_time(Fourier_times:NDArray, ntod:int) -> int:
    if ntod <= 10_000 or ntod >= 400_000:
        return ntod
    search_start = int(0.99*ntod)  # Consider sizes up to 1% smaller than ntod.
    best_ntod = np.argmin(Fourier_times[search_start:ntod+1])
    best_ntod += search_start
    assert(best_ntod <= ntod)
    return best_ntod


def read_processing_masks(band_comm: MPI.Comm,
                          band_params: Bunch) -> tuple[NDArray | None, dict[str, NDArray]]:
    """Read a band's default and named processing-mask maps once and broadcast them.

    Args:
        band_comm: The band's MPI communicator; only rank 0 touches the filesystem.
        band_params: The band's parameter block (``processing_mask`` and/or ``processing_masks``).

    Returns:
        ``(default_mask, named_masks)``: the default boolean HEALPix map (or ``None`` if the band
        defines none) and a dict of named boolean HEALPix maps (empty if none are defined). Maps are
        kept at their native nside; ``TODView`` handles any nside mismatch with the pointing.
    """
    default_mask = None
    named_masks: dict[str, NDArray] = {}
    if band_comm.Get_rank() == 0:
        filename = getattr(band_params, "processing_mask", None)
        if filename is not None:
            default_mask = hp.read_map(filename, field=0, dtype=bool)
        for name in getattr(band_params, "processing_masks", []) or []:
            named_masks[name] = hp.read_map(band_params.processing_masks[name], field=0, dtype=bool)
    # bcast returns the broadcast object (it does not fill in place), so capture the return value.
    default_mask = band_comm.bcast(default_mask, root=0)
    named_masks = band_comm.bcast(named_masks, root=0)
    return default_mask, named_masks
