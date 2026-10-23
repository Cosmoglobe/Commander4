"""Small MPI utilities shared by the setup, transfer and TOD code."""
import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray


def identify_masters(global_comm, split_comm, split_id):
    """Identify and share which global ranks are masters of each split."""
    # Prepare data: only sub-masters (local rank 0) provide meaningful data
    data = (split_id, global_comm.Get_rank()) if split_comm.Get_rank() == 0 else None

    # Gather data from all processes to all processes
    all_data = global_comm.allgather(data)

    # Process gathered data into a dictionary mapping split_ids to master ranks
    return {item[0]: item[1] for item in all_data if item is not None}


def gather_scan_distributed_array(band_comm: MPI.Comm, local_array: NDArray,
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
