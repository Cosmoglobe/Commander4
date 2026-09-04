"""One array in MPI shared memory, allocated once per node and read by every rank on it.

For data that is identical on every rank of a communicator and large enough that holding a private
copy per rank is wasteful. One rank allocates, every rank maps the same physical pages, and reads
cost nothing extra.

The memory must be released explicitly with `free()`. mpi4py does not free MPI windows when the
Python object is garbage collected -- `MPI_Win_free` is collective and cannot be called safely from
a destructor -- so an unfreed window survives until `MPI_Finalize` and the same allocation repeated
every Gibbs iteration grows without bound.
"""
import logging

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class SharedArray:
    """A numpy array backed by an MPI shared-memory window on one node.

    Rank 0 of `comm` owns the allocation and is the only rank allowed to write; every other rank
    receives a read-only view of the same memory. `comm` must therefore not span more than one
    node: build it with `comm.Split_type(MPI.COMM_TYPE_SHARED)`.

    Allocation and `free()` are both collective, so every rank of `comm` must construct and free
    the same arrays in the same order. Control flow between those two points is unconstrained.
    """

    def __init__(self, comm: MPI.Comm, shape: tuple[int, ...], dtype=np.float64,
                 name: str = "shared array"):
        """Allocate the window and map it on every rank.

        Args:
            comm: Single-node communicator; its rank 0 owns the allocation.
            shape: Shape of the resulting array.
            dtype: Element type of the resulting array.
            name: What this array holds, used to identify it if it is never freed.
        """
        self.comm = comm
        self.name = name
        self.is_owner = comm.Get_rank() == 0
        itemsize = np.dtype(dtype).itemsize
        # Only the owner requests memory; the others map its segment via Shared_query below.
        self.nbytes = int(np.prod(shape))*itemsize
        nbytes = self.nbytes if self.is_owner else 0
        self.win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)
        buf, _ = self.win.Shared_query(0)
        self.array: NDArray | None = np.ndarray(buffer=buf, dtype=dtype, shape=shape)
        if not self.is_owner:
            self.array.flags.writeable = False
        logger.debug(f"Allocated {nbytes/1024**3:.2f} GiB of shared memory for the {name} "
                     f"({shape}) across {comm.Get_size()} ranks.")


    def wait_until_filled(self) -> None:
        """Barrier marking that the owner has finished writing and the readers may start.

        Every rank of `comm` must call this after the owner has filled `array`, and none may read
        before it returns.
        """
        self.comm.Barrier()


    def free(self) -> None:
        """Release the window. Collective, and every view of `array` must already be gone.

        Reading a numpy view of the window after this returns is a use-after-free that segfaults
        rather than raising, so callers holding their own views (slices, for instance) must drop
        them before calling this.
        """
        if self.array is None:
            return
        self.array = None
        self.comm.Barrier()
        self.win.Free()


    def __del__(self):
        """Checks if memory was leaked, and let's the user know.

        The __del__ method is called when the garbage collector delets this object. If the shared
        memory array is not already freed when that happens, it will leak. We therefore check if it
        is indeed freed by now, and creates a logger error if not.
        """
        try:
            if getattr(self, "array", None) is not None:
                size = (f"{self.nbytes/1024**3:.2f} GiB" if self.nbytes >= 1024**3
                        else f"{self.nbytes/1024**2:.1f} MiB")
                logger.error(f"Shared memory for the {self.name} ({size}) was garbage collected "
                             "without free() having been called, so it stays allocated until "
                             "MPI_Finalize. Every further allocation leaks that much again.")
        except Exception:
            pass
