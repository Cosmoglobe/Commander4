import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

# MPI elementary datatypes for the float buffers exchanged by the collectives below.
_NUMPY_TO_MPI_DTYPE = {np.dtype(np.float64): MPI.DOUBLE, np.dtype(np.float32): MPI.FLOAT}

# Map distribution across the ranks of a band communicator.
#
# Commander4 builds each band map collaboratively: every rank accumulates the contributions of its
# own detector-scans into a map, and the contributions are summed onto the band master, which holds
# the full-sky result. The naive layout gives every rank a full-sky (ncomp, npix) buffer even when
# its scans only ever touch a small fraction of the sky. ``PixelDomain`` factors out the pixel
# bookkeeping and the reduction/scatter MPI so the per-rank buffers can instead hold only the pixels
# that rank observes ("sparse" mode), while the master still ends up with a full-sky map.
#
# Two modes:
#   - "full":   the historical behaviour. ``local_pix`` is the whole sky, ``to_local`` is the
#               identity, and the collectives are a plain ``Reduce`` / ``Bcast`` over npix.
#   - "sparse": each rank holds ``local_pix`` (the sorted unique global pixels its scans touch).
#               Local buffers are (ncomp, n_local). Reduction is a ``Gatherv`` of the local data to
#               the master followed by a scatter-add into the full-sky map; the symmetric scatter
#               (master -> ranks, for the CG LHS) is a ``Scatterv`` of the per-rank pixel slices.
#               The index plan (counts/displacements and the concatenated global pixels) is static
#               across Gibbs iterations and is exchanged once at construction.


class PixelDomain:
    """Owns a rank's local pixel set and the MPI that maps local map buffers to a full-sky map.

    Attributes:
        comm: Band communicator across which the map is reduced.
        nside: HEALPix nside of the full-sky map.
        npix: Full-sky pixel count (``12*nside**2``).
        mode: ``"full"`` or ``"sparse"``.
        local_pix: Sorted unique global pixel ids held by this rank (sparse mode), else ``None``.
        n_local: Length of the local buffer (``n_local == npix`` in full mode).
    """

    def __init__(self, comm: MPI.Comm, nside: int, mode: str,
                 local_pix: NDArray | None = None):
        self.comm = comm
        self.nside = nside
        self.npix = 12 * nside**2
        self.mode = mode
        rank = comm.Get_rank()

        if mode == "full":
            self.local_pix = None  # identity remapping; buffers are full-sky.
            self.n_local = self.npix
            return
        if mode != "sparse":
            raise ValueError(f"Unknown PixelDomain mode '{mode}'.")

        self.local_pix = np.ascontiguousarray(local_pix, dtype=np.int64)
        self.n_local = int(self.local_pix.size)
        # Static gather plan, exchanged once: per-rank element counts, their displacements, and the
        # concatenation of every rank's global pixels (held only on the master for the scatter-add).
        counts = np.asarray(comm.allgather(self.n_local), dtype=np.int32)
        self._recvcounts = counts
        self._displs = np.insert(np.cumsum(counts), 0, 0)[:-1].astype(np.int32)
        self._all_pix = np.empty(int(counts.sum()), dtype=np.int64) if rank == 0 else None
        recvbuf = [self._all_pix, counts, self._displs, MPI.INT64_T] if rank == 0 else None
        comm.Gatherv(self.local_pix, recvbuf, root=0)

    @classmethod
    def from_view(cls, scan_view, comm: MPI.Comm, mode: str, nside: int) -> "PixelDomain":
        """Build a domain by collecting the pixels every present detector-scan touches.

        The local pixel set is the union over *all present* detector-scans (the accept flag is
        ignored on purpose), so the domain is a static superset of both the masked and the unmasked
        pointing used by every downstream mapmaker and does not change if ``accept`` toggles between
        Gibbs iterations. Pointing is accessed through the ``TODView`` (``view.pix``), never decoded
        directly. ``"full"`` mode skips the pass entirely.
        """
        if mode == "full":
            return cls(comm, nside, "full")
        # A transient full-sky boolean hitmap (npix bytes) is the cheapest robust way to union the
        # pixels; it is freed before the large accumulation buffers are allocated.
        hit = np.zeros(12 * nside**2, dtype=bool)
        for view in scan_view.iter_focused():
            hit[view.pix] = True
        local_pix = np.flatnonzero(hit).astype(np.int64)
        return cls(comm, nside, "sparse", local_pix=local_pix)

    def to_local(self, pix: NDArray) -> NDArray:
        """Map global HEALPix indices to compact local-buffer indices (identity in full mode)."""
        if self.mode == "full":
            return pix
        # local_pix is sorted and contains every pixel this rank can pass, so searchsorted is exact.
        return np.searchsorted(self.local_pix, pix)

    def reduce_to_full(self, local_data: NDArray, root: int = 0) -> NDArray | None:
        """Sum the per-rank local buffers into a full-sky map on ``root`` (else return ``None``).

        Accepts a 1-D ``(n_local,)`` buffer (scalar maps) or a 2-D ``(ncomp, n_local)`` buffer; the
        returned full-sky map matches the input rank (``(npix,)`` or ``(ncomp, npix)``).
        """
        rank = self.comm.Get_rank()
        if self.mode == "full":
            out = np.zeros_like(local_data) if rank == root else None
            self.comm.Reduce(local_data, out, op=MPI.SUM, root=root)
            return out

        is_1d = local_data.ndim == 1
        data2d = local_data.reshape(1, -1) if is_1d else local_data
        ncomp = data2d.shape[0]
        counts, displs = self._recvcounts, self._displs
        if rank == root:
            full = np.zeros((ncomp, self.npix), dtype=np.float64)
            recv = np.empty(int(counts.sum()), dtype=np.float64)
        for c in range(ncomp):
            send = np.ascontiguousarray(data2d[c], dtype=np.float64)
            recvbuf = [recv, counts, displs, MPI.DOUBLE] if rank == root else None
            self.comm.Gatherv(send, recvbuf, root=root)
            if rank == root:
                # Pixels observed by several ranks land at repeated indices; add.at accumulates them.
                np.add.at(full[c], self._all_pix, recv)
        if rank != root:
            return None
        return full[0] if is_1d else full

    def scatter_from_full(self, full_data: NDArray | None, ncomp: int, root: int = 0,
                          dtype=np.float64) -> NDArray:
        """Send each rank the values of a full-sky map at its local pixels (inverse of reduce).

        Used by the CG LHS (full map -> per-rank operator slice) and by the sky-model distribution.
        Returns an ``(ncomp, n_local)`` buffer on every rank. In full mode this is a plain broadcast
        of the full-sky map. ``dtype`` selects the float precision of the exchanged buffers.
        """
        rank = self.comm.Get_rank()
        if self.mode == "full":
            buf = np.ascontiguousarray(full_data, dtype=dtype) if rank == root \
                else np.empty((ncomp, self.npix), dtype=dtype)
            self.comm.Bcast(buf, root=root)
            return buf

        counts, displs = self._recvcounts, self._displs
        mpi_dtype = _NUMPY_TO_MPI_DTYPE[np.dtype(dtype)]
        local = np.empty((ncomp, self.n_local), dtype=dtype)
        for c in range(ncomp):
            if rank == root:
                # Gather the full map at the concatenated per-rank pixels, then scatter the slices.
                send = np.ascontiguousarray(full_data[c][self._all_pix], dtype=dtype)
                sendbuf = [send, counts, displs, mpi_dtype]
            else:
                sendbuf = None
            self.comm.Scatterv(sendbuf, local[c], root=root)
        return local
