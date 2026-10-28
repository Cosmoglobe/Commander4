"""Decoding of the Huffman-compressed TOD, pointing and flag arrays in the experiment scan files.

The format is Commander3's (``commander_tod.py``). Each compressed array (``flag``, ``pix``,
``psi``, ``ztod``) is one opaque HDF5 scalar, which h5py reads as numpy.void:

  * The array is stored as its first differences: the first sample, then each sample minus the one
    before, taken in the array's own integer dtype.
  * Psi is first digitized into ``npsi`` bins over [0, 2*pi), numbered from 1.
  * The differences are Huffman-coded with a tree shared by all detectors of a scan:
    ``<pid>/common/hufftree`` and ``huffsymb`` for flag, pix and psi, and ``hufftree2`` and
    ``huffsymb2`` for the TOD. Nodes 1..n are the n symbols, n+1..2n-1 the internal nodes, and the
    tree array is [root, left children, right children] of the internal nodes.
  * The first byte of the bit stream is the number of padding bits after the last code word: 1..8
    from Commander3's writer, 0..7 from the SO writer.

`HuffmanCompressedArray` holds one stored array and decodes it; the bit decoding itself is the C++
``huffman_decode``. `huffman_encoding.py` writes this format.
"""
import numpy as np
from numpy.typing import NDArray

from commander4.backend import compression as cpp_compression
from commander4.diagnostics.performance import benchmark


class HuffmanCompressedArray:
    """One compressed 1-D array, decoded on demand.

    The stored values are differences between neighboring samples, and, Huffman encoded.
    If ``nbins`` is given, the original array was digitized before that, such that bin i covers
    ``[vmin + (i-1)*width, vmin + i*width)`` with ``width = (vmax - vmin)/nbins``, and `decode`
    returns the bin centers.

    Args:
        payload: The stored bytes (an HDF5 opaque scalar reads as numpy.void).
        tree: The scan's Huffman tree, as int64. The reader converts it once per scan, so all
            detectors of the scan share one array.
        symbols: The scan's Huffman symbols, in the dtype the differences were taken in.
        nsamp: Number of samples the payload decodes to (the scan's full ntod).
        nbins, vmin, vmax: The digitization, or None if the array was not digitized.
    """
    def __init__(self, payload: bytes | np.void, tree: NDArray[np.int64],
                 symbols: NDArray[np.integer], nsamp: int, nbins: int | None = None,
                 vmin: float | None = None, vmax: float | None = None):
        self.payload = np.frombuffer(payload, dtype=np.uint8)  # Zero-copy view of the bytes.
        self.tree = tree
        self.symbols = symbols
        self.nsamp = nsamp
        self.nbins = nbins
        self.vmin = vmin
        self.vmax = vmax

    def decode(self) -> NDArray:
        """Return the original array: integers in the symbol dtype, or float32 bin centers."""
        with benchmark("huffman"):
            diffs = np.empty(self.nsamp, dtype=self.symbols.dtype)
            cpp_compression.huffman_decode(self.payload, self.tree, self.symbols, diffs)
            # Undo the differences in the symbol dtype, as the writer took them in it: SO SAT
            # stores a step of -1 as uint8 255, which only sums back correctly with the same
            # wrap-around.
            values = np.cumsum(diffs, dtype=diffs.dtype)
            if self.nbins is None:  # If not quantized we're done.
                return values
            else:  # If quantized we spread it across its bin range. (typically only true for psi).
                width = (self.vmax - self.vmin)/self.nbins
                return (self.vmin + (values - 0.5)*width).astype(np.float32)
