import numpy as np
import logging
from numpy.typing import NDArray
import ducc0
import os
from commander4.cmdr4_support import utils as cpp_utils
import commander4.output.log as log

class DetectorTOD:
    """Holds time-ordered data (TOD) for a single detector within a scan.

    The raw pixel and polarization angle arrays are stored in Huffman-compressed form
    and decompressed on demand via the ``pix`` and ``psi`` properties. The TOD array,
    evaluation nside, data nside, and sampling frequency are stored as plain public
    attributes.

    Attributes:
        tod (NDArray[np.floating]): 1-D array of calibrated time-ordered samples.
        ntod (int): Number of time samples (after any Fourier-length cropping).
        nside (int): HEALPix nside at which this detector should be evaluated.
        data_nside (int): HEALPix nside at which the pixel indices are stored on disk.
        fsamp (float): Sampling frequency in Hz.
    """
    def __init__(
        self,
        tod: NDArray[np.floating],
        pix_encoded: NDArray[np.integer] | bytes,
        psi_encoded: NDArray[np.integer] | NDArray[np.floating] | bytes,
        nside: int,
        data_nside: int,
        fsamp: float,
        orb_dir_vec: NDArray[np.floating] | None,
        huffman_tree: NDArray,
        huffman_symbols: NDArray,
        npsi: int,
        processing_mask_map: NDArray[np.bool_],
        ntod_original: int,
        flag_encoded: NDArray[np.integer] | bytes | None = None,
        flag_bitmask: int | None = None,
        pix_is_compressed: bool = True,
        psi_is_compressed: bool = True,
    ):
        """Construct a DetectorTOD.

        Args:
            tod: 1-D floating-point array of calibrated time samples.
            pix_encoded: Huffman-encoded (or raw) pixel index array.
            psi_encoded: Huffman-encoded (or raw) polarization angle array.
            nside: HEALPix nside for map evaluation.
            data_nside: HEALPix nside the pixel indices are stored at on disk.
            fsamp: Sampling frequency in Hz.
            orb_dir_vec: Unit vector of the spacecraft orbital velocity (size 3),
                or None if orbital dipole is not used.
            huffman_tree: Huffman decoding tree (passed to C++ decoder).
            huffman_symbols: Huffman symbol table (passed to C++ decoder).
            npsi: Number of discretised polarization angle bins.
            processing_mask_map: Boolean HEALPix map selecting valid pixels.
            ntod_original: Original TOD length before Fourier-length cropping.
            flag_encoded: Huffman-encoded flag array, or None.
            flag_bitmask: Bitmask applied to flags to identify excluded samples.
            pix_is_compressed: Whether ``pix_encoded`` is Huffman-compressed.
            psi_is_compressed: Whether ``psi_encoded`` is Huffman-compressed.
        """
        logger = logging.getLogger(__name__)
        log.logassert_np(tod.ndim==1, "'value' must be a 1D array", logger)
        log.logassert_np(tod.dtype in [np.float64,np.float32], "TOD dtype must be floating type,"
                         f" is {tod.dtype}", logger)
        log.logassert_np(processing_mask_map.dtype == bool, "Processing mask is not boolean type",
                         logger)
        if orb_dir_vec is not None:
            log.logassert_np(orb_dir_vec.size == 3, "orb_dir_vec must be a vector of size 3.", logger)
            self._orb_dir_vec = orb_dir_vec.astype(np.float32)
        else:
            self._orb_dir_vec = None
        self.tod = tod
        self.ntod = self.tod.shape[-1]
        self._pix_encoded = pix_encoded
        self._psi_encoded = psi_encoded
        self._flag_encoded = flag_encoded
        self._flag_bitmask = flag_bitmask
        self.nside = nside
        self.data_nside = data_nside
        self.fsamp = fsamp
        self._huffman_tree = huffman_tree
        self._huffman_symbols = huffman_symbols
        self._ntod_original = ntod_original  # Size of the original TOD before Fourier cropping.
        self._npsi = npsi
        self._pix_is_compressed = pix_is_compressed
        self._psi_is_compressed = psi_is_compressed
        self._processing_mask_TOD = np.packbits(processing_mask_map[self.pix])


    @property
    def nsamples(self) -> int:
        """Number of time samples in the TOD (alias for ``ntod``)."""
        return self.tod.shape[0]

    @property
    def pix(self) -> NDArray[np.integer]:
        """Decompressed HEALPix pixel indices at the evaluation nside.

        If the stored pixel array is Huffman-compressed, it is decoded and
        cumulative-summed on each access. When ``data_nside != nside`` the
        indices are re-projected to the evaluation resolution.
        """
        if self._pix_is_compressed:
            pix = np.zeros(self._ntod_original, dtype=np.int64)
            pix = cpp_utils.huffman_decode(np.frombuffer(self._pix_encoded, dtype=np.uint8),
                                           self._huffman_tree, self._huffman_symbols, pix)
            #TODO: Include cumsum in the C++ decode, so it can't be forgotten?
            pix = np.cumsum(pix)
        else:
            pix = self._pix_encoded
        # The TOD was cropped to an ideal Fourier length, but because the pix entry is compressed,
        # we need to unpack the entire original array, and then crop it to the correct length.
        pix = pix[:self.ntod]
        
        if self.nside != self.data_nside:
            # If the data nside does not match the specified evaluation nside, we convert to it.
            # pix = hp.ang2pix(self.nside, *hp.pix2ang(self.data_nside, pix))
            nthreads = int(os.environ["OMP_NUM_THREADS"])
            geom_from = ducc0.healpix.Healpix_Base(self.data_nside, "RING")
            geom_to = ducc0.healpix.Healpix_Base(self.nside, "RING")
            ang = geom_from.pix2ang(pix, nthreads=nthreads)
            pix = geom_to.ang2pix(ang, nthreads=nthreads)
        return pix

    @property
    def psi(self) -> NDArray[np.floating]:
        """Decompressed polarization angles in radians.

        If the stored array is Huffman-compressed, it is decoded, cumulative-summed,
        and converted from integer bins to radians on each access.
        """
        if self._psi_is_compressed:
            psi = np.zeros(self._ntod_original, dtype=np.int64)
            psi = cpp_utils.huffman_decode(np.frombuffer(self._psi_encoded, dtype=np.uint8),
                                        self._huffman_tree, self._huffman_symbols, psi)
            psi = np.cumsum(psi)
            psi = psi[:self.ntod]
            psi = 2*np.pi * psi.astype(np.float32)/self._npsi
        else:
            psi = self._psi_encoded
        return psi[:self.ntod]  # Crop to actual size (might be cut to fast FFT length)
        
    @property
    def processing_mask_TOD(self) -> NDArray[np.bool_]:
        """Boolean mask selecting valid (unmasked) TOD samples.

        Stored internally as a packed bit array and unpacked on each access.
        """
        mask = np.unpackbits(self._processing_mask_TOD).view(bool)
        if mask.size > self.tod.size + 7 or mask.size < self.tod.size:
            # The bytearray is stored in multiples of 8, so it can be up to 7 elements
            # longer than the TOD. If it's even longer or shorter, something is wrong.
            raise ValueError(f"Mask size {mask.size} doesn't match TOD size {self.tod.size}.")
        return mask[:self.tod.size]

    @property
    def flags(self) -> NDArray[np.floating]:
        """
        Returns the uncompressed flag array.
        """
        flags = np.zeros(self._ntod_original, dtype=np.int64)
        flags = cpp_utils.huffman_decode(np.frombuffer(self._flag_encoded, dtype=np.uint8), 
                                        self._huffman_tree, self._huffman_symbols, flags)
        flags = np.cumsum(flags)
        flags = flags[:self.ntod]
        return flags

    @property
    def excluded_tod_mask(self) -> NDArray[np.bool_]:
        """
        Returns a mask given by the intersection between the flag array and the flag bitmask.
        """
        return (self.flags & self._flag_bitmask).astype(np.bool_)
    

    @property
    def orb_dir_vec(self) -> NDArray[np.floating]:
        """Unit vector of the spacecraft orbital velocity (size 3).

        Raises:
            ValueError: If the orbital direction vector was not set at construction.
        """
        if self._orb_dir_vec is not None:
            return self._orb_dir_vec
        else:
            raise ValueError("Attempted to access self.orb_dir_vec, which is not set.")