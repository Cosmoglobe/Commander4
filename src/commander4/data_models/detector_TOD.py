import numpy as np
import logging
from numpy.typing import NDArray

from commander4.cmdr4_support import utils as cpp_utils
from commander4.data_models.pointing import PixelPointing, DetectorBoresightPointing
import commander4.output.log as log
from commander4.logging.performance_logger import benchmark, bench_summary, start_bench,\
                                            stop_bench, log_memory, increment_count, bench_reset

logger = logging.getLogger(__name__)


class DetectorTOD:
    """Holds time-ordered data (TOD) for a single detector within a scan.

    The pointing information is stored in a ``PixelPointing`` or
    ``DetectorBoresightPointing`` object and exposed via ``get_pix()``,
    ``get_psi()``, and ``get_pix_psi()``. The TOD array, evaluation nside,
    data nside, and sampling frequency are stored as plain public attributes.

    Attributes:
        tod (NDArray[np.floating]): 1-D array of calibrated time-ordered samples.
        ntod (int): Number of time samples (after any Fourier-length cropping).
        nside (int): HEALPix nside at which this detector should be evaluated.
        data_nside (int): HEALPix nside at which the pixel indices are stored on disk.
        fsamp (float): Sampling frequency in Hz.
        init_scalars (array): 4-element array of initial guesses for gain, sigma0, fknee, and alpha.
    """
    def __init__(
        self,
        tod: NDArray[np.floating],
        pointing: PixelPointing | DetectorBoresightPointing,
        fsamp: float,
        orb_dir_vec: NDArray[np.floating] | None,
        huffman_tree: NDArray | None,
        huffman_symbols: NDArray | None,
        processing_mask_map: NDArray[np.bool_],
        ntod_original: int,
        ntod_optimal: int,
        huffman_tree2: NDArray | None = None,
        huffman_symbols2: NDArray | None = None,
        flag_encoded: NDArray[np.integer] | bytes | None = None,
        bad_data_bitmask: int | None = None,
        init_scalars: NDArray | None = None,
        tod_is_compressed: bool = True,
        flag_is_compressed: bool = True,
        det_response: NDArray | None = None,
    ):
        """Construct a DetectorTOD.

        Args:
            tod: 1-D floating-point array of calibrated time samples.
            pointing: Pointing representation for this detector. Must be a
                ``PixelPointing`` or ``DetectorBoresightPointing`` instance.
            fsamp: Sampling frequency in Hz.
            orb_dir_vec: Unit vector of the spacecraft orbital velocity (size 3),
                or None if orbital dipole is not used.
            huffman_tree: Huffman decoding tree for the flag stream, or None.
            huffman_symbols: Huffman symbol table for the flag stream, or None.
            processing_mask_map: Boolean HEALPix map selecting valid pixels.
            ntod_original: Original TOD length before Fourier-length cropping.
            flag_encoded: Huffman-encoded flag array, or None.
            flag_bitmask: Bitmask applied to flags to identify excluded samples.
        """
        if not tod_is_compressed:
            log.logassert_np(tod.ndim==1, "'value' must be a 1D array", logger)
            log.logassert_np(tod.dtype in [np.float64,np.float32], "TOD dtype must be floating "\
                             f"type, is {tod.dtype}", logger)
        log.logassert_np(processing_mask_map.dtype == bool, "Processing mask is not boolean type",
                         logger)
        log.logassert_np(
            isinstance(pointing, (PixelPointing, DetectorBoresightPointing)),
            "pointing must be a PixelPointing or DetectorBoresightPointing instance.",
            logger,
        )
        log.logassert_np(
            pointing.ntod_original == ntod_original,
            "Pointing ntod_original does not match DetectorTOD ntod_original.",
            logger,
        )
        log.logassert_np(
            pointing.ntod == ntod_optimal,
            "Pointing ntod does not match DetectorTOD ntod_optimal.",
            logger,
        )
        if flag_encoded is not None and flag_is_compressed:
            log.logassert_np(
                huffman_tree is not None and huffman_symbols is not None,
                "Compressed flags require Huffman metadata.",
                logger,
            )
        self._tod = tod
        self.ntod_original = ntod_original
        self.ntod = ntod_optimal
        self.nside = pointing.nside
        self.data_nside = pointing.data_nside
        self.fsamp = fsamp
        self.init_scalars = init_scalars
        self._flag_encoded = flag_encoded
        self._bad_data_bitmask = bad_data_bitmask
        self._huffman_tree = huffman_tree
        self._huffman_symbols = huffman_symbols
        self._huffman_tree2 = huffman_tree2
        self._huffman_symbols2 = huffman_symbols2
        self._tod_is_compressed = tod_is_compressed
        self._flag_is_compressed = flag_is_compressed
        self.processing_mask_map = processing_mask_map
        self.pointing = pointing
        processing_mask = processing_mask_map[self.get_pix()]
        self._processing_mask = np.packbits(processing_mask)
        self.det_response = det_response
        if flag_encoded is not None and bad_data_bitmask is not None:
            bad_data_mask = ~(self.flag & bad_data_bitmask)
            self._bad_data_mask = np.packbits(bad_data_mask)
            self._full_mask = np.packbits(bad_data_mask & processing_mask)
        if orb_dir_vec is not None:
            log.logassert_np(orb_dir_vec.size == 3, "orb_dir_vec must be a vector of size 3.", logger)
            self._orb_dir_vec = orb_dir_vec.astype(np.float32, copy=False)
        else:
            self._orb_dir_vec = None


    @property
    def tod(self) -> NDArray[np.floating]:
        if self._tod_is_compressed:
            tod = np.zeros(self.ntod_original, dtype=self._huffman_symbols2.dtype)
            tod[:] = cpp_utils.huffman_decode(np.frombuffer(self._tod, dtype=np.uint8),
                                    self._huffman_tree2, self._huffman_symbols2, tod)[:self.ntod]
            tod[:] = np.cumsum(tod)
            tod = tod.astype(np.float32)
        else:
            tod = self._tod
        return tod


    def get_pix(self, nside: int | None = None) -> NDArray[np.integer]:
        start_bench("pointing")
        pix = self.pointing.get_pix(nside)
        stop_bench("pointing")
        return pix

    def get_psi(self, nside: int | None = None) -> NDArray[np.integer] | NDArray[np.floating]:
        start_bench("pointing")
        psi = self.pointing.get_psi(nside)
        stop_bench("pointing")
        return psi

    def get_pix_psi(
        self,
        nside: int | None = None,
    ) -> tuple[NDArray[np.integer], NDArray[np.integer] | NDArray[np.floating]]:
        start_bench("pointing")
        pix_psi = self.pointing.get_pix_psi(nside)
        stop_bench("pointing")
        return pix_psi
        
    @property
    def flag(self) -> NDArray[np.integer]:
        if self._flag_is_compressed:
            flag = np.zeros(self.ntod_original, dtype=self._huffman_symbols.dtype)
            flag = cpp_utils.huffman_decode(np.frombuffer(self._flag_encoded, dtype=np.uint8),
                                           self._huffman_tree, self._huffman_symbols, flag)
            flag = np.cumsum(flag)
        else:
            flag = self._flag_encoded
        return flag[:self.ntod]

    @property
    def processing_mask(self) -> NDArray[np.bool_]:
        """Boolean mask selecting valid (unmasked) TOD samples.

        Stored internally as a packed bit array and unpacked on each access.
        """
        mask = np.unpackbits(self._processing_mask).view(bool)
        if mask.size > self.tod.size + 7 or mask.size < self.tod.size:
            # The bytearray is stored in multiples of 8, so it can be up to 7 elements
            # longer than the TOD. If it's even longer or shorter, something is wrong.
            raise ValueError(f"Mask size {mask.size} doesn't match TOD size {self.tod.size}.")
        return mask[:self.tod.size]


    @property
    def full_mask(self) -> NDArray[np.bool_]:
        mask = np.unpackbits(self._full_mask).view(bool)
        if mask.size > self.tod.size + 7 or mask.size < self.tod.size:
            raise ValueError(f"Mask size {mask.size} doesn't match TOD size {self.tod.size}.")
        return mask[:self.tod.size]

    @property
    def bad_data_mask(self) -> NDArray[np.bool_]:
        mask = np.unpackbits(self._bad_data_mask).view(bool)
        if mask.size > self.tod.size + 7 or mask.size < self.tod.size:
            raise ValueError(f"Mask size {mask.size} doesn't match TOD size {self.tod.size}.")
        return mask[:self.tod.size]

    # @property
    # def excluded_tod_mask(self) -> NDArray[np.bool_]:
    #     """
    #     Returns a mask given by the intersection between the flag array and the flag bitmask.
    #     """
    #     return (self.flags & self._flag_bitmask).astype(np.bool_)
    

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
        
    def IQU_response(self, psi: NDArray | None = None):
        # psi can be passed as an argument to avoid re-calculating it if already available.
        if psi is None:
            psi = self.get_psi()
        response = np.zeros((3, psi.shape[-1]))
        if self.det_response[0] != 0:
            response[0,:] = self.det_response[0]
        if self.det_response[1] != 0:
            response[1,:] = np.cos(2.0*psi)*self.det_response[1]
            response[2,:] = np.sin(2.0*psi)*self.det_response[1]
        return response