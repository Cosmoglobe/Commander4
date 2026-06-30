import numpy as np
import healpy as hp
import logging
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.cmdr4_support import utils as cpp_utils
from commander4.data_models.pointing import PixelPointing, DetectorBoresightPointing
import commander4.output.log as log
from commander4.logging.performance_logger import benchmark, bench_summary, start_bench,\
                                            stop_bench, log_memory, increment_count, bench_reset

logger = logging.getLogger(__name__)


class DetectorTOD:
    """Holds time-ordered data (TOD) for a single detector within a scan.
    This class is intended as a "data container", while the user-interaction should happen
    through the TODView class (`see tod_view.py`).

    The pointing information is stored in a ``PixelPointing`` or
    ``DetectorBoresightPointing`` object and exposed via ``get_pix()``,
    ``get_psi()``, and ``get_pix_psi()``. The TOD array, evaluation nside,
    data nside, and sampling frequency are stored as plain public attributes.

    Attributes:
        name (str): Unique name of this detector.
        tod (NDArray[np.floating]): 1-D array of calibrated time-ordered samples.
        ntod (int): Number of time samples (after any Fourier-length cropping).
        nside (int): HEALPix nside at which this detector should be evaluated.
        data_nside (int): HEALPix nside at which the pixel indices are stored on disk.
        fsamp (float): Sampling frequency in Hz.
    """
    def __init__(
        self,
        name: str,
        det_idx_fullband: int,
        det_idx_local: int,
        tod: NDArray[np.floating] | bytes | np.void,
        pointing: PixelPointing | DetectorBoresightPointing,
        fsamp: float,
        orb_dir_vec: NDArray[np.floating] | None,
        huffman_tree: NDArray | None,
        huffman_symbols: NDArray | None,
        default_proc_mask: NDArray[np.bool_] | None,
        specific_proc_masks: dict,
        ntod_original: int,
        ntod_optimal: int,
        huffman_tree2: NDArray | None = None,
        huffman_symbols2: NDArray | None = None,
        flag_encoded: NDArray[np.integer] | bytes | np.void | None = None,
        bad_data_bitmask: int | None = None,
        init_scalars: NDArray | None = None,
        tod_is_compressed: bool = False,
        flag_is_compressed: bool = True,
        det_response: NDArray | None = None,
    ):
        """Construct a DetectorTOD.

        Args:
            name: Unique name of the current detector.
            det_idx_fullband: Unique detector-index among all the detectors on the relevant band.
            det_idx_local: Unique detector-index among all detectors in the current scan, where
                some detectors might be missing from the full set of detectors on the band.
            tod: Calibrated time samples, either as a decoded 1-D floating-point
                array or as a compressed binary payload.
            pointing: Pointing representation for this detector. Must be a
                ``PixelPointing`` or ``DetectorBoresightPointing`` instance.
            fsamp: Sampling frequency in Hz.
            orb_dir_vec: Unit vector of the spacecraft orbital velocity (size 3),
                or None if orbital dipole is not used.
            huffman_tree: Huffman decoding tree for the flag stream, or None.
            huffman_symbols: Huffman symbol table for the flag stream, or None.
            default_proc_mask: Default processing-mask HEALPix map (boolean), or None if the band
                defines none. Used unless a sampling step requests a name in specific_proc_masks.
            specific_proc_masks: Mapping of operation name -> processing-mask HEALPix map for bands
                that define per-operation masks (empty dict if none).
            ntod_original: Original TOD length before Fourier-length cropping.
            flag_encoded: Flag samples, either decoded or Huffman-encoded, or None.
            flag_bitmask: Bitmask applied to flags to identify excluded samples.
        """
        if tod_is_compressed:
            log.logassert_np(
                isinstance(tod, (bytes, np.void)),
                "Compressed TOD must be provided as bytes or numpy.void.",
                logger,
            )
            log.logassert_np(
                huffman_tree2 is not None and huffman_symbols2 is not None,
                "Compressed TOD requires Huffman metadata.",
                logger,
            )
        else:
            log.logassert_np(isinstance(tod, np.ndarray), "'tod' must be a numpy array.", logger)
            log.logassert_np(tod.ndim==1, "'value' must be a 1D array", logger)
            log.logassert_np(tod.dtype in [np.float64,np.float32], "TOD dtype must be floating "\
                             f"type, is {tod.dtype}", logger)
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
        if flag_encoded is not None:
            if flag_is_compressed:
                log.logassert_np(
                    isinstance(flag_encoded, (bytes, np.void)),
                    "Compressed flags must be provided as bytes or numpy.void.",
                    logger,
                )
                log.logassert_np(
                    huffman_tree is not None and huffman_symbols is not None,
                    "Compressed flags require Huffman metadata.",
                    logger,
                )
            else:
                log.logassert_np(
                    isinstance(flag_encoded, np.ndarray),
                    "Decoded flags must be provided as a numpy array.",
                    logger,
                )
                log.logassert_np(flag_encoded.ndim == 1, "'flag' must be a 1D array.", logger)
                log.logassert_np(
                    np.issubdtype(flag_encoded.dtype, np.integer),
                    "Decoded flags must have integer dtype.",
                    logger,
                )
                log.logassert_np(
                    flag_encoded.size >= ntod_optimal,
                    f"'flag' length {flag_encoded.size} is shorter than ntod {ntod_optimal}.",
                    logger,
                )
        self.name = name
        self.det_idx_fullband = det_idx_fullband
        self.det_idx_local = det_idx_local
        self._tod = np.frombuffer(tod, dtype=np.uint8) if tod_is_compressed else tod
        self.ntod_original = ntod_original
        self.ntod = ntod_optimal
        self.nside = pointing.nside
        self.data_nside = pointing.data_nside
        self.fsamp = fsamp
        self.init_scalars = init_scalars
        self._flag_encoded = (
            np.frombuffer(flag_encoded, dtype=np.uint8)
            if flag_encoded is not None and flag_is_compressed
            else flag_encoded
        )
        self._bad_data_bitmask = bad_data_bitmask
        self._huffman_symbols = huffman_symbols
        self._huffman_tree = huffman_tree
        # C++ decoder accepts only int64 for the tree.
        if self._huffman_tree is not None:
            self._huffman_tree = self._huffman_tree.astype(np.int64, copy=False)
        self._huffman_symbols2 = huffman_symbols2
        self._huffman_tree2 = huffman_tree2
        if self._huffman_tree2 is not None:
            self._huffman_tree2 = self._huffman_tree2.astype(np.int64, copy=False)
        self._tod_is_compressed = tod_is_compressed
        self._flag_is_compressed = flag_is_compressed
        # The Huffman decoder expects uint8 arrays; for bytes and HDF5-backed
        # numpy.void payloads the internal storage is rewritten as a zero-copy
        # uint8 view once at construction.
        self.pointing = pointing
        self.det_response = det_response
        if flag_encoded is not None and bad_data_bitmask is not None:
            good_data_mask = (self.flag & bad_data_bitmask) == 0
            self._good_data_mask = np.packbits(good_data_mask)
        if orb_dir_vec is not None:
            log.logassert_np(orb_dir_vec.size == 3, "orb_dir_vec must be a vector of size 3.", logger)
            self._orb_dir_vec = orb_dir_vec.astype(np.float32, copy=False)
        else:
            self._orb_dir_vec = None
        # Band-level processing-mask HEALPix maps (shared references across the band's detectors;
        # TODView projects them onto this detector's pointing on demand). The default mask is used
        # unless a sampling step requests a name present in specific_proc_masks.
        self.default_proc_mask = default_proc_mask
        self.specific_proc_masks = specific_proc_masks


    @property
    def tod(self) -> NDArray[np.floating]:
        if self._tod_is_compressed:
            tod = np.zeros(self.ntod_original, dtype=self._huffman_symbols2.dtype)
            tod[:] = cpp_utils.huffman_decode(self._tod,
                                    self._huffman_tree2, self._huffman_symbols2, tod)
            tod[:] = np.cumsum(tod)
            tod = tod.astype(np.float32)
        else:
            tod = self._tod
        return tod[:self.ntod]


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
            flag = cpp_utils.huffman_decode(self._flag_encoded,
                                           self._huffman_tree, self._huffman_symbols, flag)
            flag = np.cumsum(flag)
        else:
            flag = self._flag_encoded
        return flag[:self.ntod]


    @property
    def good_data_mask(self) -> NDArray[np.bool_]:
        """Boolean mask keeping samples that pass the bad-data flag cut."""
        start_bench("numpy-unpack")
        mask = np.unpackbits(self._good_data_mask).view(bool)
        stop_bench("numpy-unpack")
        if mask.size > self.tod.size + 7 or mask.size < self.tod.size:
            raise ValueError(f"Mask size {mask.size} doesn't match TOD size {self.tod.size}.")
        return mask[:self.tod.size]


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
        # "None" means standard detector with [1, 1] response for intensity and polarization.
        resp_I, resp_QU = (1.0, 1.0) if self.det_response is None \
            else (self.det_response[0], self.det_response[1])
        response = np.zeros((3, psi.shape[-1]))
        if resp_I != 0:
            response[0,:] = resp_I
        if resp_QU != 0:
            response[1,:] = np.cos(2.0*psi)*resp_QU
            response[2,:] = np.sin(2.0*psi)*resp_QU
        return response