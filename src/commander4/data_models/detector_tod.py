"""`DetectorTOD`: one detector's samples within one scan, decoded on demand.

The TOD, pointing and flags are usually stored Huffman-compressed and are decompressed lazily on
first access, so a band can hold many scans without materializing all of them at once.
"""
import numpy as np
from numpy.typing import NDArray

from commander4.compression.huffman_decoding import HuffmanCompressedArray
from commander4.data_models.pointing import PixelPointing, DetectorBoresightPointing
from commander4.diagnostics.performance import start_bench, stop_bench


class DetectorTOD:
    """Holds time-ordered data (TOD) for a single detector within a scan.
    This class is intended as a "data container", while the user-interaction should happen
    through the TODView class (`see tod/view.py`).

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
        *,
        name: str,
        det_idx_fullband: int,
        tod: NDArray[np.floating] | HuffmanCompressedArray,
        pointing: PixelPointing | DetectorBoresightPointing,
        sampling_rate_hz: float,
        orbital_velocity_m_per_s: NDArray[np.floating] | None,
        default_proc_mask: NDArray[np.bool_] | None,
        specific_proc_masks: dict,
        flag: NDArray[np.integer] | HuffmanCompressedArray | None = None,
        bad_data_bitmask: int | None = None,
        init_scalars: NDArray | None = None,
        response_I_P: tuple[float, float] | NDArray | None = None,
        polang: float | None = None,
    ):
        """Construct a DetectorTOD.

        Args:
            name: Unique name of the current detector.
            det_idx_fullband: Unique detector-index among all the detectors on the relevant band.
            tod: Calibrated time samples, as a 1-D floating-point array or compressed.
            pointing: Pointing representation for this detector. Must be a
                ``PixelPointing`` or ``DetectorBoresightPointing`` instance.
            sampling_rate_hz: Sampling frequency in Hz.
            orbital_velocity_m_per_s: Spacecraft orbital velocity in metres per second, or None.
            default_proc_mask: Default processing-mask HEALPix map (boolean), or None if the band
                defines none. Used unless a sampling step requests a name in specific_proc_masks.
            specific_proc_masks: Mapping of operation name -> processing-mask HEALPix map for bands
                that define per-operation masks (empty dict if none).
            flag: Flag samples, as an integer array or compressed. None means no flags; ``flag``
                then returns all zeros.
            bad_data_bitmask: Bitmask applied to flags to identify excluded samples. None means
                no bad-data cut.
            polang: The polarization angle of this detector relative to the boresight.
            response_I_P: How sensitive this detector is to intensity and to polarization, as a
                two-element [I, QU] sequence. Only experiments whose files carry a per-detector
                response pass one; None means an ordinary [1, 1] detector. Stored as a plain pair
                of floats either way, so consumers never have to resolve the None case.
        """
        if not isinstance(tod, HuffmanCompressedArray):
            if not isinstance(tod, np.ndarray):
                raise TypeError("'tod' must be a numpy array.")
            if tod.ndim != 1:
                raise ValueError("'tod' must be a 1D array.")
            if tod.dtype not in (np.float64, np.float32):
                raise TypeError(f"TOD dtype must be float32 or float64, got {tod.dtype}.")
        if not isinstance(pointing, (PixelPointing, DetectorBoresightPointing)):
            raise TypeError("pointing must be a PixelPointing or DetectorBoresightPointing.")
        ntod_original = pointing.ntod_original
        ntod = pointing.ntod
        if flag is not None and not isinstance(flag, HuffmanCompressedArray):
            if not isinstance(flag, np.ndarray):
                raise TypeError("Decoded flags must be provided as a numpy array.")
            if flag.ndim != 1:
                raise ValueError("'flag' must be a 1D array.")
            if not np.issubdtype(flag.dtype, np.integer):
                raise TypeError("Decoded flags must have integer dtype.")
            if flag.size < ntod:
                raise ValueError(f"'flag' length {flag.size} is shorter than ntod {ntod}.")
        self.name = name
        self.det_idx_fullband = det_idx_fullband
        self._tod = tod
        self.ntod_original = ntod_original
        self.ntod = ntod
        self.nside = pointing.nside
        self.data_nside = pointing.data_nside
        self.fsamp = sampling_rate_hz
        self.init_scalars = init_scalars
        self._flag = flag
        self._bad_data_bitmask = bad_data_bitmask
        self.pointing = pointing
        # Resolved once here, so no consumer downstream has to handle a missing response.
        if response_I_P is None:
            self.response_I_P = (1.0, 1.0)
        else:
            self.response_I_P = (float(response_I_P[0]), float(response_I_P[1]))
        self._good_data_mask = None  # Stays None without flags or bitmask: nothing is cut.
        if flag is not None and bad_data_bitmask is not None:
            good_data_mask = (self.flag & bad_data_bitmask) == 0
            self._good_data_mask = np.packbits(good_data_mask)
        if orbital_velocity_m_per_s is not None:
            if orbital_velocity_m_per_s.size != 3:
                raise ValueError("orbital_velocity_m_per_s must be a vector of size 3.")
            self.orbital_velocity_m_per_s = orbital_velocity_m_per_s.astype(
                np.float32, copy=False,
            )
        else:
            self.orbital_velocity_m_per_s = None
        # Band-level processing-mask HEALPix maps (shared references across the band's detectors;
        # TODView projects them onto this detector's pointing on demand). The default mask is used
        # unless a sampling step requests a name present in specific_proc_masks.
        self.default_proc_mask = default_proc_mask
        self.specific_proc_masks = specific_proc_masks
        self.polang = polang


    @property
    def tod(self) -> NDArray[np.floating]:
        if isinstance(self._tod, HuffmanCompressedArray):
            return self._tod.decode()[:self.ntod].astype(np.float32)
        return self._tod[:self.ntod]


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
        if self._flag is None:  # Built without flags: no sample is flagged.
            return np.zeros(self.ntod, dtype=np.int64)
        if isinstance(self._flag, HuffmanCompressedArray):
            return self._flag.decode()[:self.ntod].astype(np.int64, copy=False)
        return self._flag[:self.ntod]


    @property
    def good_data_mask(self) -> NDArray[np.bool_]:
        """Boolean mask keeping samples that pass the bad-data flag cut."""
        if self._good_data_mask is None:  # No flags or no bitmask: nothing is cut.
            return np.ones(self.ntod, dtype=bool)
        mask = np.unpackbits(self._good_data_mask).view(bool)
        if mask.size > self.ntod + 7 or mask.size < self.ntod:
            raise ValueError(f"Mask size {mask.size} doesn't match TOD size {self.ntod}.")
        return mask[:self.ntod]
