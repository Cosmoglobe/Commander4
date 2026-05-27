import numpy as np
import logging
from numpy.typing import NDArray
import ducc0
import os
import healpy as hp
from pixell.bunch import Bunch
from pixell import coordsys
from commander4.cmdr4_support import utils as cpp_utils
import commander4.output.log as log

logger = logging.getLogger(__name__)


class ScanBoresightPointing:
    """Evaluate one scan's boresight once and reuse it for all detectors.

    The scan boresight is propagated for the full original TOD length in sky
    coordinates and kept as a shared object. Individual detector pointings are
    then obtained by rotating that common boresight with per-detector xi/eta
    offsets and polarization angles, which avoids recomputing the expensive
    time-dependent coordinate transform for every detector.
    """

    def __init__(self,
                 time_start_mjd: float,
                 time_end_mjd: float,
                 ntod_original: int,
                 site: NDArray,
                 bore: NDArray,
                 detoffs: NDArray,
                 polangs: NDArray | float,
                 nside: int,
                 ntod: int | None = None):
        self.site = Bunch(
            lat       = site[0],
            lon       = site[1],
            alt       = site[2],
            weather   = "toco")
        self.detoffs = np.asarray(detoffs)
        self.polangs = np.asarray(polangs)
        self.nside = nside
        self.data_nside = nside
        self.ntod_original = ntod_original
        self.ntod = ntod_original if ntod is None else ntod
        self.ndet = self.detoffs.shape[0]
        log.logassert_np(self.ntod <= self.ntod_original, "ntod cannot exceed ntod_original.", logger)
        log.logassert_np(self.detoffs.ndim == 2, "detoffs must be a 2D array.", logger)
        log.logassert_np(self.detoffs.shape[1] == 2, "detoffs must have shape (ndet, 2).", logger)
        log.logassert_np(
            self.polangs.size == self.ndet,
            "polangs must contain one polarization angle per detector.",
            logger,
        )
        # pixell's time-dependent coordinate transforms use Unix seconds.
        time_start_unix = (time_start_mjd - 40587.0) * 86400.0
        time_end_unix = (time_end_mjd - 40587.0) * 86400.0
        time_unix = np.linspace(time_start_unix, time_end_unix, ntod_original)

        # Build the boresight for the full native scan once; shorter requests
        # are handled later by slicing to self.ntod.
        self.bore_point = self.initialize_boresight(time_unix, bore, site=self.site)


    def initialize_boresight(
        self,
        ctime: NDArray[np.floating],
        bore: NDArray,
        sys: str = "cel",
        site=None,
        weather: str = "typical",
    ):
        """Transform boresight az/el/roll samples into the requested sky frame."""
        icoord = coordsys.Coords(az=bore[0], el=bore[1], roll=bore[2])
        return coordsys.transform("hor", sys, icoord, ctime=ctime, site=site, weather=weather)


    def get_det_point(
        self,
        idet: int,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        log.logassert_np(0 <= idet < self.ndet, f"Detector index {idet} out of range.", logger)
        # By slicing instead of indexing we keep the 1-sized detector dimension.
        detoff = self.detoffs[idet:idet+1]
        polang = self.polangs[idet:idet+1]
        # Apply the detector's focal-plane offset and polarization rotation on
        # top of the shared boresight quaternion.
        qdet = coordsys.rotation_xieta(detoff[:, 0], detoff[:, 1], polang)
        ocoord = self.bore_point * qdet[:, None]
        # TODO: The lines below are absurdly slow, taking 95% of the runtime of this function,
        # being almost as time-consuming as a full hp.ang2pix call. I tried replacing the call with
        # a Numba function, but couldn't achieve a speedup. Should be looked into.
        dec = np.asarray(ocoord.dec[0, :self.ntod])
        ra = np.asarray(ocoord.ra[0, :self.ntod])
        psi = np.asarray(ocoord.psi[0, :self.ntod])
        return dec, ra, psi


    def get_pix_psi(
        self,
        idet: int,
        nside: int | None = None,
    ) -> tuple[NDArray[np.integer], NDArray[np.floating]]:
        target_nside = self.nside if nside is None else nside
        dec, ra, psi = self.get_det_point(idet)
        # healpy expects co-latitude theta rather than declination.
        theta = np.pi/2.0 - dec
        pix = hp.ang2pix(target_nside, theta, ra)
        psi = psi.astype(np.float32, copy=False)[:self.ntod]
        return pix, psi


    def get_pix(self, idet: int, nside: int | None = None) -> NDArray[np.integer]:
        return self.get_pix_psi(idet, nside)[0]


    def get_psi(self, idet: int, nside: int | None = None) -> NDArray[np.floating]:
        return self.get_pix_psi(idet, nside)[1]



class DetectorBoresightPointing:
    """Detector-specific view onto a shared ScanBoresightPointing.

    This wrapper stores only the detector index and forwards all queries to the
    shared scan-level object. That keeps the per-detector interface simple while
    avoiding duplication of boresight and site state.
    """

    def __init__(self, scan_pointing: ScanBoresightPointing, idet: int):
        self.scan_pointing = scan_pointing
        self.idet = int(idet)
        log.logassert_np(
            0 <= self.idet < self.scan_pointing.ndet,
            f"Detector index {self.idet} out of range.",
            logger,
        )
        self.nside = scan_pointing.nside
        self.data_nside = scan_pointing.data_nside
        self.ntod_original = scan_pointing.ntod_original
        self.ntod = scan_pointing.ntod
    
    def get_pix(self, nside: int | None = None) -> NDArray[np.integer]:
        return self.scan_pointing.get_pix(self.idet, nside)

    def get_psi(self, nside: int | None = None) -> NDArray[np.floating]:
        return self.scan_pointing.get_psi(self.idet, nside)

    def get_pix_psi(self, nside: int | None = None) -> tuple[NDArray[np.integer], NDArray[np.floating]]:
        return self.scan_pointing.get_pix_psi(self.idet, nside)



class PixelPointing:
    """Store pixel and polarization-angle pointing for one detector TOD.

    The pointing can be supplied either as decoded 1D arrays or as Huffman-
    compressed binary payloads. Compressed payloads are kept compact in memory
    and decoded only on demand in `get_pix()` and `get_psi()`. Pixel samples are
    stored at `data_nside` and optionally remapped to another output `nside`
    after decompression.
    """

    def __init__(self,
                 pix: bytes | np.void | NDArray[np.integer],
                 psi: bytes | np.void | NDArray[np.integer] | NDArray[np.floating],
                 huffman_tree: NDArray | None,
                 huffman_symbols: NDArray | None,
                 npsi: int | None,
                 nside: int,
                 data_nside: int,
                 ntod_original: int,
                 ntod: int,
                 ):
        self.nside = nside
        self.data_nside = data_nside
        self.ntod_original = ntod_original
        self.ntod = ntod
        self.pix_encoded = pix
        self.psi_encoded = psi
        self.huffman_tree = huffman_tree
        self.huffman_symbols = huffman_symbols
        self.npsi = npsi
        self.pix_is_compressed = isinstance(pix, (bytes, np.void))
        self.psi_is_compressed = isinstance(psi, (bytes, np.void))
        # The Huffman decoder consumes uint8 arrays; for HDF5-backed np.void
        # inputs, frombuffer gives a zero-copy view over the stored payload.
        self.pix_compressed_u8 = np.frombuffer(pix, dtype=np.uint8) if self.pix_is_compressed else None
        self.psi_compressed_u8 = np.frombuffer(psi, dtype=np.uint8) if self.psi_is_compressed else None
        self._test_input()

    
    def _test_input(self):
        log.logassert_np(self.ntod <= self.ntod_original, "ntod cannot exceed ntod_original.", logger)
        log.logassert_np(
            self.pix_is_compressed or isinstance(self.pix_encoded, np.ndarray),
            "'pix' must be provided as bytes, numpy.void, or a numpy array.",
            logger,
        )
        log.logassert_np(
            self.psi_is_compressed or isinstance(self.psi_encoded, np.ndarray),
            "'psi' must be provided as bytes, numpy.void, or a numpy array.",
            logger,
        )
        if self.pix_is_compressed:
            log.logassert_np(
                self.huffman_tree is not None and self.huffman_symbols is not None,
                "Compressed pix requires Huffman metadata.",
                logger,
            )
        if self.psi_is_compressed:
            log.logassert_np(
                self.huffman_tree is not None and self.huffman_symbols is not None,
                "Compressed psi requires Huffman metadata.",
                logger,
            )
            log.logassert_np(self.npsi is not None, "Compressed psi requires npsi.", logger)
        if not self.pix_is_compressed:
            pix_array = np.asarray(self.pix_encoded)
            log.logassert_np(pix_array.ndim == 1, "'pix' must be a 1D array", logger)
            log.logassert_np(
                np.issubdtype(pix_array.dtype, np.integer),
                "'pix' array must have integer dtype.",
                logger,
            )
            log.logassert_np(
                pix_array.size >= self.ntod,
                f"'pix' length {pix_array.size} is shorter than ntod {self.ntod}.",
                logger,
            )
        if not self.psi_is_compressed:
            psi_array = np.asarray(self.psi_encoded)
            log.logassert_np(psi_array.ndim == 1, "'psi' must be a 1D array", logger)
            log.logassert_np(
                np.issubdtype(psi_array.dtype, np.integer)
                or np.issubdtype(psi_array.dtype, np.floating),
                "'psi' array must have numeric dtype.",
                logger,
            )
            log.logassert_np(
                psi_array.size >= self.ntod,
                f"'psi' length {psi_array.size} is shorter than ntod {self.ntod}.",
                logger,
            )

    def get_pix(self, nside: int | None = None) -> NDArray[np.integer]:
        """Return HEALPix pixel indices at the requested output nside."""
        target_nside = self.nside if nside is None else nside
        if self.pix_is_compressed:
            pix = np.zeros(self.ntod_original, dtype=np.int64)
            pix = cpp_utils.huffman_decode(self.pix_compressed_u8, self.huffman_tree,
                                           self.huffman_symbols, pix)
            # The compressed stream stores first differences, so reconstruct the
            # absolute pixel indices with a cumulative sum.
            pix = np.cumsum(pix)
        else:
            pix = self.pix_encoded

        pix = pix[:self.ntod]
        if target_nside != self.data_nside:
            nthreads = int(os.environ["OMP_NUM_THREADS"])
            geom_from = ducc0.healpix.Healpix_Base(self.data_nside, "RING")
            geom_to = ducc0.healpix.Healpix_Base(target_nside, "RING")
            ang = geom_from.pix2ang(pix, nthreads=nthreads)
            pix = geom_to.ang2pix(ang, nthreads=nthreads)
        return pix

    def get_psi(self, nside: int | None = None) -> NDArray[np.floating]:
        """Return polarization angles, cropped to the active TOD length."""
        if self.psi_is_compressed:
            psi = np.zeros(self.ntod_original, dtype=np.int64)
            psi = cpp_utils.huffman_decode(self.psi_compressed_u8, self.huffman_tree,
                                           self.huffman_symbols, psi)
            # psi is compressed as differences of digitized angle bins; first
            # recover the bin index stream, then convert bins back to radians.
            psi = np.cumsum(psi)
            psi = psi[:self.ntod]
            psi = 2 * np.pi * psi.astype(np.float32, copy=False) / self.npsi
        else:
            psi = self.psi_encoded
        return psi[:self.ntod]

    def get_pix_psi(self, nside: int | None = None) -> tuple[NDArray, NDArray]:
        return self.get_pix(nside), self.get_psi(nside)

