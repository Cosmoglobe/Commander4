"""Pointing containers: from a scan's boresight path to per-detector pixel and angle arrays.

`ScanBoresightPointing` holds the shared boresight for one scan, `DetectorBoresightPointing` adds a
detector's focal-plane and polarization-angle offsets, and `PixelPointing` is the resulting
(pix, psi) pair the mapmakers consume.
"""
import numpy as np
from numpy.typing import NDArray
import ducc0
import os
from pixell.bunch import Bunch
from pixell import coordsys
from commander4.backend import utils as cpp_utils


def remap_pix_nside(pix: NDArray[np.integer], nside_from: int, nside_to: int,
                    nthreads: int | None = None) -> NDArray[np.integer]:
    """Convert RING HEALPix pixel indices from one nside to another.

    Args:
        pix: RING pixel indices at ``nside_from``.
        nside_from: The nside the indices are given at.
        nside_to: The nside to convert them to.
        nthreads: Threads to use; defaults to the OMP_NUM_THREADS environment variable.
    """
    if nside_from == nside_to:
        return pix
    nthreads = int(os.environ.get("OMP_NUM_THREADS", 1)) if nthreads is None else nthreads
    geom_from = ducc0.healpix.Healpix_Base(nside_from, "RING")
    geom_to = ducc0.healpix.Healpix_Base(nside_to, "RING")
    return geom_to.ang2pix(geom_from.pix2ang(pix, nthreads=nthreads), nthreads=nthreads)


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
            lon       = site[0],
            lat       = site[1],
            alt       = site[2],
            weather   = "toco")
        self.detoffs = np.asarray(detoffs)
        self.polangs = np.asarray(polangs)
        self.nside = nside
        self.data_nside = nside
        self.ntod_original = ntod_original
        self.ntod = ntod_original if ntod is None else ntod
        self.ndet = self.detoffs.shape[0]
        if self.ntod > self.ntod_original:
            raise ValueError("ntod cannot exceed ntod_original.")
        if self.detoffs.ndim != 2:
            raise ValueError("detoffs must be a 2D array.")
        if self.detoffs.shape[1] != 2:
            raise ValueError("detoffs must have shape (ndet, 2).")
        if self.polangs.size != self.ndet:
            raise ValueError("polangs must contain one polarization angle per detector.")
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
        if not 0 <= idet < self.ndet:
            raise IndexError(f"Detector index {idet} out of range.")
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
        # ducc0 takes the two angles as one (n, 2) array of (co-latitude, longitude), and unlike
        # healpy's ang2pix it threads over the samples.
        ptg = np.empty((dec.size, 2), dtype=np.float64)
        ptg[:, 0] = np.pi/2.0 - dec
        ptg[:, 1] = ra
        nthreads = int(os.environ.get("OMP_NUM_THREADS", 1))
        geom = ducc0.healpix.Healpix_Base(target_nside, "RING")
        pix = geom.ang2pix(ptg, nthreads=nthreads)
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
        if not 0 <= self.idet < self.scan_pointing.ndet:
            raise IndexError(f"Detector index {self.idet} out of range.")
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
        # C++ decoder accepts only int64 for the tree.
        self.huffman_tree = huffman_tree.astype(np.int64, copy=False)
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
        if self.ntod > self.ntod_original:
            raise ValueError("ntod cannot exceed ntod_original.")
        if not self.pix_is_compressed and not isinstance(self.pix_encoded, np.ndarray):
            raise TypeError("'pix' must be provided as bytes, numpy.void, or a numpy array.")
        if not self.psi_is_compressed and not isinstance(self.psi_encoded, np.ndarray):
            raise TypeError("'psi' must be provided as bytes, numpy.void, or a numpy array.")
        if self.pix_is_compressed:
            if self.huffman_tree is None or self.huffman_symbols is None:
                raise ValueError("Compressed pix requires Huffman metadata.")
        if self.psi_is_compressed:
            if self.huffman_tree is None or self.huffman_symbols is None:
                raise ValueError("Compressed psi requires Huffman metadata.")
            if self.npsi is None:
                raise ValueError("Compressed psi requires npsi.")
        if not self.pix_is_compressed:
            pix_array = np.asarray(self.pix_encoded)
            if pix_array.ndim != 1:
                raise ValueError("'pix' must be a 1D array.")
            if not np.issubdtype(pix_array.dtype, np.integer):
                raise TypeError("'pix' array must have integer dtype.")
            if pix_array.size < self.ntod:
                raise ValueError(f"'pix' length {pix_array.size} is shorter than ntod {self.ntod}.")
        if not self.psi_is_compressed:
            psi_array = np.asarray(self.psi_encoded)
            if psi_array.ndim != 1:
                raise ValueError("'psi' must be a 1D array.")
            if (not np.issubdtype(psi_array.dtype, np.integer)
                    and not np.issubdtype(psi_array.dtype, np.floating)):
                raise TypeError("'psi' array must have numeric dtype.")
            if psi_array.size < self.ntod:
                raise ValueError(f"'psi' length {psi_array.size} is shorter than ntod {self.ntod}.")

    def get_pix(self, nside: int | None = None) -> NDArray[np.integer]:
        """Return HEALPix pixel indices at the requested output nside."""
        target_nside = self.nside if nside is None else nside
        if self.pix_is_compressed:
            pix = np.zeros(self.ntod_original, dtype=self.huffman_symbols.dtype)
            pix = cpp_utils.huffman_decode(self.pix_compressed_u8, self.huffman_tree,
                                           self.huffman_symbols, pix)
            # The compressed stream stores first differences, so reconstruct the
            # absolute pixel indices with a cumulative sum.
            pix = np.cumsum(pix)
        else:
            pix = self.pix_encoded

        pix = pix[:self.ntod]
        return remap_pix_nside(pix, self.data_nside, target_nside)

    def get_psi(self, nside: int | None = None) -> NDArray[np.floating]:
        """Return polarization angles, converting compressed one-based bins to their centers."""
        if self.psi_is_compressed:
            psi = np.zeros(self.ntod_original, dtype=self.huffman_symbols.dtype)
            psi = cpp_utils.huffman_decode(self.psi_compressed_u8, self.huffman_tree,
                                           self.huffman_symbols, psi)
            # Commander scan files store first differences of one-based bin numbers. Bin i covers
            # [(i-1)*width, i*width), so its representative angle is the center (i-0.5)*width.
            psi_bins = np.cumsum(psi)[:self.ntod]
            bin_width = np.float32(2*np.pi/self.npsi)
            psi = (psi_bins.astype(np.float32, copy=False) - np.float32(0.5))*bin_width
        else:
            psi = self.psi_encoded
        return psi[:self.ntod]

    def get_pix_psi(self, nside: int | None = None) -> tuple[NDArray, NDArray]:
        return self.get_pix(nside), self.get_psi(nside)
