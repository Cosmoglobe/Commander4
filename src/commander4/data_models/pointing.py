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
from commander4.compression.huffman_decoding import HuffmanCompressedArray


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

    The pix and psi samples are given either as 1D arrays or compressed, in which case they are
    decoded on demand in `get_pix()` and `get_psi()`. Pixel samples are stored at `data_nside` and
    optionally remapped to another output `nside` after decoding.
    """

    def __init__(self,
                 pix: NDArray[np.integer] | HuffmanCompressedArray,
                 psi: NDArray[np.floating] | HuffmanCompressedArray,
                 nside: int,
                 data_nside: int,
                 ntod_original: int,
                 ntod: int,
                 ):
        self.nside = nside
        self.data_nside = data_nside
        self.ntod_original = ntod_original
        self.ntod = ntod
        self.pix = pix
        self.psi = psi
        if self.ntod > self.ntod_original:
            raise ValueError("ntod cannot exceed ntod_original.")
        if not isinstance(pix, HuffmanCompressedArray):
            if not isinstance(pix, np.ndarray) or pix.ndim != 1:
                raise TypeError("'pix' must be a 1D numpy array or a HuffmanCompressedArray.")
            if not np.issubdtype(pix.dtype, np.integer):
                raise TypeError("'pix' array must have integer dtype.")
            if pix.size < self.ntod:
                raise ValueError(f"'pix' length {pix.size} is shorter than ntod {self.ntod}.")
        if not isinstance(psi, HuffmanCompressedArray):
            if not isinstance(psi, np.ndarray) or psi.ndim != 1:
                raise TypeError("'psi' must be a 1D numpy array or a HuffmanCompressedArray.")
            if not np.issubdtype(psi.dtype, np.floating):
                raise TypeError("'psi' array must have floating-point dtype.")
            if psi.size < self.ntod:
                raise ValueError(f"'psi' length {psi.size} is shorter than ntod {self.ntod}.")

    def get_pix(self, nside: int | None = None) -> NDArray[np.integer]:
        """Return HEALPix pixel indices at the requested output nside."""
        target_nside = self.nside if nside is None else nside
        if isinstance(self.pix, HuffmanCompressedArray):
            pix = self.pix.decode()[:self.ntod].astype(np.int64, copy=False)
        else:
            pix = self.pix[:self.ntod]
        return remap_pix_nside(pix, self.data_nside, target_nside)

    def get_psi(self, nside: int | None = None) -> NDArray[np.floating]:
        """Return polarization angles in radians."""
        if isinstance(self.psi, HuffmanCompressedArray):
            return self.psi.decode()[:self.ntod]
        return self.psi[:self.ntod]

    def get_pix_psi(self, nside: int | None = None) -> tuple[NDArray, NDArray]:
        return self.get_pix(nside), self.get_psi(nside)
