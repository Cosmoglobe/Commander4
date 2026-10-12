"""The binned mapmaker: accumulate P^T N^-1 d and P^T N^-1 P per pixel, then invert per pixel.

The simpler of the two mapmakers (see `mapmaking/cg.py` for the CG one). `BinnedMapmaker` holds
all binned maps of a band: the inverse-variance weights that both mapmakers need, the signal map
and the auxiliary maps. It adds each detector-scan to all of them in one pass. `tod2map`
(`mapmaking/tod2map.py`) runs the scan loop for both mapmakers.

The maps are indexed with `pix_local`: local pixel indices into the rank's map buffers
(`TODView.pix_local`), converted once per detector-scan.
"""
import numpy as np
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.backend import mapmaker as cpp_mapmaker
from commander4.data_models.pixel_domain import PixelDomain


class BinnedMapmaker:
    """All binned maps of one band: the weights P^T N^-1 P and any number of TOD maps P^T N^-1 d.

    Every map of a band is binned with the same per-sample weight, so one pass over a detector-scan
    bins the weights and all the TOD maps together, reading the pointing once. A detector-scan is
    added in two steps. `bin_scan` bins it into small maps over its own pixels; it changes nothing
    shared, so the scan workers run it at the same time (`TODView.map`). `add_scan` then adds these
    to the rank's maps, in the main thread, one scan after the other. A scan has about 20 samples
    per pixel, so this second, serial step is short, and the order of the additions, and thus the
    maps to the last bit, is the same for any number of threads.

    The buffers hold this rank's local pixels (`PixelDomain`); `finalize` sums them onto the band
    master and solves each pixel there. Afterwards the master holds (other ranks hold None):

    - `map_cov`: the summed weights, float64. For an IQU or QU band these are the 6 unique elements
      (II, IQ, IU, QQ, QU, UU) of each pixel's 3x3 matrix, (6, npix); for an I-only band (1, npix).
    - `map_inv_var`: the inverse variance 1/diag(A^-1), (ncomp, npix), float32.
    - `maps`: the solved maps by name, (ncomp, npix), float32.
    - `map_nhit`: the number of samples per pixel, (npix,), if `count_hits` was set.

    ncomp is 1 for an I-only band and 3 (I, Q, U) otherwise. A pixel that cannot be solved, because
    it is unobserved or seen at too few polarization angles, gets 0 in `map_inv_var` and `maps`.
    """
    def __init__(self, domain: PixelDomain, pols: str, names: list[str], count_hits: bool = False):
        self.domain = domain
        self.names = list(names)
        self.ncomp = 1 if pols == "I" else 3
        self._weights = np.zeros((1 if pols == "I" else 6, domain.n_local))
        self._maps = np.zeros((len(self.names), self.ncomp, domain.n_local))
        self._hits = np.zeros(domain.n_local) if count_hits else None
        self.map_cov = self.map_inv_var = self.maps = self.map_nhit = None

    def bin_scan(self, weight: float, pix_local: NDArray[np.int64], psi: NDArray,
                 tod_rows: NDArray, response_I_P: tuple[float, float] = (1.0, 1.0)) -> Bunch:
        """Bin one detector-scan into maps over its own pixels, for `add_scan`.

        Each sample adds `weight` to the weights, and `weight*tod_rows[i]` to map `names[i]`.

        Args:
            weight: The weight of every sample, the inverse noise variance.
            pix_local: Local pixel of each sample (`TODView.pix_local`).
            psi: Polarization angle of each sample.
            tod_rows: One row per map, in the order of `names`, and one column per sample.
            response_I_P: The detector's intensity and polarization response.

        Returns:
            A Bunch with `pix`, the scan's distinct local pixels, and the scan's `weights`, `maps`
            and `hits` (None without a hit map) at those pixels.
        """
        # `index` numbers the scan's pixels from 0, so the small maps hold only those pixels.
        pix, index = cpp_mapmaker.distinct_pixels(pix_local)
        weights = np.zeros((self._weights.shape[0], pix.size))
        maps = np.zeros((len(self.names), self.ncomp, pix.size))
        resp_I, resp_P = response_I_P
        cpp_mapmaker.binned_map_accumulator(weights, maps, tod_rows, float(weight), index,
                                            np.asarray(psi, dtype=np.float64),
                                            response_I=resp_I, response_P=resp_P)
        hits = None
        if self._hits is not None:
            hits = np.zeros(pix.size)
            cpp_mapmaker.hit_accumulator(hits, index)
        return Bunch(pix=pix, weights=weights, maps=maps, hits=hits)

    def add_scan(self, scan: Bunch):
        """Add a detector-scan binned by `bin_scan` to the rank's maps."""
        nrows = len(self.names) * self.ncomp   # The maps as one row per map and component.
        cpp_mapmaker.add_to_pixels(self._weights, scan.weights, scan.pix)
        cpp_mapmaker.add_to_pixels(self._maps.reshape(nrows, self.domain.n_local),
                                   scan.maps.reshape(nrows, scan.pix.size), scan.pix)
        if self._hits is not None:
            cpp_mapmaker.add_to_pixels(self._hits[None, :], scan.hits[None, :], scan.pix)

    def finalize(self):
        """Sum the buffers of all ranks onto the master and solve each pixel there (collective).

        The maps are summed one at a time, so besides the weights the master holds only one
        unsolved full-sky map at a time. The local buffers are freed.
        """
        is_master = self.domain.comm.Get_rank() == 0
        map_cov = self.domain.reduce_to_full(self._weights)
        if is_master:
            self.map_cov = map_cov
            self.maps = {}
            if self.ncomp == 1:
                # For intensity alone a pixel's normal matrix is one number: the inverse variance.
                self.map_inv_var = map_cov.astype(np.float32)
            else:
                inv_var = np.zeros((3, self.domain.npix))
                cpp_mapmaker.map_inv_var_IQU(inv_var, map_cov)
                self.map_inv_var = inv_var.astype(np.float32)
        for i, name in enumerate(self.names):
            rhs = self.domain.reduce_to_full(self._maps[i])
            if is_master:
                solved = np.zeros((self.ncomp, self.domain.npix))
                if self.ncomp == 1:
                    np.divide(rhs, map_cov, out=solved, where=map_cov != 0)
                else:
                    cpp_mapmaker.map_solve_IQU(solved, rhs, map_cov)
                self.maps[name] = solved.astype(np.float32)
        if self._hits is not None:
            map_nhit = self.domain.reduce_to_full(self._hits)
            if is_master:
                self.map_nhit = np.round(map_nhit).astype(np.int64)
        self._weights = self._maps = self._hits = None
