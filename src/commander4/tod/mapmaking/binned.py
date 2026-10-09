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

from commander4.backend import mapmaker as cpp_mapmaker
from commander4.data_models.pixel_domain import PixelDomain


class BinnedMapmaker:
    """All binned maps of one band: the weights P^T N^-1 P and any number of TOD maps P^T N^-1 d.

    Every map of a band is binned with the same per-sample weight, so one `accumulate` call per
    detector-scan adds the weights and all the TOD maps together, reading the pointing once. The
    buffers hold this rank's local pixels (`PixelDomain`); `finalize` sums them onto the band
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

    def accumulate(self, weight: float, pix_local: NDArray, psi: NDArray,
                   tods: dict[str, NDArray], response_I_P: tuple[float, float] = (1.0, 1.0)):
        """Add one detector-scan: `weight` per sample to the weights, and `weight*tods[name]` to
        each map. Every TOD has one value per entry of `pix_local` (`TODView.pix_local`)."""
        pix_local = np.asarray(pix_local, dtype=np.int64)
        # One row per map, in the order of `names`.
        tod_rows = (np.stack([tods[name] for name in self.names]) if self.names
                    else np.zeros((0, pix_local.size), dtype=np.float32))
        resp_I, resp_P = response_I_P
        cpp_mapmaker.binned_map_accumulator(self._weights, self._maps, tod_rows, float(weight),
                                            pix_local, np.asarray(psi, dtype=np.float64),
                                            response_I=resp_I, response_P=resp_P)

    def count_hits(self, pix_local: NDArray):
        """Add one hit per sample to the hit map."""
        cpp_mapmaker.hit_accumulator(self._hits, np.asarray(pix_local, dtype=np.int64))

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
