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

    `accumulate_Python` and `finalize_Python` do the same as `accumulate` and `finalize` in plain
    NumPy. They are slow, but show what the C++ kernels compute, and the tests compare the two.
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


    def accumulate_Python(self, weight: float, pix_local: NDArray, psi: NDArray,
                          tods: dict[str, NDArray], response_I_P: tuple[float, float] = (1.0, 1.0)):
        """Plain-NumPy version of `accumulate`, matching the C++ kernel `binned_map_accumulator`.

        Each sample's pointing row is [resp_I, resp_P*cos(2 psi), resp_P*sin(2 psi)] (only [resp_I]
        for an I-only band). The weights get weight*row^T row, each map weight*row*tod.
        """
        pix_local = np.asarray(pix_local, dtype=np.int64)
        resp_I, resp_P = response_I_P
        np.add.at(self._weights[0], pix_local, weight*resp_I*resp_I)  # II
        for i, name in enumerate(self.names):
            tod = np.asarray(tods[name], dtype=np.float64)
            np.add.at(self._maps[i, 0], pix_local, weight*resp_I*tod)
        # psi is only read here, so a detector without polarization response may have NaN angles.
        if self.ncomp == 3 and resp_P != 0.0:
            cos2psi = np.cos(2.0*np.asarray(psi, dtype=np.float64))
            sin2psi = np.sin(2.0*np.asarray(psi, dtype=np.float64))
            np.add.at(self._weights[1], pix_local, weight*resp_I*resp_P*cos2psi)          # IQ
            np.add.at(self._weights[2], pix_local, weight*resp_I*resp_P*sin2psi)          # IU
            np.add.at(self._weights[3], pix_local, weight*resp_P*resp_P*cos2psi*cos2psi)  # QQ
            np.add.at(self._weights[4], pix_local, weight*resp_P*resp_P*sin2psi*cos2psi)  # QU
            np.add.at(self._weights[5], pix_local, weight*resp_P*resp_P*sin2psi*sin2psi)  # UU
            for i, name in enumerate(self.names):
                tod = np.asarray(tods[name], dtype=np.float64)
                np.add.at(self._maps[i, 1], pix_local, weight*resp_P*tod*cos2psi)
                np.add.at(self._maps[i, 2], pix_local, weight*resp_P*tod*sin2psi)


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


    def finalize_Python(self):
        """Plain-NumPy version of `finalize`, matching the C++ kernels `map_inv_var_IQU` and
        `map_solve_IQU` (collective)."""
        is_master = self.domain.comm.Get_rank() == 0
        map_cov = self.domain.reduce_to_full(self._weights)
        if is_master:
            self.map_cov = map_cov
            self.maps = {}
            if self.ncomp == 1:
                self.map_inv_var = map_cov.astype(np.float32)
            else:
                # Each pixel's symmetric 3x3 matrix A from its 6 unique elements, (npix, 3, 3):
                # [[II, IQ, IU], [IQ, QQ, QU], [IU, QU, UU]].
                A = map_cov[[0, 1, 2, 1, 3, 4, 2, 4, 5]].T.reshape(-1, 3, 3)
                # Skip pixels whose A cannot be inverted reliably, by the same test as the C++ code:
                # a zero diagonal, or a determinant below 1e-12 of the product of the diagonal
                # elements (a rough estimate of 1/condition number). These pixels stay 0.
                diag_prod = A[:, 0, 0]*A[:, 1, 1]*A[:, 2, 2]
                det = np.linalg.det(A)
                good = (diag_prod > np.finfo(np.float64).tiny) & (det > 1e-12*diag_prod)
                # The variance of each Stokes parameter is the matching diagonal element of A^-1.
                var = np.diagonal(np.linalg.inv(A[good]), axis1=1, axis2=2).T
                inv_var = np.zeros((3, self.domain.npix))
                inv_var[:, good] = np.divide(1.0, var, out=np.zeros_like(var), where=var > 0)
                self.map_inv_var = inv_var.astype(np.float32)
        for i, name in enumerate(self.names):
            rhs = self.domain.reduce_to_full(self._maps[i])
            if is_master:
                solved = np.zeros((self.ncomp, self.domain.npix))
                if self.ncomp == 1:
                    np.divide(rhs, map_cov, out=solved, where=map_cov != 0)
                else:
                    # Solve A x = rhs in each good pixel.
                    solved[:, good] = np.linalg.solve(A[good], rhs[:, good].T[..., None])[..., 0].T
                self.maps[name] = solved.astype(np.float32)
        if self._hits is not None:
            map_nhit = self.domain.reduce_to_full(self._hits)
            if is_master:
                self.map_nhit = np.round(map_nhit).astype(np.int64)
        self._weights = self._maps = self._hits = None
