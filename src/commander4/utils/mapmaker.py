import numpy as np
import ctypes as ct
from mpi4py import MPI
import logging
from numpy.typing import NDArray

from commander4.output.log import logassert
from commander4.utils.ctypes_lib import load_cmdr4_ctypes_lib

# current_dir_path = os.path.dirname(os.path.realpath(__file__))
# src_dir_path = os.path.abspath(os.path.join(os.path.join(current_dir_path, os.pardir), os.pardir))


class Mapmaker:
    """Scalar (temperature-only) mapmaker using binned TOD accumulation.

    Accumulates weighted TOD samples into a map, reduces across MPI ranks,
    and normalizes with a precomputed weights map. The internal accumulation
    is performed in float64; output dtype is controlled by `dtype`.
    """
    def __init__(self, map_comm:MPI.Comm, nside:int, dtype=np.float32):
        self.logger = logging.getLogger(__name__)
        self.map_comm = map_comm
        self.nside = nside
        self.npix = 12*nside**2
        self.dtype= dtype
        self._map_signal = np.zeros(self.npix, dtype=np.float64)
        self._gathered_map = None
        self._finalized_map = None
        
        # Setting up Ctypes mapmaker
        self.maplib = load_cmdr4_ctypes_lib()
        ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_int64, ndim=1, flags="contiguous")
        ct_f64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=1, flags="contiguous")
        self.maplib.map_accumulator_f64.argtypes = [ct_f64_dim1, ct_f64_dim1, ct.c_double, ct_i64_dim1,
                                ct.c_int64]

    @property
    def final_map(self):
        if self.map_comm.Get_rank() == 0:
            logassert(self._finalized_map is not None, "Attempted to retrieve map before it was done.",
                    self.logger)
        return self._finalized_map

    def accumulate_to_map(self, tod:NDArray, weights:NDArray, pix:NDArray):
        """Accumulate weighted TOD samples into the local map buffer."""
        # Check that we are still in business, and haven't already called "gather_map".
        logassert(self._map_signal is not None, "Tried accumulating to finalized map", self.logger)
        ntod = tod.shape[0]
        tod_f64 = np.ascontiguousarray(tod, dtype=np.float64)
        weight_f64 = float(weights)
        self.maplib.map_accumulator_f64(self._map_signal, tod_f64, weight_f64,
                                    pix.astype(np.int64), ntod)

    def gather_map(self):
        """Reduce the local map buffers across MPI ranks into the root map."""
        if self.map_comm.Get_rank() == 0:
            self._gathered_map = np.zeros(self.npix, dtype=np.float64)
        self.map_comm.Reduce(self._map_signal, self._gathered_map, op=MPI.SUM, root=0)
        self._map_signal = None  # Free memory and indicate that accumulation is done.
    
    def normalize_map(self, normalization_map):
        """Normalize the gathered map by the provided weights map."""
        if self.map_comm.Get_rank() == 0:
            norm_map = np.asarray(normalization_map, dtype=np.float64)
            mask = norm_map != 0
            result = np.zeros(self.npix, dtype=np.float64)
            # Only normalize where weights are non-zero to avoid division by zero.
            result[mask] = self._gathered_map[mask] / norm_map[mask]
            self._finalized_map = result.astype(self.dtype, copy=False)
            self._gathered_map = None



class WeightsMapmaker:
    """Scalar (temperature-only) weights mapmaker.

    Accumulates per-sample weights into a map, reduces across MPI ranks,
    and exposes the gathered weights map for normalization of Mapmaker.
    """
    def __init__(self, map_comm:MPI.Comm, nside:int, dtype=np.float32):
        self.logger = logging.getLogger(__name__)
        self.map_comm = map_comm
        self.nside = nside
        self.npix = 12*nside**2
        self.dtype= dtype
        self._map_signal = np.zeros(self.npix, dtype=np.float64)
        self._gathered_map = None
        
        # Setting up Ctypes mapmaker
        self.maplib = load_cmdr4_ctypes_lib()
        ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_int64, ndim=1, flags="contiguous")
        ct_f64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=1, flags="contiguous")
        self.maplib.map_weight_accumulator_f64.argtypes = [ct_f64_dim1, ct.c_double, ct_i64_dim1,
                                   ct.c_int64]

    @property
    def final_map(self):
        if self.map_comm.Get_rank() == 0:
            logassert(self._gathered_map is not None, "Attempted to retrieve map before it was done.",
                    self.logger)
        return self._gathered_map.astype(self.dtype, copy=False)

    def accumulate_to_map(self, weight:NDArray, pix:NDArray):
        """Accumulate per-sample weights into the local map buffer."""
        # Check that we are still in business, and haven't already called "gather_map".
        logassert(self._map_signal is not None, "Tried accumulating to finalized map", self.logger)
        ntod = pix.shape[0]
        weight_f64 = float(weight)
        self.maplib.map_weight_accumulator_f64(self._map_signal, weight_f64, pix.astype(np.int64), ntod,
                                               self.npix)

    def gather_map(self):
        """Reduce the local weights buffers across MPI ranks into the root map."""
        if self.map_comm.Get_rank() == 0:
            self._gathered_map = np.zeros(self.npix, dtype=np.float64)
        self.map_comm.Reduce(self._map_signal, self._gathered_map, op=MPI.SUM, root=0)
        self._map_signal = None  # Free memory and indicate that accumulation is done.



class MapmakerIQU:
    """Binned polarized mapmaker solving per-pixel I,Q,U normal equations.

    This class accumulates the right-hand side of the mapmaking system (the
    weighted TOD projected into I,Q,U) across MPI tasks, gathers the result
    on the root rank, and solves the 3x3 system per pixel using the provided
    normalization map (A matrix) built by WeightsMapmakerIQU.

    Usage:
    - Use WeightsMapmakerIQU to accumulate the 6 unique elements of A and
        call `normalize_map()` there to produce RMS/covariance maps.
    - Use MapmakerIQU to accumulate signal maps, then call `normalize_map(A)`
        with the gathered A map to produce the finalized I,Q,U map.
    """
    def __init__(self, map_comm:MPI.Comm, nside:int, dtype=np.float32):
        self.logger = logging.getLogger(__name__)
        self.map_comm = map_comm
        self.nside = nside
        self.npix = 12*nside**2
        self.dtype= dtype
        self._map_signal = np.zeros((3, self.npix), dtype=np.float64)
        self._gathered_map = None
        self._finalized_map = None
        
        # Setting up Ctypes mapmaker
        self.maplib = load_cmdr4_ctypes_lib()
        ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_int64, ndim=1, flags="contiguous")
        ct_f64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=1, flags="contiguous")
        ct_f64_dim2 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=2, flags="contiguous")
        self.maplib.map_accumulator_IQU_f64.argtypes = [ct_f64_dim2, ct_f64_dim1, ct.c_double,
                                ct_i64_dim1, ct_f64_dim1, ct.c_int64,
                                ct.c_int64]
        self.maplib.map_solve_IQU_f64.argtypes = [ct_f64_dim2, ct_f64_dim2, ct_f64_dim2,
                      ct.c_int64]

    @property
    def final_map(self):
        if self.map_comm.Get_rank() == 0:
            logassert(self._finalized_map is not None, "Attempted to read map before it was done.",
                      self.logger)
        return self._finalized_map


    def accumulate_to_map(self, tod:NDArray, weights:NDArray, pix:NDArray, psi:NDArray):
        """Accumulate I,Q,U signal into the local map buffer."""
        # Check that we are still in business, and haven't already called "gather_map".
        logassert(self._map_signal is not None, "Tried accumulating to finalized map", self.logger)
        ntod = tod.shape[0]
        tod_f64 = np.ascontiguousarray(tod, dtype=np.float64)
        weight_f64 = float(weights)
        psi_f64 = np.ascontiguousarray(psi, dtype=np.float64)
        self.maplib.map_accumulator_IQU_f64(self._map_signal, tod_f64, weight_f64, pix.astype(np.int64),
                                            psi_f64, ntod, self.npix)

    def accumulate_to_map_Python(self, tod:NDArray, weights:NDArray, pix:NDArray, psi:NDArray):
        """Reference accumulator matching the ctypes IQU implementation."""
        # Reference implementation matching the ctypes IQU accumulator.
        logassert(self._map_signal is not None, "Tried accumulating to finalized map", self.logger)
        pix_idx = pix.astype(np.int64, copy=False)
        w_tod = np.ascontiguousarray(tod, dtype=np.float64) * float(weights)
        ang = 2.0 * np.ascontiguousarray(psi, dtype=np.float64)
        c2 = np.cos(ang)
        s2 = np.sin(ang)
        np.add.at(self._map_signal[0], pix_idx, w_tod)
        np.add.at(self._map_signal[1], pix_idx, w_tod * c2)
        np.add.at(self._map_signal[2], pix_idx, w_tod * s2)

    def gather_map(self):
        """Reduce the local IQU buffers across MPI ranks into the root map."""
        if self.map_comm.Get_rank() == 0:
            self._gathered_map = np.zeros((3, self.npix), dtype=np.float64)
        self.map_comm.Reduce(self._map_signal, self._gathered_map, op=MPI.SUM, root=0)
        self._map_signal = None  # Free memory and indicate that accumulation is done.
    
    def normalize_map(self, normalization_map):
        """Solve the per-pixel 3x3 system using the provided A matrix."""
        if self.map_comm.Get_rank() == 0:
            logassert(normalization_map.ndim == 2 and normalization_map.shape[0] == 6,
                    "Normalization map must have shape [6,NPIX] for IQU mapmaker,"
                    f"has {normalization_map.shape}", self.logger)
            norm_map = np.ascontiguousarray(normalization_map, dtype=np.float64)
            rhs_map = np.ascontiguousarray(self._gathered_map, dtype=np.float64)
            solved = np.zeros((3, self.npix), dtype=np.float64)
            self.maplib.map_solve_IQU_f64(solved, rhs_map, norm_map, self.npix)
            self._finalized_map = solved.astype(self.dtype, copy=False)
            self._gathered_map = None

    def normalize_map_Python(self, normalization_map):
        """Reference solver using NumPy for per-pixel normalization."""
        if self.map_comm.Get_rank() == 0:
            logassert(normalization_map.ndim == 2 and normalization_map.shape[0] == 6,
                    "Normalization map must have shape [6,NPIX] for IQU mapmaker,"
                    f"has {normalization_map.shape}", self.logger)
            self._finalized_map = np.zeros((3, self.npix), dtype=self.dtype)
            A = np.zeros((self.npix, 3, 3), dtype=np.float64)
            A[:, 0, 0] = normalization_map[0]
            A[:, 0, 1] = normalization_map[1]
            A[:, 1, 0] = normalization_map[1]
            A[:, 0, 2] = normalization_map[2]
            A[:, 2, 0] = normalization_map[2]
            A[:, 1, 1] = normalization_map[3]
            A[:, 1, 2] = normalization_map[4]
            A[:, 2, 1] = normalization_map[4]
            A[:, 2, 2] = normalization_map[5]

            # Test whether the matrix is singular or ill-conditioned.
            det = np.linalg.det(A)
            diag_prod = A[:, 0, 0] * A[:, 1, 1] * A[:, 2, 2]
            eps = np.finfo(np.float64).eps
            # If a diagonal entry is 0 the matrix is singular, and if negative the matrix is not SPD
            # If the determinant is very small the matrix is ill-conditioned.
            mask = (diag_prod > 0) & (np.abs(det) > eps * diag_prod)

            # Ill-conditioned pixels stay as 0.0.
            if np.any(mask):
                rhs = self._gathered_map[:, mask].T[..., np.newaxis]
                sol = np.linalg.solve(A[mask], rhs)
                self._finalized_map[:, mask] = sol[..., 0].T.astype(self.dtype, copy=False)
            self._gathered_map = None


class WeightsMapmakerIQU:
    """Binned polarized weight/covariance mapmaker for I,Q,U.

    This class accumulates the left-hand side (A matrix) of the mapmaking
    system, storing the 6 unique elements per pixel. The gathered A map is
    then inverted per pixel to provide RMS/covariance information and used as
    normalization input for MapmakerIQU.

    Usage:
    - Call `accumulate_to_map()` for each scan to build the A elements.
    - Call `gather_map()` to reduce across MPI tasks.
    - Call `normalize_map()` to compute RMS maps and expose `final_cov_map`
        for MapmakerIQU normalization.
    """
    def __init__(self, map_comm:MPI.Comm, nside:int, dtype=np.float32):
        self.logger = logging.getLogger(__name__)
        self.map_comm = map_comm
        self.nside = nside
        self.npix = 12*nside**2
        self.dtype= dtype
        self._map_signal = np.zeros((6, self.npix), dtype=np.float64)
        self._gathered_map = None
        self._finalized_rms_map = None
        
        # Setting up Ctypes mapmaker
        self.maplib = load_cmdr4_ctypes_lib()
        ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_int64, ndim=1, flags="contiguous")
        ct_f64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=1, flags="contiguous")
        ct_f64_dim2 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=2, flags="contiguous")
        self.maplib.map_weight_accumulator_IQU_f64.argtypes = [ct_f64_dim2, ct.c_double, ct_i64_dim1,
                                       ct_f64_dim1, ct.c_int64, ct.c_int64]
        self.maplib.map_invdiag_IQU_f64.argtypes = [ct_f64_dim2, ct_f64_dim2, ct.c_int64]

    @property
    def final_rms_map(self):
        if self.map_comm.Get_rank() == 0:
            logassert(self._finalized_rms_map is not None, "Attempted to read map before it was done.",
                      self.logger)
        return self._finalized_rms_map
    
    @property
    def final_cov_map(self):
        if self.map_comm.Get_rank() == 0:
            logassert(self._gathered_map is not None, "Attempted to read map before it was done.",
                      self.logger)
        return self._gathered_map

    def accumulate_to_map(self, weight:float, pix:NDArray, psi:NDArray):
        """Accumulate IQU weight/covariance elements into the local buffer."""
        # Check that we are still in business, and haven't already called "gather_map".
        logassert(self._map_signal is not None, "Tried accumulating to finalized map", self.logger)
        ntod = pix.shape[0]
        weight_f64 = float(weight)
        psi_f64 = np.ascontiguousarray(psi, dtype=np.float64)
        self.maplib.map_weight_accumulator_IQU_f64(self._map_signal, weight_f64, pix.astype(np.int64),
                                                   psi_f64, ntod, self.npix)

    def accumulate_to_map_Python(self, weight:float, pix:NDArray, psi:NDArray):
        """Reference accumulator matching the ctypes IQU weights implementation."""
        # Reference implementation matching the ctypes IQU weight accumulator.
        logassert(self._map_signal is not None, "Tried accumulating to finalized map", self.logger)
        pix_idx = pix.astype(np.int64, copy=False)
        ang = 2.0 * np.ascontiguousarray(psi, dtype=np.float64)
        c2 = np.cos(ang)
        s2 = np.sin(ang)
        weight_f64 = float(weight)
        np.add.at(self._map_signal[0], pix_idx, weight_f64)
        np.add.at(self._map_signal[1], pix_idx, weight_f64 * c2)
        np.add.at(self._map_signal[2], pix_idx, weight_f64 * s2)
        np.add.at(self._map_signal[3], pix_idx, weight_f64 * c2 * c2)
        np.add.at(self._map_signal[4], pix_idx, weight_f64 * s2 * c2)
        np.add.at(self._map_signal[5], pix_idx, weight_f64 * s2 * s2)

    @property
    def inv_N_diag(self):
        """
        Gives the diagonal of the accumulated inverse covariance matrix, per each pixel.

        It is useful as a preconditioner for the CG mapmaker.

        Note: call it before `gather_map` wipes the inverse covariance matrix away. 
        """
        logassert(self._map_signal is not None, "Attempted to access inv cov map after finalization.",
                      self.logger)
        return self._map_signal[(0,3,5), :]

    def gather_map(self):
        """Reduce the local IQU weight buffers across MPI ranks into the root map."""
        if self.map_comm.Get_rank() == 0:
            self._gathered_map = np.zeros((6, self.npix), dtype=np.float64)
        self.map_comm.Reduce(self._map_signal, self._gathered_map, op=MPI.SUM, root=0)
        self._map_signal = None  # Free memory and indicate that accumulation is done.

    def normalize_map(self):
        """Compute RMS maps from the per-pixel inverse covariance diagonals."""
        if self.map_comm.Get_rank() == 0:
            norm_map = np.ascontiguousarray(self._gathered_map, dtype=np.float64)
            rms = np.zeros((3, self.npix), dtype=np.float64)
            self.maplib.map_invdiag_IQU_f64(rms, norm_map, self.npix)
            self._finalized_rms_map = rms.astype(self.dtype, copy=False)

    def normalize_map_Python(self):
        """Reference RMS computation using NumPy inversion."""
        if self.map_comm.Get_rank() == 0:
            self._finalized_rms_map = np.zeros((3, self.npix), dtype=self.dtype)
 
            # `A` matrix is float64 no matter what, to get very accurate inversion.
            A = np.zeros((self.npix, 3, 3), dtype=np.float64)
            A[:, 0, 0] = self._gathered_map[0]
            A[:, 0, 1] = self._gathered_map[1]
            A[:, 1, 0] = self._gathered_map[1]
            A[:, 0, 2] = self._gathered_map[2]
            A[:, 2, 0] = self._gathered_map[2]
            A[:, 1, 1] = self._gathered_map[3]
            A[:, 1, 2] = self._gathered_map[4]
            A[:, 2, 1] = self._gathered_map[4]
            A[:, 2, 2] = self._gathered_map[5]

            # Check if matrix is non-SPD, singular, or ill-conditioned.
            det = np.linalg.det(A)
            diag_prod = A[:, 0, 0] * A[:, 1, 1] * A[:, 2, 2]
            eps = np.finfo(np.float64).eps
            mask = (diag_prod > 0) & (np.abs(det) > eps * diag_prod)

            # If any of the above, RMS is set to inf.
            if np.any(mask):
                A_inv = np.linalg.inv(A[mask])
                diag = np.diagonal(A_inv, axis1=1, axis2=2)
                diag = np.where(diag >= 0, np.sqrt(diag), np.inf)
                self._finalized_rms_map[:, mask] = diag.T.astype(self.dtype, copy=False)