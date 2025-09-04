import numpy as np
import os
import healpy as hp
import ctypes as ct
from pixell import bunch
from mpi4py import MPI
import logging
from numpy.typing import NDArray

from src.python.output.log import logassert
from src.python.data_models.scan_TOD import ScanTOD
from src.python.data_models.detector_TOD import DetectorTOD
from src.python.utils.map_utils import get_static_sky_TOD, get_s_orb_TOD

current_dir_path = os.path.dirname(os.path.realpath(__file__))
src_dir_path = os.path.abspath(os.path.join(os.path.join(current_dir_path, os.pardir), os.pardir))


class Mapmaker:
    def __init__(self, map_comm:MPI.Comm, nside:int, dtype=np.float32):
        self.logger = logging.getLogger(__name__)
        self.map_comm = map_comm
        self.nside = nside
        self.npix = 12*nside**2
        self.dtype= dtype
        self._map_signal = np.zeros(self.npix, dtype=dtype)
        self._gathered_map = None
        self._finalized_map = None
        
        # Setting up Ctypes mapmaker
        self.maplib = ct.cdll.LoadLibrary(os.path.join(src_dir_path, "cpp/mapmaker.so"))
        ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_int64, ndim=1, flags="contiguous")
        ct_f32_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_float, ndim=1, flags="contiguous")
        self.maplib.map_accumulator_f32.argtypes = [ct_f32_dim1, ct_f32_dim1, ct.c_double, ct_i64_dim1,
                                                    ct.c_int64, ct.c_int64]

    @property
    def final_map(self):
        if self.map_comm.Get_rank() == 0:
            logassert(self._finalized_map is not None, "Attempted to retrieve map before it was done.",
                    self.logger)
        return self._finalized_map

    def accumulate_to_map(self, tod:NDArray, weights:NDArray, pix:NDArray):
        # Check that we are still in business, and haven't already called "gather_map".
        logassert(self._map_signal is not None, "Tried accumulating to finalized map", self.logger)
        ntod = tod.shape[0]
        self.maplib.map_accumulator_f32(self._map_signal, tod, weights,
                                    pix.astype(np.int64), ntod, self.npix)

    def gather_map(self):
        if self.map_comm.Get_rank() == 0:
            self._gathered_map = np.zeros(self.npix, dtype=self.dtype)
        self.map_comm.Reduce(self._map_signal, self._gathered_map, op=MPI.SUM, root=0)
        self._map_signal = None  # Free memory and indicate that accumulation is done.
    
    def normalize_map(self, normalization_map):
        if self.map_comm.Get_rank() == 0:
            mask = normalization_map != 0
            self._finalized_map = self._gathered_map[mask] / normalization_map[mask]
            self._finalized_map[~mask] = 0.0
            self._gathered_map = None



class WeightsMapmaker:
    def __init__(self, map_comm:MPI.Comm, nside:int, dtype=np.float32):
        self.logger = logging.getLogger(__name__)
        self.map_comm = map_comm
        self.nside = nside
        self.npix = 12*nside**2
        self.dtype= dtype
        self._map_signal = np.zeros(self.npix, dtype=dtype)
        self._gathered_map = None
        
        # Setting up Ctypes mapmaker
        self.maplib = ct.cdll.LoadLibrary(os.path.join(src_dir_path, "cpp/mapmaker.so"))
        ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_int64, ndim=1, flags="contiguous")
        ct_f32_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_float, ndim=1, flags="contiguous")
        self.maplib.map_weight_accumulator_f32.argtypes = [ct_f32_dim1, ct.c_float, ct_i64_dim1,
                                                           ct.c_int64, ct.c_int64]

    @property
    def final_map(self):
        if self.map_comm.Get_rank() == 0:
            logassert(self._gathered_map is not None, "Attempted to retrieve map before it was done.",
                    self.logger)
        return self._gathered_map

    def accumulate_to_map(self, weight:NDArray, pix:NDArray):
        # Check that we are still in business, and haven't already called "gather_map".
        logassert(self._map_signal is not None, "Tried accumulating to finalized map", self.logger)
        ntod = pix.shape[0]
        self.maplib.map_weight_accumulator_f32(self._map_signal, weight, pix.astype(np.int64), ntod,
                                               self.npix)

    def gather_map(self):
        if self.map_comm.Get_rank() == 0:
            self._gathered_map = np.zeros(self.npix, dtype=self.dtype)
        self.map_comm.Reduce(self._map_signal, self._gathered_map, op=MPI.SUM, root=0)
        self._map_signal = None  # Free memory and indicate that accumulation is done.



class MapmakerIQU:
    def __init__(self, map_comm:MPI.Comm, nside:int, dtype=np.float32):
        self.logger = logging.getLogger(__name__)
        self.map_comm = map_comm
        self.nside = nside
        self.npix = 12*nside**2
        self.dtype= dtype
        self._map_signal = np.zeros((3, self.npix), dtype=dtype)
        self._gathered_map = None
        self._finalized_map = None
        
        # Setting up Ctypes mapmaker
        self.maplib = ct.cdll.LoadLibrary(os.path.join(src_dir_path, "cpp/mapmaker.so"))
        ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_int64, ndim=1, flags="contiguous")
        ct_f32_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_float, ndim=1, flags="contiguous")
        ct_f64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=1, flags="contiguous")
        ct_f32_dim2 = np.ctypeslib.ndpointer(dtype=ct.c_float, ndim=2, flags="contiguous")
        self.maplib.map_accumulator_IQU_f32.argtypes = [ct_f32_dim2, ct_f32_dim1, ct.c_double,
                                                        ct_i64_dim1, ct_f64_dim1, ct.c_int64,
                                                        ct.c_int64]

    @property
    def final_map(self):
        if self.map_comm.Get_rank() == 0:
            logassert(self._finalized_map is not None, "Attempted to read map before it was done.",
                      self.logger)
        return self._finalized_map


    def accumulate_to_map(self, tod:NDArray, weights:NDArray, pix:NDArray, psi:NDArray):
        # Check that we are still in business, and haven't already called "gather_map".
        logassert(self._map_signal is not None, "Tried accumulating to finalized map", self.logger)
        ntod = tod.shape[0]
        self.maplib.map_accumulator_IQU_f32(self._map_signal, tod, weights, pix.astype(np.int64),
                                            psi.astype(np.float64), ntod, self.npix)

    def gather_map(self):
        if self.map_comm.Get_rank() == 0:
            self._gathered_map = np.zeros((3, self.npix), dtype=self.dtype)
        self.map_comm.Reduce(self._map_signal, self._gathered_map, op=MPI.SUM, root=0)
        self._map_signal = None  # Free memory and indicate that accumulation is done.
    
    def normalize_map(self, normalization_map):
        if self.map_comm.Get_rank() == 0:
            logassert(normalization_map.ndim == 2 and normalization_map.shape[0] == 6,
                    "Normalization map must have shape [6,NPIX] for IQU mapmaker,"
                    f"has {normalization_map.shape}", self.logger)
            # Set up A-matrix for IQU mapmaking
            A = np.zeros((self.npix, 3, 3), dtype=self.dtype)
            A[:,0,0] = normalization_map[0]
            A[:,0,1] = normalization_map[1]
            A[:,1,0] = normalization_map[1]
            A[:,0,2] = normalization_map[2]
            A[:,2,0] = normalization_map[2]
            A[:,1,1] = normalization_map[3]
            A[:,1,2] = normalization_map[4]
            A[:,2,1] = normalization_map[4]
            A[:,2,2] = normalization_map[5]

            A += np.eye(3)*1e-12  # Regularization, avoid singular matrix.
            # Solve the Ax=b mapmaking problem for IQU mapmaking, where A is a 3x3 matrix per pixel.
            # The transposes, newaxis, and [0] is just to get the right dimensions etc.
            self._finalized_map = np.linalg.solve(A, self._gathered_map.T[..., np.newaxis]).T[0]
            self._gathered_map = None


class WeightsMapmakerIQU:
    def __init__(self, map_comm:MPI.Comm, nside:int, dtype=np.float32):
        self.logger = logging.getLogger(__name__)
        self.map_comm = map_comm
        self.nside = nside
        self.npix = 12*nside**2
        self.dtype= dtype
        self._map_signal = np.zeros((6, self.npix), dtype=dtype)
        self._gathered_map = None
        self._finalized_map = None
        
        # Setting up Ctypes mapmaker
        self.maplib = ct.cdll.LoadLibrary(os.path.join(src_dir_path, "cpp/mapmaker.so"))
        ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_int64, ndim=1, flags="contiguous")
        ct_f64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=1, flags="contiguous")
        ct_f32_dim2 = np.ctypeslib.ndpointer(dtype=ct.c_float, ndim=2, flags="contiguous")
        self.maplib.map_weight_accumulator_IQU_f32.argtypes = [ct_f32_dim2, ct.c_float, ct_i64_dim1,
                                                               ct_f64_dim1, ct.c_int64, ct.c_int64]

    @property
    def final_map(self):
        if self.map_comm.Get_rank() == 0:
            logassert(self._finalized_map is not None, "Attempted to read map before it was done.",
                      self.logger)
        return self._finalized_map
    
    @property
    def final_cov_map(self):
        if self.map_comm.Get_rank() == 0:
            logassert(self._gathered_map is not None, "Attempted to read map before it was done.",
                      self.logger)
        return self._gathered_map

    def accumulate_to_map(self, weight:float, pix:NDArray, psi:NDArray):
        # Check that we are still in business, and haven't already called "gather_map".
        logassert(self._map_signal is not None, "Tried accumulating to finalized map", self.logger)
        ntod = pix.shape[0]
        self.maplib.map_weight_accumulator_IQU_f32(self._map_signal, weight, pix.astype(np.int64),
                                                   psi.astype(np.float64), ntod, self.npix)

    def gather_map(self):
        if self.map_comm.Get_rank() == 0:
            self._gathered_map = np.zeros((6, self.npix), dtype=self.dtype)
        self.map_comm.Reduce(self._map_signal, self._gathered_map, op=MPI.SUM, root=0)
        self._map_signal = None  # Free memory and indicate that accumulation is done.


    def normalize_map(self):
        if self.map_comm.Get_rank() == 0:
            self._finalized_map = np.zeros((3, self.npix), dtype=self.dtype)
            # Set up A-matrix for IQU mapmaking
            A = np.zeros((self.npix, 3, 3), dtype=self.dtype)
            A[:,0,0] = self._gathered_map[0]
            A[:,0,1] = self._gathered_map[1]
            A[:,1,0] = self._gathered_map[1]
            A[:,0,2] = self._gathered_map[2]
            A[:,2,0] = self._gathered_map[2]
            A[:,1,1] = self._gathered_map[3]
            A[:,1,2] = self._gathered_map[4]
            A[:,2,1] = self._gathered_map[4]
            A[:,2,2] = self._gathered_map[5]

            A += np.eye(3)*1e-12  # Regularization, avoid singular matrix.
            # The inverse-variance of the I,Q,U maps are the diagonal elemenents of the inverse of A.
            A_inv = np.linalg.pinv(A)
            self._finalized_map[0] = A_inv[:,0,0]
            self._finalized_map[1] = A_inv[:,1,1]
            self._finalized_map[2] = A_inv[:,2,2]