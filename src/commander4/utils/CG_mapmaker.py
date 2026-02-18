import numpy as np
import ctypes as ct
from mpi4py import MPI
import logging
from numpy.typing import NDArray
import ducc0
import healpy as hp
from pixell import utils
from typing import Callable
import time

from commander4.output.log import logassert
from commander4.utils.ctypes_lib import load_cmdr4_ctypes_lib
from commander4.data_models.detector_TOD import DetectorTOD
from commander4.data_models.scan_TOD import ScanTOD
from commander4.data_models.detector_samples import DetectorSamples
from commander4.utils.math_operations import inplace_scale, MPI_dot

# I need to CG-solve P^T T^T N^−1 T P m = P^T T^T N^-1 d
# Where:
# - m is the final map [npix]
# - P is the pointing matrix: [ntod, npix]
# - T is the bolometer trnasfer function operator (check Artem's code)
# - N^-1 is the inverse noise covariance matrix (inv_var in tod_processing.tod2map) which is diagonal in tod space [ntod]
# - d is the calibrated TODs [ntod].
# Notes: T is non-local so each rank must hold all the scans, but only for one detector.


class CG_Mapmaker:
    """
    CG mapmaker for solving the general P^T T^T N^-1 T P m = P^T T^T N^-1 d problem
    """
    def __init__(self, 
                detector_tod:DetectorTOD, detector_samples:DetectorSamples, 
                map_comm:MPI.Comm, T_omega:Callable = np.ones_like, preconditioner:Callable = np.copy,
                nthreads:int=1, double_prec:bool = True, CG_maxiter:int=200, CG_tol:float=1e-10, CG_check_interval:int = 1):
        self.logger = logging.getLogger(__name__)
        self.detector_tod = detector_tod
        self.detector_samples = detector_samples
        self.double_perc = double_prec
        #output map to be solved for, serves as buffer for the last mpi.Reduce call 
        self._map_signal = np.zeros((3,hp.nside2npix(detector_tod._eval_nside)), 
            dtype=np.float64 if double_prec else np.float32) if map_comm.Get_rank() == 0 else None
        self.nthreads = nthreads
        self.T_omega = T_omega
        self.map_comm = map_comm
        self.CG_maxiter = CG_maxiter
        self.CG_tol = CG_tol
        self.CG_check_interval = CG_check_interval
        self.M = preconditioner
        self.maplib = load_cmdr4_ctypes_lib()
        ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_int64, ndim=1, flags="contiguous")
        ct_f64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=1, flags="contiguous")
        ct_f64_dim2 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=2, flags="contiguous")
        self.maplib.map_accumulator_IQU_f64.argtypes = [ct_f64_dim2, ct_f64_dim1, ct.c_double,
                                ct_i64_dim1, ct_f64_dim1, ct.c_int64,
                                ct.c_int64]
        self.maplib.map2tod_IQU_f64.argtypes = [ct_f64_dim2, ct_f64_dim1,
                                            ct_i64_dim1, ct_f64_dim1, 
                                            ct.c_int64, ct.c_int64]

    @property
    def solved_map(self):
        if self.map_comm.Get_rank() == 0:
            logassert(self._map_signal is not None, "Attempted to read map on master rank before it was solved.",
                      self.logger)
        return self._map_signal

    def apply_inv_N(self, scan:ScanTOD, sigma0:float, scan_tod_arr=None):
        """
        Applies inplace the N^-1 operator to one scan, given the corresponding noise variance sigma0.

        if a `scan_tod_arr` is passed, it will be used to overwrite the result instead of using `scan.tod`.
        """
        scan_tod_arr = scan._tod if scan_tod_arr is None else scan_tod_arr
        inplace_scale(scan_tod_arr, 1/sigma0**2)
        # logger = logging.getLogger(__name__)

        # if self.map_comm.Get_rank() == 0:
        #     logger.info(f"## Inv N sigma0 {sigma0} mean: {np.mean(scan_tod_arr)}")
        return scan_tod_arr

    def apply_P(self, in_map: NDArray, out_scan:ScanTOD, scan_tod_arr=None):
        """
        Applies the pointing matrix operator to one scan.
        
        It takes an array of pixels in input and projects them over a time ordered data scan.

        if a `scan_tod_arr` is passed, it will be used to overwrite the result instead of using `scan.tod`.
        """
        scan_tod_arr = out_scan._tod if scan_tod_arr is None else scan_tod_arr
        npix_out = hp.nside2npix(out_scan._eval_nside)
        assert npix_out == in_map.shape[1] 
        pix = out_scan.pix
        psi = out_scan.psi
        ntod = out_scan.tod.shape[0]
        tod_f64 = np.ascontiguousarray(scan_tod_arr, dtype=np.float64) #FIXME: check if this is actually updating tods in place
        psi_f64 = np.ascontiguousarray(psi, dtype=np.float64)
        self.maplib.map2tod_IQU_f64(in_map, tod_f64,  
                                            pix.astype(np.int64), 
                                            psi_f64, ntod, npix_out)
        return tod_f64.astype(dtype=np.float32)

    def apply_P_adjoint(self, in_scan: ScanTOD, out_map:NDArray, scan_tod_arr=None):
        """
        Applies the adjoint, or transpose in matrix-notation, of the pointing matrix operator to one scan, updating out_map inplace.

        It takes in input a time ordered data scan and accumulates them over a map in output.

        if a `scan_tod_arr` is passed, it will be used to compute the result instead of using `scan.tod`.
        """
        scan_tod_arr = in_scan._tod if scan_tod_arr is None else scan_tod_arr
        npix_out = out_map.shape[1]
        assert npix_out == hp.nside2npix(in_scan._eval_nside)
        pix = in_scan.pix  #FIXME: maybe move this in LHS so we uncompress pix and psi only once.
        psi = in_scan.psi
        ntod = in_scan.tod.shape[0]
        tod_f64 = np.ascontiguousarray(scan_tod_arr, dtype=np.float64)
        psi_f64 = np.ascontiguousarray(psi, dtype=np.float64)
        self.maplib.map_accumulator_IQU_f64(out_map, tod_f64, 1, 
                                            pix.astype(np.int64), 
                                            psi_f64, ntod, npix_out)
        return out_map

    def _apply_T(self, scan: ScanTOD, adjoint=False, scan_tod_arr=None):
        """
        General function for T and T^T, inplace.
        """
        scan_tod_arr = scan._tod if scan_tod_arr is None else scan_tod_arr
        #F d
        d = ducc0.fft.r2c(scan_tod_arr, forward=True, inorm=1)
        #T F d
        freqs = np.fft.rfftfreq(len(scan_tod_arr)) #if we want freqs in Hz fro T(omega) the use scan.fsamp.
        if adjoint:
            freqs = np.flip(freqs)
        d = d * self.T_omega(freqs)
        #F^-1 T F
        scan_tod_arr = ducc0.fft.c2r(d, forward=False, inorm=1)
        return scan_tod_arr

    def apply_T(self, scan: ScanTOD, scan_tod_arr=None):
        """
        Applies INPLACE the T operator defined as T := F^-1 T(omega) F, to one scan.

        If a `scan_tod_arr` is passed, it will be used to overwrite the result instead of using `scan.tod`.

        Notes:
        - T(omega) is a filter which must respect the Hermitian symmetry T*(omega) = T(-omega) 
        in order to give real-valued outputs in time space, and F and F^-1 are RFFT and RFFT inverse.

        - F returns an array of frequencies `omega_s` and an array of complex amplitudes, 1 for each of those frequencies.
        T(omega) must be evaluated at each of those omega's and the result (a complex amplitude for each omega) is also a complex array
        that must be multiplicated element-wise with the output of F.
        The result is then given in input to F^-1, which is a irfft taking the complex array back to real tod domain. 

        - The "ortho" normalization of the FFT corresponds to option 1 in ducc0.
        """
        return self._apply_T(scan, adjoint=False, scan_tod_arr=scan_tod_arr)

    def apply_T_adjoint(self, scan: ScanTOD, scan_tod_arr=None):
        """
        Applies INPLACE the adjoint, or transpose in matrix-notation, of the T operator to one scan. 
        It is defined as T^T := (F^-1 T(omega) F)^T = F^T T*(omega) (F^-1)^T = F^-1 T(-omega) F.

        If a `scan_tod_arr` is passed, it will be used to overwrite the result instead of using `scan.tod`.

        Notes:
        - T(omega) is a filter which must respect the Hermitian symmetry T*(omega) = T(-omega) 
        in order to give real-valued outputs in time space, and F and F^-1 := F^T are RFFT and
        RFFT inverse, which corresponds to the adjoint, which in this notation is improperly written as F^T.

        - Numerically the procedure is identical to apply_T, with the difference that the complex array
        given by T(omega) is flipped, as we are evaluating T(-omega).
        """
        return self._apply_T(scan, adjoint=True, scan_tod_arr=scan_tod_arr)
    
    def calc_RHS(self, out_map=None):
        """
        Computes the RHS of the mapmaking problem: P^T T^T N^-1 d.

        if no `out_map` is given, a new array is allocated.
        """
        out_map = np.zeros((3,hp.nside2npix(self.detector_tod.nside)), dtype=np.float64 if self.double_perc else np.float32) if out_map is None else out_map
        for scan, sample in zip(self.detector_tod.scans, self.detector_samples.scans):
            scan_tod_arr_aux = np.copy(scan._tod) #aux array to not modify scan._tod
            #N^-1 d
            scan_tod_arr_aux = self.apply_inv_N(scan, sample.sigma0, scan_tod_arr=scan_tod_arr_aux)
            
            logger = logging.getLogger(__name__)

            # if self.map_comm.Get_rank() == 0:
            #     logger.info(f"##RHS 1 mean: {np.mean(scan_tod_arr_aux)}")
                
            #T^T N^-1 d
            scan_tod_arr_aux = self.apply_T_adjoint(scan, scan_tod_arr=scan_tod_arr_aux)

            # if self.map_comm.Get_rank() == 0:
            #     logger.info(f"##RHS 1 mean: {np.mean(scan_tod_arr_aux)}")

            #P^T T^T N^-1 d
            out_map = self.apply_P_adjoint(scan, out_map, scan_tod_arr=scan_tod_arr_aux)

            # if self.map_comm.Get_rank() == 0:
            #     logger.info(f"##RHS 3 mean: {np.mean(out_map)}")
        return out_map

    def apply_LHS(self, m_map: NDArray, out_map=None):
        """
        Applies the LHS of the mapmaking problem P^T T^T N^-1 T P m to an input map, without modifying it,
        and updates the result in the out_tods object.
        """
        logger = logging.getLogger(__name__)
        out_map = np.zeros_like(m_map) if out_map is None else out_map
        for scan, sample in zip(self.detector_tod.scans, self.detector_samples.scans):
            scan_tod_arr_aux = np.zeros_like(scan._tod) #aux array to not modify scan._tod
            # if self.map_comm.Get_rank() == 0:
            #     logger.info(f"##LHS 1 mean: {np.mean(scan_tod_arr_aux)}")
            #P m
            scan_tod_arr_aux = self.apply_P(m_map, scan, scan_tod_arr=scan_tod_arr_aux)
            # if self.map_comm.Get_rank() == 0:
            #     logger.info(f"##LHS 2 mean: {np.mean(scan_tod_arr_aux)}")
            #T P m
            scan_tod_arr_aux = self.apply_T(scan, scan_tod_arr=scan_tod_arr_aux)
            # if self.map_comm.Get_rank() == 0:
            #     logger.info(f"##LHS 3 mean: {np.mean(scan_tod_arr_aux)}")
            #N^-1 T P m
            scan_tod_arr_aux = self.apply_inv_N(scan, sample.sigma0, scan_tod_arr=scan_tod_arr_aux)
            # if self.map_comm.Get_rank() == 0:
            #     logger.info(f"##LHS 4 mean: {np.mean(scan_tod_arr_aux)}")
            #T^T N^-1 T P m
            scan_tod_arr_aux = self.apply_T_adjoint(scan, scan_tod_arr=scan_tod_arr_aux)
            # if self.map_comm.Get_rank() == 0:
            #     logger.info(f"##LHS 5 mean: {np.mean(scan_tod_arr_aux)}")
            #P^T T^T N^-1 T P
            out_map = self.apply_P_adjoint(scan, out_map, scan_tod_arr=scan_tod_arr_aux)
            # if self.map_comm.Get_rank() == 0:
            #     logger.info(f"##LHS 6 mean: {np.mean(scan_tod_arr_aux)}")
        return out_map

    def solve(self):
        """
        Solves the CG to compute the target sky map.
        """
        logger = logging.getLogger(__name__)
        RHS_map = self.calc_RHS()
        ismaster = self.map_comm.Get_rank() == 0
        # logger.info(f"##RHS map mean: {np.mean(RHS_map)}")

        my_dot = lambda arr1, arr2: MPI_dot(arr1, arr2, self.map_comm, double_prec=self.double_perc)
        CG_solver = utils.CG(self.apply_LHS, RHS_map, M = self.M, dot = my_dot)
        if ismaster:
            logger.info(f"Mapmaker CG starting up!")
        for i in range(self.CG_maxiter):
            CG_solver.step()
            if i%self.CG_check_interval == 0:
                if ismaster:
                    logger.info(f"Mapmaker CG iter {i:3d} - Residual {CG_solver.err:.6e}")
            if CG_solver.err < self.CG_tol:
                break
        local_map = CG_solver.x
        self.map_comm.Reduce(local_map, self._map_signal, op=MPI.SUM, root=0)
        local_map = None #free memory.

