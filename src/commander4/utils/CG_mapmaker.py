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
import matplotlib.pyplot as plt

from commander4.output.log import logassert
from commander4.utils.ctypes_lib import load_cmdr4_ctypes_lib
from commander4.data_models.detector_TOD import DetectorTOD
from commander4.data_models.scan_TOD import ScanTOD
from commander4.solvers.CG_driver import distributed_CG_arr
from commander4.data_models.detector_samples import DetectorSamples
from commander4.data_models.scan_samples import ScanSamples
from commander4.utils.math_operations import inplace_scale, dot, norm

# I need to CG-solve P^T T^T N^−1 T P m = P^T T^T N^-1 d
# Where:
# - m is the final map [npix]
# - P is the pointing matrix: [ntod, npix]
# - T is the bolometer trnasfer function operator (check Artem's code)
# - N^-1 is the inverse noise covariance matrix (inv_var in tod_processing.tod2map) which is diagonal in tod space [ntod]
# - d is the calibrated TODs [ntod].
# Notes: T is non-local so each rank must hold the whole scan, but only for one detector.


class CG_Mapmaker:
    """
    CG mapmaker for solving the general P^T T^T N^-1 T P m = P^T T^T N^-1 d problem
    """
    def __init__(self, 
                detector_tod:DetectorTOD, 
                detector_samples:DetectorSamples, 
                map_comm:MPI.Comm,
                is_IQU:bool, #else only I is assumed
                #optionals:
                T_omega:Callable = np.ones_like, 
                preconditioner:Callable = np.copy,
                nthreads:int=1, 
                double_prec:bool = True, 
                CG_maxiter:int=200, 
                CG_tol:float=1e-10, 
                CG_check_interval:int = 1):
        self.logger = logging.getLogger(__name__)
        self.detector_tod = detector_tod
        self.detector_samples = detector_samples
        self.double_perc = double_prec
        self.is_IQU = is_IQU
        self.map_comm = map_comm
        self.ismaster = self.map_comm.Get_rank() == 0
        self.f_dtype = np.float64 if double_prec else np.float32
        #output map to be solved for
        self._map_signal = np.zeros(
            (3 if is_IQU else 1,hp.nside2npix(detector_tod._eval_nside)), 
            dtype=self.f_dtype) if self.ismaster else None
        #local RHS map
        self._rhs_loca_map = None
        #RHS map to be accumulated on master rank
        self._rhs_finalized_map = np.zeros(
            (3 if is_IQU else 1,hp.nside2npix(detector_tod._eval_nside)), 
            dtype=self.f_dtype) if self.ismaster else None
        self.nthreads = nthreads
        self.T_omega = T_omega
        self.CG_maxiter = CG_maxiter
        self.CG_tol = CG_tol
        self.CG_check_interval = CG_check_interval
        self.M = preconditioner
        self.maplib = load_cmdr4_ctypes_lib()
        ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_int64, ndim=1, flags="contiguous")
        if double_prec:
            ct_f64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=1, flags="contiguous")
            ct_f64_dim2 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=2, flags="contiguous")
            if is_IQU:
                self.maplib.map_accumulator_IQU_f64.argtypes = [ct_f64_dim2, ct_f64_dim1, ct.c_double,
                                        ct_i64_dim1, ct_f64_dim1, ct.c_int64,
                                        ct.c_int64]
                self.maplib.map2tod_IQU_f64.argtypes = [ct_f64_dim2, ct_f64_dim1,
                                                    ct_i64_dim1, ct_f64_dim1, 
                                                    ct.c_int64, ct.c_int64]
                self.map_accumulator = self.maplib.map_accumulator_IQU_f64
                self.map2tod = self.maplib.map2tod_IQU_f64
            else:
                self.maplib.map_accumulator_f64.argtypes = [ct_f64_dim2, ct_f64_dim1, ct.c_double,
                                        ct_i64_dim1, ct.c_int64]
                self.maplib.map2tod_f64.argtypes = [ct_f64_dim2, ct_f64_dim1,
                                                    ct_i64_dim1, ct.c_int64]
                self.map_accumulator = self.maplib.map_accumulator_f64
                self.map2tod = self.maplib.map2tod_f64
        else:
            ct_f32_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_float, ndim=1, flags="contiguous")
            ct_f32_dim2 = np.ctypeslib.ndpointer(dtype=ct.c_float, ndim=2, flags="contiguous")
            if is_IQU:
                self.maplib.map_accumulator_IQU_f32.argtypes = [ct_f32_dim2, ct_f32_dim1, ct.c_double,
                                        ct_i64_dim1, ct_f64_dim1, ct.c_int64,
                                        ct.c_int64]
                self.maplib.map2tod_IQU_f32.argtypes = [ct_f32_dim2, ct_f32_dim1,
                                                    ct_i64_dim1, ct_f64_dim1, 
                                                    ct.c_int64, ct.c_int64]
                self.map_accumulator = self.maplib.map_accumulator_IQU_f32
                self.map2tod = self.maplib.map2tod_IQU_f32
            else:
                self.maplib.map_accumulator_f32.argtypes = [ct_f32_dim2, ct_f32_dim1, ct.c_double,
                                        ct_f64_dim1, ct.c_int64]
                self.maplib.map2tod_f32.argtypes = [ct_f32_dim2, ct_f32_dim1,
                                                    ct_i64_dim1, ct.c_int64]
                self.map_accumulator = self.maplib.map_accumulator_f32
                self.map2tod = self.maplib.map2tod_f32
       
    @property
    def solved_map(self):
        if self.map_comm.Get_rank() == 0:
            logassert(self._map_signal is not None, "Attempted to read solution map on master rank before it was solved.",
                      self.logger)
        return self._map_signal
    
    @property
    def RHS_map(self):
        if self.map_comm.Get_rank() == 0:
            logassert(self._rhs_finalized_map is not None, "Attempted to read RHS map on master rank before it was finalized.",
                      self.logger)
            return self._rhs_finalized_map
        else:
            return np.empty(())

    def apply_inv_N(self, scan_tod_arr:NDArray, sigma0:float):
        """
        Applies inplace the N^-1 operator to one scan, given the corresponding noise variance sigma0.

        Args:
        - `scan_tod_arr`: array of TODs corresponding to the scan.
        - `sigma0`: noise variance for that scan.
        """
        inplace_scale(scan_tod_arr, 1.0/sigma0**2)
        return scan_tod_arr

    def apply_P(self, in_map: NDArray, out_scan:ScanTOD, pix=None, scan_tod_arr=None):
        """
        Applies the pointing matrix operator to one scan.
        
        It takes in input a time ordered data scan and accumulates them over a map in output. if a 
        `pix` is passed, it will be used to compute the result instead of decompressing a new one 
        from `out_scan`. If a `scan_tod_arr` is passed it is used instead of overwriting `out_scan`
        """
        #scan_tod_arr = out_scan._tod if scan_tod_arr is None else scan_tod_arr
        npix_out = hp.nside2npix(out_scan._eval_nside)
        assert npix_out == in_map.shape[1] 
        pix = out_scan.pix if pix is None else pix
        ntod = out_scan.tod.shape[0]
        self.map2tod(in_map, scan_tod_arr, pix.astype(np.int64), ntod)
        return scan_tod_arr

    def apply_P_adjoint(self, in_scan: ScanTOD, out_map:NDArray, pix=None, scan_tod_arr=None):
        """
        Applies the adjoint, or transpose in matrix-notation, of the pointing matrix operator to one
        scan, updating out_map inplace.

        It takes in input a time ordered data scan and accumulates them over a map in output. if a 
        `pix` is passed, it will be used to compute the result instead of decompressing a new one 
        from `in_scan`. If a `scan_tod_arr` is passed it is used instead of overwriting `in_scan`
        """
        #scan_tod_arr = in_scan._tod if scan_tod_arr is None else scan_tod_arr
        npix_out = out_map.shape[1]
        assert npix_out == hp.nside2npix(in_scan._eval_nside)
        pix = in_scan.pix if pix is None else pix
        ntod = in_scan.tod.shape[0]
        self.map_accumulator(out_map, scan_tod_arr, 1, pix.astype(np.int64), ntod)
        return out_map

    def _apply_T(self, scan_tod_arr, adjoint=False):
        """
        General function for T and T^T, inplace.
        """
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

    def apply_T(self, scan_tod_arr):
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
        return self._apply_T(scan_tod_arr, adjoint=False)

    def apply_T_adjoint(self, scan_tod_arr):
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
        return self._apply_T(scan_tod_arr, adjoint=True)
    
    def accum_to_RHS(self, scan_tod: ScanTOD, scan_samp: ScanSamples, 
                     pix = None, scan_tod_arr = None):
        """
        Computes the contribution to RHS of the mapmaking problem: P^T T^T N^-1 d for one scan. 
        Both scan TOD and samples must be given. This allows to compute the RHS contirbutions in an
        external loop together with the correlated noise sampling, pix can be passed already
        uncompressed from an external loop to avoid double uncompression.
        """
        # if self.map_comm.Get_rank():
        #     self.logger.info(f"##RHS called!")
        # out_map = np.zeros(
        #     (3 if self.is_IQU else 1,hp.nside2npix(self.detector_tod._eval_nside)), 
        #     dtype=self.f_dtype)
        # for scan, sample in zip(self.detector_tod.scans, self.detector_samples.scans):

        if self._rhs_loca_map is None:
            #if not done already, allocate memory for local maps
            self._rhs_loca_map = np.zeros(
                (3 if self.is_IQU else 1,hp.nside2npix(self.detector_tod._eval_nside)), 
                dtype=self.f_dtype)

        if scan_tod_arr is None:
            scan_tod_arr = np.copy(scan_tod._tod) #aux array to not modify scan._tod
        #N^-1 d
        # if self.ismaster:
        #     self.logger.info(f"RHS_1: {scan_tod_arr}")
        scan_tod_arr = self.apply_inv_N(scan_tod_arr, scan_samp.sigma0)
        #T^T N^-1 d
        # if self.ismaster:
        #     self.logger.info(f"RHS_2: {scan_tod_arr}")
        scan_tod_arr = self.apply_T_adjoint(scan_tod_arr)
        # if self.ismaster:
        #     self.logger.info(f"RHS_3: {scan_tod_arr}")
        #P^T T^T N^-1 d
        self._rhs_loca_map = self.apply_P_adjoint(scan_tod, self._rhs_loca_map, pix=pix, scan_tod_arr=scan_tod_arr)

    def finalize_RHS(self, root=0):
        """
        Reduces RHS map on main rank, summing up all the contributions.
        """
        logassert(self._rhs_loca_map is not None, 
                "Attempted to reduce RHS map on master rank before its contributions have been computed.",
                self.logger)
        if self.map_comm.Get_rank() == 0:
            send, recv = (self._rhs_loca_map, self._rhs_finalized_map)  
        else: 
            send, recv = (self._rhs_loca_map, np.empty(()))
        self.map_comm.Reduce(send, recv, op=MPI.SUM, root=root)
        self.map_comm.Barrier()

        #free memory
        self._rhs_loca_map = None
        return recv

    def apply_LHS(self, in_map: NDArray):
        """
        Applies the LHS of the mapmaking problem P^T T^T N^-1 T P m to an input map, without modifying it,
        and updates the result in the out_tods object.
        """
        ismaster = self.map_comm.Get_rank() == 0
        if in_map.shape==():
            if ismaster:
                raise ValueError("input map can not be empty on master rank.")
            else:
                in_map = np.zeros(
                    (3 if self.is_IQU else 1, hp.nside2npix(self.detector_tod._eval_nside)), 
                    dtype=self.f_dtype)

        if ismaster:
            self.logger.info("## LHS Bcasting!!")

        # self.logger.info(f"## on rank {self.map_comm.Get_rank()} in_map is {in_map.shape} {in_map.dtype}")
        self.map_comm.Bcast(in_map, root=0)
        if ismaster:
            self.logger.info("## LHS Bcasting Done")
        out_map = np.zeros_like(in_map)
        pri = True
        for scan, sample in zip(self.detector_tod.scans, self.detector_samples.scans):
            scan_tod_arr_aux = np.zeros_like(scan._tod, dtype=self.f_dtype) #aux array to not modify scan._tod
            # if self.map_comm.Get_rank() == 0 and pri:
            #     self.logger.info(f"##LHS 1 mean: {np.mean(in_map)}")
            #P m
            scan_tod_arr_aux = self.apply_P(in_map, scan, scan_tod_arr=scan_tod_arr_aux)
            # if self.map_comm.Get_rank() == 0 and pri:
            #     self.logger.info(f"##LHS 2 mean: {np.mean(scan_tod_arr_aux)}")
            #T P m
            scan_tod_arr_aux = self.apply_T(scan_tod_arr_aux)
            # if self.map_comm.Get_rank() == 0 and pri:
            #     self.logger.info(f"##LHS 3 mean: {np.mean(scan_tod_arr_aux)}")
            #N^-1 T P m
            scan_tod_arr_aux = self.apply_inv_N(scan_tod_arr_aux, sample.sigma0)
            # if self.map_comm.Get_rank() == 0 and pri:
            #     self.logger.info(f"##LHS 4 mean: {np.mean(scan_tod_arr_aux)}")
            #T^T N^-1 T P m
            scan_tod_arr_aux = self.apply_T_adjoint(scan_tod_arr_aux)
            # if self.map_comm.Get_rank() == 0 and pri:
            #     self.logger.info(f"##LHS 5 mean: {np.mean(scan_tod_arr_aux)}")
            #P^T T^T N^-1 T P
            out_map = self.apply_P_adjoint(scan, out_map, scan_tod_arr=scan_tod_arr_aux)
            # if self.map_comm.Get_rank() == 0 and pri:
            #     self.logger.info(f"##LHS 6 mean: {np.mean(out_map)}")
            pri=False
        send, recv = (MPI.IN_PLACE, out_map) if self.map_comm.Get_rank() == 0 else (out_map, None)
        self.map_comm.Reduce(send, recv, op=MPI.SUM, root=0)
        if not ismaster:
            in_map = np.empty(())
            out_map = None
        return recv

    def solve(self, x_true=None):
        """
        Solves the CG to compute the target sky map.
        """
        RHS_map = self.RHS_map
        ismaster = self.map_comm.Get_rank() == 0


        def mydot(a, b):
            return np.dot(a.flatten(), b.flatten())
        my_dot = mydot # lambda arr1, arr2: MPI_dot(arr1, arr2, self.map_comm, double_prec=self.double_perc)
        CG_solver = distributed_CG_arr(self.apply_LHS, 
                                       RHS_map, 
                                       ismaster,
                                       M = self.M, 
                                       dot = my_dot,
                                       destroy_b=True)
        
        if ismaster:
            self.logger.info(f"Mapmaker CG starting up!")
        for i in range(self.CG_maxiter):
            CG_solver.step()
            if i%self.CG_check_interval == 0:
                if ismaster:
                    self.logger.info(f"Mapmaker CG iter {i:3d} - Residual {CG_solver.err:.6e}")
                    if x_true is not None:
                        CG_errors_true = norm(CG_solver.x - x_true)/norm(x_true)
                        A_residual = self.apply_LHS(CG_solver.x - x_true)
                        #A_residual = np.concatenate(A_residual, axis=-1)
                        CG_Anorm_error = dot(CG_solver.x - x_true, A_residual)
                        # A-norm error is only defined for the full vector.
                        self.logger.info(f"CG iter {i:3d} - Mean X: {norm(CG_solver.x)}")

                        self.logger.info(f"CG iter {i:3d} - Mean X true: {norm(x_true)}")

                        self.logger.info(f"CG iter {i:3d} - True A-norm error: {CG_Anorm_error:.3e}")
                        # We can print the individual component L2 errors.
                        self.logger.info(f"CG iter {i:3d} - True L2 error: {CG_errors_true:.3e}")

                        #check if LHS works:
                        #rhs_comp = self.apply_LHS(x_true.astype(np.float64))
                        #self.logger.info(f"CG iter {i:3d} - Check of rhs - rhs_comp {np.allclose(self.RHS_map, rhs_comp, rtol=1e-5, atol=1e-8)} max diff {np.max(self.RHS_map - rhs_comp)}")

                    # for s in ["I", "Q", "U"]:
                    #     plt.figure()
                    #     hp.mollview(CG_solver.x[0,:], min=-1e4, max=1e4, cmap = 'RdBu_r')
                    #     plt.savefig(f"/mn/stornext/u3/leoab/cmdr4_plots/x_sol_iter{i}_pol{s}.png")
                    #     plt.close()
                    if self.is_IQU:
                        plt.figure(figsize=(8.5*3, 5.4))
                        npol = 3 if self.is_IQU else 1
                        for i in range(npol):
                            limup   = np.nanpercentile(CG_solver.x[i,:], 99)
                            limdown = np.nanpercentile(CG_solver.x[i,:], 1)
                            hp.mollview(CG_solver.x[i,:], cmap='RdBu_r', title='CG sol',
                                        sub=(1,npol,i+1), min=limdown, max=limup)
                        plt.savefig(f"/mn/stornext/u3/leoab/cmdr4_plots/x_sol_IQU_iter{i}.png")
                        plt.close()
                    else:
                        plt.figure()
                        limup   = np.nanpercentile(CG_solver.x[0,:], 99)
                        limdown = np.nanpercentile(CG_solver.x[0,:], 1)
                        hp.mollview(CG_solver.x[0,:], cmap='RdBu_r', title='CG sol',
                                        min=limdown, max=limup)
                        plt.savefig(f"/mn/stornext/u3/leoab/cmdr4_plots/x_sol_iter{i}_Akari.png")
                        plt.close()

            if CG_solver.err < self.CG_tol:
                break
        self._map_signal = CG_solver.x