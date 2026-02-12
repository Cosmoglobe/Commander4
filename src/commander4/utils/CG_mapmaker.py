import numpy as np
import ctypes as ct
from mpi4py import MPI
import logging
from numpy.typing import NDArray
import ducc0
import healpy as hp

from commander4.output.log import logassert
from commander4.utils.ctypes_lib import load_cmdr4_ctypes_lib
from commander4.data_models.detector_TOD import DetectorTOD
from commander4.data_models.detector_samples import DetectorSamples
from commander4.utils.math_operations import inplace_scale

# current_dir_path = os.path.dirname(os.path.realpath(__file__))
# src_dir_path = os.path.abspath(os.path.join(os.path.join(current_dir_path, os.pardir), os.pardir))



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
    def __init__(self, detector_tod:DetectorTOD, detector_samples:DetectorSamples, nside:int, T_omega:function = np.ones_like, nthreads=1, double_prec = True):
        self.detector_tod = detector_tod
        self.detector_samples = detector_samples
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.double_perc = double_prec
        self.map = np.zeros((3, self.npix), dtype=np.float64 if double_prec else np.float32)
        self.nthreads = nthreads
        self.T_omega = T_omega
        self.maplib = load_cmdr4_ctypes_lib()
        ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_int64, ndim=1, flags="contiguous")
        ct_f64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=1, flags="contiguous")
        ct_f64_dim2 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=2, flags="contiguous")
        self.maplib.map_accumulator_IQU_f64.argtypes = [ct_f64_dim2, ct_f64_dim1, ct.c_double,
                                ct_i64_dim1, ct_f64_dim1, ct.c_int64,
                                ct.c_int64]
        self.maplib.map2tod_IQU_f64.argtypes = [ct_f64_dim1, ct_f64_dim2, 
                                            ct_i64_dim1, ct_f64_dim1, 
                                            ct.c_int64, ct.c_int64]

    def compute_RHS(self):
        """
        Compute the RHS of the mapmaking problem:
        P^T T^T N^-1 d
        """
        return NotImplementedError
    
    def apply_inv_N(self, tods:DetectorTOD):
        """
        Applies inplace the N^-1 operator.
        """
        for scan, scan_samples in zip(tods.scans, self.detector_samples.scans):
            inplace_scale(scan, 1/scan_samples.sigma0**2)
        return tods

    def apply_P(self, in_map: NDArray, out_tods:DetectorTOD):
        """
        Applies the pointing matrix operator.
        
        It takes an array of pixels in input and projects them over a stream of time ordered data.
        """
        npix_out = hp.nside2npix(out_tods._eval_nside)
        assert npix_out == in_map.shape[1] 
        for scan in out_tods.scans:
            pix = scan.pix
            psi = scan.psi
            ntod = scan.tod.shape[0]
            tod_f64 = np.ascontiguousarray(scan._tod, dtype=np.float64) #FIXME: check if this is actually updating tods in place
            psi_f64 = np.ascontiguousarray(psi, dtype=np.float64)
            self.maplib.map_accumulator_IQU_f64(in_map, tod_f64, 1, 
                                                pix.astype(np.int64), 
                                                psi_f64, ntod, npix_out)
        return out_tods

    def apply_P_adjoint(self, in_tods: DetectorTOD, out_map:NDArray):
        """
        Applies the adjoint, or transpose in matrix-notation, of the pointing matrix operator, updating out_map inplace.

        It takes in input a stream of time ordered data and accumulates them over a map in output.
        """
        npix_out = out_map.shape[1]
        assert npix_out == hp.nside2npix(in_tods._eval_nside)
        for scan in in_tods.scans:
            pix = scan.pix
            psi = scan.psi
            ntod = scan.tod.shape[0]
            tod_f64 = np.ascontiguousarray(scan._tod, dtype=np.float64)
            psi_f64 = np.ascontiguousarray(psi, dtype=np.float64)
            self.maplib.map2tod_IQU_f64(tod_f64, out_map.astype(np.float64), 
                                                pix.astype(np.int64), 
                                                psi_f64, ntod, npix_out)
        return out_map

    def _apply_T(self, tods: DetectorTOD, adjoint=False):
        """
        General function for T and T^T
        """
        for scan in tods.scans:
            #F d
            d = ducc0.fft.r2c(scan._tod, forward=True, inorm=1)
            #T F d
            freqs = np.fft.rfftfreq(len(scan._tod)) #if we want freqs in Hz fro T(omega) the use scan.fsamp.
            if adjoint:
                freqs = np.flip(freqs)
            d = d * self.T_omega(freqs)
            #F^-1 T F
            scan._tods = ducc0.fft.r2c(d, forward=False, inorm=1)
        return tods

    def apply_T(self, tods: DetectorTOD):
        """
        Applies INPLACE the T operator defined as T := F^-1 T(omega) F

        Where T(omega) is a filter which must respect the Hermitian symmetry T*(omega) = T(-omega) 
        in order to give real-valued outputs in time space, and F and F^-1 are RFFT and RFFT inverse.

        Numerically: F returns an array of frequencies `omega_s` and an array of complex amplitudes, 1 for each of those frequencies.
        T(omega) must be evaluated at each of those omega's and the result (a complex amplitude for each omega) is also a complex array
        that must be multiplicated element-wise with the output of F.
        The result is then given in input to F^-1, which is a irfft taking the complex array back to real tod domain. 

        Note: the "ortho" normalization of the FFT corresponds to option 1 in ducc0.
        """
        return self._apply_T(tods, adjoint=False)

    def apply_T_adjoint(self, tods: DetectorTOD):
        """
        Applies INPLACE the adjoint, or transpose in matrix-notation, of the T operator. 
        It is defined as T^T := (F^-1 T(omega) F)^T = F^T T*(omega) (F^-1)^T = F^-1 T(-omega) F

        Where T(omega) is a filter which must respect the Hermitian symmetry T*(omega) = T(-omega) 
        in order to give real-valued outputs in time space, and F and F^-1 := F^T are RFFT and
        RFFT inverse, which corresponds to the adjoint, which in this notation is improperly written as F^T.

        Numerically the procedure is identical to apply_T, with the difference that the complex array
        given by T(omega) is flipped, as we are evaluating T(-omega).
        """
        return self._apply_T(tods, adjoint=True)
    
    def calc_RHS(self, d_tods: DetectorTOD, out_map=None):
        """
        Computes the RHS of the mapmaking problem: P^T T^T N^-1 d
        """
        #N^-1 d
        self.apply_inv_N(d_tods)
        #T^T N^-1 d
        self.apply_T_adjoint(d_tods)
        #P^T T^T N^-1 d
        out_map = np.zeros((3,hp.nside2npix(d_tods.nside)), dtype=np.float64 if self.double_perc else np.float32) if out_map is None else out_map
        self.apply_P_adjoint(d_tods, out_map)
        return out_map

    def applies_RHS(self, m_map: NDArray, aux_tods=DetectorTOD):
        """
        Applies the LHS of the mapmaking problem P^T T^T N^-1 T P m to an input map, without modifying it,
        and updates the result in the out_tods object.
        """
        #P m
        self.apply_P(m_map, aux_tods)
        #T P m
        self.apply_T(aux_tods)
        #N^-1 T P m
        self.apply_inv_N(aux_tods)
        #T^T N^-1 T P m
        self.apply_T_adjoint(aux_tods)
        #P^T T^T N^-1 T P
        out_map = np.zeros_like(m_map)
        self.apply_P_adjoint(aux_tods, out_map)
        return out_map

    



