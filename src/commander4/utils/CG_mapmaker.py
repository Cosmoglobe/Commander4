import numpy as np
import ctypes as ct
from mpi4py import MPI
import logging
from numpy.typing import NDArray
import healpy as hp
from typing import Callable

from commander4.output.log import logassert
from commander4.utils.ctypes_lib import load_cmdr4_ctypes_lib
from commander4.data_models.detector_TOD import DetectorTOD
from commander4.data_models.scan_TOD import ScanTOD
from commander4.data_models.tod_view import TODView
from commander4.solvers.CG_driver import distributed_CG_arr
from commander4.data_models.detector_samples import DetectorSamples
from commander4.utils.pixel_domain import PixelDomain
from commander4.utils.math_operations import inplace_scale, dot, norm, forward_rfft, backward_rfft

# I need to CG-solve P^T T^T N^−1 T P m = P^T T^T N^-1 d
# Where:
# - m is the final map [npix]
# - P is the pointing matrix: [ntod, npix]
# - T is the bolometer trnasfer function operator (check Artem's code)
# - N^-1 is the inverse noise covariance matrix (inv_var in tod_processing.tod2map) which is diagonal in tod space [ntod]
# - d is the calibrated TODs [ntod].
# Notes: T is non-local so each rank must hold the whole scan, but only for one detector.


#FIXME: check threading

class CGMapmaker:
    """

    Super-class of a CG mapmaker solving the general P^T T^T N^-1 T P m = P^T T^T N^-1 d problem.

    To solve for a map, an instance of the inherited CGMapmakerI or CGMapmakerIQU must be used.
    """
    def __init__(self, 
                detector_tod:DetectorTOD, 
                detector_samples:DetectorSamples, 
                map_comm:MPI.Comm,
                #optionals:
                T_omega:Callable = np.ones_like, 
                preconditioner:Callable = np.copy,
                nthreads:int=1, 
                double_prec:bool = True,
                CG_maxiter:int=60,
                CG_tol:float=1e-6,
                CG_check_interval:int = 1,
                pixel_domain:PixelDomain|None = None):
        """Initialise the CG mapmaker.

        Args:
            detector_tod: Detector-group TOD data (``DetGroupTOD``).
            detector_samples: Sampled noise and gain parameters.
            map_comm: MPI communicator shared by ranks contributing to
                the same output map.
            T_omega: Bolometer transfer function T(omega). Must accept a
                real-frequency array and return a complex filter.
            preconditioner: Preconditioner callable ``M(x) -> x'``.
            nthreads: Number of threads for FFT and HEALPix operations.
            double_prec: If True, use float64 for internal maps.
            CG_maxiter: Maximum number of CG iterations.
            CG_tol: Convergence tolerance on the CG residual.
            CG_check_interval: Check convergence every this many iterations.
            pixel_domain: Pixel-distribution domain. When ``None`` a full-sky domain is built, in
                which case every rank holds full-sky local maps (the historical behaviour). In
                sparse mode each rank's RHS/LHS buffers cover only its observed pixels, and the
                full-sky iterate held by the master is scattered/gathered to the ranks each
                iteration.
        """

        self.logger = logging.getLogger(__name__)
        self.detector_tod = detector_tod
        self.detector_samples = detector_samples
        self.double_perc = double_prec
        self.map_comm = map_comm
        self.ismaster = self.map_comm.Get_rank() == 0
        self.f_dtype = np.float64 if double_prec else np.float32
        self.nthreads = nthreads
        self.T_omega = T_omega
        # Native sampling rate [Hz] of the mapmaking TODs, so the transfer function T_omega(omega) is
        # evaluated on a physical-frequency grid (a `tau` in seconds means seconds, not samples). The
        # CG's own noise model is white, so unlike apply_N_inv this rate is only needed for apply_T.
        self.fsamp = detector_tod.fsamp
        self.CG_maxiter = CG_maxiter
        self.CG_tol = CG_tol
        self.CG_check_interval = CG_check_interval
        self.M = preconditioner
        self.domain = pixel_domain if pixel_domain is not None \
            else PixelDomain(map_comm, detector_tod.nside, "full")
        # The sparse gather/scatter collectives operate in float64; the float32 map path is only
        # supported full-sky (and is unused in production, which always runs double_prec).
        if self.domain.mode == "sparse" and not double_prec:
            raise NotImplementedError("Sparse CG maps require double_prec=True.")
        self._nloc = self.domain.n_local
        # View over the band's detector-scans, used to access pointing (pix/psi) when applying the
        self._scan_view = TODView(detector_tod, detector_samples)
        self._rhs_loca_map = None
        self._rhs_finalized_map = None
        self.res_s = []

        self.maplib = load_cmdr4_ctypes_lib()
        self.ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_int64, ndim=1, flags="contiguous")
        self.ct_f64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=1, flags="contiguous")
        self.ct_f64_dim2 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=2, flags="contiguous")
        self.ct_f32_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_float, ndim=1, flags="contiguous")
        self.ct_f32_dim2 = np.ctypeslib.ndpointer(dtype=ct.c_float, ndim=2, flags="contiguous")
       
    @property
    def solved_map(self):
        """The solved sky map. Only valid on the master rank after ``solve()``."""
        if self.map_comm.Get_rank() == 0:
            logassert(self._map_signal is not None, "Attempted to read solution map on master rank before it was solved.",
                      self.logger)
        return self._map_signal
    
    @property
    def RHS_map(self):
        """The finalised RHS map. Only valid on master rank after ``finalize_RHS()``."""
        if self.map_comm.Get_rank() == 0:
            logassert(self._rhs_finalized_map is not None, "Attempted to read RHS map on master rank before it was finalized.",
                      self.logger)
            return self._rhs_finalized_map
        else:
            return np.empty(())

    def apply_P(self, in_map: NDArray, out_scan:ScanTOD, pix=None, psi=None, scan_tod_arr=None):
        raise NotImplementedError("Subclasses must implement apply_P()")

    def apply_P_adjoint(self, in_map: NDArray, out_scan:ScanTOD, pix=None, psi=None, scan_tod_arr=None):
        raise NotImplementedError("Subclasses must implement apply_P()")

    def apply_inv_N(self, scan_tod_arr:NDArray, sigma0:float):
        """
        Applies inplace the N^-1 operator to one scan, given the corresponding noise variance sigma0.

        Args:
        - `scan_tod_arr`: array of TODs corresponding to the scan.
        - `sigma0`: noise variance for that scan.
        """
        inplace_scale(scan_tod_arr, 1.0/sigma0**2)
        return scan_tod_arr

    def _apply_T(self, scan_tod_arr, adjoint=False):
        """Apply the transfer-function operator T (or its transpose T^T) to one scan; returns a new
        length-N array.

        The forward operator is ``T = R F^-1 diag(H) F E`` where ``E`` reflect-extends the scan to
        length ``2N`` (``x -> [x, x[::-1]]``), ``H = T_omega`` is the filter evaluated on the ``2N``
        frequency grid, and ``R`` restricts back to the first ``N`` samples. The grid is in physical
        Hz (``rfftfreq(2N, d=1/fsamp)``), so ``T_omega`` sees true frequencies -- a time constant is
        in seconds, not samples. Mirroring makes the scan boundary continuous so a causal ``H`` does
        not wrap the scan's end onto its start (matching ``apply_N_inv`` and the simulator that bakes
        ``H`` in).

        The transpose is ``T^T = E^T F^-1 diag(H*) F R^T``: ``R^T`` zero-pads (``x -> [x, 0]``), the
        filter is **conjugated** (``H*`` -- a frequency flip is *not* the transpose for a non-trivial
        ``H``), and ``E^T`` folds the mirror back (``v -> v[:N] + v[N:][::-1]``). Implementing the two
        directions as this exact adjoint pair keeps the mapmaking operator ``P^T T^T N^-1 T P``
        symmetric, so the CG solve stays well-posed. At ``T_omega = 1`` both reduce to the identity.
        """
        n = scan_tod_arr.shape[-1]
        freqs = np.fft.rfftfreq(2 * n, d=1.0 / self.fsamp)  # physical frequency grid [Hz]
        if adjoint:
            ext = np.concatenate([scan_tod_arr, np.zeros_like(scan_tod_arr)])  # R^T: zero-pad
            filt = np.conj(self.T_omega(freqs))                               # H*
        else:
            ext = np.concatenate([scan_tod_arr, scan_tod_arr[::-1]])          # E: reflect-extend
            filt = self.T_omega(freqs)                                        # H
        out = backward_rfft(forward_rfft(ext, nthreads=self.nthreads) * filt, 2 * n,
                            nthreads=self.nthreads)
        if adjoint:
            return out[:n] + out[n:][::-1]                                    # E^T: fold the mirror back
        return np.ascontiguousarray(out[:n])                                 # R: keep the first N samples

    def apply_T(self, scan_tod_arr):
        """Apply the transfer-function operator ``T = R F^-1 diag(T_omega) F E`` to one scan.

        ``T_omega`` is the (Hermitian-symmetric) filter ``H(omega)``; the mirrored FFT (reflect-extend
        to ``2N``, filter, keep the first ``N`` samples) suppresses boundary wrap-around. Returns a new
        array of the same length; see ``_apply_T`` for the full definition.
        """
        return self._apply_T(scan_tod_arr, adjoint=False)

    def apply_T_adjoint(self, scan_tod_arr):
        """Apply the transpose ``T^T`` of the transfer-function operator to one scan.

        This is the exact numerical transpose of ``apply_T``: zero-pad, filter with the **conjugated**
        symbol ``T_omega*``, and fold the mirror back (``v -> v[:N] + v[N:][::-1]``). Conjugating the
        filter -- not flipping the frequency array -- is what makes it the true adjoint for a
        non-trivial ``T_omega`` (and hence keeps ``P^T T^T N^-1 T P`` symmetric). Returns a new array
        of the same length; see ``_apply_T``.
        """
        return self._apply_T(scan_tod_arr, adjoint=True)
    
    def accum_to_RHS(self, scan_tod: DetectorTOD, sigma0: float,
                     pix=None, psi=None, scan_tod_arr=None):
        """
        Computes the contribution to RHS of the mapmaking problem: P^T T^T N^-1 d for one scan. 
        Both scan TOD and the white noise level sigma0 must be given. This allows to compute the RHS
        contributions in an external loop together with the correlated noise sampling, pix can be
        passed already uncompressed from an external loop to avoid double uncompression.
        """
        if self._rhs_loca_map is None:
            #if not done already, allocate memory for local maps
            self._rhs_loca_map = self._zeros_map

        if scan_tod_arr is None:
            scan_tod_arr = np.copy(scan_tod.tod) #aux array to not modify scan.tod
        # Guard against pathological scans that slip past read-in: an empty scan crashes the FFT,
        # and a single non-finite sample is spread across the whole scan by apply_T (and then across
        # every pixel that scan hits), making the CG residual NaN. Readers should discard these, but
        # not all of them do, so fail loudly here identifying the offending detector-scan.
        logassert(scan_tod_arr.shape[-1] > 0,
                  f"Empty TOD passed to CG RHS for detector {getattr(scan_tod, 'name', '?')}.",
                  self.logger)
        logassert(np.isfinite(scan_tod_arr).all(),
                  f"Non-finite samples in CG RHS for detector {getattr(scan_tod, 'name', '?')} "
                  "(check gain, sigma0, and that flagged/non-finite samples are gap-filled).",
                  self.logger)
        #N^-1 d
        # if self.ismaster:
        #     self.logger.info(f"RHS_1: {scan_tod_arr.shape}")
        scan_tod_arr = self.apply_inv_N(scan_tod_arr, sigma0)
        #T^T N^-1 d
        # if self.ismaster:
        #     self.logger.info(f"RHS_2: {scan_tod_arr.shape}")
        scan_tod_arr = self.apply_T_adjoint(scan_tod_arr)
        # if self.ismaster:
        #     self.logger.info(f"RHS_3: {scan_tod_arr.shape}")
        #P^T T^T N^-1 d
        self._rhs_loca_map = self.apply_P_adjoint(scan_tod, self._rhs_loca_map, 
                                                  pix=pix, psi=psi, scan_tod_arr=scan_tod_arr)

    def finalize_RHS(self, root=0):
        """
        Reduces the local RHS contributions onto the full-sky RHS map held by the master rank.
        """
        # Check for None, which indicates a rank without any scans. Give it a zero-map.
        if self._rhs_loca_map is None:
            self._rhs_loca_map = self._zeros_map
        full = self.domain.reduce_to_full(self._rhs_loca_map, root=root)
        if self.map_comm.Get_rank() == root:
            self._rhs_finalized_map = full
        self.map_comm.Barrier()
        self._rhs_loca_map = None  # free memory
        return self._rhs_finalized_map if self.map_comm.Get_rank() == root else np.empty(())

    def apply_LHS(self, in_map: NDArray):
        """
        Applies the LHS of the mapmaking problem P^T T^T N^-1 T P m to an input map.

        The master holds the full-sky iterate ``in_map``; each rank receives only the values at its
        locally-observed pixels (a broadcast of the full map in full mode), applies its block of the
        operator into a local buffer, and the contributions are summed back into a full-sky map on
        the master.
        """
        ismaster = self.map_comm.Get_rank() == 0
        # Distribute the iterate to the ranks' local pixel domains (master -> ranks).
        local_in = self.domain.scatter_from_full(in_map if ismaster else None, self._ncomp,
                                                  dtype=self.f_dtype)
        out_local = self._zeros_map
        # The LHS operator P^T T^T N^-1 T P and the RHS P^T T^T N^-1 d must span the same set of
        # detector-scans AND the same samples, or the CG solves an inconsistent (A, b). We iterate
        # the same accept-gated TODView path the RHS loop uses, on the *full-length* pointing: the
        # RHS gap-fills flagged samples rather than removing them (apply_T needs a continuous TOD),
        # so both sides run over every sample of each accepted detector-scan.
        for view in self._scan_view.iter_focused(accepted_only=True):
            pix = view.pix
            psi = view.psi
            sigma0 = view.sigma0
            scan_tod_arr_aux = np.zeros(pix.shape[0], dtype=self.f_dtype)  # full-length, as RHS
            #P m
            scan_tod_arr_aux = self.apply_P(local_in, view.detector, pix=pix, psi=psi, scan_tod_arr=scan_tod_arr_aux)
            #T P m
            scan_tod_arr_aux = self.apply_T(scan_tod_arr_aux)
            #N^-1 T P m
            scan_tod_arr_aux = self.apply_inv_N(scan_tod_arr_aux, sigma0)
            #T^T N^-1 T P m
            scan_tod_arr_aux = self.apply_T_adjoint(scan_tod_arr_aux)
            #P^T T^T N^-1 T P
            out_local = self.apply_P_adjoint(view.detector, out_local, pix=pix, psi=psi, scan_tod_arr=scan_tod_arr_aux)
        # Sum the local contributions back to the full-sky map on the master (None on other ranks).
        return self.domain.reduce_to_full(out_local)

    def solve(self, x_true=None):
        """
        Solves the CG to compute the target sky map.
        """
        RHS_map = self.RHS_map
        ismaster = self.map_comm.Get_rank() == 0

        CG_solver = distributed_CG_arr(self.apply_LHS,
                                       RHS_map,
                                       ismaster,
                                       M = self.M,
                                       dot = dot,
                                       destroy_b=True)
        if ismaster:
            self.logger.info("Mapmaker CG starting up!")
        for i in range(self.CG_maxiter):
            CG_solver.step()
            if i%self.CG_check_interval == 0:
                if ismaster:
                    self.logger.info(f"Mapmaker CG iter {i:3d} - Residual {CG_solver.err:.6e}")
                    self.res_s.append(CG_solver.err)
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
            
            converged = self.map_comm.bcast(CG_solver.err < self.CG_tol if ismaster else None, root=0)
            if converged:
                break

        self._map_signal = CG_solver.x

    
class CGMapmakerI(CGMapmaker):
    """Intensity-only (temperature) CG mapmaker.

    Inherits from ``CGMapmaker`` and implements the pointing matrix operators
    ``apply_P`` and ``apply_P_adjoint`` for a scalar (I-only) map.
    """

    def __init__(self, 
                 detector_tod, 
                 detector_samples,
                 map_comm, T_omega = np.ones_like, 
                 preconditioner = np.copy, 
                 nthreads = 1,
                 double_prec = True,
                 CG_maxiter = 200,
                 CG_tol = 1e-10,
                 CG_check_interval = 1,
                 pixel_domain = None):

        super().__init__(detector_tod, detector_samples, map_comm, T_omega, preconditioner,
                         nthreads, double_prec, CG_maxiter, CG_tol, CG_check_interval, pixel_domain)

        self._ncomp = 1
        # Master holds the full-sky solution and RHS; the iterate is scattered to the ranks' local
        # domains each iteration (see apply_LHS).
        self._map_signal = np.zeros((1,hp.nside2npix(detector_tod.nside)),
            dtype=self.f_dtype) if self.ismaster else None
        #RHS map to be accumulated on master rank
        self._rhs_finalized_map = np.zeros((1,hp.nside2npix(detector_tod.nside)),
            dtype=self.f_dtype) if self.ismaster else None
        
        #RHS map to be accumulate
        if double_prec:
            self.maplib.map_accumulator_f64.argtypes = [self.ct_f64_dim2, #map
                                                        self.ct_f64_dim1, #tod
                                                        ct.c_double,      #weight
                                                        self.ct_i64_dim1, #pix
                                                        ct.c_int64]       #scan_len
            self.maplib.map2tod_f64.argtypes = [self.ct_f64_dim2, #map
                                                self.ct_f64_dim1, #tod
                                                self.ct_i64_dim1, #pix
                                                ct.c_int64]       #scan_len
            self.map_accumulator = self.maplib.map_accumulator_f64
            self.map2tod = self.maplib.map2tod_f64
        else:
            self.maplib.map_accumulator_f32.argtypes = [self.ct_f32_dim2, 
                                                        self.ct_f32_dim1, 
                                                        ct.c_double, 
                                                        self.ct_i64_dim1, 
                                                        ct.c_int64]
            self.maplib.map2tod_f32.argtypes = [self.ct_f32_dim2, 
                                                self.ct_f32_dim1,
                                                self.ct_i64_dim1, 
                                                ct.c_int64]
            self.map_accumulator = self.maplib.map_accumulator_f32
            self.map2tod = self.maplib.map2tod_f32

    def apply_P(self, in_map: NDArray, out_scan:ScanTOD, pix=None, psi=None, scan_tod_arr=None):
        """
        Applies the pointing matrix operator to one scan.
        
        It takes in input a time ordered data scan and accumulates them over a map in output. if a 
        `pix` is passed, it will be used to compute the result instead of decompressing a new one 
        from `out_scan`. If a `scan_tod_arr` is passed it is used instead of overwriting `out_scan`.
        In the CGMapmakerI the psi will be ignored.
        """
        scan_tod_arr = out_scan.tod if scan_tod_arr is None else scan_tod_arr
        # in_map is indexed by pix, so its pixel axis defines the domain (full-sky or rank-local).
        pix = self.domain.to_local(out_scan.pix if pix is None else pix)
        assert pix.shape == scan_tod_arr.shape, "pix shape must match scan_tod_arr."
        # Use the passed array length, not the full detector ntod: apply_LHS masks pix/scan_tod_arr
        # down to good samples, so this must match (mirrors apply_P_adjoint).
        ntod = scan_tod_arr.shape[-1]
        self.map2tod(in_map, scan_tod_arr, pix.astype(np.int64, copy=False), ntod)
        return scan_tod_arr

    def apply_P_adjoint(self, in_scan: ScanTOD, out_map:NDArray, pix=None, psi=None, scan_tod_arr=None):
        """
        Applies the adjoint, or transpose in matrix-notation, of the pointing matrix operator to one
        scan, updating out_map inplace.

        It takes in input a time ordered data scan and accumulates them over a map in output. if a 
        `pix` is passed, it will be used to compute the result instead of decompressing a new one 
        from `in_scan`. If a `scan_tod_arr` is passed it is used instead of overwriting `in_scan`.
        In the CGMapmakerI the psi will be ignored.
        """
        scan_tod_arr = in_scan.tod if scan_tod_arr is None else scan_tod_arr
        # out_map is indexed by pix, so its pixel axis defines the domain (full-sky or rank-local).
        pix = self.domain.to_local(in_scan.pix if pix is None else pix)
        assert pix.shape == scan_tod_arr.shape, "pix shape must match scan_tod_arr."
        ntod = scan_tod_arr.shape[-1]
        self.map_accumulator(out_map, scan_tod_arr, 1, pix.astype(np.int64, copy=False), ntod)
        return out_map

    @property
    def _zeros_map(self):
        """Allocate a zero local map buffer (full-sky in full mode, rank-local in sparse mode)."""
        return np.zeros((1, self._nloc), dtype=self.f_dtype)

class CGMapmakerIQU(CGMapmaker):
    """Polarised (I, Q, U) CG mapmaker.

    Inherits from ``CGMapmaker`` and implements the pointing matrix operators
    ``apply_P`` and ``apply_P_adjoint`` for a full I/Q/U map.
    """

    def __init__(self, 
                 detector_tod, 
                 detector_samples, 
                 map_comm, 
                 T_omega = np.ones_like, 
                 preconditioner = np.copy, 
                 nthreads = 1,
                 double_prec = True,
                 CG_maxiter = 200,
                 CG_tol = 1e-10,
                 CG_check_interval = 1,
                 pixel_domain = None):

        super().__init__(detector_tod, detector_samples, map_comm, T_omega, preconditioner,
                         nthreads, double_prec, CG_maxiter, CG_tol, CG_check_interval, pixel_domain)

        self._ncomp = 3
        # Master holds the full-sky solution and RHS; the iterate is scattered to the ranks' local
        # domains each iteration (see apply_LHS).
        self._map_signal = np.zeros((3,hp.nside2npix(detector_tod.nside)),
            dtype=self.f_dtype) if self.ismaster else None
        #local RHS map
        self._rhs_loca_map = None
        #RHS map to be accumulated on master rank
        self._rhs_finalized_map = np.zeros((3,hp.nside2npix(detector_tod.nside)),
            dtype=self.f_dtype) if self.ismaster else None
        
        if double_prec:
            self.maplib.map_accumulator_IQU_f64.argtypes = [self.ct_f64_dim2, #map
                                                            self.ct_f64_dim1, #tod
                                                            ct.c_double,      #weight
                                                            self.ct_i64_dim1, #pix
                                                            self.ct_f64_dim1, #psi
                                                            ct.c_int64,       #scan_len
                                                            ct.c_int64]       #num_pix
            self.maplib.map2tod_IQU_f64.argtypes = [self.ct_f64_dim2, #map
                                                    self.ct_f64_dim1, #tod
                                                    self.ct_i64_dim1, #pix
                                                    self.ct_f64_dim1, #psi
                                                    ct.c_int64,       #scan_len
                                                    ct.c_int64]       #num_pix
            self.map_accumulator_IQU = self.maplib.map_accumulator_IQU_f64
            self.map2tod_IQU = self.maplib.map2tod_IQU_f64
        else:
            self.maplib.map_accumulator_IQU_f32.argtypes = [self.ct_f32_dim2, 
                                                            self.ct_f32_dim1, 
                                                            ct.c_double,
                                                            self.ct_i64_dim1, 
                                                            self.ct_f64_dim1, 
                                                            ct.c_int64, 
                                                            ct.c_int64]
            self.maplib.map2tod_IQU_f32.argtypes = [self.ct_f32_dim2, 
                                                    self.ct_f32_dim1,
                                                    self.ct_i64_dim1, 
                                                    self.ct_f64_dim1, 
                                                    ct.c_int64, 
                                                    ct.c_int64]
            self.map_accumulator_IQU = self.maplib.map_accumulator_IQU_f32
            self.map2tod_IQU = self.maplib.map2tod_IQU_f32

    def apply_P(self, in_map: NDArray, out_scan:ScanTOD, pix=None, psi=None, scan_tod_arr=None):
        """
        Applies the pointing matrix operator to one scan.
        
        It takes in input a time ordered data scan and accumulates them over a map in output. if a 
        `pix` or `psi` is passed, it will be used to compute the result instead of decompressing 
        a new one from `out_scan`. 
        If a `scan_tod_arr` is passed it is used instead of overwriting `out_scan`
        """
        scan_tod_arr = out_scan.tod if scan_tod_arr is None else scan_tod_arr
        # in_map is indexed by pix and strided by its pixel axis, which defines the domain
        # (full-sky or rank-local); num_pix is that axis length.
        npix_out = in_map.shape[-1]
        pix = self.domain.to_local(out_scan.pix if pix is None else pix)
        psi = out_scan.psi if psi is None else psi
        # Use the passed array length, not the full detector ntod: apply_LHS masks pix/psi/scan_tod_arr
        # down to good samples, so this must match (mirrors apply_P_adjoint).
        ntod = scan_tod_arr.shape[-1]
        self.map2tod_IQU(in_map, scan_tod_arr, pix.astype(np.int64, copy=False),
                         psi.astype(np.float64, copy=False), ntod, npix_out)
        return scan_tod_arr
    
    def apply_P_adjoint(self, in_scan: ScanTOD, out_map:NDArray, pix=None, psi=None, scan_tod_arr=None):
        """
        Applies the adjoint, or transpose in matrix-notation, of the pointing matrix operator to one
        scan, updating out_map inplace.

        It takes in input a time ordered data scan and accumulates them over a map in output. if a 
        `pix` or `psi` is passed, it will be used to compute the result instead of decompressing 
        a new one from `out_scan`. 
        If a `scan_tod_arr` is passed it is used instead of overwriting `out_scan`
        """
        scan_tod_arr = in_scan.tod if scan_tod_arr is None else scan_tod_arr
        # out_map is indexed by pix and strided by its pixel axis, which defines the domain
        # (full-sky or rank-local); num_pix is that axis length.
        npix_out = out_map.shape[-1]
        pix = self.domain.to_local(in_scan.pix if pix is None else pix)
        psi = in_scan.psi if psi is None else psi
        ntod = scan_tod_arr.shape[-1]
        self.map_accumulator_IQU(out_map, scan_tod_arr, 1, pix.astype(np.int64, copy=False),
                                 psi.astype(np.float64, copy=False), ntod, npix_out)
        return out_map

    @property
    def _zeros_map(self):
        """Allocate a zero local map buffer (full-sky in full mode, rank-local in sparse mode)."""
        return np.zeros((3, self._nloc), dtype=self.f_dtype)
