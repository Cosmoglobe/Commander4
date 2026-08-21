"""The CG mapmaker: solves P^T T^T N^-1 T P m = P^T T^T N^-1 d for one band's sky map, where

- m is the final map [npix],
- P is the pointing matrix [ntod, npix],
- T is the bolometer transfer-function operator,
- N^-1 is the inverse noise covariance, diagonal in TOD space [ntod],
- d is the calibrated TOD [ntod].

Unlike the binned mapmaker (`mapmaking/binned.py`), this one can deconvolve T, since the operator
is applied iteratively rather than inverted per pixel. T is non-local along the scan, so each rank
must hold a whole scan, though only for one detector at a time.

`tod2map_CG` drives the whole per-band iteration: one fused scan loop samples correlated noise and
sigma0 and accumulates every sigma0-dependent quantity, and the CG solve follows.
"""
import numpy as np
import ctypes as ct
from mpi4py import MPI
import logging
from numpy.typing import NDArray
import healpy as hp
from pixell import utils
from typing import Callable

from commander4.diagnostics.log import logassert
from commander4.backend.ctypes_lib import load_cmdr4_ctypes_lib
from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.scan_tod import ScanTOD
from commander4.tod.view import TODView
from commander4.compsep.cg_driver import DistributedCGArray
from commander4.compsep.preconditioners import InvNPreconditionerI, InvNPreconditionerIQU
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.detector_map import DetectorMap
from commander4.data_models.tod_samples import TODSamples
from commander4.tod.mapmaking.binned import Mapmaker, MapmakerIQU, WeightsMapmaker,\
    WeightsMapmakerIQU
from commander4.tod.noise.gap_filling import fill_all_masked
from commander4.tod.noise.sample_ncorr import sample_correlated_noise, log_corr_noise_stats,\
    CorrelatedNoiseConfig
from commander4.tod.noise.sigma0 import _estimate_standalone_sigma0
from commander4.tod.scan_diagnostics import _record_tod_diagnostics
from commander4.data_models.pixel_domain import PixelDomain
from commander4.math_utils.arithmetic import inplace_scale, dot, norm
from commander4.math_utils.fft import forward_rfft, backward_rfft
from commander4.tod.mapmaking.config import MapmakingConfig
from commander4.tod.data_selection import DataSelectionConfig

logger = logging.getLogger(__name__)


class CGMapmaker:
    """Super-class of a CG mapmaker solving the general P^T T^T N^-1 T P m = P^T T^T N^-1 d problem.

    To solve for a map, an instance of the inherited CGMapmakerI or CGMapmakerIQU must be used.
    """
    def __init__(self,
                detector_tod:DetectorGroupTOD,
                detector_samples:TODSamples,
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
            detector_tod: Detector-group TOD data for this band.
            detector_samples: Sampled noise and gain parameters for the current chain state.
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
        self.double_prec = double_prec
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
        # pointing matrix and its adjoint.
        self._scan_view = TODView(detector_tod, detector_samples)
        self._rhs_loca_map = None
        self._rhs_finalized_map = None

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
        raise NotImplementedError("Subclasses must implement apply_P_adjoint()")

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
        Hz (``rfftfreq(2N, d=1/fsamp)``), so ``T_omega`` sees true frequencies and a time constant
        is in seconds, not samples. Mirroring makes the scan boundary continuous so a causal ``H`` does
        not wrap the scan's end onto its start (matching ``apply_N_inv`` and the simulator that bakes
        ``H`` in).

        The transpose is ``T^T = E^T F^-1 diag(H*) F R^T``: ``R^T`` zero-pads (``x -> [x, 0]``), the
        filter is **conjugated** (``H*``; a frequency flip is *not* the transpose for a non-trivial
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
        filter (rather than flipping the frequency array) is what makes it the true adjoint for a
        non-trivial ``T_omega``, and hence keeps ``P^T T^T N^-1 T P`` symmetric. Returns a new array
        of the same length; see ``_apply_T``.
        """
        return self._apply_T(scan_tod_arr, adjoint=True)

    def accum_to_RHS(self, scan_tod: DetectorTOD, sigma0: float,
                     pix=None, psi=None, scan_tod_arr=None):
        """ Computes the contribution to the RHS of the mapmaking problem, P^T T^T N^-1 d, for one
            scan.
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
        # N^-1 d
        scan_tod_arr = self.apply_inv_N(scan_tod_arr, sigma0)
        # T^T N^-1 d
        scan_tod_arr = self.apply_T_adjoint(scan_tod_arr)
        # P^T T^T N^-1 d
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

        CG_solver = DistributedCGArray(self.apply_LHS,
                                       RHS_map,
                                       ismaster,
                                       M = self.M,
                                       dot = dot,
                                       destroy_b=True)

        if ismaster:
            self.logger.info("Mapmaker CG starting up!")
        for i in range(self.CG_maxiter):
            CG_solver.step()
            if i % self.CG_check_interval == 0 and ismaster:
                self.logger.info(f"Mapmaker CG iter {i:3d} - Residual {CG_solver.err:.6e}")
                if x_true is not None:  # Optional error against a known solution (testing only).
                    CG_L2_error = norm(CG_solver.x - x_true)/norm(x_true)
                    CG_Anorm_error = dot(CG_solver.x - x_true, self.apply_LHS(CG_solver.x - x_true))
                    self.logger.info(f"CG iter {i:3d} - True A-norm error: {CG_Anorm_error:.3e} "
                                     f"- True L2 error: {CG_L2_error:.3e}")
            # Only the master updates CG_solver.err, so the stopping decision has to be broadcast.
            if self.map_comm.bcast(CG_solver.err < self.CG_tol, root=0):
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


def called_on_non_master(arr):
    logger.debug("Dummy precond has been called")
    return np.copy(arr)

def tod2map_CG(band_comm: MPI.Comm, experiment_data: DetectorGroupTOD, compsep_output: NDArray,
               tod_samples: TODSamples, iteration: int,
               mapmaking: MapmakingConfig, correlated_noise: CorrelatedNoiseConfig,
               data_selection: DataSelectionConfig,
               ) -> tuple[dict[str, DetectorMap], dict[str, NDArray]]:
    """ Commander4 CG mapmaking. All ranks on the provided MPI communicator collaborates on creating
        the band maps (sky signal, inverse variance, possibly also aux maps like orbital dipole).
    Args:
        band_comm (Comm): The communicator consisting of all MPI ranks which holds TOD data that
                          should go into the same map.
        experiment_data (DetectorGroupTOD): TOD data class to be made into maps.
        compsep_output (NDArray): The sky model at our band. Not used, but written to chain file.
        tod_samples (TODSamples): Sampled TOD parameters, such as gain.
        iteration: Current Gibbs iteration.
        mapmaking: Validated mapmaking settings.
        correlated_noise: Validated correlated-noise settings.
        data_selection: Validated detector-scan selection settings.
    Output:
        Detector maps for component separation and maps selected for chain output.

    """
    ismaster = band_comm.Get_rank() == 0
    corr_noise_active = correlated_noise.is_active(iteration)
    selection_active = data_selection.cuts_are_active(iteration, correlated_noise)
    ### CG MAPMAKER ###
    # Single fused scan loop (mirrors Commander3's process_TOD): each detector-scan samples
    # correlated noise / sigma0 *first*, then accumulates every sigma0-dependent quantity with that
    # freshly-sampled sigma0. Those are the inverse-variance weights (preconditioner + rms/cov), the
    # orbital-dipole and corr-noise maps, and the CG RHS. Building the inverse-variance map in a
    # separate up-front pass instead leaves it on the previous iteration's sigma0, which makes the
    # CG RHS inconsistent with its own A (the LHS operator and preconditioner read the live sigma0).
    pols = experiment_data.pols
    scan_view = TODView(experiment_data, tod_samples, compsep_output=compsep_output)
    # Optional per-experiment sparse map storage: each rank holds only its locally-observed pixels
    # rather than a full sky map. The band master still ends up with full-sky maps.
    domain = experiment_data.get_pixel_domain(scan_view, band_comm, mapmaking.sparse_maps)
    # The inverse-variance map (preconditioner + rms/cov) is accumulated inside the fused loop below,
    # so neither it nor cg_mapmaker.M can be finalized until afterwards. cg_mapmaker is constructed
    # here with a placeholder preconditioner; M is unused until solve() and accum_to_RHS never reads
    # it, so it is reassigned to the real Jacobi preconditioner after the loop.
    if pols == "IQU":
        mapmaker_invvar = WeightsMapmakerIQU(band_comm, experiment_data.nside, pixel_domain=domain)
        cg_mapmaker = CGMapmakerIQU(experiment_data, tod_samples, band_comm,
                    preconditioner=called_on_non_master, nthreads=mapmaking.num_threads,
                    CG_maxiter=mapmaking.cg.max_iter, CG_tol=mapmaking.cg.err_tol,
                    pixel_domain=domain)
    elif pols == "I":
        mapmaker_invvar = WeightsMapmaker(band_comm, experiment_data.nside, pixel_domain=domain)
        cg_mapmaker = CGMapmakerI(experiment_data, tod_samples, band_comm,
                    preconditioner=called_on_non_master, nthreads=mapmaking.num_threads,
                    CG_maxiter=mapmaking.cg.max_iter, CG_tol=mapmaking.cg.err_tol,
                    pixel_domain=domain)
    else:
        raise ValueError(f"specified polarizations {pols} is notsupported yet.")

    BinMapmaker = MapmakerIQU if pols == "IQU" else Mapmaker  # General bin mapmaker class.
    mapmaker_orbdipole = BinMapmaker(band_comm, experiment_data.nside, pixel_domain=domain)

    if corr_noise_active:
        mapmaker_ncorr = BinMapmaker(band_comm, experiment_data.nside, pixel_domain=domain)
        sampled_params = []
        residuals = []
        niters = []
        num_failed_convergences_ncorr = 0
        num_too_high_var_ncorr = 0
        worst_residual_ncorr = 0

    ### MAIN SCAN LOOP ###
    for view in scan_view.iter_focused(accepted_only=True):
        # Full-length pointing (no good_data_mask compaction): the CG operator gap-fills flagged
        # samples rather than removing them, so every sample carries weight (gain/sigma0)^2 and the
        # inverse-variance / preconditioner must count them all to match the A operator.
        pix, psi = view.pix, view.psi
        good_data_mask = view.get_mask(proc_mask=False)
        gain = view.get_gain()
        response = view.det_response if pols == "IQU" else None

        ### DATA-SELECTION VETO 1 (too little unflagged data).
        good_frac = good_data_mask.mean()
        tod_samples.good_fraction[view.iscan, view.idet] = good_frac
        if selection_active and good_frac < data_selection.min_good_fraction:
            tod_samples.accept[view.iscan, view.idet] = False
            continue

        ### CORRELATED NOISE / SIGMA0 SAMPLING (first, so the weights below use the new sigma0) ###
        n_corr_est = None
        if corr_noise_active:
            sky_subtracted_TOD = view.get_tod(
                subtract=(("sky", TODView._ALL_GAIN_TERMS),
                          ("orbital_dipole", TODView._ALL_GAIN_TERMS)),
            )
            res = sample_correlated_noise(
                sky_subtracted_TOD, view.get_mask(proc_mask_type="ncorr"),
                np.array(view.noise_params, copy=True),
                experiment_data.noise_model, view.fsamp, cg_err_tol=correlated_noise.cg.err_tol,
                cg_max_iter=correlated_noise.cg.max_iter,
                sample_params=correlated_noise.sample_psd_params,
                sample_sigma0=correlated_noise.sample_sigma0,
                sigma0_method=correlated_noise.sigma0_method,
                nomono=correlated_noise.nomono,
                onlymono=correlated_noise.onlymono,
                sigma0_dec=correlated_noise.sigma0_decimation,
                psd_fit_nu_min=correlated_noise.psd_fit_nu_min,
                psd_fit_nu_max=correlated_noise.psd_fit_nu_max,
                psd_bin=correlated_noise.psd_bin)
            n_corr_est = res.n_corr
            tod_samples.noise_params[view.iscan, view.idet, :] = res.noise_params
            if correlated_noise.sample_psd_params:
                sampled_params.append(np.array(res.noise_params, copy=True))
            if not res.converged:
                num_failed_convergences_ncorr += 1
            if res.high_var:
                num_too_high_var_ncorr += 1
            worst_residual_ncorr = max(worst_residual_ncorr, res.residual)
            residuals.append(res.residual)
            niters.append(res.niter)
        elif correlated_noise.sample_sigma0:
            # No correlated noise this iteration: estimate sigma0 here, at the same point in the
            # chain (after gain) as the n_corr-coupled estimate, instead of a separate pre-gain pass.
            tod_samples.noise_params[view.iscan, view.idet, 0] = _estimate_standalone_sigma0(
                view, correlated_noise.sigma0_method)

        # Diagnostics (incl. chisq_z) before any accumulation, so a veto below leaves this
        # detector-scan out of every map product (the CG operator passes re-read `accept`).
        _record_tod_diagnostics(tod_samples, view.iscan, view.idet, view, n_corr_est)

        ### DATA-SELECTION VETO 2 (catastrophic chi^2), applied in-loop so this iteration's maps
        ### already exclude the scan.
        if selection_active:
            z = tod_samples.chisq_z[view.iscan, view.idet]
            if not (np.isfinite(z) and abs(z) <= data_selection.chisq_abs_threshold):
                tod_samples.accept[view.iscan, view.idet] = False
                continue

        # sigma0 now reflects this iteration's estimate; every weight below is consistent with it.
        sigma0 = view.sigma0
        inv_var = (gain/sigma0)**2

        ### INVERSE-VARIANCE WEIGHTS (preconditioner + rms/cov) ###
        if pols == "IQU":
            mapmaker_invvar.accumulate_to_map(inv_var, pix, psi, response=response)
        else:
            mapmaker_invvar.accumulate_to_map(inv_var, pix)

        ### ORBITAL DIPOLE ###
        sky_orb_dipole = view.get_orbital_dipole_tod()
        d_sky = view.get_tod(subtract=(("orbital_dipole", TODView._ALL_GAIN_TERMS),))
        if pols == "IQU":
            mapmaker_orbdipole.accumulate_to_map(sky_orb_dipole, inv_var, pix, psi,
                                                 response=response)
        else:
            mapmaker_orbdipole.accumulate_to_map(sky_orb_dipole, inv_var, pix, psi)

        ### CORRELATED-NOISE MAP ###
        if corr_noise_active:
            if pols == "IQU":
                mapmaker_ncorr.accumulate_to_map(
                    (n_corr_est/gain).astype(np.float32, copy=False),
                    inv_var, pix, psi, response=response)
            else:
                mapmaker_ncorr.accumulate_to_map(
                    (n_corr_est/gain).astype(np.float32, copy=False),
                    inv_var, pix, psi)
            d_sky -= n_corr_est

        # Gap-fill flagged samples instead of compacting them away. The CG operator applies a
        # Fourier transform (apply_T), which requires a continuous, full-length TOD: removing masked
        # samples corrupts the FFT, and a single non-finite sample (or an empty compacted scan)
        # otherwise poisons/crashes the whole solve. fill_all_masked (linear interpolation + white
        # noise) is the same gap-filling used in correlated-noise sampling; the filled samples are
        # noisy realizations carrying weight 1/sigma0^2, consistent with the full-length A operator.
        fill_all_masked(d_sky, good_data_mask, sigma0)

        cg_mapmaker.accum_to_RHS(
                    scan_tod=view.detector,
                    sigma0=sigma0,
                    pix=pix,
                    psi=psi,
                    scan_tod_arr=d_sky/gain
                    )

    ### PRINT NOISE SAMPLING STATS ###
    if corr_noise_active:
        log_corr_noise_stats(band_comm, experiment_data.nu, experiment_data.noise_model,
                             sampled_params, residuals, niters, num_failed_convergences_ncorr,
                             num_too_high_var_ncorr, worst_residual_ncorr,
                             sum(len(s.detectors) for s in experiment_data.scans))


    ### FINALIZE INVERSE-VARIANCE MAP, BUILD PRECONDITIONER, GATHER/NORMALIZE ###
    # The inverse-variance map is now complete (accumulated with this iteration's sigma0); finalize
    # it and assign cg_mapmaker.M before solving, so the preconditioner matches the RHS and the LHS
    # operator (which reads the live sigma0 too).
    mapmaker_invvar.gather_map()
    if pols == "IQU":
        mapmaker_invvar.normalize_map()
        if ismaster:
            # Jacobi preconditioner M = 1/diag(A), where A is the accumulated inverse-noise matrix
            # (final_cov_map holds its 6 unique elements; [0,3,5] are A_II, A_QQ, A_UU). Using
            # diag(A^-1) via rms**2 instead blows up at near-singular pixels (poor per-pixel
            # polarization-angle coverage, where the 3x3 inverse is inflated by a vanishing
            # determinant), wrecking the conditioning and making PCG diverge. 1/diag(A) stays bounded
            # by the actual per-component inverse variance; without_nan zeros unobserved pixels.
            A_diag = mapmaker_invvar.final_cov_map[(0, 3, 5), :]
            cg_mapmaker.M = InvNPreconditionerIQU(utils.without_nan(1.0 / A_diag))
        map_rms = mapmaker_invvar.final_rms_map
        map_cov = mapmaker_invvar.final_cov_map
    else:
        if ismaster:
            cg_mapmaker.M = InvNPreconditionerI(utils.without_nan(1./mapmaker_invvar.final_map))
        map_cov = mapmaker_invvar.final_map
        map_rms = 1./np.sqrt(map_cov)

    mapmaker_orbdipole.gather_map()
    mapmaker_orbdipole.normalize_map(map_cov)
    map_orbdipole = mapmaker_orbdipole.final_map
    cg_mapmaker.finalize_RHS()
    cg_mapmaker.solve()
    map_signal = cg_mapmaker.solved_map

    if corr_noise_active:
        mapmaker_ncorr.gather_map()
        mapmaker_ncorr.normalize_map(map_cov)
        map_corrnoise = mapmaker_ncorr.final_map

    ### FINAL CLEANUP ON MASTER RANK ###
    detmap_dict_out = {}
    maps_to_file = {}
    if band_comm.Get_rank() == 0:
        #Here we split here between I and QU
        # Smooth maps to the common analysis resolution after mapmaking; 0 leaves bands at their
        # native beam.
        common_res_fwhm = mapmaking.common_res_fwhm
        if "I" in pols:
            detmap_I = DetectorMap(map_signal[0,:], map_rms[0,:], experiment_data.nu,
                                experiment_data.fwhm, experiment_data.nside,
                                lmax=mapmaking.band_lmax)
            detmap_I.g0 = tod_samples.abs_gain
            if common_res_fwhm:
                detmap_I.smooth_to_resolution(common_res_fwhm)
            detmap_dict_out.update({"I": detmap_I})
        if "QU" in pols:
            detmap_QU = DetectorMap(map_signal[1:3,:], map_rms[1:3,:], experiment_data.nu,
                                experiment_data.fwhm, experiment_data.nside,
                                lmax=mapmaking.band_lmax)
            detmap_QU.g0 = tod_samples.abs_gain
            if common_res_fwhm:
                detmap_QU.smooth_to_resolution(common_res_fwhm)
            detmap_dict_out.update({"QU": detmap_QU})

        maps_to_file["map_observed_sky"] = map_signal
        maps_to_file["map_rms"] = map_rms
        if mapmaking.include_orbital_dipole_maps:
            maps_to_file["map_orbdipole"] = map_orbdipole
        if mapmaking.include_corr_noise_maps and corr_noise_active:
            maps_to_file["map_corrnoise"] = map_corrnoise
        if mapmaking.include_sky_model_maps:
            maps_to_file["map_skymodel"] = compsep_output

    return detmap_dict_out, maps_to_file
