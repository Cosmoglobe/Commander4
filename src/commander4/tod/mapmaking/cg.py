"""The CG mapmaker: solves P^T T^T N^-1 T P m = P^T T^T N^-1 d for one band's sky map, where

- m is the final map [npix],
- P is the pointing matrix [ntod, npix],
- T is the bolometer transfer-function operator,
- N^-1 is the inverse noise covariance, diagonal in TOD space [ntod],
- d is the calibrated TOD [ntod].

Unlike the binned mapmaker (`mapmaking/binned.py`), this one can deconvolve T, since the operator
is applied iteratively rather than inverted per pixel. T is non-local along the scan, so each rank
must hold a whole scan, though only for one detector at a time.

Flagged samples cannot simply be removed, since T is a Fourier filter that needs the whole time
axis. Instead N^-1 gives them zero weight. This is the maximum-likelihood map of the good samples
alone: T still smears the model over every sample, but flagged data never enters the map.

`tod2map` (`mapmaking/tod2map.py`) runs the scan loop: it adds each detector-scan to the right-hand
side (`accum_to_RHS`) and calls `solve` after the loop.
"""
import numpy as np
from mpi4py import MPI
import logging
from numpy.typing import NDArray

from commander4.backend import mapmaker as cpp_mapmaker
from commander4.tod.view import TODView
from commander4.compsep.cg_driver import DistributedCGArray
from commander4.tod.mapmaking.preconditioners import InvNPreconditionerI,\
    BlockInvNPreconditionerIQU
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.tod_samples import TODSamples
from commander4.math_utils.arithmetic import inplace_scale, dot, norm
from commander4.math_utils.fft import forward_rfft, backward_rfft
from commander4.math_utils.transfer_func import SinglePole
from commander4.tod.config import MapmakingConfig


class CGMapmaker:
    """CG mapmaker solving P^T T^T N^-1 T P m = P^T T^T N^-1 d for one band's sky map.

    The map has the rows I, Q, U, (3, npix), or intensity alone, (1, npix), for an I-only band.
    """
    def __init__(self,
                detector_tod:DetectorGroupTOD,
                detector_samples:TODSamples,
                map_comm:MPI.Comm,
                mapmaking_cfg:MapmakingConfig,
                #optionals:
                W_mat:NDArray|None = None,
                CG_check_interval:int = 1):
        """Initialise the CG mapmaker.

        Args:
            detector_tod: Detector-group TOD data for this band. Its bolometer time constant
                (`tf_tau_sec`, None for none) sets the transfer function T.
            detector_samples: Sampled noise and gain parameters for the current chain state.
            map_comm: MPI communicator shared by ranks contributing to
                the same output map.
            mapmaking_cfg: Mapmaking settings: the thread count (FFTs) and the CG iteration limit
                and tolerance on the squared relative preconditioned residual.
            CG_check_interval: Check convergence every this many iterations.

        The rank's RHS/LHS buffers hold the pixels of the band's ``PixelDomain``
        (``detector_tod.pixel_domain``, which must be built), and the full-sky iterate held by the
        master is scattered/gathered to the ranks each iteration.
        """
        if detector_tod.pols not in ("I", "IQU"):
            raise ValueError(f"The CG mapmaker supports I and IQU bands, not {detector_tod.pols}.")
        self.logger = logging.getLogger(__name__)
        self.detector_tod = detector_tod
        self.detector_samples = detector_samples
        self.map_comm = map_comm
        self.ismaster = self.map_comm.Get_rank() == 0
        self.nthreads = mapmaking_cfg.num_threads
        # Bolometer transfer function T(f), a function of the frequency in Hz; None means identity.
        self.T_omega = (SinglePole(detector_tod.tf_tau_sec).response
                        if detector_tod.tf_tau_sec is not None else None)
        self.W_mat = W_mat
        # Native sampling rate [Hz] of the mapmaking TODs, so the transfer function T_omega(omega) is
        # evaluated on a physical-frequency grid (a `tau` in seconds means seconds, not samples). The
        # CG's own noise model is white, so unlike apply_N_inv this rate is only needed for apply_T.
        self.fsamp = detector_tod.fsamp
        self.CG_maxiter = mapmaking_cfg.cg.max_iter
        self.CG_tol = mapmaking_cfg.cg.err_tol
        self.CG_check_interval = CG_check_interval
        self._ncomp = 1 if detector_tod.pols == "I" else 3
        # The same domain the view converts the pointing with, so the two always agree.
        self.domain = detector_tod.pixel_domain
        self._nloc = self.domain.n_local
        # View over the band's detector-scans, used to access pointing (pix/psi) when applying the
        # pointing matrix and its adjoint.
        self._scan_view = TODView(detector_tod, detector_samples)
        self._rhs_loca_map = None
        # The full-sky RHS and solution exist only on the master, and only once `finalize_RHS` and
        # `solve` produce them; the iterate is scattered to the ranks each iteration (apply_LHS).
        self._rhs_finalized_map = None
        self._map_signal = None

    @property
    def solved_map(self):
        """The solved sky map. Only valid on the master rank after ``solve()``."""
        if self.map_comm.Get_rank() == 0:
            if self._map_signal is None:
                raise RuntimeError("Attempted to read the solution map before it was solved.")
        return self._map_signal
    
    @property
    def RHS_map(self):
        """The finalised RHS map. Only valid on master rank after ``finalize_RHS()``."""
        if self.map_comm.Get_rank() == 0:
            if self._rhs_finalized_map is None:
                raise RuntimeError("Attempted to read the RHS map before it was finalized.")
            return self._rhs_finalized_map
        else:
            return np.empty(())

    # The pointing operators take `pix_local`: local pixel indices into the rank's map buffers
    # (`TODView.pix_local`), converted once per detector-scan by the caller, and the detector's
    # `response_I_P`, which scales the intensity and polarization parts of its pointing row.
    def apply_P(self, in_map: NDArray, scan_tod_arr: NDArray, pix_local: NDArray, psi: NDArray,
                response_I_P: tuple[float, float] = (1.0, 1.0)) -> NDArray:
        """Read the local map `in_map` along one detector-scan into `scan_tod_arr` (P m), with
        the pointing row [response_I, response_P cos(2 psi), response_P sin(2 psi)], or
        [response_I] for an I-only map."""
        cpp_mapmaker.map2tod(in_map, scan_tod_arr, np.asarray(pix_local, dtype=np.int64),
                             np.asarray(psi, dtype=np.float64), response_I=response_I_P[0],
                             response_P=response_I_P[1])
        return scan_tod_arr

    def apply_P_adjoint(self, scan_tod_arr: NDArray, out_map: NDArray, pix_local: NDArray,
                        psi: NDArray, response_I_P: tuple[float, float] = (1.0, 1.0)) -> NDArray:
        """Add one detector-scan's TOD into the local map `out_map` (P^T d), with the same
        pointing row as `apply_P`."""
        cpp_mapmaker.tod2map(out_map, scan_tod_arr, np.asarray(pix_local, dtype=np.int64),
                             np.asarray(psi, dtype=np.float64), response_I=response_I_P[0],
                             response_P=response_I_P[1])
        return out_map

    def apply_inv_N(self, scan_tod_arr: NDArray, inv_var: float, good_mask: NDArray) -> NDArray:
        """Apply N^-1 in place to one scan: weight `inv_var` at good samples, zero at flagged ones.

        Args:
            scan_tod_arr: The scan's TOD.
            inv_var: Inverse noise variance per sample of the gain-calibrated TOD,
                (gain/sigma0)^2, the same weight the binned maps and the preconditioner use.
            good_mask: True at the good samples (`TODView.get_mask(proc_mask=False)`).
        """
        inplace_scale(scan_tod_arr, inv_var)
        # Assigned rather than multiplied, so whatever a flagged sample holds, even NaN, is gone.
        scan_tod_arr[~good_mask] = 0.0
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
        if self.T_omega is None:
            return np.ascontiguousarray(scan_tod_arr, dtype=np.float64)
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

    def apply_W(self, scan_tod_arr):
        """
        Applies the cross-talk operator to a tod.
        """

        # TODO: how to deal with multiple detectors here???
        # in the CG mapmaker tod processing we only load a detector at a time.
        if self.W_mat is None:
            return scan_tod_arr
        else:
            return self.W_mat @ scan_tod_arr

    def apply_W_adjoint(self, scan_tod_arr):
        """
        Applies the cross-talk operator to a tod.
        (self-adjoint as the matrix is symmetric)
        """
        return self.apply_W(scan_tod_arr)

    def accum_to_RHS(self, view: TODView, scan_tod_arr: NDArray):
        """Add one detector-scan's contribution P^T T^T W^T N^-1 d to the right-hand side.

        Args:
            view: The view focused on the detector-scan. Its pointing, response, gain and sigma0 are
                read exactly as `apply_LHS` reads them, so the right-hand side and the operator
                always agree.
            scan_tod_arr: The detector-scan's calibrated TOD d (detector TOD divided by the gain),
                full length. It is modified in place.
        """
        if self._rhs_loca_map is None:
            #if not done already, allocate memory for local maps
            self._rhs_loca_map = np.zeros((self._ncomp, self._nloc))

        # Guard against pathological scans that slip past read-in: an empty scan crashes the FFT,
        # and a single non-finite good sample is spread across the whole scan by apply_T (and then
        # across every pixel that scan hits), making the CG residual NaN. Readers should discard
        # these, but not all of them do, so fail loudly here identifying the offending
        # detector-scan. Flagged samples get zero weight below, so they may hold anything.
        good_mask = view.get_mask(proc_mask=False)
        if scan_tod_arr.shape[-1] == 0:
            raise ValueError(f"Empty TOD passed to CG RHS for detector {view.detector.name}.")
        if not np.isfinite(scan_tod_arr[good_mask]).all():
            raise ValueError(f"Non-finite good samples in CG RHS for detector "
                             f"{view.detector.name} (check gain and sigma0).")
        # N^-1 d
        scan_tod_arr = self.apply_inv_N(scan_tod_arr, (view.get_gain()/view.sigma0)**2, good_mask)
        # W^T N^-1 d
        scan_tod_arr = self.apply_W_adjoint(scan_tod_arr)
        # T^T W^T N^-1 d
        scan_tod_arr = self.apply_T_adjoint(scan_tod_arr)
        # P^T T^T W^T N^-1 d
        self._rhs_loca_map = self.apply_P_adjoint(scan_tod_arr, self._rhs_loca_map, view.pix_local,
                                                  view.psi, view.response_I_P)

    def finalize_RHS(self, root=0):
        """
        Reduces the local RHS contributions onto the full-sky RHS map held by the master rank.
        """
        # Check for None, which indicates a rank without any scans. Give it a zero-map.
        if self._rhs_loca_map is None:
            self._rhs_loca_map = np.zeros((self._ncomp, self._nloc))
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
                                                  dtype=np.float64)
        out_local = np.zeros((self._ncomp, self._nloc))
        # The LHS operator P^T T^T N^-1 T P and the RHS P^T T^T N^-1 d must span the same set of
        # detector-scans AND weight the samples the same way, or the CG solves an inconsistent
        # (A, b). We iterate the same accept-gated TODView path the RHS loop uses, on the
        # *full-length* pointing (apply_T needs the whole time axis), with the same zero weight
        # for flagged samples as accum_to_RHS.
        for view in self._scan_view.iter_focused(accepted_only=True):
            pix_local = view.pix_local  # converted once, shared by P and P^T below
            psi = view.psi
            response_I_P = view.response_I_P
            inv_var = (view.get_gain()/view.sigma0)**2  # as in accum_to_RHS and the binned weights
            good_mask = view.get_mask(proc_mask=False)
            scan_tod_arr_aux = np.zeros(pix_local.shape[0], dtype=np.float64)  # full-length, as RHS
            #P m
            scan_tod_arr_aux = self.apply_P(local_in, scan_tod_arr_aux, pix_local, psi,
                                            response_I_P)
            #T P m
            scan_tod_arr_aux = self.apply_T(scan_tod_arr_aux)
            #W T P m
            scan_tod_arr_aux = self.apply_W(scan_tod_arr_aux)
            #N^-1 W T P m
            scan_tod_arr_aux = self.apply_inv_N(scan_tod_arr_aux, inv_var, good_mask)
            #W^T N^-1 W T P m
            scan_tod_arr_aux = self.apply_W_adjoint(scan_tod_arr_aux)
            #T^T W^T N^-1 W T P m
            scan_tod_arr_aux = self.apply_T_adjoint(scan_tod_arr_aux)
            #P^T T^T W^T N^-1 W T P m
            out_local = self.apply_P_adjoint(scan_tod_arr_aux, out_local, pix_local, psi,
                                             response_I_P)
        # Sum the local contributions back to the full-sky map on the master (None on other ranks).
        return self.domain.reduce_to_full(out_local)

    def solve(self, map_cov: NDArray | None, x_true=None):
        """Sum the right-hand side onto the master and solve for the sky map there (collective).

        The result is in `solved_map` on the master.

        Args:
            map_cov: The binned weights P^T N^-1 P on the master (`BinnedMapmaker.map_cov`), None
                on the other ranks. Their inverse per pixel is the (block) Jacobi preconditioner.
            x_true: Optional known solution to report the error against, for testing. Only the
                master's copy is used, but every rank must agree on whether one was passed, since
                evaluating the error applies the (collective) LHS operator.
        """
        self.finalize_RHS()
        RHS_map = self.RHS_map
        ismaster = self.map_comm.Get_rank() == 0
        # Only the master runs the CG arithmetic, so only it needs the preconditioner.
        if ismaster:
            M = (BlockInvNPreconditionerIQU(map_cov) if self._ncomp == 3
                 else InvNPreconditionerI(map_cov))
        else:
            M = None

        CG_solver = DistributedCGArray(self.apply_LHS,
                                       RHS_map,
                                       ismaster,
                                       M = M,
                                       dot = dot,
                                       destroy_b=True)

        # Every collective below has to be entered by all ranks, so the two conditions guarding one
        # are made rank-independent up front: the iteration index already is, and whether a true
        # solution was supplied is broadcast (the caller may only have one on the master).
        check_x_true = self.map_comm.bcast(x_true is not None, root=0)
        if ismaster:
            self.logger.verbose("Mapmaker CG starting up.")
        converged = False
        niter = 0
        for i in range(self.CG_maxiter):
            CG_solver.step()
            niter = i + 1
            log_this_iter = i % self.CG_check_interval == 0
            if log_this_iter and ismaster:
                self.logger.verbose(
                    f"Mapmaker CG iter {i:3d} - squared residual {CG_solver.err:.6e}")
            if log_this_iter and check_x_true:
                # apply_LHS is collective, so it is called outside the master-only branch; it takes
                # its input from, and returns its result to, the master alone.
                error_map = CG_solver.x - x_true if ismaster else None
                A_error_map = self.apply_LHS(error_map)
                if ismaster:
                    CG_L2_error = norm(error_map)/norm(x_true)
                    CG_Anorm_error = dot(error_map, A_error_map)
                    self.logger.debug(f"CG iter {i:3d} - True A-norm error: {CG_Anorm_error:.3e} "
                                      f"- True L2 error: {CG_L2_error:.3e}")
            # Only the master updates CG_solver.err, so the stopping decision has to be broadcast.
            converged = self.map_comm.bcast(CG_solver.err < self.CG_tol, root=0)
            if converged:
                break
        self._map_signal = CG_solver.x
        if ismaster:
            # CG_tol == 0 means running to max_iter was on purpose, and is not concerning.
            if not converged and self.CG_tol > 0.0:
                self.logger.warning(f"Mapmaker CG reached its maximum of {self.CG_maxiter} "
                                    f"iterations with squared residual {CG_solver.err:.3e}.")
            self.logger.info(f"Mapmaker CG finished after {niter} iterations with squared residual "
                             f"{CG_solver.err:.3e} (tolerance {self.CG_tol:.3e}).")
