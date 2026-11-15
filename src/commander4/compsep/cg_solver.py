"""The CG amplitude solver: `CompSepSolver` builds and solves the component-amplitude system.

One MPI rank per band, with the LHS operator and its right-hand side both assembled across ranks.
Depending on the sampling group's `optimize` flag the solution is either the posterior maximum or a
constrained realization drawn from the amplitude posterior. See the class docstring for the system
being solved.
"""
import numpy as np
import time
import logging
from mpi4py import MPI
from typing import TYPE_CHECKING
from collections.abc import Callable
from copy import deepcopy

from commander4.data_models.detector_map import DetectorMap
from commander4.sky.comp_list import CompList
from commander4.math_utils.alm import gaussian_random_alm
from commander4.math_utils.sht import alm_to_map, alm_to_map_adjoint
from commander4.sky.comp_list import complist_dot, complist_norm
from commander4.compsep.dense_matrix_debug import DenseMatrix
from commander4.math_utils.cg_driver import DistributedCG
from commander4.compsep import preconditioners
from commander4.data_models.band import Band
from commander4.diagnostics.performance import benchmark, log_memory

if TYPE_CHECKING:
    from commander4.compsep.processing import CGSamplingGroupConfig

logger = logging.getLogger(__name__)

MPI_LIMIT_32BIT = 2**31 - 1


class CompSepSolver:
    """ Class for performing global component separation using the preconditioned conjugate gradient
        method. After initializing the class, the solve() method should be called to perform the
        component separation. Note that the solve() method will in-place update (as well as return)
        the 'comp_list' argument passed to the constructor.

        The component separation problem is an Ax = b equation on the form
        (S^-1 + Y^T M^T Y^-1^T N^-1 B M Y) a
            = Y^T M^T Y^-1^T B^T d + Y^T M^T Y^-1^T B^T N^-{1/2} eta_1 + S^-1 mu + S^{-1/2} eta_2,
        where B is the beam smoothing, M is the mixing matrix, N is the noise covariance matrix,
        Y is alm->map spherical harmonic synthesis, d is the observed frequency maps (as alms),
        a is the component maps we want to solve for (as alms), and eta_1 and eta_2 are random
        numbers drawn from N(0,1). The sampling group's `optimize` switches between a full
        constrained realization (`optimize: false`, the default) and just optimizing for MAP.
        For numerical stability, we actually solve the equivalent
        (1 + S^{1/2} Y^T M^T Y^-1^T B^T N^-1 B Y^-1 M Y S^{1/2})[S^{-1/2} a]
            = S^{1/2} Y^T M^T {Y^{-1}}^T B^T N^{-1} d
            + S^{1/2}A^TN^{-1/2} eta_1 + S^{-1/2} mu + eta_2.
    """
    def __init__(self, det_map: DetectorMap, CompSep_comm: MPI.Comm,
                 group: "CGSamplingGroupConfig", double_precision: bool,
                 nthreads: int):
        """
        Args:
            det_map: This rank's band data (the conditional residual for this sampling group).
            CompSep_comm: The communicator of the ranks collaborating on this solve.
            group: Validated solver settings for this CG sampling group.
            double_precision: Whether maps and alms use double precision.
            nthreads: Number of computational threads used on this rank.
        """
        self.CompSep_comm = CompSep_comm
        self.my_rank = CompSep_comm.Get_rank()
        self.det_map = det_map
        self.config = group
        self.double_precision = double_precision
        self.my_band = Band.init_from_detector(det_map = det_map,
                                               double_precision = double_precision)

        if not double_precision:
            self.float_dtype = np.float32
            self.complex_dtype = np.complex64
        else:
            self.float_dtype = np.float64
            self.complex_dtype = np.complex128

        self.alm_dtype = self.complex_dtype

        # For simplicity all array will have shapes (1, ...) for non-polarization
        # (and then (2, ...) for polarization).
        self.npol = det_map.npol
        self.pol = det_map.pol
        self.spin = 2 if self.pol else 0

        self.nthreads = nthreads

        # Whether to add the two fluctuation terms to the RHS, or just optimize for MAP. Note the
        # inverted polarity: `optimize: false` (the default) draws a constrained realization.
        self.sample_amplitudes = not group.optimize
    

    def apply_LHS_matrix(self, comp_list_in: CompList) -> CompList:
        #This a_in should become a list of Component objects instead. 
        """ Applies the A matrix to inputed component alms a, where A represents the entire LHS of
            the Ax=b system for global component separation. The full A matrix is:
            (1 + S^{1/2} Y^T M^T Y^-1^T B^T N^-1 B Y^-1 M Y S^{1/2}).
            For a component without a prior (`Cl_prior_amplitude: null`, C3's `CL_TYPE = none`)
            the "1 +" is left out and S = 1. For point sources, `B Y^-1 M Y` is replaced by painting
            the sources into the band map, and the Y and Y^T around N^-1 are left out.
            This function should be called by all ranks holding a frequency map, even if they do
            not hold a compoenent, as they are still needed to compute the LHS operation.
            Args:
                comp_list_in (CompList): List of Components contributing to the local band
            Returns:
                Aa (np.array): The result of applying A to the input alms. Will return a zero-sized
                               array if this MPI rank does not hold a component.                               
        """
        myrank = self.CompSep_comm.Get_rank()
        
        comp_list = deepcopy(comp_list_in)
        if myrank == 0:  # this task actually holds a component
            # S^{1/2} a
            comp_list.apply_Cl_prior_sqrt()

        # Spread initial a to all ranks from master.
        # NB: For some stupid reason the non-blocking mpi4py calls do not have 64-bit length support
        # and are therefore limited to <2GB arrays... We have to fallback to blocking communication
        # for >2GB arrays. In the future we should probably implement chunking instead.

        with benchmark("lhs-bcast"):
            comp_list.bcast_data_blocking(self.CompSep_comm)

        # B Y^-1 M Y S^{1/2} a
        with benchmark("lhs-project"):
            comp_list.comp2band(self.my_band, nthreads=self.nthreads)

        # Y^T N^-1 Y B Y^-1 M Y S^{1/2} a. Point sources are added in pixel space, as in C3:
        # comp2band painted them into `band.map` (None without point sources), and they read
        # N^-1 (...) back from it. Diffuse components go through Y and Y^T.
        with benchmark("lhs-inv-N"):
            band, det_map = self.my_band, self.det_map
            band_map = alm_to_map(band.alms, det_map.nside, det_map.lmax, spin=det_map.spin,
                                  nthreads=self.nthreads)
            if band.map is not None:
                band_map += band.map
            det_map.apply_inv_N_map(band_map)
            if band.map is not None:
                band.map = band_map
            band.alms = alm_to_map_adjoint(band_map, det_map.nside, det_map.lmax,
                                           spin=det_map.spin, nthreads=self.nthreads)

        # Y^T M^T Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y S^{1/2} a
        with benchmark("lhs-eval"):
            comp_list.band2comp(self.my_band, nthreads=self.nthreads)

        # Accumulate solution on master
        with benchmark("lhs-accum"):
            biggest_size_bytes = np.max([comp.amps.nbytes for comp in comp_list])
            use_blocking = biggest_size_bytes > MPI_LIMIT_32BIT
            if use_blocking:
                logger.debug(f"Using blocking CompSep communication for array size "
                             f"{biggest_size_bytes:.2e} B.")
                comp_list.accum_data_blocking(self.CompSep_comm)

            else:
                requests = comp_list.accum_data_non_blocking(self.CompSep_comm)

            if myrank == 0:
                for icomp in range(len(comp_list)):
                    # Since we used non-blocking reduce, master rank can start working on components
                    # as they are received instead of waiting for all to be received.
                    if not use_blocking:
                        # Wait until all data for component icomp has been received.
                        MPI.Request.Wait(requests[icomp])
                    # S^{1/2} Y^T M^T Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y S^{1/2} a
                    comp = comp_list.components[icomp]
                    comp.apply_Cl_prior_sqrt(comp.amps)
                    # The prior term adds the input vector: (1 + S^{1/2}...)a = a + (S^{1/2}...)a.
                    if comp.Cl_prior_amplitude is not None:
                        comp += comp_list_in.components[icomp]
            else: # Worker ranks just wait for all their sends to complete.
                if not use_blocking:
                    for icomp in range(len(comp_list)):
                        MPI.Request.Wait(requests[icomp])
                comp_list = CompList([])

        return comp_list


    def calc_RHS(self, comp_list: CompList, rng: np.random.Generator) -> CompList:
        """ Calculates the right-hand-side b-vector of the Ax=b CompSep equation, overwriting and
            returning `comp_list`:

                b = S^{1/2} A^T (N^-1 d + N^{-1/2} eta_1) + S^{-1/2} mu + eta_2,

            where the two eta terms are included only when `sample_amplitudes` is set. Without them
            b gives the deterministic Wiener-filtered (maximum a posteriori) mean, whose power
            spectrum is biased low; with them the CG solution is a constrained realization, i.e. a
            draw from the full conditional posterior of the amplitudes.

            The data and the eta_1 fluctuation are summed in *map* space, before the single shared
            application of S^{1/2}A^T: they differ only in which map-space vector they start from,
            and the operator between there and the component alms is linear, so one adjoint SHT, one
            MPI reduce and one prior application serve both terms. The prior mean mu and the prior
            fluctuation eta_2 need none of that (they live directly in the component alm space)
            so they are added afterwards, on the rank that holds the accumulated result. A component
            without a prior (`Cl_prior_amplitude: null`) gets neither of the two.
        """
        #FIXME: how will eta_2 work for point sources?
        myrank = self.CompSep_comm.Get_rank()

        # N^-1 d
        b_map = self.det_map.apply_inv_N_map(self.det_map.map_sky, inplace=False)
        if self.sample_amplitudes:
            # + N^{-1/2} eta_1. Drawn per band and independently on every rank, as it must be:
            # eta_1 is the white-noise realization of *this* band's data. Zero-weight pixels
            # (unobserved, map_inv_var = 0) get no fluctuation, matching their absence from the LHS.
            eta_1 = rng.normal(0.0, 1.0, b_map.shape)*np.sqrt(self.det_map.map_inv_var)
            logger.debug(f"RHS |N^-1 d| = {np.mean(np.abs(b_map)):.2e}, "
                         f"|N^-1/2 eta_1| = {np.mean(np.abs(eta_1)):.2e}")
            b_map += eta_1.astype(b_map.dtype, copy=False)

        # Y^T (N^-1 d + N^{-1/2} eta_1)
        b_band = Band.init_from_detector(det_map=self.det_map,
                                         double_precision=self.double_precision)
        b_alm = alm_to_map_adjoint(b_map, self.my_band.nside, self.my_band.lmax, spin=self.spin,
                                   nthreads=self.nthreads)
        b_band.alms = b_alm.astype(b_band.alms.dtype, copy=False)
        # Point sources read the pixel map itself, as in the LHS.
        b_band.map = b_map

        # (Y^T M^T Y^-1^T B^T) Y^T (...)
        comp_list.band2comp(b_band, nthreads=self.nthreads)

        # Accumulate on master and apply the prior there: S^{1/2} Y^T M^T Y^-1^T B^T Y^T (...)
        for comp in comp_list:
            comp.accum_data_blocking(self.CompSep_comm)
            if myrank == 0:
                comp.apply_Cl_prior_sqrt(comp.amps)

        if myrank == 0:
            for comp in comp_list:
                if comp.Cl_prior_amplitude is None:
                    continue  # No prior: no prior mean and no prior draw.
                # + S^{-1/2} mu, the prior mean. Added whether or not we are sampling: unlike the
                # eta terms this belongs to the *mean* solution, so a non-zero prior mean shifts the
                # Wiener filter itself and must be present in the MAP solve too. mu is an alm
                # vector, None unless the component sets `amp_prior_mean_map` (read as a sky map
                # and transformed once at startup by CompList.load_amp_prior_means); the usual
                # zero-mean prior therefore costs one None check, not a full-size scale-and-add.
                amp_prior_mean = comp.amp_prior_mean
                if amp_prior_mean is not None:
                    comp.amps += comp.apply_Cl_prior_inv_sqrt(amp_prior_mean)
                if self.sample_amplitudes:
                    # + eta_2, the prior's own fluctuation. In the preconditioned variable it enters
                    # unmodified: substituting a = S^{1/2}x into the S^{-1/2}eta_2 term and
                    # left-multiplying by S^{1/2} leaves it alone.
                    for ipol in range(self.npol):
                        comp.amps[ipol] += gaussian_random_alm(comp.lmax, comp.lmax,
                                                               self.spin, 1, rng)[0]
                logger.debug(f"RHS comp-{comp.shortname}: {np.mean(np.abs(comp.amps)):.2e}")

        return comp_list


    def solve_CG(self, LHS: Callable, RHS: CompList, M: Callable, x0 = None,
                 x_true = None) -> CompList:
        """ Solves the equation Ax=b for x given A (LHS) and b (RHS) using CG.
            Assumes that both x and b are in alm space.

            Args:
                LHS (callable): A function/callable taking x as argument and returning Ax.
                RHS (list): A list of alm-vectors representing the right-hand-side of the equation.
                x0 (list): Initial guess for x as list of alm-vectors for each component (optional).
                M (callable): Preconditioner: A callable which approximates A^-1.
                x_true (np.array): True solution for x, in order to print the true error (optional).
            Returns:
                m_bestfit (list): The resulting best-fit solution, for the component held by this
                                  rank, as a list of alm-vectors for each component.
        """
        if self.det_map.pol:
            max_iter = self.config.max_iter_pol
        else:
            max_iter = self.config.max_iter

        checkpt_int = 30
        master = self.CompSep_comm.Get_rank() == 0
        mycomp = self.CompSep_comm.Get_rank()

        CG_solver = DistributedCG(LHS, RHS, master, x0=x0, M=M, destroy_b=True)
        self.CG_residuals = np.zeros(max_iter)
        if x_true is not None:
            if master:
                self.xtrue_A_xtrue = complist_dot(x_true, LHS(x_true))
            else:
                LHS(CompList([]))
        if master:
            logger.verbose(f"{'QU' if self.det_map.pol else 'Intensity'} CG starting up.")
        iter = 0
        t0 = time.time()
        stop_CG = False
        while not stop_CG:
            # Counted per step, so the report's N column is the number of CG iterations taken.
            with benchmark("cg-iter"):
                CG_solver.step()
            log_memory("cg-iter")
            self.CG_residuals[iter] = CG_solver.err
            iter += 1
            if iter%checkpt_int == 0:
                if master:
                    logger.verbose(
                        f"{'QU' if self.det_map.pol else 'Intensity'} CG iter {iter:3d} - "
                        f"Squared residual {np.mean(self.CG_residuals[iter-checkpt_int:iter]):.6e} "
                        f"({(time.time() - t0)/checkpt_int:.2f}s/iter)")
                    t0 = time.time()
                    if x_true is not None:
                        CG_errors_true = complist_norm(CG_solver.x - x_true)/complist_norm(x_true)
                        A_residual = LHS([x - y for x,y in zip(CG_solver.x, x_true)])
                        #A_residual = np.concatenate(A_residual, axis=-1)
                        CG_Anorm_error = complist_dot([x - y for x,y in zip(CG_solver.x, x_true)],
                                                      A_residual)
                        # A-norm error is only defined for the full vector.
                        logger.verbose(f"CG iter {iter:3d} - True A-norm error: "
                                     f"{CG_Anorm_error:.3e}")
                        # We can print the individual component L2 errors.
                        logger.verbose(f"CG iter {iter:3d} - {self.comp_list[mycomp].comp_name} - "
                                     f"True L2 error: {CG_errors_true:.3e}")
                else:
                    if x_true is not None:
                        LHS(CompList([]))  # Matching LHS call for the calculation of LHS(CG_solver.x-x_true).
            if iter >= max_iter:
                # Only log warning if err_tol != 0, as this means user wants to do max_iter.
                if master and self.config.err_tol > 0.0:
                    logger.warning(f"Maximum number of iterations ({max_iter}) reached in CG.")
                stop_CG = True
            if CG_solver.err <= self.config.err_tol:
                stop_CG = True
            stop_CG = self.CompSep_comm.bcast(stop_CG, root=0)
        self.CG_residuals = self.CG_residuals[:iter]
        if master:
            logger.info(f"{'QU' if self.det_map.pol else 'Intensity'} CG finished after {iter} "\
                        f"iterations with a squared residual of {CG_solver.err:.3e} "\
                        f"(err tol = {self.config.err_tol})")

        complist_sol = CG_solver.x
        for comp in complist_sol:
            if master:
                comp.apply_Cl_prior_sqrt(comp.amps)
            comp.bcast_data_blocking(self.CompSep_comm)

        return complist_sol


    def solve(self, comp_list: CompList, rng: np.random.Generator) -> CompList:
        """Solve  the amplitudes of `comp_list` for this rank's band.

        If `sample_amplitudes==True` the solution is a constrained realization, i.e the Wiener mean
        plus two fluctuation terms, i.e. the amplitudes are a draw from the conditional posterior.
        If `sample_amplitudes==False`, only the mean term is used and the result is the
        deterministic maximum a posteriori solution, which is biased low in power.

        `rng` is this rank's random generator. Each rank must have its own stream: eta_1 is *this*
        band's white-noise realization, so shared draws would correlate the fluctuation across
        bands.

        The CG starts from the amplitudes that `comp_list` holds on entry.
        """
        # Set comp beams to 0 (in theory they could be non-zero if the per-pix solver has ran).
        for comp in comp_list:
            comp.amp_fwhm_rad = 0.0
        # Set starting guess to x0 = S^{-1/2} a. It is copied, because `calc_RHS` overwrites
        # `comp_list`. Only the master's values are used; the other ranks need the list as a buffer
        # of the right shape.
        x0 = deepcopy(comp_list)
        if self.my_rank == 0:
            for comp in x0:
                comp.apply_Cl_prior_inv_sqrt(comp.amps)

        # `calc_RHS` overwrites the CompList it is handed, so comp_list doubles as the RHS buffer
        # (the preconditioner below reads only its metadata, never its alms).
        with benchmark("cg-rhs"):
            RHS = self.calc_RHS(comp_list, rng)

        # Initialize the precondidioner class, which is in the module "solvers.preconditioners",
        # and has the name specified by this sampling group's preconditioner setting.
        with benchmark("cg-precond-init"):
            precond = getattr(preconditioners, self.config.preconditioner)(self, comp_list)

        # For testing preconditioner with a true solution as reference,
        # first solving for exact solution with dense matrix math.
        if self.config.dense_matrix_debug_mode:
            M_A_matrix = lambda a : self.apply_LHS_matrix(precond(a))

            # Testing the initial LHS (A) matrix
            dense_matrix = DenseMatrix(self, self.apply_LHS_matrix, "A")
            dense_matrix.test_matrix_hermitian()
            dense_matrix.print_sing_vals()
            dense_matrix.test_matrix_eigenvalues()
            dense_matrix.print_matrix_diag()

            # Testing the preconditioning matrix (M) alone
            dense_matrix = DenseMatrix(self, precond, "M")
            dense_matrix.test_matrix_hermitian()  # Preconditioner matrix needs to be Hermitian.
            dense_matrix.test_matrix_eigenvalues()  # and to have positive and real eigenvalues

            # Testing the combined preconditioned system (MA)
            # (this combined matrix will generally not be Hermitian positive-definite).
            dense_matrix = DenseMatrix(self, M_A_matrix, "MA")
            x_true = dense_matrix.solve_by_inversion(precond(RHS))
            # Check how much singular values (condition number) of preconditioned system improved.
            dense_matrix.print_sing_vals()
            dense_matrix.print_matrix_diag()

        return self.solve_CG(self.apply_LHS_matrix, RHS, x0=x0, M=precond,
                             x_true=x_true if self.config.dense_matrix_debug_mode else None)
