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
from pixell.bunch import Bunch

from commander4.data_models.detector_map import DetectorMap
from commander4.sky.comp_list import CompList
from commander4.math_utils.alm import gaussian_random_alm
from commander4.math_utils.sht import alm_to_map, alm_to_map_adjoint
from commander4.math_utils.cg_driver import DistributedCG
from commander4.compsep import preconditioners
from commander4.data_models.band import Band
from commander4.diagnostics.performance import benchmark, log_memory

logger = logging.getLogger(__name__)


class CompSepSolver:
    """ Class for performing global component separation using the preconditioned conjugate gradient
        method. After initializing the class, the solve() method should be called to perform the
        component separation. solve() leaves the solution in the 'comp_list' argument passed to it,
        and returns the CG residuals.

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
    def __init__(self, band_map: DetectorMap, solver_comm: MPI.Comm, group: Bunch,
                 double_precision: bool, nthreads: int):
        """
        Args:
            band_map: This rank's band data (the conditional residual for this sampling group).
            solver_comm: The communicator of the ranks collaborating on this solve.
            group: The settings of this CG sampling group (`compsep.cg_sampling_groups.<name>`
                in the parameter file, with defaults filled in). `group.optimize` chooses the
                maximum a posteriori solution (True) or a constrained realization (False).
            double_precision: Whether maps and alms use double precision.
            nthreads: Number of computational threads used on this rank.
        """
        self.solver_comm = solver_comm
        self.is_master = solver_comm.Get_rank() == 0
        self.band_map = band_map
        self.group = group
        self.double_precision = double_precision
        self.nthreads = nthreads
        # This rank's band in harmonic space: the alms (and the pixel map of the point sources)
        # that the components are projected to and read back from in each CG iteration.
        self.band_alms = Band.init_from_detector(band_map, double_precision=double_precision)


    def apply_LHS_matrix(self, comp_list_in: CompList) -> CompList:
        """Apply A, the left-hand side of the Ax=b system, to the vector `comp_list_in`.

        The full A matrix is (1 + S^{1/2} Y^T M^T Y^-1^T B^T N^-1 B Y^-1 M Y S^{1/2}).
        For a component without a prior (`Cl_prior_amplitude: null`, C3's `CL_TYPE = none`)
        the "1 +" is left out and S = 1. For point sources, `B Y^-1 M Y` is replaced by painting
        the sources into the band map, and the Y and Y^T around N^-1 are left out.

        Every rank of the solver's communicator must call this, because each rank applies the
        part of A that belongs to its band.

        Args:
            comp_list_in: The vector to apply A to. Only the master's amplitudes are used, and
                they are not changed. On the other ranks the list is a work buffer, and its
                amplitudes are overwritten.
        Returns:
            On the master, a new list holding A applied to the input. On the other ranks, an
            empty list.
        """

        # `comp_list` is the "working" copy of the components. For the master rank we must retain
        # the original `comp_list_in`, because master adds the (1 + ...) part of the LHS equation,
        # and because the CG uses the vector again after this call.
        # The worker ranks are free to destroy their `comp_list_in`, so we point `comp_list` at it.
        comp_list = comp_list_in.copy() if self.is_master else comp_list_in

        if self.is_master:  # Only the master holds the values of the vector.
            # S^{1/2} a
            for comp in comp_list:
                comp.apply_Cl_prior_sqrt(comp.amps)

        # Spread initial a to all ranks from master.
        with benchmark("lhs-bcast"):
            for comp in comp_list:
                comp.bcast_data(self.solver_comm)

        # B Y^-1 M Y S^{1/2} a. Each component adds itself to the emptied band: a diffuse component
        # to its alms, point sources to its pixel map.
        with benchmark("lhs-project"):
            self.band_alms.alms[:] = 0
            if self.band_alms.map is not None:
                self.band_alms.map[:] = 0
            for comp in comp_list:
                comp.comp2band(self.band_alms, nthreads=self.nthreads)

        # Y^T N^-1 Y B Y^-1 M Y S^{1/2} a. Point sources are added in pixel space, as in C3:
        # comp2band painted them into `band_alms.map` (None without point sources), and they read
        # N^-1 (...) back from it. Diffuse components go through Y and Y^T.
        with benchmark("lhs-inv-N"):
            band_alms, band_map = self.band_alms, self.band_map
            pix_map = alm_to_map(band_alms.alms, band_map.nside, band_map.lmax, spin=band_map.spin,
                                 nthreads=self.nthreads)
            if band_alms.map is not None:
                pix_map += band_alms.map
            band_map.apply_inv_N_map(pix_map)
            if band_alms.map is not None:
                band_alms.map = pix_map
            band_alms.alms = alm_to_map_adjoint(pix_map, band_map.nside, band_map.lmax,
                                                spin=band_map.spin, nthreads=self.nthreads)

        # Y^T M^T Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y S^{1/2} a
        with benchmark("lhs-eval"):
            for comp in comp_list:
                comp.band2comp(self.band_alms, nthreads=self.nthreads)

        # Accumulate the result on the master. The reduces are non-blocking, so the master can work
        # on each component as soon as it has arrived, while later ones are still on their way.
        with benchmark("lhs-accum"):
            requests = [comp.accum_data_non_blocking(self.solver_comm) for comp in comp_list]
            if self.is_master:
                for icomp in range(len(comp_list)):
                    requests[icomp].Wait()
                    # S^{1/2} Y^T M^T Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y S^{1/2} a
                    comp = comp_list[icomp]
                    comp.apply_Cl_prior_sqrt(comp.amps)
                    # The prior term adds the input vector: (1 + S^{1/2}...)a = a + (S^{1/2}...)a.
                    if comp.Cl_prior_amplitude is not None:
                        comp.amps += comp_list_in[icomp].amps
            else: # Worker ranks just wait for all their sends to complete.
                MPI.Request.Waitall(requests)
                comp_list = CompList([])

        return comp_list


    def calc_RHS(self, comp_list: CompList, rng: np.random.Generator) -> CompList:
        """ Calculates the right-hand-side b-vector of the Ax=b CompSep equation, and returns it as
            a new list with the same components as `comp_list`, which is not changed:

                b = S^{1/2} A^T (N^-1 d + N^{-1/2} eta_1) + S^{-1/2} mu + eta_2,

            where the two eta terms are left out when `group.optimize` is set. Without them
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

        # N^-1 d
        b_map = self.band_map.apply_inv_N_map(self.band_map.map_sky, inplace=False)
        if not self.group.optimize:
            # + N^{-1/2} eta_1. Drawn per band and independently on every rank, as it must be:
            # eta_1 is the white-noise realization of *this* band's data. Zero-weight pixels
            # (unobserved, map_inv_var = 0) get no fluctuation, matching their absence from the LHS.
            eta_1 = rng.normal(0.0, 1.0, b_map.shape)*np.sqrt(self.band_map.map_inv_var)
            logger.debug(f"RHS |N^-1 d| = {np.mean(np.abs(b_map)):.2e}, "
                         f"|N^-1/2 eta_1| = {np.mean(np.abs(eta_1)):.2e}")
            b_map += eta_1.astype(b_map.dtype, copy=False)

        # Y^T (N^-1 d + N^{-1/2} eta_1)
        b_band = Band.init_from_detector(band_map=self.band_map,
                                         double_precision=self.double_precision)
        b_alm = alm_to_map_adjoint(b_map, self.band_alms.nside, self.band_alms.lmax,
                                   spin=self.band_map.spin, nthreads=self.nthreads)
        b_band.alms = b_alm.astype(b_band.alms.dtype, copy=False)
        # Point sources read the pixel map itself, as in the LHS.
        b_band.map = b_map

        # (Y^T M^T Y^-1^T B^T) Y^T (...)
        RHS = comp_list.zeros_like()
        for comp in RHS:
            comp.band2comp(b_band, nthreads=self.nthreads)

        # Accumulate on master and apply the prior there: S^{1/2} Y^T M^T Y^-1^T B^T Y^T (...)
        for comp in RHS:
            comp.accum_data_blocking(self.solver_comm)
            if self.is_master:
                comp.apply_Cl_prior_sqrt(comp.amps)

        if self.is_master:
            for comp in RHS:
                if comp.Cl_prior_amplitude is None:
                    continue  # No prior: no prior mean and no prior draw.
                # + S^{-1/2} mu, the prior mean. Added whether or not we are sampling: unlike the
                # eta terms this belongs to the *mean* solution, so a non-zero prior mean shifts the
                # Wiener filter itself and must be present in the MAP solve too. mu is an alm
                # vector, None unless the component sets `amp_prior_mean_map` (read as a sky map
                # and transformed once at startup by `load_amp_prior_mean`); the usual
                # zero-mean prior therefore costs one None check, not a full-size scale-and-add.
                amp_prior_mean = comp.amp_prior_mean
                if amp_prior_mean is not None:
                    comp.amps += comp.apply_Cl_prior_inv_sqrt(amp_prior_mean)
                if not self.group.optimize:
                    # + eta_2, the prior's own fluctuation. In the preconditioned variable it enters
                    # unmodified: substituting a = S^{1/2}x into the S^{-1/2}eta_2 term and
                    # left-multiplying by S^{1/2} leaves it alone.
                    for ipol in range(self.band_map.npol):
                        comp.amps[ipol] += gaussian_random_alm(comp.lmax, comp.lmax,
                                                               self.band_map.spin, 1, rng)[0]
                logger.debug(f"RHS comp-{comp.shortname}: {np.mean(np.abs(comp.amps)):.2e}")

        return RHS


    def solve(self, comp_list: CompList, rng: np.random.Generator) -> list[float]:
        """Solve the amplitudes of `comp_list` for this rank's band, in place, with the CG.

        The CG starts from the amplitudes that `comp_list` holds on entry, and leaves the solution
        in them. It stops at the sampling group's `err_tol`, or after `max_iter` iterations
        (`max_iter_pol` for QU). Returns the squared residual after each iteration; only the
        master's values are meaningful.

        Every rank of the solver's communicator must call this. Rank 0 (the master) runs the CG,
        and the other ranks apply their band's part of the operator.

        With `group.optimize` False (the default) the solution is a constrained realization, i.e.
        the Wiener mean plus two fluctuation terms, i.e. the amplitudes are a draw from the
        conditional posterior. With `group.optimize` True, only the mean term is used and the
        result is the deterministic maximum a posteriori solution, which is biased low in power.

        `rng` is this rank's random generator. Each rank must have its own stream: eta_1 is *this*
        band's white-noise realization, so shared draws would correlate the fluctuation across
        bands.
        """
        pol_name = "QU" if self.band_map.pol == "QU" else "Intensity"
        max_iter = self.group.max_iter_pol if self.band_map.pol == "QU" else self.group.max_iter

        # Set comp beams to 0 (in theory they could be non-zero if the per-pix solver has ran).
        for comp in comp_list:
            comp.amp_fwhm_rad = 0.0
        # `comp_list` is the solution vector of the CG: it starts as the guess and ends as the
        # solution. The CG works in x = S^{-1/2} a, so the guess is scaled here, and the solution
        # is scaled back at the end. Only the master's values are used; the other ranks need the
        # list as a buffer of the right shape.
        if self.is_master:
            for comp in comp_list:
                comp.apply_Cl_prior_inv_sqrt(comp.amps)

        with benchmark("cg-rhs"):
            RHS = self.calc_RHS(comp_list, rng)

        # Find the chosen preconditioner, as named in the parameter file.
        with benchmark("cg-precond-init"):
            precond = getattr(preconditioners, self.group.preconditioner)(
                self.solver_comm, self.band_map, comp_list)

        # The CG changes `comp_list` (its x) and `RHS` (its residual) in place.
        CG_solver = DistributedCG(self.apply_LHS_matrix, RHS, self.is_master, x0=comp_list,
                                  M=precond, destroy_b=True)
        if self.is_master:
            logger.verbose(f"{pol_name} CG starting up.")
        checkpt_int = 30
        residuals = []
        t0 = time.time()
        for iteration in range(1, max_iter + 1):
            # Counted per step, so the report's N column is the number of CG iterations taken.
            with benchmark("cg-iter"):
                CG_solver.step()
            log_memory("cg-iter")
            residuals.append(CG_solver.err)
            if iteration%checkpt_int == 0 and self.is_master:
                logger.verbose(f"{pol_name} CG iter {iteration:3d} - Squared residual "
                               f"{np.mean(residuals[-checkpt_int:]):.6e} "
                               f"({(time.time() - t0)/checkpt_int:.2f}s/iter)")
                t0 = time.time()
            # Only the master holds the error, so it decides for all ranks.
            if self.solver_comm.bcast(CG_solver.err <= self.group.err_tol, root=0):
                break
        if self.is_master:
            # With err_tol 0 the user asks for exactly max_iter iterations, so that is no warning.
            if self.group.err_tol > 0.0 and CG_solver.err > self.group.err_tol:
                logger.warning(f"Maximum number of iterations ({max_iter}) reached in CG.")
            logger.info(f"{pol_name} CG finished after {len(residuals)} iterations with a squared "
                        f"residual of {CG_solver.err:.3e} (err tol = {self.group.err_tol})")

        # a = S^{1/2} x on the master, which then sends the solution to the other ranks.
        for comp in comp_list:
            if self.is_master:
                comp.apply_Cl_prior_sqrt(comp.amps)
            comp.bcast_data(self.solver_comm)
        return residuals
