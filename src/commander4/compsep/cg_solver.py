"""The CG amplitude solver: `CompSepSolver` builds and solves the component-amplitude system.
"""
from __future__ import annotations

import numpy as np
import time
import logging
from typing import TYPE_CHECKING
from mpi4py import MPI

from commander4.data_models.detector_map import DetectorMap
from commander4.sky.comp_list import CompList
from commander4.math_utils.alm import gaussian_random_alm
from commander4.math_utils.sht import alm_to_map, alm_to_map_adjoint
from commander4.math_utils.cg_driver import DistributedCG
from commander4.compsep import preconditioners
from commander4.data_models.band import Band
from commander4.diagnostics.performance import benchmark, log_memory

if TYPE_CHECKING:  # For the type hint only; a real import would be circular.
    from commander4.compsep.amplitude_sampling import AmplitudeGroup

logger = logging.getLogger(__name__)


class CompSepSolver:
    """ Class for performing global component separation using the preconditioned CG method.
    
    After initializing the class, the solve() method should be called to perform the component
    separation. solve() leaves the solution in the 'comp_list' argument passed to it, and returns
    the CG residuals.

    The component separation problem is an Hx = b equation on the form
        (S^-1 + A^T N^-1 A) a = A^T N^-1 d + A^T N^{-1/2} eta_1 + S^-1 mu + S^{-1/2} eta_2,
    where:
        - A = B M, and equivalently A^T = M^T B^T
        - a is the sky solution, mostly diffuse component alms, but also e.g. point sources.
        - B is the beam smoothing
        - M is the mixing matrix
        - N is the noise covariance matrix
        - mu is the mean spatial prior, typically 0
        - S is the power spectrum prior on each component (i.e. a prior on the variance around mu)

    For numerical stability, we rewrite the above equation to:
        (1 + S^{1/2} A^T N^-1 A S^{1/2}) a' = S^{1/2} A^T N^{-1} d
                                            + S^{1/2} A^T N^{-1/2} eta_1 + S^{-1/2} mu + eta_2
    which is solved for a' = [S^{-1/2} a], from which we can recover a = S^{1/2} a'.
    
    The eta_1 and eta_2 terms are random Gaussians. If `group.cfg.optimize=True` they are set to 0,
    which gives the maximum posterior (Wiener filtered) map solution, with a PS that is biased low.

    For components with no prior (S -> infinity) we do *not* rewrite the equation, and instead get:
        (A^T N^-1 A) a = A^T N^-1 d + A^T N^{-1/2} eta_1

    The solution, a, is in spherical harmonic (alm) space for all diffuse components, while some of
    the matrices (N^-1 and sometimes M) have to be applied in real-space. There are therefore
    several implicit spherical harmonic transformations in this equation, around these matrices.
    """
    def __init__(self, band_map: DetectorMap, group: AmplitudeGroup, nthreads: int):
        """
        Args:
            band_map: This rank's band data (the conditional residual for this sampling group).
            group: The config class holding the settings for this solve.
            nthreads: Number of computational threads used on this rank.
        """
        self.solver_comm = group.solver_comm
        self.is_master = self.solver_comm.Get_rank() == 0
        self.band_map = band_map
        self.group = group
        self.nthreads = nthreads


    def apply_LHS_matrix(self, comp_list_in: CompList) -> CompList:
        """Apply H, the left-hand side of the Hx = b system, to the vector `comp_list_in`.

        Every rank of the solver's communicator must call this, because each rank applies the
        part of H that belongs to its band. See class docstrings for details on the equation.

        Args:
            comp_list_in: The vector to apply H to. Only the master's amplitudes are used, and
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
            # a' = S^{1/2} [a']
            for comp in comp_list:
                comp.apply_Cl_prior_sqrt(comp.amps)

        # Spread initial a to all ranks from master.
        with benchmark("lhs-bcast"):
            for comp in comp_list:
                comp.bcast_amps(self.solver_comm)

        # B M [S^{1/2} a']
        # Each component adds itself to an empty band, in the components' precision: a diffuse
        # component to its alms, point sources to its pixel map.
        with benchmark("lhs-project"):
            band_alms = Band.init_from_detector(self.band_map,
                                                double_precision=comp_list[0].double_precision)
            for comp in comp_list:
                comp.comp2band(band_alms, nthreads=self.nthreads)

        # N^-1 [B M S^{1/2} a']
        with benchmark("lhs-inv-N"):
            band_map = self.band_map
            pix_map = alm_to_map(band_alms.alms, band_map.nside, band_map.lmax, spin=band_map.spin,
                                 nthreads=self.nthreads)
            if band_alms.map is not None:
                pix_map += band_alms.map
            band_map.apply_inv_N_map(pix_map)
            if band_alms.map is not None:
                band_alms.map = pix_map
            band_alms.alms = alm_to_map_adjoint(pix_map, band_map.nside, band_map.lmax,
                                                spin=band_map.spin, nthreads=self.nthreads)

        # M^T B^T [N^-1 B M S^{1/2} a']
        with benchmark("lhs-eval"):
            for comp in comp_list:
                comp.band2comp(band_alms, nthreads=self.nthreads)

        # Accumulate the result on the master. The reduces are non-blocking, so the master can work
        # on each component as soon as it has arrived, while later ones are still on their way.
        with benchmark("lhs-accum"):
            requests = [comp.ireduce_amps(self.solver_comm) for comp in comp_list]
            if self.is_master:
                for icomp in range(len(comp_list)):
                    requests[icomp].Wait()
                    # S^{1/2} [M^T B^T N^-1 B M S^{1/2} a']
                    comp = comp_list[icomp]
                    comp.apply_Cl_prior_sqrt(comp.amps)
                    # The +1 term is equivalent to simply adding the input a' vector to the result:
                    # (1 + S^{1/2}...)a' = a' + (S^{1/2}...)a'
                    if comp.Cl_prior_amplitude is not None:
                        comp.amps += comp_list_in[icomp].amps
            else: # Worker ranks just wait for all their sends to complete.
                MPI.Request.Waitall(requests)
                comp_list = CompList([])

        return comp_list


    @benchmark("cg-rhs")
    def calc_RHS(self, comp_list: CompList, rng: np.random.Generator) -> CompList:
        """ Calculates RHS of the Hx = b CompSep equation, and returns it as a new CompList.

        The RHS can be written as
            b = S^{1/2} A^T Y^T (N^-1 d + N^{-1/2} eta_1) + S^{-1/2} mu + eta_2,
        (explicitly including the Y^T SHT needed to get from map to alm space).
        """
        # N^-1 [d]
        b_map = self.band_map.apply_inv_N_map(self.band_map.map_sky, inplace=False)
        
        # + N^{-1/2} eta_1, unless we want the Wiener filtered solution.
        if not self.group.cfg.optimize:
            #  Drawn per band and independently on every rank, as it must be:
            # eta_1 is the white-noise realization of *this* band's data. Zero-weight pixels
            # (unobserved, map_inv_var = 0) get no fluctuation, matching their absence from the LHS.
            eta_1 = rng.normal(0.0, 1.0, b_map.shape)*np.sqrt(self.band_map.map_inv_var)
            logger.debug(f"RHS |N^-1 d| = {np.mean(np.abs(b_map)):.2e}, "
                         f"|N^-1/2 eta_1| = {np.mean(np.abs(eta_1)):.2e}")
            # The data and the eta_1 fluctuation are summed in *map* space, before the single shared
            # application of S^{1/2}A^T
            b_map += eta_1.astype(b_map.dtype, copy=False)

        # Y^T [N^-1 d + N^{-1/2} eta_1]
        b_alm = alm_to_map_adjoint(b_map, self.band_map.nside, self.band_map.lmax,
                                   spin=self.band_map.spin, nthreads=self.nthreads)
        b_band = Band.init_from_detector(self.band_map,
                                         double_precision=comp_list[0].double_precision)
        b_band.alms[:] = b_alm
        # Point sources read the pixel map itself, as in the LHS.
        b_band.map = b_map

        # M^T B^T [Y^T N^-1 d + N^{-1/2} eta_1]
        RHS = comp_list.zeros_like()
        for comp in RHS:
            comp.band2comp(b_band, nthreads=self.nthreads)

        # Accumulate on master and apply the prior there: S^{1/2} Y^T M^T Y^-1^T B^T Y^T (...)
        for comp in RHS:
            comp.reduce_amps(self.solver_comm)
            if self.is_master:
                comp.apply_Cl_prior_sqrt(comp.amps)

        if self.is_master:
            for comp in RHS:
                if comp.Cl_prior_amplitude is None:  # No prior: no prior mean and no prior draw.
                    continue

                # + S^{-1/2} mu, the prior mean, added whether or not we are sampling: unlike the
                # eta terms this belongs to the *mean* solution, so a non-zero prior mean shifts the
                # Wiener filter itself and must be present in the MAP solve too.
                amp_prior_mean = comp.amp_prior_mean
                if amp_prior_mean is not None:
                    comp.amps += comp.apply_Cl_prior_inv_sqrt(amp_prior_mean)

                #FIXME: how will eta_2 work for point sources?
                # + eta_2, the prior's own fluctuation, unless we want the Wiener filtered solution.
                if not self.group.cfg.optimize:
                    comp.amps += gaussian_random_alm(comp.lmax, comp.lmax, self.band_map.spin,
                                                     self.band_map.npol, rng)
                logger.debug(f"RHS comp-{comp.comp_name}: {np.mean(np.abs(comp.amps)):.2e}")

        return RHS


    def solve(self, comp_list: CompList, rng: np.random.Generator) -> list[float]:
        """Solve the amplitudes of `comp_list` for this rank's band, in place, with a CG.

        The CG starts from the amplitudes that `comp_list` holds on entry, and places the 
        solution in them. It stops at the sampling group's `err_tol`, or after `max_iter` iterations
        (`max_iter_pol` for QU). Returns the squared residual after each iteration; only the
        master's values are meaningful.

        Every rank of the solver's communicator must call this function. Rank 0 (the master) runs
        the CG, and the other ranks apply their band's part of the operator.
        """
        pol_name = self.band_map.pol
        max_iter = self.group.cfg.max_iter_pol if pol_name == "QU" else self.group.cfg.max_iter

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

        RHS = self.calc_RHS(comp_list, rng)

        # Find the chosen preconditioner, as named in the parameter file.
        with benchmark("cg-precond-init"):
            precond = getattr(preconditioners, self.group.cfg.preconditioner)(
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
            if self.solver_comm.bcast(CG_solver.err <= self.group.cfg.err_tol, root=0):
                break
        if self.is_master:
            # With err_tol 0 the user asks for exactly max_iter iterations, so that is no warning.
            if self.group.cfg.err_tol > 0.0 and CG_solver.err > self.group.cfg.err_tol:
                logger.warning(f"Maximum number of iterations ({max_iter}) reached in CG.")
            logger.info(f"{pol_name} CG finished after {len(residuals)} iterations with a squared "
                        f"residual of {CG_solver.err:.3e} (err tol = {self.group.cfg.err_tol})")

        # a = S^{1/2} x on the master, which then sends the solution to the other ranks.
        for comp in comp_list:
            if self.is_master:
                comp.apply_Cl_prior_sqrt(comp.amps)
            comp.bcast_amps(self.solver_comm)
        return residuals
