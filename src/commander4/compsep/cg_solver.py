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

MPI_LIMIT_32BIT = 2**31 - 1


class CompSepSolver:
    """ Class for performing global component separation using the preconditioned conjugate gradient
        method. After initializing the class, the solve() method should be called to perform the
        component separation. Note that the solve() method will in-place update (as well as return)
        the 'comp_list' argument passed to it.

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
    def __init__(self, band_map: DetectorMap, CompSep_comm: MPI.Comm, group: Bunch,
                 double_precision: bool, nthreads: int):
        """
        Args:
            band_map: This rank's band data (the conditional residual for this sampling group).
            CompSep_comm: The communicator of the ranks collaborating on this solve.
            group: The settings of this CG sampling group (`compsep.cg_sampling_groups.<name>`
                in the parameter file, with defaults filled in).
            double_precision: Whether maps and alms use double precision.
            nthreads: Number of computational threads used on this rank.
        """
        self.CompSep_comm = CompSep_comm
        self.is_master = CompSep_comm.Get_rank() == 0
        self.band_map = band_map
        self.config = group
        self.double_precision = double_precision
        self.my_band = Band.init_from_detector(band_map = band_map,
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
        self.npol = band_map.npol
        self.spin = band_map.spin

        self.nthreads = nthreads

        # Whether to add the two fluctuation terms to the RHS, or just optimize for MAP. Note the
        # inverted polarity: `optimize: false` (the default) draws a constrained realization.
        self.sample_amplitudes = not group.optimize
    

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
        # NB: For some stupid reason the non-blocking mpi4py calls do not have 64-bit length support
        # and are therefore limited to <2GB arrays... We have to fallback to blocking communication
        # for >2GB arrays. In the future we should probably implement chunking instead.

        with benchmark("lhs-bcast"):
            for comp in comp_list:
                comp.bcast_data_blocking(self.CompSep_comm)

        # B Y^-1 M Y S^{1/2} a. Each component adds itself to the emptied band: a diffuse component
        # to its alms, point sources to its pixel map.
        with benchmark("lhs-project"):
            self.my_band.alms[:] = 0
            if self.my_band.map is not None:
                self.my_band.map[:] = 0
            for comp in comp_list:
                comp.comp2band(self.my_band, nthreads=self.nthreads)

        # Y^T N^-1 Y B Y^-1 M Y S^{1/2} a. Point sources are added in pixel space, as in C3:
        # comp2band painted them into `band.map` (None without point sources), and they read
        # N^-1 (...) back from it. Diffuse components go through Y and Y^T.
        with benchmark("lhs-inv-N"):
            band, band_map = self.my_band, self.band_map
            pix_map = alm_to_map(band.alms, band_map.nside, band_map.lmax, spin=band_map.spin,
                                 nthreads=self.nthreads)
            if band.map is not None:
                pix_map += band.map
            band_map.apply_inv_N_map(pix_map)
            if band.map is not None:
                band.map = pix_map
            band.alms = alm_to_map_adjoint(pix_map, band_map.nside, band_map.lmax,
                                           spin=band_map.spin, nthreads=self.nthreads)

        # Y^T M^T Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y S^{1/2} a
        with benchmark("lhs-eval"):
            for comp in comp_list:
                comp.band2comp(self.my_band, nthreads=self.nthreads)

        # Accumulate solution on master
        with benchmark("lhs-accum"):
            biggest_size_bytes = np.max([comp.amps.nbytes for comp in comp_list])
            use_blocking = biggest_size_bytes > MPI_LIMIT_32BIT
            if use_blocking:
                logger.debug(f"Using blocking CompSep communication for array size "
                             f"{biggest_size_bytes:.2e} B.")
                for comp in comp_list:
                    comp.accum_data_blocking(self.CompSep_comm)
            else:
                requests = [comp.accum_data_non_blocking(self.CompSep_comm) for comp in comp_list]

            if self.is_master:
                for icomp in range(len(comp_list)):
                    # Since we used non-blocking reduce, master rank can start working on components
                    # as they are received instead of waiting for all to be received.
                    if not use_blocking:
                        # Wait until all data for component icomp has been received.
                        MPI.Request.Wait(requests[icomp])
                    # S^{1/2} Y^T M^T Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y S^{1/2} a
                    comp = comp_list[icomp]
                    comp.apply_Cl_prior_sqrt(comp.amps)
                    # The prior term adds the input vector: (1 + S^{1/2}...)a = a + (S^{1/2}...)a.
                    if comp.Cl_prior_amplitude is not None:
                        comp.amps += comp_list_in[icomp].amps
            else: # Worker ranks just wait for all their sends to complete.
                if not use_blocking:
                    for icomp in range(len(comp_list)):
                        MPI.Request.Wait(requests[icomp])
                comp_list = CompList([])

        return comp_list


    def calc_RHS(self, comp_list: CompList, rng: np.random.Generator) -> CompList:
        """ Calculates the right-hand-side b-vector of the Ax=b CompSep equation, and returns it as
            a new list with the same components as `comp_list`, which is not changed:

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

        # N^-1 d
        b_map = self.band_map.apply_inv_N_map(self.band_map.map_sky, inplace=False)
        if self.sample_amplitudes:
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
        b_alm = alm_to_map_adjoint(b_map, self.my_band.nside, self.my_band.lmax, spin=self.spin,
                                   nthreads=self.nthreads)
        b_band.alms = b_alm.astype(b_band.alms.dtype, copy=False)
        # Point sources read the pixel map itself, as in the LHS.
        b_band.map = b_map

        # (Y^T M^T Y^-1^T B^T) Y^T (...)
        RHS = comp_list.zeros_like()
        for comp in RHS:
            comp.band2comp(b_band, nthreads=self.nthreads)

        # Accumulate on master and apply the prior there: S^{1/2} Y^T M^T Y^-1^T B^T Y^T (...)
        for comp in RHS:
            comp.accum_data_blocking(self.CompSep_comm)
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
                if self.sample_amplitudes:
                    # + eta_2, the prior's own fluctuation. In the preconditioned variable it enters
                    # unmodified: substituting a = S^{1/2}x into the S^{-1/2}eta_2 term and
                    # left-multiplying by S^{1/2} leaves it alone.
                    for ipol in range(self.npol):
                        comp.amps[ipol] += gaussian_random_alm(comp.lmax, comp.lmax,
                                                               self.spin, 1, rng)[0]
                logger.debug(f"RHS comp-{comp.shortname}: {np.mean(np.abs(comp.amps)):.2e}")

        return RHS


    def solve(self, comp_list: CompList, rng: np.random.Generator) -> CompList:
        """Solve the amplitudes of `comp_list` for this rank's band, in place, with the CG.

        The CG starts from the amplitudes that `comp_list` holds on entry, and leaves the solution
        in them. The same list is returned. It stops at the sampling group's `err_tol`, or after
        `max_iter` iterations (`max_iter_pol` for QU). `self.CG_residuals` then holds the error
        after each iteration.

        Every rank of the solver's communicator must call this. Rank 0 (the master) runs the CG,
        and the other ranks apply their band's part of the operator.

        If `sample_amplitudes==True` the solution is a constrained realization, i.e the Wiener mean
        plus two fluctuation terms, i.e. the amplitudes are a draw from the conditional posterior.
        If `sample_amplitudes==False`, only the mean term is used and the result is the
        deterministic maximum a posteriori solution, which is biased low in power.

        `rng` is this rank's random generator. Each rank must have its own stream: eta_1 is *this*
        band's white-noise realization, so shared draws would correlate the fluctuation across
        bands.
        """
        pol_name = "QU" if self.band_map.pol == "QU" else "Intensity"
        max_iter = self.config.max_iter_pol if self.band_map.pol == "QU" else self.config.max_iter

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

        # Initialize the precondidioner class, which is in the module "solvers.preconditioners",
        # and has the name specified by this sampling group's preconditioner setting. It reads only
        # the settings of the components, never their amplitudes.
        with benchmark("cg-precond-init"):
            precond = getattr(preconditioners, self.config.preconditioner)(self, comp_list)

        # The CG changes `comp_list` (its x) and `RHS` (its residual) in place.
        CG_solver = DistributedCG(self.apply_LHS_matrix, RHS, self.is_master, x0=comp_list,
                                  M=precond, destroy_b=True)
        self.CG_residuals = np.zeros(max_iter)
        if self.is_master:
            logger.verbose(f"{pol_name} CG starting up.")
        checkpt_int = 30
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
            if iter%checkpt_int == 0 and self.is_master:
                logger.verbose(
                    f"{pol_name} CG iter {iter:3d} - "
                    f"Squared residual {np.mean(self.CG_residuals[iter-checkpt_int:iter]):.6e} "
                    f"({(time.time() - t0)/checkpt_int:.2f}s/iter)")
                t0 = time.time()
            if iter >= max_iter:
                # Only log warning if err_tol != 0, as this means user wants to do max_iter.
                if self.is_master and self.config.err_tol > 0.0:
                    logger.warning(f"Maximum number of iterations ({max_iter}) reached in CG.")
                stop_CG = True
            if CG_solver.err <= self.config.err_tol:
                stop_CG = True
            stop_CG = self.CompSep_comm.bcast(stop_CG, root=0)
        self.CG_residuals = self.CG_residuals[:iter]
        if self.is_master:
            logger.info(f"{pol_name} CG finished after {iter} iterations with a squared residual "
                        f"of {CG_solver.err:.3e} (err tol = {self.config.err_tol})")

        # a = S^{1/2} x on the master, which then sends the solution to the other ranks.
        for comp in comp_list:
            if self.is_master:
                comp.apply_Cl_prior_sqrt(comp.amps)
            comp.bcast_data_blocking(self.CompSep_comm)
        return comp_list
