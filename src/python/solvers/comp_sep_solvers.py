import numpy as np
import healpy as hp
import time
from pixell import utils, curvedsky
from pixell.bunch import Bunch
import logging
from mpi4py import MPI
from numpy.typing import NDArray
from typing import Callable

from src.python.output.log import logassert
from src.python.model.component import CMB, ThermalDust, Synchrotron, DiffuseComponent
from src.python.utils.math_operations import alm_to_map, alm_to_map_adjoint, alm_real2complex, alm_complex2real
from src.python.solvers.dense_matrix_math import DenseMatrix
import src.python.solvers.preconditioners as preconditioners


def amplitude_sampling_per_pix(map_sky: NDArray, map_rms: NDArray, freqs: NDArray) -> NDArray:
    logger = logging.getLogger(__name__)
    ncomp = 3
    nband, npix = map_sky.shape
    comp_maps = np.zeros((ncomp, npix))
    M = np.empty((nband, ncomp))
    M[:,0] = CMB().get_sed(freqs)
    M[:,1] = ThermalDust().get_sed(freqs)
    M[:,2] = Synchrotron().get_sed(freqs)
    from time import time
    t0 = time()
    rand = np.random.randn(npix,nband)
    logger.info(f"time for random numbers: {time()-t0}s.")
    t0 = time()
    for i in range(npix):
        xmap = 1/map_rms[:,i]
        x = M.T.dot((xmap**2*map_sky[:,i]))
        x += M.T.dot(rand[i]*xmap)
        A = (M.T.dot(np.diag(xmap**2)).dot(M))
        try:
            comp_maps[:,i] = np.linalg.solve(A, x)
        except np.linalg.LinAlgError:
            comp_maps[:,i] = 0
    logger.info(f"Time for Python solution: {time()-t0}s.")
    # import cmdr4_support
    # t0 = time()
    # comp_maps2 = cmdr4_support.utils.amplitude_sampling_per_pix_helper(map_sky, map_rms, M, rand, nnthreads=1)
    # logger.info(f"Time for native solution: {time()-t0}s.")
    # import ducc0
    # logger.info(f"L2 error between solutions: {ducc0.misc.l2error(comp_maps, comp_maps2)}.")
    return comp_maps




class CompSepSolver:
    """ Class for performing global component separation using the conjugate gradient method.
        After initializing the class, the solve() method should be called to perform the component separation.
        Note that the solve() method will in-place update (as well as return) the 'comp_list' argument passed to the constructor.
    """
    def __init__(self, comp_list: list[DiffuseComponent], map_sky: NDArray, map_rms: NDArray, freq: float, fwhm: float, params: Bunch, CompSep_comm: MPI.Comm):
        self.logger = logging.getLogger(__name__)
        self.CompSep_comm = CompSep_comm
        self.params = params
        self.map_sky = map_sky
        self.map_rms = map_rms
        self.freqs = np.array(CompSep_comm.allgather(freq))
        self.npix = map_rms.shape[0]
        self.nband = len(self.freqs)
        self.my_rank = CompSep_comm.Get_rank()
        self.nside = np.sqrt(self.npix//12)
        logassert(self.nside.is_integer(), f"Npix dimension of map ({self.npix}) resulting in a non-integer nside ({self.nside}).", self.logger)
        self.nside = int(self.nside)
        self.lmax_bands = (self.nside*5)//2  # Slightly higher than 2*NSIDE to avoid accumulation of numeric junk.
        self.alm_len_bands = ((self.lmax_bands+1)*(self.lmax_bands+2))//2
        self.comp_list = comp_list
        self.comps_SED = np.array([comp.get_sed(self.freqs) for comp in comp_list])
        self.ncomp = len(self.comps_SED)
        self.is_holding_comp = self.my_rank < self.ncomp
        self.lmax_per_comp = np.array([comp.lmax for comp in comp_list])
        # self.alm_len_percomp = np.array([((lmax+1)*(lmax+2))//2 for lmax in self.lmax_per_comp])
        self.alm_len_percomp = np.array([(lmax+1)**2 for lmax in self.lmax_per_comp])
        self.my_band_fwhm_rad = fwhm/60.0*(np.pi/180.0)
        if self.is_holding_comp:
            self.my_comp_lmax = comp_list[self.my_rank].lmax
            self.my_comp_alm_len = self.alm_len_percomp[self.my_rank] #((self.my_comp_lmax+1)*(self.my_comp_lmax+2))//2
            color = 0
        else:
            self.my_comp_lmax = 0
            self.my_comp_alm_len = 0
            color = MPI.UNDEFINED  # If we are not holding a component, we will not be part of the component communicator.
        self.CompSep_holdingcomp_comm = self.CompSep_comm.Split(color, key=self.my_rank)  # Split off a new communicator for ranks holding components.



    def apply_LHS_matrix(self, a_array: NDArray) -> NDArray:
        """ Applies the A matrix to inputed component alms a, where A represents the entire LHS of the Ax=b system for global component separation.
            The full A matrix can be written B^T Y^T M^T N^-1 M Y B, where B is the beam smoothing, M is the mixing matrix, and N is the noise covariance matrix.

            Args:
                a_array: the a_lm of the component residing on this task (may be zero-sized).
            Returns:
                Aa: the result of A(a_array) of the component residing on this task (may be zero-sized).
        """
        logger = logging.getLogger(__name__)

        logassert(a_array.dtype == np.float64, f"Provided component array is of type {a_array.dtype} and not np.float64. This operator takes and returns real alms (and converts to and from complex interally).", logger)
        logassert(a_array.shape[0] == self.my_comp_alm_len, f"Provided component array is of length {a_array.shape[0]}, not {self.my_comp_alm_len}.", logger)

        mycomp = self.CompSep_comm.Get_rank()
        mythreads = self.params.nthreads_compsep

        if mycomp < self.ncomp:  # this task actually holds a component
            a_array = alm_real2complex(a_array, self.my_comp_lmax) # Convert the real input alms to complex alms.
            # Y a
            a = alm_to_map(a_array, self.nside, self.my_comp_lmax, nthreads=mythreads)
        else:
            a = None

        # M Y a
        a_old = a
        a = np.zeros(self.npix)
        for icomp in range(self.ncomp):
            # successively broadcast a_old and build a from it
            tmp = a_old if mycomp == icomp else np.empty(self.npix)
            self.CompSep_comm.Bcast(tmp, root=icomp)
            a += self.comps_SED[icomp,self.my_rank]*tmp
        del tmp

        # Y^-1 M Y a
        a_old = a
        a = np.empty((self.alm_len_bands,), dtype=np.complex128)
        curvedsky.map2alm_healpix(a_old, a, niter=3, spin=0, nthread=mythreads)
        del a_old

        # B Y^-1 M Y a
        hp.smoothalm(a, self.my_band_fwhm_rad, inplace=True)

        # Y B Y^-1 M Y a
        a = alm_to_map(a, self.nside, self.lmax_bands, nthreads=mythreads)

        # N^-1 Y B Y^-1 M Y a
        a /= self.map_rms**2

        # Y^T N^-1 Y B Y^-1 M Y a
        a = alm_to_map_adjoint(a, self.nside, self.lmax_bands, nthreads=mythreads)

        # B^T Y^T N^-1 Y B Y^-1 M Y a
        hp.smoothalm(a, self.my_band_fwhm_rad, inplace=True)

        # Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y a
        a_old = a
        a = np.empty((self.npix,))
        curvedsky.map2alm_healpix(a, a_old, niter=3, adjoint=True, spin=0, nthread=mythreads)

        # M^T Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y a
        a_old = a
        for icomp in range(self.ncomp):
            tmp = a_old * self.comps_SED[icomp,self.my_rank]
            # accumulate tmp onto the relevant task
            send, recv = (MPI.IN_PLACE, tmp) if icomp == mycomp else (tmp, None)
            self.CompSep_comm.Reduce(send, recv, op=MPI.SUM, root=icomp)
            if icomp == mycomp:
                a = tmp
        del a_old

        if mycomp < self.ncomp:
            # Y^T M^T Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y a
            a = alm_to_map_adjoint(a, self.nside, self.my_comp_lmax, nthreads=mythreads)
 
            a = alm_complex2real(a, self.my_comp_lmax) # Convert complex alm back to real before returning.
        else:
            a = np.zeros((0,), dtype=np.float64)  # zero-sized array

        return a


    def solve_CG(self, LHS: Callable, RHS: NDArray, x0: NDArray, M = None, x_true = None) -> NDArray:
        """ Solves the equation Ax=b for x given A (LHS) and b (RHS) using CG from the pixell package.
            Assumes that both x and b are in alm space.

            Args:
                LHS (callable): A function/callable taking x as argument and returning Ax.
                RHS (np.array): A Numpy array representing b, in alm space.
                x0 (np.array): Initial guess for x.
                M (callable): Preconditioner for the CG solver. A function/callable which approximates A^-1 (optional).
                x_true (np.array): True solution for x, in order to print the true error (optional, used for testing).
            Returns:
                m_bestfit (np.array): The resulting best-fit solution, in alm space, for the component held by this rank.
                                      A zero-sized array for ranks holding no component.
        """
        logger = logging.getLogger(__name__)
        checkpoint_interval = 10
        master = self.CompSep_comm.Get_rank() == 0
        mycomp = self.CompSep_comm.Get_rank()

        mydot = lambda a,b: self._calc_dot(a,b)
        if M is None:
            CG_solver = utils.CG(LHS, RHS, dot=mydot, x0=x0)
        else:
            CG_solver = utils.CG(LHS, RHS, dot=mydot, x0=x0, M=M)
        self.CG_residuals = np.zeros((self.params.CG_max_iter))
        if x_true is not None:
            # self.x_true_allcomps = self.CompSep_comm.allgather()
            self.xtrue_A_xtrue = x_true.dot(LHS(x_true))  # The normalization factor for the true error (contribution from my rank).
            self.xtrue_A_xtrue = self.CompSep_comm.allreduce(self.xtrue_A_xtrue, MPI.SUM)  # Dot product is linear, so we can just sum the contributions.
        if master:
            logger.info("CG starting up!")
        iter = 0
        t0 = time.time()
        stop_CG = False
        while not stop_CG:
            CG_solver.step()
            self.CG_residuals[iter] = CG_solver.err
            iter += 1
            if iter%checkpoint_interval == 0:
                if mycomp < self.ncomp:
                    if master:
                        logger.info(f"CG iter {iter:3d} - Residual {np.mean(self.CG_residuals[iter-10:iter]):.3e} ({(time.time() - t0)/10.0:.1f}s/iter)")
                    if x_true is not None:
                        CG_errors_true = np.linalg.norm(CG_solver.x-x_true)/np.linalg.norm(x_true)
                        CG_Anorm_error = (CG_solver.x-x_true).dot(LHS(CG_solver.x-x_true))
                        CG_Anorm_error = self.CompSep_holdingcomp_comm.allreduce(CG_Anorm_error, MPI.SUM)/self.xtrue_A_xtrue  # Collect error contributions from all ranks.
                        time.sleep(0.01*mycomp)  # Getting the prints in the same order every time.
                        if master:
                            logger.info(f"CG iter {iter:3d} - True A-norm error: {CG_Anorm_error:.3e}")  # A-norm error is only defined for the full vector.
                        logger.info(f"CG iter {iter:3d} - {self.comp_list[mycomp].longname} - True L2 error: {CG_errors_true:.3e}")  # We can print the individual component L2 errors.
                    t0 = time.time()
                else:
                    if x_true is not None:
                        LHS(np.zeros((0,), dtype=np.float64))  # Matching LHS call for the calculation of LHS(CG_solver.x-x_true).
            if iter >= self.params.CG_max_iter:
                if master:
                    logger.warning(f"Maximum number of iterations ({self.params.CG_max_iter}) reached in CG.")
                stop_CG = True
            if CG_solver.err < self.params.CG_err_tol:
                stop_CG = True
            stop_CG = self.CompSep_comm.bcast(stop_CG, root=0)
        self.CG_residuals = self.CG_residuals[:iter]
        if master:
            logger.info(f"CG finished after {iter} iterations with a residual of {CG_solver.err:.3e} (err tol = {self.params.CG_err_tol})")
        if self.is_holding_comp:
            s_bestfit = CG_solver.x
            s_bestfit = alm_real2complex(s_bestfit, self.my_comp_lmax)  # CG search uses real-valued alms, convert to complex, which is used outside CG.
        else:
            s_bestfit = np.zeros((0,), dtype=np.complex128)
        return s_bestfit


    def _calc_dot(self, a: NDArray, b: NDArray):
        """ Calculates the dot product of two sets of real a_lms which are distributed across the
            self.ncomp first ranks of the self.CompSep_comm communicator.
        """
        res = 0.
        if self.is_holding_comp:
            res = np.dot(a, b)
        res = self.CompSep_comm.allreduce(res, op=MPI.SUM)
        return res


    def _calc_dot_complex(self, a: NDArray, b: NDArray):
        """ Calculates the dot product of two sets of complex a_lms which are distributed across the
            self.ncomp first ranks of the self.CompSep_comm communicator.
            NB: This function is currently NOT USED, as we transitioned to using real alms in the CG.
        """
        res = 0.
        if self.is_holding_comp:
            lmax = self.my_comp_lmax
            res = np.dot(a[0:lmax+1].real, b[0:lmax+1].real)
            res += 2 * np.dot(a[lmax+1:].real, b[lmax+1:].real)
            res += 2 * np.dot(a[lmax+1:].imag, b[lmax+1:].imag)
        res = self.CompSep_comm.allreduce(res, op=MPI.SUM)
        return res


    def _calc_RHS_from_input_array(self, b: NDArray) -> NDArray:
        """ Applies the matrices Y^T M^T Y^-1^T B^T to a vector b, which is the terms in common
            for b_mean and b_fluct of the RHS of the CompSep Ax=b equation.
        """
        mycomp = self.CompSep_comm.Get_rank()
        mythreads = self.params.nthreads_compsep

        # Y^T N^-1 d
        b = alm_to_map_adjoint(b, self.nside, self.lmax_bands, nthreads=mythreads)

        # B^T Y^T N^-1 d
        hp.smoothalm(b, self.my_band_fwhm_rad, inplace=True)

        # Y^-1^T B^T Y^T N^-1 d
        b_old = b
        b = np.empty((self.npix,))
        curvedsky.map2alm_healpix(b, b_old, adjoint=True, niter=3, spin=0, nthread=self.params.nthreads_compsep)

        # M^T Y^-1^T B^T Y^T N^-1 d
        b_old = b
        for icomp in range(self.ncomp):
            tmp = self.comps_SED[icomp,self.my_rank]*b_old
            send, recv = (MPI.IN_PLACE, tmp) if icomp == mycomp else (tmp, None)
            self.CompSep_comm.Reduce(send, recv, op=MPI.SUM, root=icomp)
            if icomp == mycomp:
                b = tmp

        # Y^T M^T Y^-1^T B^T Y^T N^-1 d
        if self.is_holding_comp:  # This task actually holds a component
            b = alm_to_map_adjoint(b, self.nside, self.my_comp_lmax, nthreads=mythreads)
            b = alm_complex2real(b, self.my_comp_lmax)
        else:
            b = np.zeros((0,), dtype=np.float64)

        return b


    def calc_RHS_mean(self) -> NDArray:
        """ Caculates the right-hand-side b-vector of the Ax=b CompSep equation for the Wiener filtered (or mean-field) solution.
            If used alone on the right-hand-side, gives the deterministic maximum likelihood map-space solution, but a biased PS solution.
        """
        # N^-1 b_mean
        b_mean = self.map_sky/self.map_rms**2

        return self._calc_RHS_from_input_array(b_mean)


    def calc_RHS_fluct(self) -> NDArray:
        """ Calculates the right-hand-side fluctuation vector. Provides unbiased realizations (of foregrounds or the CMB) if added
            together with the right-hand-side of the Wiener filtered solution : Ax = b_mean + b_fluct.
        """
        # b_fluct
        b_fluct = np.random.normal(0.0, 1.0, self.map_rms.shape)

        # N^-1/2 b_fluct
        b_fluct /= self.map_rms

        return self._calc_RHS_from_input_array(b_fluct)


    def solve(self, seed=None) -> list[DiffuseComponent]:
        mycomp = self.CompSep_comm.Get_rank()

        RHS = self.calc_RHS_mean() + self.calc_RHS_fluct()
        # Initialize the precondidioner class, which is in the module "solvers.preconditioners", and has a name specified by self.params.compsep.preconditioner.
        precond = getattr(preconditioners, self.params.compsep.preconditioner)(self)

        if self.params.compsep.dense_matrix_debug_mode:  # For testing preconditioner with a true solution as reference, first solving for exact solution with dense matrix math.
            M_A_matrix = lambda a : self.apply_LHS_matrix(precond(a))

            # Testing the initial LHS (A) matrix
            dense_matrix = DenseMatrix(self.CompSep_comm, self.apply_LHS_matrix, self.alm_len_percomp, matrix_name="A")
            x_true = dense_matrix.solve_by_inversion(RHS)
            dense_matrix.test_matrix_hermitian()
            dense_matrix.print_sing_vals()
            dense_matrix.test_matrix_eigenvalues()
            dense_matrix.print_matrix_diag()

            # Testing the preconditioning matrix (M) alone
            dense_matrix = DenseMatrix(self.CompSep_comm, precond, self.alm_len_percomp, matrix_name="M")
            dense_matrix.test_matrix_hermitian()  # Preconditioner matrix needs to be Hermitian.
            dense_matrix.test_matrix_eigenvalues()  # and to have positive and real eigenvalues

            # Testing the combined preconditioned system (MA)  (this combined matrix will generally not be Hermitian positive-definite).
            dense_matrix = DenseMatrix(self.CompSep_comm, M_A_matrix, self.alm_len_percomp, matrix_name="MA")
            dense_matrix.print_sing_vals()  # Check how much singular values (condition number) of preconditioned system improved.
            dense_matrix.print_matrix_diag()


        if seed is not None:
            np.random.seed(seed)
        if mycomp < self.ncomp:
            x0 = np.zeros((self.alm_len_percomp[mycomp],))
        else:
            x0 = np.zeros((0,), dtype=np.float64)
        sol_array = self.solve_CG(self.apply_LHS_matrix, RHS, x0, M=precond, x_true=x_true if self.params.compsep.dense_matrix_debug_mode else None)
        for icomp in range(self.ncomp):
            tmp = self.CompSep_comm.bcast(sol_array, root=icomp)
            self.comp_list[icomp].component_alms = tmp

        return self.comp_list
