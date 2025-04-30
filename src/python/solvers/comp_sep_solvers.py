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
from src.python.output.plotting import alm_plotter
from src.python.model.component import CMB, ThermalDust, Synchrotron, Component
from src.python.utils.math_operations import alm_to_map, alm_to_map_adjoint
from src.python.solvers.dense_matrix_math import DenseMatrix
import src.python.solvers.preconditioners as preconditioners


def amplitude_sampling_per_pix(map_sky: np.array, map_rms: np.array, freqs: np.array) -> np.array:
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
    def __init__(self, comp_list: list[Component], map_sky: NDArray, map_rms: NDArray, freq: float, fwhm: float, params: Bunch, CompSep_comm: MPI.Comm):
        self.logger = logging.getLogger(__name__)
        self.CompSep_comm = CompSep_comm
        self.params = params
        self.map_sky = map_sky
        self.map_rms = map_rms
        self.freqs = np.array(CompSep_comm.allgather(freq))
        self.npix = map_rms.shape[0]
        self.nband = len(self.freqs)
        self.my_band_idx = CompSep_comm.Get_rank()
        self.nside = np.sqrt(self.npix//12)
        logassert(self.nside.is_integer(), f"Npix dimension of map ({self.npix}) resulting in a non-integer nside ({self.nside}).", self.logger)
        self.nside = int(self.nside)
        self.lmax = 3*self.nside-1
        self.alm_len_complex = ((self.lmax+1)*(self.lmax+2))//2
        self.alm_len_real = (self.lmax+1)**2
        self.comp_list = comp_list
        self.comps_SED = np.array([comp.get_sed(self.freqs) for comp in comp_list])
        self.ncomp = len(self.comps_SED)
        self.lmax_per_comp = np.array([comp.lmax for comp in comp_list])
        self.alm_len_complex_percomp = np.array([((lmax+1)*(lmax+2))//2 for lmax in self.lmax_per_comp])
        self.alm_len_real_percomp = np.array([(lmax+1)**2 for lmax in self.lmax_per_comp])
        self.fwhm_rad = fwhm/60.0*(np.pi/180.0)
        self.fwhm_rad_allbands = np.array(CompSep_comm.allgather(fwhm))


    def alm_complex2real(self, alm: NDArray[np.complex128], lmax: int) -> NDArray[np.float64]:
        """ Coverts from the complex convention of storing alms when the map is real, to the real convention.
            In the real convention, the all m modes are stored, but they are all stored as real values, not complex.
            Args:
                alm (np.array): Complex alm array of length ((lmax+1)*(lmax+2))/2.
                lmax (int): The lmax of the alm array.
            Returns:
                x (np.array): Real alm array of length (lmax+1)^2.
        """
        ainfo = curvedsky.alm_info(lmax=lmax)
        i = int(ainfo.mstart[1]+1)
        return np.concatenate([alm[:i].real,np.sqrt(2.)*alm[i:].view(np.float64)])


    def alm_real2complex(self, x: NDArray[np.float64], lmax: int) -> NDArray[np.complex128]:
        """ Coverts from the real convention of storing alms when the map is real, to the complex convention.
            In the complex convention, the only m>=0 is stored, but are stored as complex numbers (m=0 still always real).
            Args:
                x (np.array): Real alm array of length (lmax+1)^2.
                lmax (int): The lmax of the alm array.
            Returns:
                oalm (np.array): Complex alm array of length ((lmax+1)*(lmax+2))/2.
        """
        ainfo = curvedsky.alm_info(lmax=lmax)
        i    = int(ainfo.mstart[1]+1)
        oalm = np.zeros(ainfo.nelem, np.complex128)
        oalm[:i] = x[:i]
        oalm[i:] = x[i:].view(np.complex128)/np.sqrt(2.)
        return oalm


    def apply_LHS_matrix(self, a_array: NDArray) -> NDArray:
        """ Applies the A matrix to inputed component alms a, where A represents the entire LHS of the Ax=b system for global component separation.
            The full A matrix can be written B^T Y^T M^T N^-1 M Y B, where B is the beam smoothing, M is the mixing matrix, and N is the noise covariance matrix.

            Args:
                a: (ncomp*alm_len_real,) array containing real, flattened alms of each component.
            Returns:
                Aa: (nband*alm_len_rea,) array from applying the full A matrix to a.
        """
        logger = logging.getLogger(__name__)

        a_array = self.CompSep_comm.bcast(a_array, root=0)  # Send a to all worker ranks (which are called with a dummy a).
        logassert(a_array.dtype == np.float64, "Provided component array is not of type np.float64. This operator takes and returns real alms (and converts to and from complex interally).", logger)

# MR: General idea of the changes:
# I try to never hold lists or arrays of a_lm/maps; it should be sufficient to
# work on a single one at any time.
# For function input/output this currently does not work; for that we need
# to change the interface, but let's first see how this works out.
# I also try to replace sum reductions with only a single contributor
# by broadcasts, which should be more efficient.
# (Currently these are lowercase "bcast"s, so they may be slow, but we can fix that.)

# shorthands
        mycomp = self.CompSep_comm.Get_rank()
        mythreads = self.params.nthreads_compsep

# split the input; we only need "our" component
        if mycomp < self.ncomp:  # this task actually holds a component
            idx_start = sum(self.alm_len_real_percomp[:mycomp])
            idx_stop = idx_start + self.alm_len_real_percomp[mycomp]
            # directly convert to complex a_lm
            a = self.alm_real2complex(a_array[idx_start:idx_stop],
                                      lmax=self.lmax_per_comp[mycomp])

            # Y a
            a = alm_to_map(a, self.nside, self.lmax_per_comp[mycomp], nthreads=mythreads)
        else:
            a = None

        # M Y a
        a_old = a
        a = np.zeros(self.npix)
        for icomp in range(self.ncomp):
            # successively broadcast a_old and build a from it
            tmp = a_old if mycomp == icomp else np.empty(self.npix)
            self.CompSep_comm.Bcast(tmp, root=icomp)
            a += self.comps_SED[icomp,self.my_band_idx]*tmp
        del tmp

        # Y^-1 M Y a
        a_old = a
        a = np.empty((self.alm_len_complex,), dtype=np.complex128)
        curvedsky.map2alm_healpix(a_old, a, niter=3, spin=0, nthread=mythreads)
        del a_old

        # B Y^-1 M Y a
        hp.smoothalm(a, self.fwhm_rad, inplace=True)

        # Y B Y^-1 M Y a
        a = alm_to_map(a, self.nside, self.lmax, nthreads=mythreads)

        # N^-1 Y B Y^-1 M Y a
        a /= self.map_rms**2

        # Y^T N^-1 Y B Y^-1 M Y a
        a = alm_to_map_adjoint(a, self.nside, self.lmax, nthreads=mythreads)

        # B^T Y^T N^-1 Y B Y^-1 M Y a
        hp.smoothalm(a, self.fwhm_rad, inplace=True)

        # Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y a
        a_old = a
        a = np.empty((self.npix,))
        curvedsky.map2alm_healpix(a, a_old, niter=3, adjoint=True, spin=0, nthread=mythreads)

        # M^T Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y a
        a_old = a
        for icomp in range(self.ncomp):
            tmp = a_old * self.comps_SED[icomp,self.my_band_idx]
            # accumulate tmp onto the relevant task
            send, recv = (MPI.IN_PLACE, tmp) if icomp == mycomp else (tmp, None)
            self.CompSep_comm.Reduce(send, recv, op=MPI.SUM, root=icomp)
            if icomp == mycomp:
                a = tmp
        del a_old

        # Y^T M^T Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y a
        if mycomp < self.ncomp:
            a = alm_to_map_adjoint(a, self.nside, self.lmax_per_comp[mycomp], nthreads=mythreads)

            # Converting back from complex alms to real alms
            a = self.alm_complex2real(a, lmax=self.lmax_per_comp[mycomp])
        else:
            a = None

        # For now, every task holds every a_lm, so let's gather them together
        a_old = a
        a=np.empty(a_array.shape)
        idx_start = 0
        for icomp in range(self.ncomp):
            idx_stop = idx_start + self.alm_len_real_percomp[icomp]
            if icomp == mycomp:
                a[idx_start:idx_stop] = a_old
            self.CompSep_comm.Bcast(a[idx_start:idx_stop], root=icomp)
            idx_start = idx_stop

        return a#.flatten()


    def solve_CG(self, LHS: Callable, RHS: NDArray, x0: NDArray, M = None, x_true = None) -> NDArray|None:
        """ Solves the equation Ax=b for x given A (LHS) and b (RHS) using CG from the pixell package.
            Assumes that both x and b are in alm space.

            Args:
                LHS (callable): A function/callable taking x as argument and returning Ax.
                RHS (np.array): A Numpy array representing b, in alm space.
                x0 (np.array): Initial guess for x.
                M (callable): Preconditioner for the CG solver. A function/callable which approximates A^-1 (optional).
                x_true (np.array): True solution for x, in order to print the true error (optional, used for testing).
            Returns:
                if self.CompSep_comm.Get_rank() == 0:
                    m_bestfit (np.array): The resulting best-fit solution, in alm space.
                else:
                    None
        """
        logger = logging.getLogger(__name__)
        checkpoint_interval = 10
        if self.CompSep_comm.Get_rank() == 0:
            if M is None:
                CG_solver = utils.CG(LHS, RHS, x0=x0)
            else:
                CG_solver = utils.CG(LHS, RHS, x0=x0, M=M)
            self.CG_residuals = np.zeros((self.params.CG_max_iter))
            if not x_true is None:
                self.CG_errors_true = np.zeros((self.params.CG_max_iter//checkpoint_interval))
                self.CG_Anorm_error = np.zeros((self.params.CG_max_iter//checkpoint_interval))
                self.xtrue_A_xtrue = x_true.dot(LHS(x_true))  # The normalization factor for the true error.
            logger.info(f"CG starting up!")
            iter = 0
            t0 = time.time()
            stop_CG = False
            while not stop_CG:
                CG_solver.step()
                self.CG_residuals[iter] = CG_solver.err
                iter += 1
                if iter%checkpoint_interval == 0:
                    logger.info(f"CG iter {iter:3d} - Residual {np.mean(self.CG_residuals[iter-10:iter]):.3e} ({(time.time() - t0)/10.0:.1f}s/iter)")
                    if not x_true is None:
                        self.CG_errors_true[iter//checkpoint_interval-1] = np.linalg.norm(CG_solver.x-x_true)/np.linalg.norm(x_true)
                        self.CG_Anorm_error[iter//checkpoint_interval-1] = (CG_solver.x-x_true).dot(LHS(CG_solver.x-x_true))/self.xtrue_A_xtrue
                        logger.info(f"True error: {self.CG_errors_true[iter//checkpoint_interval-1]:.3e} - Anorm error: {self.CG_Anorm_error[iter//checkpoint_interval-1]:.3e}")
                    t0 = time.time()
                if iter >= self.params.CG_max_iter:
                    logger.warning(f"Maximum number of iterations ({self.params.CG_max_iter}) reached in CG.")
                    stop_CG = True
                if CG_solver.err < self.params.CG_err_tol:
                    stop_CG = True
                stop_CG = self.CompSep_comm.bcast(stop_CG, root=0)
            self.CG_residuals = self.CG_residuals[:iter]
            logger.info(f"CG finished after {iter} iterations with a residual of {CG_solver.err:.3e} (err tol = {self.params.CG_err_tol})")
            s_bestfit = CG_solver.x
            return s_bestfit
        else:
            LHS(None)  # Calling the LHS operator because the initialization of the CG driver will call it.
            if not x_true is None:
                LHS(None)  # Second call, for the xtrue_A_xtrue calculation.
            stop_CG = False
            iter = 0
            while not stop_CG:
                LHS(None)
                iter += 1
                if not x_true is None and iter%checkpoint_interval == 0:  # Every nth iteration, if we have a true solution, we do an extra LHS calculation in order to determine the true error.
                    LHS(None)
                stop_CG = self.CompSep_comm.bcast(stop_CG, root=0)


    def _calc_RHS_from_input_array(self, b: NDArray) -> NDArray:
# shorthands
        mycomp = self.CompSep_comm.Get_rank()
        mythreads = self.params.nthreads_compsep

        # Y^T N^-1 d
        b = alm_to_map_adjoint(b, self.nside, self.lmax, nthreads=mythreads)

        # B^T Y^T N^-1 d
        hp.smoothalm(b, self.fwhm_rad, inplace=True)

        # Y^-1^T B^T Y^T N^-1 d
        b_old = b
        b = np.empty((self.npix,))
        curvedsky.map2alm_healpix(b, b_old, adjoint=True, niter=3, spin=0, nthread=self.params.nthreads_compsep)

        # M^T Y^-1^T B^T Y^T N^-1 d
        b_old = b
        for icomp in range(self.ncomp):
            tmp = self.comps_SED[icomp,self.my_band_idx]*b_old
            send, recv = (MPI.IN_PLACE, tmp) if icomp == mycomp else (tmp, None)
            self.CompSep_comm.Reduce(send, recv, op=MPI.SUM, root=icomp)
            if icomp == mycomp:
                b = tmp

        # Y^T M^T Y^-1^T B^T Y^T N^-1 d
        if mycomp < self.ncomp:  # This task actually holds a component
            b = alm_to_map_adjoint(b, self.nside, self.lmax_per_comp[mycomp], nthreads=mythreads)

            # complex to real
            b = self.alm_complex2real(b, lmax=self.lmax_per_comp[mycomp])
        else:
            b = None

        # For now, every task holds every a_lm, so let's gather them together
        b_old = b
        b = np.empty(sum(self.alm_len_real_percomp))
        idx_start = 0
        for icomp in range(self.ncomp):
            idx_stop = idx_start + self.alm_len_real_percomp[icomp]
            if icomp == mycomp:
                b[idx_start:idx_stop] = b_old
            self.CompSep_comm.Bcast(b[idx_start:idx_stop], root=icomp)
            idx_start = idx_stop

        return b


    def calc_RHS_mean(self) -> NDArray:
        # N^-1 d
        b = self.map_sky/self.map_rms**2

        return self._calc_RHS_from_input_array(b)


    def calc_RHS_fluct(self) -> NDArray:
        # d
        b = np.random.normal(0.0, 1.0, self.map_rms.shape)

        # N^-1 d
        b /= self.map_rms

        return self._calc_RHS_from_input_array(b)


    def solve(self, seed=None) -> list[Component]:

        RHS = self.calc_RHS_mean() + self.calc_RHS_fluct()
        debug_mode = self.params.compsep.dense_matrix_debug_mode

        # Initialize the precondidioner class, which is in the module "solvers.preconditioners", and has a name specified by self.params.compsep.preconditioner.
        precond = getattr(preconditioners, self.params.compsep.preconditioner)(self)

        if debug_mode:  # For testing preconditioner with a true solution as reference, first solving for exact solution with dense matrix math.
            dense_matrix = DenseMatrix(self.CompSep_comm, self.apply_LHS_matrix, np.sum(self.alm_len_real_percomp))
            x_true = None
            if self.CompSep_comm.Get_rank() == 0:
                x_true = dense_matrix.solve_by_inversion(RHS)
            x_true = self.CompSep_comm.bcast(x_true, root=0)
            if self.CompSep_comm.Get_rank() == 0:
                sing_vals = dense_matrix.get_sing_vals()
                self.logger.info(f"Condition number of regular (A) matrix: {sing_vals[0]/sing_vals[-1]:.3e}")
                self.logger.info(f"Sing-vals: {sing_vals[0]:.1e} .. {sing_vals[sing_vals.size//4]:.1e} .. {sing_vals[sing_vals.size//2]:.1e} .. {sing_vals[3*sing_vals.size//4]:.1e} .. {sing_vals[-1]:.1e}")
            def M_A_matrix(a):
                if self.CompSep_comm.Get_rank() == 0:
                    a = precond(a)
                a = self.apply_LHS_matrix(a)
                return a

            dense_matrix = DenseMatrix(self.CompSep_comm, M_A_matrix, np.sum(self.alm_len_real_percomp))
            if self.CompSep_comm.Get_rank() == 0:
                sing_vals = dense_matrix.get_sing_vals()
                self.logger.info(f"Condition number of preconditioned (MA) matrix: {sing_vals[0]/sing_vals[-1]:.3e}")
                self.logger.info(f"Sing-vals: {sing_vals[0]:.1e} .. {sing_vals[sing_vals.size//4]:.1e} .. {sing_vals[sing_vals.size//2]:.1e} .. {sing_vals[3*sing_vals.size//4]:.1e} .. {sing_vals[-1]:.1e}")


        if not seed is None:
            np.random.seed(seed)
        x0 = [np.random.normal(0.0, 1.0, self.alm_len_real_percomp[icomp]) for icomp in range(self.ncomp)]
        x0 = np.concatenate(x0)
        sol_array = self.solve_CG(self.apply_LHS_matrix, RHS, x0, M=precond, x_true=x_true if debug_mode else None)
        sol_array = self.CompSep_comm.bcast(sol_array, root=0)
        sol = []
        idx_start = 0
        idx_stop = 0
        for icomp in range(self.ncomp):
            idx_stop += self.alm_len_real_percomp[icomp]
            sol.append(sol_array[idx_start:idx_stop])
            idx_start = idx_stop
        for icomp in range(self.ncomp):
            self.comp_list[icomp].component_alms = self.alm_real2complex(sol[icomp], lmax=self.lmax_per_comp[icomp])
        return self.comp_list
