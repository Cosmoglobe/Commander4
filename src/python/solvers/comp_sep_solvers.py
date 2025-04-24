import numpy as np
import healpy as hp
import time
from pixell import utils, curvedsky
from pixell import curvedsky as pixell_curvedsky
import logging
from mpi4py.MPI import Comm
from mpi4py import MPI

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
    def __init__(self, comp_list: list[Component], map_sky, map_rms, freq, fwhm, params, CompSep_comm: Comm):
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


    def alm_imag2real(self, alm, lmax):
        ainfo = curvedsky.alm_info(lmax=lmax)
        i = int(ainfo.mstart[1]+1)
        return np.concatenate([alm[:i].real,np.sqrt(2.)*alm[i:].view(np.float64)])


    def alm_real2imag(self, x, lmax):
        ainfo = curvedsky.alm_info(lmax=lmax)
        i    = int(ainfo.mstart[1]+1)
        oalm = np.zeros(ainfo.nelem, np.complex128)
        oalm[:i] = x[:i]
        oalm[i:] = x[i:].view(np.complex128)/np.sqrt(2.)
        return oalm


    def apply_LHS_matrix(self, a_array: np.array):
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

        a = []
        idx_start = 0
        idx_stop = 0
        for icomp in range(self.ncomp):
            idx_stop += self.alm_len_real_percomp[icomp]
            a.append(a_array[idx_start:idx_stop])
            idx_start = idx_stop

        a_old = a.copy()
        a = []
        for icomp in range(self.ncomp):
            a.append(self.alm_real2imag(a_old[icomp], lmax=self.lmax_per_comp[icomp]))

        # Y a
        a_old = a.copy()
        a = []
        for icomp in range(self.ncomp):
            if icomp == self.CompSep_comm.Get_rank():
                a.append(alm_to_map(a_old[icomp], self.nside, self.lmax_per_comp[icomp], nthreads=self.params.nthreads_compsep))
            else:
                a.append(np.zeros((self.npix), dtype=np.float64))
        for icomp in range(self.ncomp):
            self.CompSep_comm.Allreduce(MPI.IN_PLACE, a[icomp], op=MPI.SUM)

        # M Y a
        a_old = a.copy()
        a = np.zeros((self.npix))
        for icomp in range(self.ncomp):
            a += self.comps_SED[icomp,self.my_band_idx]*a_old[icomp]

        # Y^-1 M Y a
        a_old = a.copy()
        a = np.zeros((self.alm_len_complex), dtype=np.complex128)
        pixell_curvedsky.map2alm_healpix(a_old, a, niter=3, spin=0, nthread=self.params.nthreads_compsep)

        # B Y^-1 M Y a
        hp.smoothalm(a, self.fwhm_rad, inplace=True)

        # Y B Y^-1 M Y a
        a_old = a.copy()
        a = np.zeros((self.npix))
        a = alm_to_map(a_old, self.nside, self.lmax, nthreads=self.params.nthreads_compsep)

        # N^-1 Y B Y^-1 M Y a
        a = a/self.map_rms**2

        # Y^T N^-1 Y B Y^-1 M Y a
        a_old = a.copy()
        a = np.zeros((self.alm_len_complex), dtype=np.complex128)
        a = alm_to_map_adjoint(a_old, self.nside, self.lmax, nthreads=self.params.nthreads_compsep)

        # B^T Y^T N^-1 Y B Y^-1 M Y a
        hp.smoothalm(a, self.fwhm_rad, inplace=True)

        # Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y a
        a_old = a.copy()
        a = np.zeros((self.npix))
        pixell_curvedsky.map2alm_healpix(a, a_old, niter=3, adjoint=True, spin=0, nthread=self.params.nthreads_compsep)

        # M^T Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y a
        a_old = a.copy()
        a = np.zeros((self.ncomp, self.npix))
        for iband in range(self.nband):
            for icomp in range(self.ncomp):
                if iband == self.my_band_idx:
                    a[icomp] += self.comps_SED[icomp,iband]*a_old
        self.CompSep_comm.Allreduce(MPI.IN_PLACE, a, op=MPI.SUM)

        # Y^T M^T Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y a
        a_old = a.copy()
        a = []
        for icomp in range(self.ncomp):
            if icomp == self.CompSep_comm.Get_rank():
                a.append(alm_to_map_adjoint(a_old[icomp], self.nside, self.lmax_per_comp[icomp], nthreads=self.params.nthreads_compsep))
            else:
                a.append(np.zeros((self.alm_len_complex_percomp[icomp]), dtype=np.complex128))
        for icomp in range(self.ncomp):
            self.CompSep_comm.Allreduce(MPI.IN_PLACE, a[icomp], op=MPI.SUM)

        # Converting back from complex alms to real alms
        a_old = a.copy()
        a = []
        for icomp in range(self.ncomp):
            a.append(self.alm_imag2real(a_old[icomp], lmax=self.lmax_per_comp[icomp]))
        a = np.concatenate(a)

        return a#.flatten()


    def solve_CG(self, LHS, RHS, x0, M=None, x_true=None):
        """ Solves the equation Ax=b for x given A (LHS) and b (RHS) using CG from the pixell package.
            Assumes that both x and b are in alm space.

            Args:
                LHS (callable): A function/callable taking x as argument and returning Ax.
                RHS (np.array): A Numpy array representing b, in alm space.
                x0 (np.array): Initial guess for x.
                M (callable): Preconditioner for the CG solver. A function/callable which approximates A^-1 (optional).
                x_true (np.array): True solution for x, in order to print the true error (optional, used for testing).
            Returns:
                m_bestfit: The resulting best-fit solution, in alm space.
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


    def calc_RHS_mean(self):
        # d
        b = self.map_sky.copy()

        # N^-1 d
        b = b/self.map_rms**2

        # Y^T N^-1 d
        b_old = b.copy()
        b = np.zeros((self.alm_len_complex), dtype=np.complex128)
        b = alm_to_map_adjoint(b_old, self.nside, self.lmax, nthreads=self.params.nthreads_compsep)

        # B^T Y^T N^-1 d
        hp.smoothalm(b, self.fwhm_rad, inplace=True)

        # Y^-1^T B^T Y^T N^-1 d
        b_old = b.copy()
        b = np.zeros((self.npix))
        pixell_curvedsky.map2alm_healpix(b, b_old, adjoint=True, niter=3, spin=0, nthread=self.params.nthreads_compsep)

        # M^T Y^-1^T B^T Y^T N^-1 d
        b_old = b.copy()
        b = np.zeros((self.ncomp, self.npix))
        for icomp in range(self.ncomp):
            b[icomp] += self.comps_SED[icomp,self.my_band_idx]*b_old
        b = self.CompSep_comm.allreduce(b, op=MPI.SUM)

        # Y^T M^T Y^-1^T B^T Y^T N^-1 d
        b_old = b.copy()
        b = []
        for icomp in range(self.ncomp):
            if icomp == self.CompSep_comm.Get_rank():
                b.append(alm_to_map_adjoint(b_old[icomp], self.nside, self.lmax_per_comp[icomp], nthreads=self.params.nthreads_compsep))
            else:
                b.append(np.zeros((self.alm_len_complex_percomp[icomp]), dtype=np.complex128))
        for icomp in range(self.ncomp):
            b[icomp] = self.CompSep_comm.allreduce(b[icomp], op=MPI.SUM)

        b_old = b.copy()
        b = []
        for icomp in range(self.ncomp):
            b.append(self.alm_imag2real(b_old[icomp], lmax=self.lmax_per_comp[icomp]))
        b = np.concatenate(b)

        return b


    def calc_RHS_fluct(self):
        # d
        b = np.random.normal(0.0, 1.0, self.map_rms.shape)

        # N^-1 d
        b = b/self.map_rms

        # Y^T N^-1 d
        b_old = b.copy()
        b = np.zeros((self.alm_len_complex), dtype=np.complex128)
        b = alm_to_map_adjoint(b_old, self.nside, self.lmax, nthreads=self.params.nthreads_compsep)

        # B^T Y^T N^-1 d
        hp.smoothalm(b, self.fwhm_rad, inplace=True)

        # Y^-1^T B^T Y^T N^-1 d
        b_old = b.copy()
        b = np.zeros((self.npix))
        pixell_curvedsky.map2alm_healpix(b, b_old, adjoint=True, niter=3, spin=0, nthread=self.params.nthreads_compsep)

        # M^T Y^-1^T B^T Y^T N^-1 d
        b_old = b.copy()
        b = np.zeros((self.ncomp, self.npix))
        for icomp in range(self.ncomp):
            b[icomp] += self.comps_SED[icomp,self.my_band_idx]*b_old
        b = self.CompSep_comm.allreduce(b, op=MPI.SUM)

        # Y^T M^T Y^-1^T B^T Y^T N^-1 d
        b_old = b.copy()
        b = []
        for icomp in range(self.ncomp):
            if icomp == self.CompSep_comm.Get_rank():
                b.append(alm_to_map_adjoint(b_old[icomp], self.nside, self.lmax_per_comp[icomp], nthreads=self.params.nthreads_compsep))
            else:
                b.append(np.zeros((self.alm_len_complex_percomp[icomp]), dtype=np.complex128))
        for icomp in range(self.ncomp):
            b[icomp] = self.CompSep_comm.allreduce(b[icomp], op=MPI.SUM)

        b_old = b.copy()
        b = []
        for icomp in range(self.ncomp):
            b.append(self.alm_imag2real(b_old[icomp], lmax=self.lmax_per_comp[icomp]))
        b = np.concatenate(b)

        return b


    def solve(self, seed=None) -> np.array:

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
            self.comp_list[icomp].component_alms = self.alm_real2imag(sol[icomp], lmax=self.lmax_per_comp[icomp])
        return self.comp_list