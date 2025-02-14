import numpy as np
import healpy as hp
import time
from pixell import utils, curvedsky
import logging

import output
from model.component import CMB, ThermalDust, Synchrotron
from utils.math_operations import alm_to_map, alm_to_map_adjoint
from pixell import curvedsky as pixell_curvedsky

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
    def __init__(self, map_sky, map_rms, freqs, params):
        logger = logging.getLogger(__name__)
        self.params = params
        self.map_sky = map_sky
        self.map_rms = map_rms
        self.freqs = freqs
        self.nband, self.npix = map_rms.shape
        self.nside = np.sqrt(self.npix//12)
        output.logassert(self.nside.is_integer(), f"Npix dimension of map ({self.npix}) resulting in a non-integer nside ({self.nside}).", logger)
        self.nside = int(self.nside)
        self.lmax = 3*self.nside-1
        self.alm_len_complex = ((self.lmax+1)*(self.lmax+2))//2
        self.alm_len_real = (self.lmax+1)**2
        self.ainfo = curvedsky.alm_info(lmax=self.lmax)
        self.comps_SED = np.array([CMB().get_sed(freqs), ThermalDust().get_sed(freqs), Synchrotron().get_sed(freqs)])
        self.ncomp = 3  # Should be in parameter file, but also needs to match length of above list.
        output.logassert(len(self.params.fwhm) == len(self.freqs), f"Number of bands {len(freqs)} does not match length of FWHM ({len(self.params.fwhm)}).", logger)
        self.fwhm = np.array(self.params.fwhm)/60.0*(np.pi/180.0)  # Converting arcmin to radians.


    def alm_imag2real(self, alm):
        i = int(self.ainfo.mstart[1]+1)
        return np.concatenate([alm[:i].real,np.sqrt(2.)*alm[i:].view(np.float64)])


    def alm_real2imag(self, x):
        i    = int(self.ainfo.mstart[1]+1)
        oalm = np.zeros(self.ainfo.nelem, np.complex128)
        oalm[:i] = x[:i]
        oalm[i:] = x[i:].view(np.complex128)/np.sqrt(2.)
        return oalm


    def apply_LHS_matrix(self, a: np.array):
        """ Applies the A matrix to inputed component alms a, where A represents the entire LHS of the Ax=b system for global component separation.
            The full A matrix can be written B^T Y^T M^T N^-1 M Y B, where B is the beam smoothing, M is the mixing matrix, and N is the noise covariance matrix.

            Args:
                a: (ncomp*alm_len_real,) array containing real, flattened alms of each component.
            Returns:
                Aa: (nband*alm_len_rea,) array from applying the full A matrix to a.
        """
        logger = logging.getLogger(__name__)
        # B^T Y^T M^T N^-1 M Y B a
        a = a.reshape((self.ncomp, self.alm_len_real))
        output.logassert(a.dtype == np.float64, "Provided component array is not of type np.float64. This operator takes and returns real alms (and converts to and from complex interally).", logger)
        # output.logassert(a.dtype == np.complex128, "Provided component array is not of type np.complex128, which is required.", logger)
        a_old = a.copy()
        a = np.zeros((self.ncomp, self.alm_len_complex), dtype=np.complex128)
        for icomp in range(self.ncomp):
            a[icomp] = self.alm_real2imag(a_old[icomp])

        # Y a
        a_old = a.copy()
        a = np.zeros((self.ncomp, self.npix))
        for icomp in range(self.ncomp):
            # a[icomp] = hp.alm2map(a_old[icomp], self.nside, self.lmax)
            a[icomp] = alm_to_map(a_old[icomp], self.nside, self.lmax, nthreads=self.params.nthreads_compsep)

        # M Y a
        a_old = a.copy()
        a = np.zeros((self.nband, self.npix))
        for iband in range(self.nband):
            for icomp in range(self.ncomp):
                a[iband] += self.comps_SED[icomp,iband]*a_old[icomp]

        # Y^-1 M Y a
        a_old = a.copy()
        a = np.zeros((self.nband, self.alm_len_complex), dtype=np.complex128)
        pixell_curvedsky.map2alm_healpix(a_old, a, niter=3, spin=0, nthread=self.params.nthreads_compsep)

        # B Y^-1 M Y a
        for iband in range(self.nband):
            hp.smoothalm(a[iband], self.fwhm[iband], inplace=True)

        # Y B Y^-1 M Y a
        a_old = a.copy()
        a = np.zeros((self.nband, self.npix))
        for iband in range(self.nband):
            # a[icomp] = hp.alm2map(a_old[icomp], self.nside, self.lmax)
            a[iband] = alm_to_map(a_old[iband], self.nside, self.lmax, nthreads=self.params.nthreads_compsep)

        # N^-1 Y B Y^-1 M Y a
        a = a/self.map_rms**2

        # Y^T N^-1 Y B Y^-1 M Y a
        a_old = a.copy()
        a = np.zeros((self.nband, self.alm_len_complex), dtype=np.complex128)
        for iband in range(self.nband):
            # a[icomp] = hp.alm2map(a_old[icomp], self.nside, self.lmax)
            a[iband] = alm_to_map_adjoint(a_old[iband], self.nside, self.lmax, nthreads=self.params.nthreads_compsep)

        # B^T Y^T N^-1 Y B Y^-1 M Y a
        for iband in range(self.nband):
            hp.smoothalm(a[iband], self.fwhm[iband], inplace=True)

        # Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y a
        a_old = a.copy()
        a = np.zeros((self.nband, self.npix))
        pixell_curvedsky.map2alm_healpix(a, a_old, niter=3, adjoint=True, spin=0, nthread=self.params.nthreads_compsep)

        # M^T Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y a
        a_old = a.copy()
        a = np.zeros((self.ncomp, self.npix))
        for iband in range(self.nband):
            for icomp in range(self.ncomp):
                a[icomp] += self.comps_SED[icomp,iband]*a_old[iband]

        # Y^T M^T Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y a
        a_old = a.copy()
        a = np.zeros((self.ncomp, self.alm_len_complex), dtype=np.complex128)
        for icomp in range(self.ncomp):
            a[icomp] = alm_to_map_adjoint(a_old[icomp], self.nside, self.lmax, nthreads=self.params.nthreads_compsep)

        # Converting back from complex alms to real alms
        a_old = a.copy()
        a = np.zeros((self.ncomp, self.alm_len_real), dtype=np.float64)
        for icomp in range(self.ncomp):
            a[icomp] = self.alm_imag2real(a_old[icomp])

        return a.flatten()


    def solve_CG(self, LHS, RHS, x0):
        """ Solves the equation Ax=b for x given A (LHS) and b (RHS) using CG from the pixell package.
            Assumes that both x and b are in alm space.

            Args:
                LHS: A callable taking x as argument and returning Ax.
                RHS: A Numpy array representing b, in alm space.
            Returns:
                m_bestfit: The resulting best-fit solution, in alm space.
        """
        logger = logging.getLogger(__name__)
        # CG_solver = utils.CG(LHS, RHS, x0=x0, dot=self.alm_dot_product)
        CG_solver = utils.CG(LHS, RHS, x0=x0)
        iter = 0
        self.CG_residuals = np.zeros((self.params.CG_max_iter))
        t0 = time.time()
        while CG_solver.err > self.params.CG_err_tol:
            CG_solver.step()
            self.CG_residuals[iter] = CG_solver.err
            iter += 1
            if iter%10 == 0:
                logger.info(f"CG iter {iter:3d} - Residual {np.mean(self.CG_residuals[iter-10:iter]):.3e} ({(time.time() - t0)/10.0:.1f}s/iter)")
                t0 = time.time()
            if iter >= self.params.CG_max_iter:
                logger.warning(f"Maximum number of iterations ({self.params.CG_max_iter}) reached in CG.")
                break
        self.CG_residuals = self.CG_residuals[:iter]
        logger.info(f"CG finished after {iter} iterations with a residual of {CG_solver.err:.3e} (err tol = {self.params.CG_err_tol})")
        s_bestfit = CG_solver.x
        return s_bestfit


    def solve(self, seed=None) -> np.array:

        # d
        b = self.map_sky.copy()

        # N^-1 d
        b = b/self.map_rms**2

        # Y^T N^-1 d
        b_old = b.copy()
        b = np.zeros((self.nband, self.alm_len_complex), dtype=np.complex128)
        for iband in range(self.nband):
            b[iband] = alm_to_map_adjoint(b_old[iband], self.nside, self.lmax, nthreads=self.params.nthreads_compsep)

        # B^T Y^T N^-1 d
        for iband in range(self.nband):
            hp.smoothalm(b[iband], self.fwhm[iband], inplace=True)

        # Y^-1^T B^T Y^T N^-1 d
        b_old = b.copy()
        b = np.zeros((self.nband, self.npix))
        pixell_curvedsky.map2alm_healpix(b, b_old, adjoint=True, niter=3, spin=0, nthread=self.params.nthreads_compsep)

        # M^T Y^-1^T B^T Y^T N^-1 d
        b_old = b.copy()
        b = np.zeros((self.ncomp, self.npix))
        for iband in range(self.nband):
            for icomp in range(self.ncomp):
                b[icomp] += self.comps_SED[icomp,iband]*b_old[iband]

        # Y^T M^T Y^-1^T B^T Y^T N^-1 d
        b_old = b.copy()
        b = np.zeros((self.ncomp, self.alm_len_complex), dtype=np.complex128)
        for icomp in range(self.ncomp):
            b[icomp] = alm_to_map_adjoint(b_old[icomp], self.nside, self.lmax, nthreads=self.params.nthreads_compsep)
            # b[icomp] = hp.map2alm(b_old[icomp], self.lmax)
        
        b_old = b.copy()
        b = np.zeros((self.ncomp, self.alm_len_real), dtype=np.float64)
        for icomp in range(self.ncomp):
            b[icomp] = self.alm_imag2real(b_old[icomp])

        b = b.flatten()

        if not seed is None:
            np.random.seed(seed)
        x0 = np.random.normal(0.0, 1.0, (self.ncomp, self.alm_len_real))
        x0 = x0.flatten()

        sol = self.solve_CG(self.apply_LHS_matrix, b, x0)
        sol = sol.reshape((self.ncomp, self.alm_len_real))
        sol_map = np.zeros((self.ncomp, self.npix))
        for icomp in range(self.ncomp):
            sol_map[icomp] = alm_to_map(self.alm_real2imag(sol[icomp]), self.nside, self.lmax, nthreads=self.params.nthreads_compsep)
        return sol_map
