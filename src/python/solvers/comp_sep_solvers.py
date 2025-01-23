import numpy as np
import healpy as hp
from pixell import utils, curvedsky

from model.component import CMB, ThermalDust, Synchrotron
from utils.math_operations import alm_to_map, alm_to_map_adjoint

def amplitude_sampling_per_pix(map_sky: np.array, map_rms: np.array, freqs: np.array) -> np.array:
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
    print(f"time for random numbers: {time()-t0}s.")
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
    print(f"Time for Python solution: {time()-t0}s.")
    # import cmdr4_support
    # t0 = time()
    # comp_maps2 = cmdr4_support.utils.amplitude_sampling_per_pix_helper(map_sky, map_rms, M, rand, nthreads=1)
    # print(f"Time for native solution: {time()-t0}s.")
    # import ducc0
    # print(f"L2 error between solutions: {ducc0.misc.l2error(comp_maps, comp_maps2)}.")
    return comp_maps




class CompSepSolver:
    def __init__(self, map_sky, map_rms, freqs, fwhm):
        self.map_sky = map_sky
        self.map_rms = map_rms
        self.freqs = freqs
        self.nband, self.npix = map_rms.shape
        self.nside = np.sqrt(self.npix//12)
        assert self.nside.is_integer(), f"Npix dimension of map ({self.npix}) resulting in a non-integer nside ({self.nside})."
        self.nside = int(self.nside)
        self.lmax = 3*self.nside-1
        self.alm_len_imag = ((self.lmax+1)*(self.lmax+2))//2
        self.alm_len_real = (self.lmax+1)**2
        self.ainfo = curvedsky.alm_info(lmax=self.lmax)
        self.comps_SED = np.array([CMB().get_sed(freqs), ThermalDust().get_sed(freqs), Synchrotron().get_sed(freqs)])
        self.ncomp = 3  # Should be in parameter file, but also needs to match length of above list.
        self.fwhm = fwhm/60.0*(np.pi/180.0)  # Converting arcmin to radians.


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
        # B^T Y^T M^T N^-1 M Y B a
        a = a.reshape((self.ncomp, self.alm_len_real))
        assert a.dtype == np.float64
        # assert a.dtype == np.complex128, "Provided component array is not of type np.complex128, which is required."
        a_old = a.copy()
        a = np.zeros((self.ncomp, self.alm_len_imag), dtype=np.complex128)
        for icomp in range(self.ncomp):
            a[icomp] = self.alm_real2imag(a_old[icomp])

        # B a
        for icomp in range(self.ncomp):
            hp.smoothalm(a[icomp], self.fwhm, inplace=True)

        # Y B a
        a_old = a.copy()
        a = np.zeros((self.ncomp, self.npix))
        for icomp in range(self.ncomp):
            # a[icomp] = hp.alm2map(a_old[icomp], self.nside, self.lmax)
            a[icomp] = alm_to_map(a_old[icomp], self.nside, self.lmax)

        # M Y B a
        a_old = a.copy()
        a = np.zeros((self.nband, self.npix))
        for iband in range(self.nband):
            for icomp in range(self.ncomp):
                a[iband] += self.comps_SED[icomp,iband]*a_old[icomp]

        # for icomp in range(self.nband):
        #     a[icomp] = hp.smoothing(a[icomp], fwhm=self.fwhm)

        # N^-1 M Y B a
        a = a/self.map_rms**2

        # for icomp in range(self.nband):
        #     a[icomp] = hp.smoothing(a[icomp], fwhm=self.fwhm)

        # M^T N^-1 M Y B a
        a_old = a.copy()
        a = np.zeros((self.ncomp, self.npix))
        for iband in range(self.nband):
            for icomp in range(self.ncomp):
                a[icomp] += self.comps_SED[icomp,iband]*a_old[iband]

        # Y^T M^T N^-1 M Y B a
        a_old = a.copy()
        a = np.zeros((self.ncomp, self.alm_len_imag), dtype=np.complex128)
        for icomp in range(self.ncomp):
            # a[icomp] = hp.map2alm(a_old[icomp], self.lmax)
            a[icomp] = alm_to_map_adjoint(a_old[icomp], self.nside, self.lmax)

        # B^T Y^T M^T N^-1 M Y B a
        for icomp in range(self.ncomp):
            hp.smoothalm(a[icomp], self.fwhm, inplace=True)

        a_old = a.copy()
        a = np.zeros((self.ncomp, self.alm_len_real), dtype=np.float64)
        for icomp in range(self.ncomp):
            a[icomp] = self.alm_imag2real(a_old[icomp])

        return a.flatten()


    def solve_CG(self, LHS, RHS, x0, maxiter):
        """ Solves the equation Ax=b for x given A (LHS) and b (RHS) using CG from the pixell package.
            Assumes that both x and b are in alm space.

            Args:
                LHS: A callable taking x as argument and returning Ax.
                RHS: A Numpy array representing b, in alm space.
            Returns:
                m_bestfit: The resulting best-fit solution, in alm space.
        """
        # CG_solver = utils.CG(LHS, RHS, x0=x0, dot=self.alm_dot_product)
        CG_solver = utils.CG(LHS, RHS, x0=x0)
        err_tol = 1e-6
        iter = 0
        while CG_solver.err > err_tol:
            CG_solver.step()
            iter += 1
            print(f"CG iter {iter:3d} - Residual {CG_solver.err:.3e}")
            if iter >= maxiter:
                print(f"Warning: Maximum number of iterations ({maxiter}) reached in CG.")
                break
        print(f"CG finished after {iter} iterations with a residual of {CG_solver.err:.3e} (err tol = {err_tol})")
        s_bestfit = CG_solver.x
        return s_bestfit


    def solve(self) -> np.array:

        # d
        b = self.map_sky.copy()

        # N^-1 d
        b = b/self.map_rms**2

        # M^T N^-1 d
        b_old = b.copy()
        b = np.zeros((self.ncomp, self.npix))
        for iband in range(self.nband):
            for icomp in range(self.ncomp):
                b[icomp] += self.comps_SED[icomp,iband]*b_old[iband]

        # for icomp in range(self.ncomp):  # TODO: Everything breaks if I do the beam convolution in real space, why???
        #     b[icomp] = hp.smoothing(b[icomp], fwhm=self.fwhm)

        # Y^T M^T N^-1 d
        b_old = b.copy()
        b = np.zeros((self.ncomp, self.alm_len_imag), dtype=np.complex128)
        for icomp in range(self.ncomp):
            b[icomp] = alm_to_map_adjoint(b_old[icomp], self.nside, self.lmax)
            # b[icomp] = hp.map2alm(b_old[icomp], self.lmax)

        # B^T Y^T M^T N^-1 d
        for icomp in range(self.ncomp):
            hp.smoothalm(b[icomp], self.fwhm, inplace=True)
        
        b_old = b.copy()
        b = np.zeros((self.ncomp, self.alm_len_real), dtype=np.float64)
        for icomp in range(self.ncomp):
            b[icomp] = self.alm_imag2real(b_old[icomp])

        print("b", b.shape)
        b = b.flatten()

        # A = utils.ubash()

        # N = self.ncomp*self.alm_len
        # print(self.ncomp, self.alm_len)
        # v = np.arange(0, N, N//10)
        # a = None
        # for ind, i in enumerate(v):
        #     u = np.zeros((N), dtype=np.complex128)
        #     u.real = utils.uvec(N, i)
        #     print(u.shape)
        #     temp = self.apply_whatever_matrix(u)
        #     if a is None:
        #         a = np.zeros((temp.shape[-1], len(v)))
        #     a[:,ind] = temp
        # a = a[v]
        # print(np.sum(np.abs(a - a.T))/np.sum(np.abs(a)))
        # print(np.std(a - a.T)/np.std(a))
        # for i in range(10):
        #     print([f"{a[i,j]:.4f}" for j in range(10)])

        # x0 = np.zeros((self.ncomp, self.alm_len), dtype=np.complex128)
        # x0.real = np.random.normal(0.0, 1.0, (self.ncomp, self.alm_len))
        # x0.imag = np.random.normal(0.0, 1.0, (self.ncomp, self.alm_len))
        x0 = np.random.normal(0.0, 1.0, (self.ncomp, self.alm_len_real))

        x0 = x0.flatten()

        # test = self.apply_whatever_matrix(x0)
        # print("test", test.shape)

        sol = self.solve_CG(self.apply_LHS_matrix, b, x0, 150)
        sol = sol.reshape((self.ncomp, self.alm_len_real))
        sol_map = np.zeros((self.ncomp, self.npix))
        for icomp in range(self.ncomp):
            sol_map[icomp] = alm_to_map(self.alm_real2imag(sol[icomp]), self.nside, self.lmax)
        # print(sol_map.shape)
        return sol_map
