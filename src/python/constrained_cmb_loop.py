import numpy as np
import ducc0
import healpy as hp
import logging
from pixell import utils
from mpi4py import MPI
import matplotlib.pyplot as plt
import os

nthreads = 32  # Number of threads to use for ducc S
VERBOSE = False


def alm2map(alm, nside, lmax):
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    return ducc0.sht.synthesis(alm=alm.reshape((1,-1)),
                               lmax=lmax,
                               spin=0,
                               nthreads=nthreads, **geom).reshape((-1,))


def alm2map_adjoint(map, nside, lmax):
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    return ducc0.sht.adjoint_synthesis(map=map.reshape((1,-1)),
                                       lmax=lmax,
                                       spin=0,
                                       nthreads=nthreads, **geom).reshape((-1,))


class ConstrainedCMB:
    def __init__(self, map_sky, map_rms, iter):
        self.iter = iter
        self.map_sky = map_sky
        self.map_rms = map_rms
        self.nband, self.npix = map_sky.shape
        self.fwhm = 1.0/60.0*np.pi/180.0*np.ones(self.nband)
        self.nside = hp.npix2nside(self.npix)
        self.lmax = 3*self.nside-1
        self.alm_len = ((self.lmax+1)*(self.lmax+2))//2

        # TEMPORARY. Set Cl prior to true CMB Cls.
        import camb
        pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06, As=2e-9, ns=0.965, halofit_version='mead', lmax=self.lmax)
        results = camb.get_results(pars)
        powers =results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
        totCL=powers['total']
        self.ell = np.arange(self.lmax+1)
        self.Cl_true = totCL[self.ell,0]

        self.Cl_prior = self.Cl_true.copy()
        # self.Cl_prior[:2] = 1e6
        self.Cl_prior[:] = 1e6  # We currently "turn off" the prior by setting it very high.
                                # In the future, the C(ell)s will be sampled and used as a prior here.


    def dot_alm(self, alm1, alm2):
        """ Function calculating the dot product of two alms, given that they follow the Healpy standard,
            where alms are represented as complex numbers, but with the conjugate 'negative' ms missing.
        """        
        return np.sum((alm1[:self.lmax]*alm2[:self.lmax]).real) + np.sum((alm1[self.lmax:]*np.conj(alm2[self.lmax:])).real*2)


    def LHS_func(self, x):
        """ The LHS of equations 5 and 6 from Eriksen 2004, implemented as a function on the alm-vector x.
            The equation can be written as (C^-1 x + A^T Y^T N Y A x), where Y is a map->alm conversion, and Y^T is map-> alm,
            A is the beam-smoothing, and C is the current C(ell) sample.
        """
        LHS_sum = np.zeros_like(x)
        LHS_sum += hp.almxfl(x, 1.0/self.Cl_prior)
        for iband in range(self.nband):
            Ax = hp.smoothalm(x, self.fwhm[iband], inplace=False)
            YAx = alm2map(Ax, self.nside, self.lmax)
            NYAx = YAx.copy()/self.map_rms[iband]**2
            YTNYAx = alm2map_adjoint(NYAx, self.nside, self.lmax)
            ATYTNYAx = hp.smoothalm(YTNYAx, self.fwhm[iband], inplace=False)
            LHS_sum += ATYTNYAx
        return LHS_sum


    def get_RHS_eqn_mean(self):
        """ Calculates and returns the RHS of the mean-field (Wiener filtered) map equation (eqn 5 from Eriksen 2004).
            This RHS can be written as (A^T Y^T N d), where d is the observed sky, and Y^T is a map->alm conversion,
            N is the noise covariance, and A is the beam.
        """
        RHS_sum = np.zeros(self.alm_len, dtype=np.complex128)
        for iband in range(self.nband):
            Nd = self.map_sky[iband]/self.map_rms[iband]**2
            YTNd = alm2map_adjoint(Nd, self.nside, self.lmax)
            ATYTNd = hp.smoothalm(YTNd, self.fwhm[iband], inplace=False)
            RHS_sum += ATYTNd
        return RHS_sum


    def get_RHS_eqn_fluct(self):
        """ Calculates and returns the RHS of the map fluctuation equation (eqn 6 from Eriksen 2004).
            This RHS can be written as (C^-1/2 Y^T omega0 + A^T Y^T N^-1/2 omega1), where omega0 and omega1 are N(0,1) maps,
            and C is the currentl C(ell) sample.
        """
        # YTomega0 = hp.map2alm(np.random.normal(0, 1, self.npix), iter=0)#*(4*np.pi/self.npix)
        # CYTomega0 = hp.almxfl(YTomega0, np.sqrt(1.0/self.Cl_sample))
        RHS_sum = np.zeros(self.alm_len, dtype=np.complex128)
        CYTomega0 = hp.synalm(1.0/self.Cl_prior, self.lmax)
        RHS_sum += CYTomega0

        for iband in range(self.nband):
            omega1 = np.random.normal(0, 1, self.npix)
            Nomega1 = omega1/self.map_rms[iband]
            YTNomega1 = alm2map_adjoint(Nomega1, self.nside, self.lmax)
            ATYTNomega1 = hp.smoothalm(YTNomega1, self.fwhm[iband], inplace=False)
            RHS_sum += ATYTNomega1
        return RHS_sum


    def solve_CG(self, LHS, RHS):
        """ Solves the equation Ax=b for x given A (LHS) and b (RHS) using CG from the pixell package.
            Assumes that both x and b are in alm space.

            Args:
                LHS: A callable taking x as argument and returning Ax.
                RHS: A Numpy array representing b, in alm space.
            Returns:
                m_bestfit: The resulting best-fit solution, in alm space.
        """
        CG_solver = utils.CG(LHS, RHS, dot=self.dot_alm)
        err_tol = 1e-6
        if self.iter == 0:
            maxiter = 21
        else:
            maxiter = 351
        while CG_solver.err > err_tol:
            CG_solver.step()
            self.iter += 1
            if VERBOSE and self.iter%10 == 1:
                logger.info(f"CG iter {self.iter:3d} - Residual {CG_solver.err:.3e}")
            if self.iter >= maxiter:
                logger.warning(f"Maximum number of iterations ({maxiter}) reached in CG.")
                break
        logger.info(f"CG finished after {self.iter} iterations with a residual of {CG_solver.err:.3e} (err tol = {err_tol})")
        s_bestfit = CG_solver.x

        return s_bestfit


def constrained_cmb_loop(comm, compsep_master: int, params: dict):
    master = comm.Get_rank() == 0
    logger = logging.getLogger(__name__)
    if master:
        if not os.path.isdir(params["output_paths"]["plots"] + "maps_CMB/"):
            os.mkdir(params["output_paths"]["plots"] + "maps_CMB/")
        if not os.path.isdir(params["output_paths"]["plots"] + "plots/"):
            os.mkdir(params["output_paths"]["plots"] + "plots/")

    while True:
        # check for simulation end
        stop = MPI.COMM_WORLD.recv(source=compsep_master) if master else False
        stop = comm.bcast(stop, root=0)
        if stop:
            if master:
                logger.warning("CMB: stop requested; exiting")
            return
        if master:
            logger.info("CMB: new job obtained")

        data, iter, chain = MPI.COMM_WORLD.recv(source=compsep_master) if master else None
        # Broadcast te data to all tasks, or do anything else that's appropriate
        data = comm.bcast(data, root=0)
        if master:
            logger.info("CMB: successfully got data.")
        if master:
            signal_maps, rms_maps = data
            signal_maps = signal_maps[:2]  # Ignore highest frequency band - very dust contaminated.
            rms_maps = rms_maps[:2]
            constrained_cmb_solver = ConstrainedCMB(signal_maps, rms_maps, iter)
            logger.info("CMB: Solving for mean-field map")
            RHS_mean_field = constrained_cmb_solver.get_RHS_eqn_mean()
            CMB_mean_field_alms = constrained_cmb_solver.solve_CG(constrained_cmb_solver.LHS_func, RHS_mean_field)
            CMB_mean_field_Cl = hp.alm2cl(CMB_mean_field_alms)
            CMB_mean_field_map = alm2map(CMB_mean_field_alms, constrained_cmb_solver.nside, constrained_cmb_solver.lmax)

            constrained_cmb_solver = ConstrainedCMB(signal_maps, rms_maps, iter)
            logger.info("CMB: Solving for fluctuation map")
            RHS_fluct = constrained_cmb_solver.get_RHS_eqn_fluct()
            CMB_fluct_alms = constrained_cmb_solver.solve_CG(constrained_cmb_solver.LHS_func, RHS_fluct)
            CMB_fluct_Cl = hp.alm2cl(CMB_fluct_alms)
            CMB_fluct_map = alm2map(CMB_fluct_alms, constrained_cmb_solver.nside, constrained_cmb_solver.lmax)

            # Plotting stuff
            ell = constrained_cmb_solver.ell
            Z = ell*(ell+1)/(2*np.pi)
            hp.mollview(CMB_mean_field_map, cmap="RdBu_r", title=f"Constrained mean field CMB realization chain{chain} iter{iter}")
            plt.savefig(params["output_paths"]["plots"] + f"maps_CMB/CMB_mean_field_chain{chain}_iter{iter}.png")
            plt.close()

            hp.mollview(CMB_fluct_map, cmap="RdBu_r", title=f"Constrained fluctuation CMB realization chain{chain} iter{iter}")
            plt.savefig(params["output_paths"]["plots"] + f"maps_CMB/CMB_fluct_chain{chain}_iter{iter}.png")
            plt.close()

            hp.mollview(CMB_mean_field_map+CMB_fluct_map, cmap="RdBu_r", title=f"Joint constrained CMB realization chain{chain} iter{iter}")
            plt.savefig(params["output_paths"]["plots"] + f"maps_CMB/CMB_joint_realization_chain{chain}_iter{iter}.png")
            plt.close()

            plt.figure()
            plt.plot(ell, Z*CMB_mean_field_Cl, label="Cl CMB mean field")
            plt.plot(ell, Z*CMB_fluct_Cl, label="Cl CMB fluct")
            plt.plot(ell, Z*CMB_mean_field_Cl + CMB_fluct_Cl, label="Cl CMB joint")
            plt.plot(ell, Z*hp.alm2cl(hp.map2alm(signal_maps[0])), label="CL observed sky")
            plt.plot(ell, Z*constrained_cmb_solver.Cl_true, label="True CMB Cl", c="k")
            plt.legend()
            plt.xscale("log")
            plt.yscale("log")
            plt.ylim(1e-2, 1e6)
            plt.savefig(params["output_paths"]["plots"] + f"plots/Cl_CMB_chain{chain}_iter{iter}.png")
