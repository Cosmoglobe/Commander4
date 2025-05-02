import numpy as np
import ducc0
import logging
import healpy as hp
from pixell import utils
from mpi4py import MPI
from utils.math_operations import alm_to_map, alm_to_map_adjoint
from output import plotting

nthreads = 32  # Number of threads to use for ducc SHTs.
VERBOSE = False


class ConstrainedCMB:
    def __init__(self, map_sky, map_rms, iter, comm):
        self.comm = comm
        self.nprocs = comm.Get_size()
        self.iter = iter
        self.map_sky = map_sky
        self.map_rms = map_rms
        self.npix = map_sky.shape[0]
        self.fwhm = 1.0/60.0*np.pi/180.0
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


    def master_LHS_func(self, x):
        """ The LHS of equations 5 and 6 from Eriksen 2004, implemented as a function on the alm-vector x.
            The equation can be written as (C^-1 x + A^T Y^T N Y A x), where Y is a map->alm conversion, and Y^T is map->alm,
            A is the beam-smoothing, and C is the current C(ell) sample.
        """
        self.comm.Bcast(x, root=0)  # Sending up-to-date x 
        LHS_sum = np.zeros_like(x)
        Ax = hp.smoothalm(x, self.fwhm, inplace=False)
        YAx = alm_to_map(Ax, self.nside, self.lmax, nthreads=nthreads)
        NYAx = YAx/self.map_rms**2
        YTNYAx = alm_to_map_adjoint(NYAx, self.nside, self.lmax, nthreads=nthreads)
        ATYTNYAx = hp.smoothalm(YTNYAx, self.fwhm, inplace=False)

        self.comm.Reduce(ATYTNYAx,  # Receiving the LHS contribution from other ranks.
                        LHS_sum,
                        op=MPI.SUM,
                        root=0)
        LHS_sum += hp.almxfl(x, 1.0/self.Cl_prior)
        return LHS_sum


    def worker_LHS_func(self):
        """ Function for calculating a single-channel contribution to the Ax LHS shown above,
            and send the result to rank 0. The current x is first received from rank 0.
        """
        x = np.zeros(self.alm_len, dtype=np.complex128)
        self.comm.Bcast(x, root=0)
        Ax = hp.smoothalm(x, self.fwhm, inplace=False)
        YAx = alm_to_map(Ax, self.nside, self.lmax, nthreads=nthreads)
        NYAx = YAx/self.map_rms**2
        YTNYAx = alm_to_map_adjoint(NYAx, self.nside, self.lmax, nthreads=nthreads)
        ATYTNYAx = hp.smoothalm(YTNYAx, self.fwhm, inplace=False)
        self.comm.Reduce(ATYTNYAx,  # Sending our part of the LHS equation to rank 0.
                        None,
                        op=MPI.SUM,
                        root=0)


    def get_RHS_eqn_mean(self):
        """ Calculates and returns the RHS of the mean-field (Wiener filtered) map equation (eqn 5 from Eriksen 2004).
            This RHS can be written as (A^T Y^T N d), where d is the observed sky, and Y^T is a map->alm conversion,
            N is the noise covariance, and A is the beam.
        """
        RHS_sum = np.zeros(self.alm_len, dtype=np.complex128)
        Nd = self.map_sky/self.map_rms**2
        YTNd = alm_to_map_adjoint(Nd, self.nside, self.lmax, nthreads=nthreads)
        ATYTNd = hp.smoothalm(YTNd, self.fwhm, inplace=False)
        self.comm.Allreduce([ATYTNd, MPI.DOUBLE],
                         [RHS_sum, MPI.DOUBLE],
                         op=MPI.SUM)
        return RHS_sum


    def get_RHS_eqn_fluct(self):
        """ Calculates and returns the RHS of the map fluctuation equation (eqn 6 from Eriksen 2004).
            This RHS can be written as (C^-1/2 Y^T omega0 + A^T Y^T N^-1/2 omega1), where omega0 and omega1 are N(0,1) maps,
            and C is the currentl C(ell) sample.
        """
        # YTomega0 = hp.map2alm(np.random.normal(0, 1, self.npix), iter=0)#*(4*np.pi/self.npix)
        # CYTomega0 = hp.almxfl(YTomega0, np.sqrt(1.0/self.Cl_sample))
        RHS_sum = np.zeros(self.alm_len, dtype=np.complex128)

        omega1 = np.random.normal(0, 1, self.npix)
        Nomega1 = omega1/self.map_rms
        YTNomega1 = alm_to_map_adjoint(Nomega1, self.nside, self.lmax, nthreads=nthreads)
        ATYTNomega1 = hp.smoothalm(YTNomega1, self.fwhm, inplace=False)
        self.comm.Allreduce([ATYTNomega1, MPI.DOUBLE],
                         [RHS_sum, MPI.DOUBLE],
                         op=MPI.SUM)
        CYTomega0 = hp.synalm(1.0/self.Cl_prior, self.lmax)
        RHS_sum += CYTomega0

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
        logger = logging.getLogger(__name__)
        CG_solver = utils.CG(LHS, RHS, dot=self.dot_alm)
        err_tol = 1e-6
        if self.iter == 0:
            maxiter = 21
        else:
            maxiter = 351
        while CG_solver.err > err_tol:
            for irank in range(1, self.nprocs):
                self.comm.send(False, irank)  # Send "don't stop" signal to worked tasks.

            CG_solver.step()
            self.iter += 1
            if VERBOSE and self.iter%10 == 1:
                logger.info(f"CG iter {self.iter:3d} - Residual {CG_solver.err:.3e}")
            if self.iter >= maxiter:
                logger.warning(f"Maximum number of iterations ({maxiter}) reached in CG.")
                break
        for irank in range(1, self.nprocs):
            self.comm.send(True, irank)  # Send "stop" signal to worker tasks.
        logger.info(f"CG finished after {self.iter} iterations with a residual of {CG_solver.err:.3e} (err tol = {err_tol})")
        s_bestfit = CG_solver.x

        return s_bestfit


def constrained_cmb_loop_MPI(comm, compsep_master: int, params: dict):
    master = comm.Get_rank() == 0
    logger = logging.getLogger(__name__)

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

        # data, iter, chain = MPI.COMM_WORLD.recv(source=compsep_master) if master else None
        data, iter, chain = MPI.COMM_WORLD.recv(source=compsep_master) 
        if master:
            logger.info("CMB: successfully got data.")
        # Broadcast te data to all tasks, or do anything else that's appropriate
        # data = comm.bcast(data, root=0)

        signal_maps, rms_maps = data
        constrained_cmb_solver = ConstrainedCMB(signal_maps, rms_maps, iter, comm)
        RHS_mean_field = constrained_cmb_solver.get_RHS_eqn_mean()
        RHS_fluct = constrained_cmb_solver.get_RHS_eqn_fluct()

        if master:
            logger.info("CMB: Solving for mean-field map")
            CMB_mean_field_alms = constrained_cmb_solver.solve_CG(constrained_cmb_solver.master_LHS_func, RHS_mean_field)
            CMB_mean_field_Cl = hp.alm2cl(CMB_mean_field_alms)
            CMB_mean_field_map = alm_to_map(CMB_mean_field_alms, constrained_cmb_solver.nside, constrained_cmb_solver.lmax, nthreads=nthreads)
        else:
            while not comm.recv(source=0):  # Looking for "stop" signal.
                constrained_cmb_solver.worker_LHS_func()  # If not asked to stop, compute LHS.

        constrained_cmb_solver = ConstrainedCMB(signal_maps, rms_maps, iter, comm)
        if master:
            logger.info("CMB: Solving for fluctuation map")
            CMB_fluct_alms = constrained_cmb_solver.solve_CG(constrained_cmb_solver.master_LHS_func, RHS_fluct)
            CMB_fluct_Cl = hp.alm2cl(CMB_fluct_alms)
            CMB_fluct_map = alm_to_map(CMB_fluct_alms, constrained_cmb_solver.nside, constrained_cmb_solver.lmax, nthreads=nthreads)
        else:
            while not comm.recv(source=0):  # Looking for "stop" signal.
                constrained_cmb_solver.worker_LHS_func()  # If not asked to stop, compute LHS.

        if master and params.make_plots:
            plotting.plot_constrained_cmb_results(
                master, params, detector, chain, iter,
                constrained_cmb_solver.ell, CMB_mean_field_map, CMB_fluct_map,
                signal_maps[0], constrained_cmb_solver.Cl_true)
