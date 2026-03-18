import numpy as np
import ducc0
import healpy as hp
import logging
from pixell import utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import os
import argparse
import re
from astropy.io import fits
from pixell.bunch import Bunch

from commander4.sky_models.component import ThermalDust, FreeFree, Synchrotron
from commander4.sky_models.sky_model import SkyModel

logger = logging.getLogger("cmb_realizations")


CHAIN_ITER_RE = re.compile(r"chain(?P<chain>\d+)_iter(?P<iter>\d+)\.h5$")
BAND_CHAIN_ITER_RE = re.compile(r"(?:(?P<prefix>.+)_)?chain(?P<chain>\d+)_iter(?P<iter>\d+)\.h5$")

def _extract_chain_iter(filename: str) -> tuple[int | None, int | None]:
    match = CHAIN_ITER_RE.search(filename)
    if not match:
        return None, None
    return int(match.group("chain")), int(match.group("iter"))

def _extract_band_chain_iter(filename: str) -> tuple[str | None, int | None, int | None]:
    match = BAND_CHAIN_ITER_RE.search(filename)
    if not match:
        return None, None, None
    return str(match.group("prefix")), int(match.group("chain")), int(match.group("iter"))


nthreads = 32  # Number of threads to use for ducc S
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
    def __init__(self, map_sky, map_rms, cmb_Cell, maxiter=100):
        self.maxiter = maxiter
        self.map_sky = map_sky
        self.map_rms = map_rms
        self.nband, self.npix = map_sky.shape
        self.fwhm = 1.0/60.0*np.pi/180.0*np.ones(self.nband)
        self.nside = hp.npix2nside(self.npix)
        self.lmax = 2*self.nside
        self.alm_len = ((self.lmax+1)*(self.lmax+2))//2
        self.Cl_prior = cmb_Cell

        # Precompute C_ell^{1/2} for the square-root reformulation.
        # Any negative values are clamped to 0.
        Cl_safe = self.Cl_prior[:self.lmax+1].copy()
        Cl_safe[Cl_safe < 0] = 0.0
        self.Cl_sqrt = np.sqrt(Cl_safe)

        # Build diagonal preconditioner in harmonic space
        self._build_preconditioner()

        # # TEMPORARY. Set Cl prior to true CMB Cls.
        # import camb
        # pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06, As=2e-9, ns=0.965, halofit_version='mead', lmax=self.lmax)
        # results = camb.get_results(pars)
        # powers =results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
        # totCL=powers['total']
        # self.ell = np.arange(self.lmax+1)
        # self.Cl_true = totCL[self.ell,0]

        # self.Cl_prior = 3*self.Cl_true.copy()
        # self.Cl_prior[:2] = 1e6
        # self.Cl_prior[:] = 1e6  # We currently "turn off" the prior by setting it very high.
        #                         # In the future, the C(ell)s will be sampled and used as a prior here.


    def _build_preconditioner(self):
        """Build a diagonal (in ell) preconditioner for the renormalized CG system.

        The renormalized LHS is  (I + C^{1/2} sum_i B_i^T Y^T N_i^{-1} Y B_i C^{1/2}).
        Approximating pixel-space noise by its sky-average makes this diagonal
        in harmonic space:

            d_ell = 1 + C_ell * sum_i  b_ell_i^2 * <1/sigma_i^2>

        The preconditioner is M = 1 / d_ell.
        """
        Cl = self.Cl_sqrt ** 2  # = Cl_safe, the clamped version
        diag = np.ones(self.lmax + 1)  # identity contribution

        # The LHS uses ducc0 synthesis (Y) and adjoint_synthesis (Y^T),
        # which are un-normalized: Y^T Y ≈ (Npix/4π) I in harmonic space.
        # The preconditioner must include this factor to match the true diagonal.
        pix_factor = self.npix / (4.0 * np.pi)

        for iband in range(self.nband):
            bl = hp.gauss_beam(self.fwhm[iband], lmax=self.lmax)

            inv_noise_var = 1.0 / self.map_rms[iband] ** 2
            inv_noise_var = np.where(np.isfinite(inv_noise_var), inv_noise_var, 0.0)
            avg_inv_noise_var = np.mean(inv_noise_var)

            diag += Cl * bl ** 2 * avg_inv_noise_var * pix_factor

        self._precond_ell = 1.0 / diag
        logger.info(
            "Preconditioner dynamic range: %.3e  (min/max of M_ell)",
            self._precond_ell.min() / self._precond_ell.max(),
        )

    def preconditioner(self, x):
        """Apply the diagonal harmonic-space preconditioner."""
        return hp.almxfl(x, self._precond_ell)

    def dot_alm(self, alm1, alm2):
        """ Function calculating the dot product of two alms, given that they follow the Healpy standard,
            where alms are represented as complex numbers, but with the conjugate 'negative' ms missing.
        """        
        # return np.sum((alm1[:self.lmax]*alm2[:self.lmax]).real) + np.sum((alm1[self.lmax:]*np.conj(alm2[self.lmax:])).real*2)
        n_m0 = self.lmax + 1  # number of m=0 modes
        return np.sum((alm1[:n_m0]*alm2[:n_m0]).real) + np.sum((alm1[n_m0:]*np.conj(alm2[n_m0:])).real*2)


    def LHS_func(self, x_tilde):
        """ The LHS of the C^{1/2}-renormalized system:
            (I + C^{1/2} sum_i B_i^T Y^T N_i^{-1} Y B_i C^{1/2}) x_tilde
            where x_tilde = C^{-1/2} x.
            All inputs are assumed to be in CMB uK units.
        """
        # Start with the identity contribution
        LHS_sum = x_tilde.copy()

        # C^{1/2} x_tilde
        Cx = hp.almxfl(x_tilde, self.Cl_sqrt)

        for iband in range(self.nband):
            # B C^{1/2} x_tilde
            BCx = hp.smoothalm(Cx.copy(), self.fwhm[iband], inplace=False)
            # Y B C^{1/2} x_tilde
            YBCx = alm2map(BCx, self.nside, self.lmax)
            # N^{-1} Y B C^{1/2} x_tilde
            NYBCx = YBCx / self.map_rms[iband]**2
            # Y^T N^{-1} Y B C^{1/2} x_tilde
            YTNYBCx = alm2map_adjoint(NYBCx, self.nside, self.lmax)
            # B^T Y^T N^{-1} Y B C^{1/2} x_tilde
            BTYTNYBCx = hp.smoothalm(YTNYBCx, self.fwhm[iband], inplace=False)
            # C^{1/2} B^T Y^T N^{-1} Y B C^{1/2} x_tilde
            LHS_sum += hp.almxfl(BTYTNYBCx, self.Cl_sqrt)

        return LHS_sum


    def get_RHS_eqn_mean(self):
        """ Calculates and returns the RHS of the renormalized mean-field equation:
            C^{1/2} sum_i B_i^T Y^T N_i^{-1} d_i
            All inputs are assumed to be in CMB uK units.
        """
        RHS_sum = np.zeros(self.alm_len, dtype=np.complex128)
        for iband in range(self.nband):
            Nd = self.map_sky[iband] / self.map_rms[iband]**2
            YTNd = alm2map_adjoint(Nd, self.nside, self.lmax)
            BTYTNd = hp.smoothalm(YTNd, self.fwhm[iband], inplace=False)
            RHS_sum += BTYTNd
        # Apply C^{1/2}
        RHS_sum = hp.almxfl(RHS_sum, self.Cl_sqrt)
        return RHS_sum


    def get_RHS_eqn_fluct(self):
        """ Calculates and returns the RHS of the renormalized fluctuation equation:
            omega_0 + C^{1/2} sum_i B_i^T Y^T N_i^{-1/2} omega_1
            where omega_0 and omega_1 are drawn from N(0, I).
            All inputs are assumed to be in CMB uK units.
        """
        RHS_sum = np.zeros(self.alm_len, dtype=np.complex128)
        # omega_0 term: in the renormalized system this is simply a unit-variance random alm.
        # hp.synalm(Cl) draws alms with <|a_lm|^2> = Cl, so synalm(ones) gives unit variance.
        omega0 = hp.synalm(np.ones(self.lmax + 1), self.lmax)
        RHS_sum += omega0

        for iband in range(self.nband):
            omega1 = np.random.normal(0, 1, self.npix)
            Nomega1 = omega1 / self.map_rms[iband]  # N^{-1/2} omega_1
            YTNomega1 = alm2map_adjoint(Nomega1, self.nside, self.lmax)
            BTYTNomega1 = hp.smoothalm(YTNomega1, self.fwhm[iband], inplace=False)
            RHS_sum += hp.almxfl(BTYTNomega1, self.Cl_sqrt)  # C^{1/2} B^T Y^T N^{-1/2} omega_1
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
        # logger = logging.getLogger(__name__)
        CG_solver = utils.CG(LHS, RHS, dot=self.dot_alm, M=self.preconditioner)
        err_tol = 1e-100
        iter = 0
        while CG_solver.err > err_tol:
            CG_solver.step()
            iter += 1
            logger.info(f"CG iter {iter:3d} - Residual {CG_solver.err:.3e}")
            print(f"CG iter {iter:3d} - Residual {CG_solver.err:.3e}")
            if iter >= self.maxiter:
                logger.warning(f"Maximum number of iterations ({self.maxiter}) reached in CG.")
                break
        print(f"CG finished after {iter} iterations with a residual of {CG_solver.err:.3e} (err tol = {err_tol})")

        # Recover physical solution: x = C^{1/2} x_tilde
        s_bestfit = hp.almxfl(CG_solver.x, self.Cl_sqrt)

        return s_bestfit


# def constrained_cmb_loop(comm, compsep_master: int, params: dict):
#     master = comm.Get_rank() == 0
#     logger = logging.getLogger(__name__)

#     while True:
#         # check for simulation end
#         stop = MPI.COMM_WORLD.recv(source=compsep_master) if master else False
#         stop = comm.bcast(stop, root=0)
#         if stop:
#             if master:
#                 logger.warning("CMB: stop requested; exiting")
#             return
#         if master:
#             logger.info("CMB: new job obtained")

#         data, iter, chain = MPI.COMM_WORLD.recv(source=compsep_master) if master else None
#         # Broadcast te data to all tasks, or do anything else that's appropriate
#         data = comm.bcast(data, root=0)
#         if master:
#             logger.info("CMB: successfully got data.")
#         if master:
#             signal_maps, rms_maps = data
#             signal_maps = signal_maps[:2]  # Ignore highest frequency band - very dust contaminated.
#             rms_maps = rms_maps[:2]
#             constrained_cmb_solver = ConstrainedCMB(signal_maps, rms_maps, iter)
#             logger.info("CMB: Solving for mean-field map")
#             RHS_mean_field = constrained_cmb_solver.get_RHS_eqn_mean()
#             CMB_mean_field_alms = constrained_cmb_solver.solve_CG(constrained_cmb_solver.LHS_func, RHS_mean_field)
#             CMB_mean_field_Cl = hp.alm2cl(CMB_mean_field_alms)
#             CMB_mean_field_map = alm2map(CMB_mean_field_alms, constrained_cmb_solver.nside, constrained_cmb_solver.lmax)

#             constrained_cmb_solver = ConstrainedCMB(signal_maps, rms_maps, iter)
#             logger.info("CMB: Solving for fluctuation map")
#             RHS_fluct = constrained_cmb_solver.get_RHS_eqn_fluct()
#             CMB_fluct_alms = constrained_cmb_solver.solve_CG(constrained_cmb_solver.LHS_func, RHS_fluct)
#             CMB_fluct_Cl = hp.alm2cl(CMB_fluct_alms)
#             CMB_fluct_map = alm2map(CMB_fluct_alms, constrained_cmb_solver.nside, constrained_cmb_solver.lmax)

#             if params.general.make_plots:
#                 plotting.plot_constrained_cmb_results(
#                     master, params, detector, chain, iter,
#                     constrained_cmb_solver.ell, CMB_mean_field_map,
#                     CMB_fluct_map, signal_maps[0],
#                     constrained_cmb_solver.Cl_true)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Commander4 chain outputs from disk.")
    parser.add_argument(
        "cmb_dir",
        help="Path to a cmb chain output directory (by default called compsep/).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots. Defaults to <cmb_dir>/cmb_realizations.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose printing/logging during run.",
    )

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logger.handlers.clear()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("[cmb_real] %(levelname)s: %(message)s"))
    logger.addHandler(stream_handler)
    logger.setLevel(level)
    logger.propagate = False

    chain_dir = os.path.abspath(args.cmb_dir)
    compsep_dir = os.path.join(chain_dir, "compsep")
    datamaps_dir = os.path.join(chain_dir, "datamaps")

    if not os.path.isdir(compsep_dir) or not os.path.isdir(datamaps_dir):
        logger.error(f"Chain directory not found: {compsep_dir} or {datamaps_dir}")
        return 1

    compsep_files = sorted(os.listdir(compsep_dir))
    datamaps_files = sorted(os.listdir(datamaps_dir))

    bands = []
    iters = []
    for datamaps_file in datamaps_files:
        band, chain, iter = _extract_band_chain_iter(datamaps_file)
        bands.append(band)
        iters.append(iter)
    iters = np.unique(iters)
    bands = np.unique(bands)

    chain = 1
    for iter in iters:
        signal_maps = []
        rms_maps = []
        foreground_maps = []
        for band in bands:
            compsep_filename = f"chain01_iter{iter:04d}.h5"
            compsep_filepath = os.path.join(compsep_dir, compsep_filename)
            datamaps_filename = f"{band}_chain01_iter{iter:04d}.h5"
            datamaps_filepath = os.path.join(datamaps_dir, datamaps_filename)

            band_freq = float(datamaps_filename.split("GHz")[0].split("LiteBIRD")[-1])

            print(compsep_filepath, datamaps_filename, chain, iter, band_freq)

            with h5py.File(datamaps_filepath, "r") as f:
                map_observed_sky = f["map_observed_sky"][0]
                map_rms = f["map_rms"][0]
            map_nside = hp.npix2nside(map_rms.shape[-1])
            print(f"map_nside: {map_nside}")

            with h5py.File(compsep_filepath, "r") as f:
                cmb_alm = f["/comps/cmb_I/alms"][()]
                dust_alm = f["/comps/dust_I/alms"][()]
                ff_alm = f["/comps/ff_I/alms"][()]
                sync_alm = f["/comps/sync_I/alms"][()]

            global_params = Bunch()
            global_params.CG_float_precision = -3.1
            dust_params = Bunch()
            dust_params.beta = 1.56
            dust_params.T = 20
            dust_params.nu0 = 545
            dust_params.lmax = 1024
            dust_params.spatially_varying_MM = False
            dust_params.smoothing_prior_FWHM = 0
            dust_params.smoothing_prior_amplitude = 1.0
            dust_params.polarized = False 
            dust_params.longname = "Thermal Dust"
            dust_params.shortname = "dust"
            dust = ThermalDust(dust_params, global_params)
            dust.alms = dust_alm
            ff_params = Bunch()
            ff_params.T = 7000
            ff_params.nu0 = 0.408
            ff_params.lmax = 1024
            ff_params.spatially_varying_MM = False
            ff_params.smoothing_prior_FWHM = 0
            ff_params.smoothing_prior_amplitude = 1.0
            ff_params.polarized = False 
            ff_params.longname = "Free Free"
            ff_params.shortname = "ff"
            ff = FreeFree(ff_params, global_params)
            ff.alms = ff_alm
            sync_params = Bunch()
            sync_params.beta = -3.1
            sync_params.nu0 = 30
            sync_params.lmax = 1024
            sync_params.spatially_varying_MM = False
            sync_params.smoothing_prior_FWHM = 0
            sync_params.smoothing_prior_amplitude = 1.0
            sync_params.polarized = False 
            sync_params.longname = "Synchrotron"
            sync_params.shortname = "sync"
            sync = Synchrotron(sync_params, global_params)
            sync.alms = sync_alm

            components = [dust,ff,sync]
            sky = SkyModel(components)
            foreground_map = sky.get_sky_at_nu(band_freq, map_nside, fwhm=0)[0]

            # lmax = hp.Alm.getlmax(cmb_alm.shape[0])
            # for idx in [hp.Alm.getidx(lmax, 0, 0),
            #             hp.Alm.getidx(lmax, 1, 0),
            #             hp.Alm.getidx(lmax, 1, 1)]:
            #     cmb_alm[:,idx] = 0.0
            cmb_map = hp.alm2map(cmb_alm[0], map_nside)
            # foreground_map = hp.alm2map(dust_alm, map_nside)
            map_observed_sky -= foreground_map

            cmb_Cell = hp.alm2cl(cmb_alm[0].astype(np.complex128))
            # Loose prior on monopole/dipole: large enough that data dominates,
            # small enough to avoid floating-point issues with C^{1/2} renormalization.
            cmb_Cell[:2] = 1000*np.max(cmb_Cell[2:])

            hdul = fits.open("/mn/stornext/d5/data/duncanwa/WMAP/data/mask_proc_030_res_v5.fits")
            binary_mask = hdul[1].data["TEMPERATURE"].flatten().astype(bool)
            binary_mask = hp.ud_grade(binary_mask, map_nside)
            fwhm_rad = np.radians(3.0)
            smoothed_mask = hp.smoothing(binary_mask, fwhm=fwhm_rad)
            smoothed_mask[smoothed_mask < 0.0] = 0.0

            # inv_var = np.ones_like(cmb_map)*1e-10
            # name = "1e-10"

            # rms[:,~mask] = np.inf
            # inv_var *= smoothed_mask
            # rms = 1.0/np.sqrt(inv_var)
            map_rms /= smoothed_mask

            # Convert data and RMS from RJ uK to CMB uK before feeding
            # into the solver, so the solver stays unit-agnostic.
            import pysm3.units as pysm3u
            from astropy import units as u
            g_nu = (1.0 * pysm3u.uK_CMB).to(
                pysm3u.uK_RJ,
                equivalencies=pysm3u.cmb_equivalencies(band_freq * u.GHz)
            ).value  # CMB -> RJ factor
            # Divide by g to go from RJ -> CMB:  T_CMB = T_RJ / g
            map_observed_sky /= g_nu
            map_rms /= g_nu

            signal_maps.append(map_observed_sky)
            rms_maps.append(map_rms)
            foreground_maps.append(foreground_map)
        
        signal_maps = np.array(signal_maps)
        rms_maps = np.array(rms_maps)
        dir = "plots/precond_v2/"
        os.makedirs(dir, exist_ok=True)
        plt.figure()
        hp.mollview(signal_maps[5] + foreground_maps[5])
        plt.savefig(f"{dir}map_observed_sky.png")
        plt.close()
        plt.figure()
        hp.mollview(foreground_maps[5])
        plt.savefig(f"{dir}foreground_map.png")
        plt.close()
        for i in range(len(signal_maps)):
            plt.figure()
            hp.mollview(signal_maps[5])
            plt.savefig(f"{dir}signal_map{i}.png")
        plt.close()
        plt.figure()
        hp.mollview(signal_maps[5]/rms_maps[5])
        plt.savefig(f"{dir}signal_map_norm.png")
        plt.close()

        CMB = ConstrainedCMB(signal_maps, rms_maps, cmb_Cell, maxiter=100)
        rhs = CMB.get_RHS_eqn_mean()
        rhs += CMB.get_RHS_eqn_fluct()

        cmb_alms_bestfit = CMB.solve_CG(CMB.LHS_func, rhs)
        cmb_map_bestfit = hp.alm2map(cmb_alms_bestfit, 512)

        print(cmb_alm.shape, cmb_alms_bestfit.shape)

        plt.figure()
        plt.loglog(hp.alm2cl(cmb_alm[0].astype(np.complex128)), label="prior")
        plt.loglog(hp.alm2cl(cmb_alms_bestfit), label="bestfit")
        plt.legend()
        plt.savefig(f"{dir}Cell_cmb.png")
        plt.close()
        plt.figure()
        hp.mollview(cmb_map_bestfit, cmap="RdBu_r")
        plt.savefig(f"{dir}test_cmb.png")
        plt.close()
        plt.figure()
        hp.mollview(cmb_map, cmap="RdBu_r")
        plt.savefig(f"{dir}initial_cmb.png")
        plt.close()
        plt.figure()
        hp.mollview(rms_maps[0])
        plt.savefig(f"{dir}rms_cmb.png")
        plt.close()

        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())