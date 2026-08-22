"""Offline tool: draw constrained CMB realizations from a finished chain's band maps.

Reads the datamaps written by a run, solves the constrained-realization system for the CMB alms
with a supplied C_l prior, and writes the resulting maps. The in-process versions of the same solve
live in ``compsep/constrained_cmb_loop`` and ``compsep/constrained_cmb_loop_mpi``.
"""
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
import glob
import re
import yaml
from astropy.io import fits
from pixell.bunch import Bunch

from commander4.file_io import paths
from commander4.parameters.bunch import as_bunch_recursive
from commander4.sky.comp_io import _read_view_alms_from_chain
from commander4.sky.comp_list import CompList
from commander4.sky.diffuse_components import CMB
from commander4.sky.sky_model import SkyModel
from commander4.units import rj_to_band_unit_factor

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
    """The constrained-realization system for the CMB alms, with an externally supplied C_l."""

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
        logger.debug(
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


    def solve_CG(self, LHS, RHS, err_tol: float = 1e-6):
        """ Solves the equation Ax=b for x given A (LHS) and b (RHS) using CG from the pixell package.
            Assumes that both x and b are in alm space.

            Args:
                LHS: A callable taking x as argument and returning Ax.
                RHS: A Numpy array representing b, in alm space.
                err_tol: Residual below which CG stops; `maxiter` caps it either way.
            Returns:
                m_bestfit: The resulting best-fit solution, in alm space.
        """
        CG_solver = utils.CG(LHS, RHS, dot=self.dot_alm, M=self.preconditioner)
        iter = 0
        while CG_solver.err > err_tol:
            CG_solver.step()
            iter += 1
            logger.debug(f"CG iter {iter:3d} - Residual {CG_solver.err:.3e}")
            if iter >= self.maxiter:
                logger.warning(f"Maximum number of iterations ({self.maxiter}) reached in CG "
                               f"at residual {CG_solver.err:.3e} (tolerance {err_tol:.1e}).")
                break
        else:
            logger.info(f"CG converged after {iter} iterations "
                        f"(residual {CG_solver.err:.3e} < {err_tol:.1e}).")

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

#             if params.output.plots.enabled:
#                 plotting.plot_constrained_cmb_results(
#                     master, params, detector, chain, iter,
#                     constrained_cmb_solver.ell, CMB_mean_field_map,
#                     CMB_fluct_map, signal_maps[0],
#                     constrained_cmb_solver.Cl_true)


def _load_params_from_chain(run_dir: str) -> Bunch | None:
    """The parameter file the run was launched with, read back out of its chain files.

    Every chain file stores the parameter file verbatim under `metadata/parameter_file_as_string`,
    which is what lets this tool rebuild the run's components without being told anything about
    them. Returns None if no chain file in `run_dir` carries it.
    """
    patterns = [os.path.join(run_dir, paths.CHAINS_COMPSEP, "*.h5"),
                os.path.join(run_dir, paths.CHAINS_DATAMAPS, "*.h5")]
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            try:
                with h5py.File(path, "r") as f:
                    if "metadata/parameter_file_as_string" not in f:
                        continue
                    raw_yaml = f["metadata/parameter_file_as_string"][()]
            except OSError:
                continue
            if isinstance(raw_yaml, bytes):
                raw_yaml = raw_yaml.decode("utf-8")
            return as_bunch_recursive(yaml.safe_load(raw_yaml))
    return None


def _band_frequencies(params: Bunch) -> dict[str, float]:
    """Map every band name in the parameter file onto its centre frequency in GHz."""
    freqs = {}
    for experiment_name in params.experiments:
        experiment = params.experiments[experiment_name]
        if "bands" not in experiment:
            continue
        for band_name in experiment.bands:
            freqs[band_name] = float(experiment.bands[band_name].freq)
    return freqs


def _build_intensity_components(params: Bunch, compsep_path: str) -> list:
    """The run's components, with their intensity amplitudes read from one compsep chain file.

    The components are constructed by the same code the main program uses, so their SEDs and
    reference frequencies come from the run's own parameter file rather than being restated here.
    """
    comp_list = CompList.init_from_params(params.components, params)
    intensity_comps = comp_list.components_for_eval_pol("I")
    for comp in intensity_comps:
        alms = _read_view_alms_from_chain(comp, compsep_path)
        if alms is None:
            raise ValueError(f"Component {comp.comp_name!r} has no intensity alms in "
                             f"{compsep_path!r}.")
        comp.alms = alms
    return intensity_comps


def _read_mask(mask_path: str, nside: int, smoothing_fwhm_deg: float) -> np.ndarray:
    """A smoothed apodization mask at `nside`, from a binary mask in a FITS file.

    The mask divides the RMS, so masked pixels get a large (eventually infinite) RMS and are
    effectively excluded from the solve.
    """
    with fits.open(mask_path) as hdul:
        binary_mask = hdul[1].data["TEMPERATURE"].flatten().astype(bool)
    binary_mask = hp.ud_grade(binary_mask, nside)
    smoothed_mask = hp.smoothing(binary_mask.astype(np.float64),
                                 fwhm=np.radians(smoothing_fwhm_deg))
    smoothed_mask[smoothed_mask < 0.0] = 0.0
    return smoothed_mask


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draw constrained CMB realizations from a finished Commander4 run.")
    parser.add_argument(
        "run_dir",
        help=f"Path to a Commander4 run's output directory (its `output.dir`, containing "
             f"{paths.CHAINS_COMPSEP}/ and {paths.CHAINS_DATAMAPS}/).")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for outputs. Defaults to <run_dir>/cmb_realizations.")
    parser.add_argument("--iter", type=int, default=None, dest="only_iter",
                        help="Process only this Gibbs iteration (default: all found).")
    parser.add_argument("--chain", type=int, default=1, help="Chain number to read (default 1).")
    parser.add_argument("--maxiter", type=int, default=100,
                        help="Maximum CG iterations (default 100).")
    parser.add_argument("--err-tol", type=float, default=1e-6,
                        help="CG residual to stop at (default 1e-6).")
    parser.add_argument("--mask", default=None,
                        help="FITS binary mask (TEMPERATURE column) dividing the RMS. Optional.")
    parser.add_argument("--mask-fwhm-deg", type=float, default=3.0,
                        help="FWHM in degrees the mask is smoothed by (default 3).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug-level logging.")
    args = parser.parse_args()

    logger.handlers.clear()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("[cmb_real] %(levelname)s: %(message)s"))
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logger.propagate = False

    run_dir = os.path.abspath(args.run_dir)
    compsep_dir = os.path.join(run_dir, paths.CHAINS_COMPSEP)
    datamaps_dir = os.path.join(run_dir, paths.CHAINS_DATAMAPS)
    if not os.path.isdir(compsep_dir) or not os.path.isdir(datamaps_dir):
        logger.error(f"Run output directory not found: {compsep_dir} or {datamaps_dir}")
        return 1

    params = _load_params_from_chain(run_dir)
    if params is None:
        logger.error(f"No chain file in {run_dir} carries the parameter file, so the run's "
                     f"components cannot be reconstructed.")
        return 1
    band_freqs = _band_frequencies(params)

    output_dir = args.output_dir or os.path.join(run_dir, "cmb_realizations")
    os.makedirs(output_dir, exist_ok=True)

    # Which (band, iteration) pairs the run actually wrote maps for.
    bands_by_iter = {}
    for filename in sorted(os.listdir(datamaps_dir)):
        band, chain, iteration = _extract_band_chain_iter(filename)
        if band is None or chain != args.chain:
            continue
        bands_by_iter.setdefault(iteration, []).append((band, filename))
    iterations = sorted(bands_by_iter)
    if args.only_iter is not None:
        iterations = [it for it in iterations if it == args.only_iter]
    if not iterations:
        logger.error(f"No datamaps found in {datamaps_dir} for chain {args.chain}.")
        return 1

    for iteration in iterations:
        compsep_path = os.path.join(compsep_dir, f"chain{args.chain:02d}_iter{iteration:04d}.h5")
        if not os.path.isfile(compsep_path):
            logger.warning(f"No compsep chain for iteration {iteration}; skipping.")
            continue

        components = _build_intensity_components(params, compsep_path)
        cmb_comps = [comp for comp in components if isinstance(comp, CMB)]
        foreground_comps = [comp for comp in components if not isinstance(comp, CMB)]
        if len(cmb_comps) != 1:
            logger.error(f"Expected exactly one CMB component, found {len(cmb_comps)}.")
            return 1
        foreground_sky = SkyModel(foreground_comps)

        signal_maps = []
        rms_maps = []
        used_bands = []
        for band, filename in bands_by_iter[iteration]:
            band_name = band.split("_")[-1]
            if band_name not in band_freqs:
                logger.warning(f"Band {band_name!r} is not in the parameter file; skipping.")
                continue
            nu = band_freqs[band_name]
            with h5py.File(os.path.join(datamaps_dir, filename), "r") as f:
                map_observed_sky = f["map_observed_sky"][0].astype(np.float64)
                map_rms = f["map_rms"][0].astype(np.float64)
                stored_unit = f["metadata/band_unit"][()] if "metadata/band_unit" in f else b"uK_RJ"
            if isinstance(stored_unit, bytes):
                stored_unit = stored_unit.decode("utf-8")
            nside = hp.npix2nside(map_rms.shape[-1])

            # The foreground model is evaluated at this band's frequency and subtracted, leaving
            # CMB + noise for the solver.
            foreground_map = foreground_sky.get_sky_at_nu(nu, nside, "I", fwhm=0)[0]
            map_observed_sky -= foreground_map

            # The solver works in thermodynamic units, the maps are written in the band's own unit.
            to_uK_CMB = (rj_to_band_unit_factor(nu, "uK_CMB")
                         / rj_to_band_unit_factor(nu, stored_unit))
            map_observed_sky *= to_uK_CMB
            map_rms *= to_uK_CMB

            if args.mask is not None:
                map_rms = map_rms / _read_mask(args.mask, nside, args.mask_fwhm_deg)

            signal_maps.append(map_observed_sky)
            rms_maps.append(map_rms)
            used_bands.append((band_name, nu))
            logger.info(f"iter {iteration}: read {band_name} ({nu:g} GHz, {stored_unit}) "
                        f"at nside {nside}.")

        if not signal_maps:
            logger.warning(f"No usable bands for iteration {iteration}; skipping.")
            continue

        cmb_alms_in = np.ascontiguousarray(cmb_comps[0].alms[0]).astype(np.complex128)
        cmb_Cell = hp.alm2cl(cmb_alms_in)
        # Loose prior on monopole/dipole: large enough that the data dominates, small enough to
        # avoid floating-point trouble in the C^{1/2} renormalization.
        cmb_Cell[:2] = 1000 * np.max(cmb_Cell[2:])

        solver = ConstrainedCMB(np.array(signal_maps), np.array(rms_maps), cmb_Cell,
                                maxiter=args.maxiter)
        rhs = solver.get_RHS_eqn_mean() + solver.get_RHS_eqn_fluct()
        cmb_alms_bestfit = solver.solve_CG(solver.LHS_func, rhs, err_tol=args.err_tol)

        nside = hp.npix2nside(signal_maps[0].shape[-1])
        cmb_map_bestfit = hp.alm2map(cmb_alms_bestfit, nside)
        out_base = os.path.join(output_dir, f"chain{args.chain:02d}_iter{iteration:04d}")
        hp.write_map(f"{out_base}_cmb_realization.fits", cmb_map_bestfit, overwrite=True)

        plt.figure()
        plt.loglog(hp.alm2cl(cmb_alms_in), label="compsep CMB")
        plt.loglog(hp.alm2cl(cmb_alms_bestfit), label="constrained realization")
        plt.xlabel("multipole $\\ell$")
        plt.ylabel("$C_\\ell$")
        plt.legend()
        plt.savefig(f"{out_base}_Cell.png", dpi=120, bbox_inches="tight")
        plt.close()

        plt.figure()
        hp.mollview(cmb_map_bestfit, cmap="RdBu_r", title=f"Constrained CMB, iter {iteration}")
        plt.savefig(f"{out_base}_cmb_realization.png", dpi=120, bbox_inches="tight")
        plt.close()
        logger.info(f"iter {iteration}: wrote {out_base}_cmb_realization.fits (+ 2 figures) from "
                    f"{len(used_bands)} bands.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
