"""Offline tool: draw constrained CMB realizations from a finished chain's band maps.

Subtracts each saved sample's foregrounds from its band maps, then draws CMB alms conditioned on
those residuals and a fixed theoretical C_l prior. The mask and low-ell preconditioner are shared
across Gibbs samples and realizations. Only intensity is sampled; all calculations use uK_CMB.
"""
import argparse
import glob
import logging
import os
import re
from collections.abc import Callable, Sequence

import ducc0
import h5py
import healpy as hp
import matplotlib
import numpy as np
import yaml
from numpy.typing import NDArray
from pixell import enmap, utils
from pixell.bunch import Bunch
from scipy.linalg import cho_factor, cho_solve

# Select a non-interactive backend before importing pyplot: this tool also runs in batch jobs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from commander4.file_io import paths
from commander4.parameters.bunch import as_bunch_recursive
from commander4.sky.comp_io import _read_view_alms_from_chain
from commander4.sky.comp_list import CompList
from commander4.sky.diffuse_components import CMB, DiffuseComponent
from commander4.sky.sky_model import SkyModel
from commander4.units import rj_to_band_unit_factor

logger = logging.getLogger("cmb_realizations")


BAND_CHAIN_ITER_RE = re.compile(r"(?:(?P<prefix>.+)_)?chain(?P<chain>\d+)_iter(?P<iter>\d+)\.h5$")
nthreads = 32  # Threads per ducc spherical-harmonic transform.


def _extract_band_chain_iter(filename: str) -> tuple[str | None, int | None, int | None]:
    """Read the experiment/band prefix, chain number and iteration from a band filename."""
    match = BAND_CHAIN_ITER_RE.search(filename)
    if not match:
        return None, None, None
    return str(match.group("prefix")), int(match.group("chain")), int(match.group("iter"))


def alm2map(alm: NDArray, nside: int, lmax: int) -> NDArray:
    """Synthesize a scalar RING map from healpy-packed alms (the operator Y)."""
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    return ducc0.sht.synthesis(alm=alm.reshape(1, -1), lmax=lmax, spin=0,
                               nthreads=nthreads, **geom).reshape(-1)


def alm2map_adjoint(sky_map: NDArray, nside: int, lmax: int) -> NDArray:
    """Apply Y^T, the adjoint of synthesis, without pixel-area normalization.

    This is not an inverse map-to-alm transform. The likelihood needs the sum over weighted
    pixels, so multiplying by the pixel area here would change the noise normalization.
    """
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    return ducc0.sht.adjoint_synthesis(map=sky_map.reshape(1, -1), lmax=lmax, spin=0,
                                       nthreads=nthreads, **geom).reshape(-1)


class ConstrainedCMB:
    """Sample CMB alms in uK_CMB, with C_l for ell = 0 through the CMB lmax.

    Sky maps must be in uK_CMB, inverse variances in uK_CMB**-2, and the prior in uK_CMB**2.
    Maps, inverse variances and masks are sequences of one-dimensional arrays, one per band.
    Each band keeps its native HEALPix resolution; all bands use the same CMB harmonic range.
    Masks are dimensionless and beam FWHMs are in radians.

    We solve for x_tilde with CMB alms s = sqrt(C) x_tilde. In these coordinates the prior
    precision is the identity, avoiding explicit division by small or zero C_l values.
    """

    def __init__(self, map_sky: Sequence[NDArray], map_ivar: Sequence[NDArray], cmb_Cell: NDArray,
                 masks: Sequence[NDArray] | None = None, beam_fwhm: NDArray | None = None,
                 maxiter: int = 100, precond_lmax: int = 32) -> None:
        if precond_lmax < 0:
            raise ValueError("Preconditioner lmax must be nonnegative; 0 selects diagonal only.")
        self.maxiter = maxiter
        self.precond_lmax = precond_lmax
        self.map_sky = list(map_sky)
        self.map_ivar = list(map_ivar)
        self.nband = len(self.map_sky)
        self.masks = [] if masks is None else list(masks)
        self.npix: list[int] = []
        self.nsides: list[int] = []
        if masks is None:
            for sky in self.map_sky:
                self.masks.append(np.ones_like(sky))
        if len(self.map_ivar) != self.nband or len(self.masks) != self.nband:
            raise ValueError("Maps, inverse variances and masks must have the same band count.")
        for band, sky in enumerate(self.map_sky):
            if (sky.ndim != 1 or self.map_ivar[band].shape != sky.shape
                    or self.masks[band].shape != sky.shape):
                raise ValueError("Each band needs matching map, inverse-variance and mask shapes.")
            self.npix.append(sky.size)
            self.nsides.append(hp.npix2nside(sky.size))
        self.fwhm = np.full(self.nband, np.radians(1.0 / 60.0))
        if beam_fwhm is not None:
            self.fwhm = beam_fwhm
        # The component's harmonic range is independent of the bands' pixel resolutions.
        self.lmax = len(cmb_Cell) - 1
        self.alm_len = hp.Alm.getsize(self.lmax)
        self.Cl_prior = cmb_Cell

        # Zero-variance modes stay fixed at zero; negative prior values are treated as zero.
        self.Cl_sqrt = np.sqrt(np.maximum(self.Cl_prior, 0.0))

        # Use a coupled low-ell block for masked skies, and a diagonal approximation above it.
        self._build_preconditioner()

    def update_data(self, map_sky: Sequence[NDArray], map_ivar: Sequence[NDArray],
                    beam_fwhm: NDArray) -> None:
        """Update a Gibbs sample's likelihood while retaining the shared low-ell factor.

        Map shapes, mask and C_l prior must stay fixed. Noise and beams may change: the original
        low-ell factor remains a valid positive-definite preconditioner, though its effectiveness
        can change. Only the inexpensive harmonic diagonal is refreshed.
        """
        if len(map_sky) != self.nband or len(map_ivar) != self.nband:
            raise ValueError("Reuse requires the same band count and map nsides.")
        for band, sky in enumerate(map_sky):
            if sky.shape != (self.npix[band],) or map_ivar[band].shape != sky.shape:
                raise ValueError("Reuse requires the same band count and map nsides.")
        self.map_sky = list(map_sky)
        self.map_ivar = list(map_ivar)
        self.fwhm = beam_fwhm
        self._build_preconditioner(rebuild_lowell=False)

    def _build_preconditioner(self, rebuild_lowell: bool = True) -> None:
        """Build the harmonic diagonal and, for masked data, a coupled low-ell correction.

        Replacing each band's pixel weights by their whole-sky average gives a diagonal
        approximation to LHS_func:

            d_ell = 1 + C_ell * sum_i (Npix_i / 4pi) * b_ell_i^2 * <mask_i / sigma_i^2>

        The high-ell preconditioner is M = 1 / d_ell. A dense block replaces its low-ell entries
        when pixels have zero weight. This captures mask-induced coupling between harmonic modes.
        """
        cl = self.Cl_sqrt**2
        diag = np.ones(self.lmax + 1)  # identity contribution

        # The LHS uses ducc0 synthesis (Y) and adjoint_synthesis (Y^T),
        # which are unnormalized: Y^T Y ≈ (Npix/4π) I in harmonic space.
        # The preconditioner must include this factor to match the true diagonal.
        self.beams = []
        weights = []
        has_zero_weight = False
        has_positive_weight = False

        for iband in range(self.nband):
            bl = hp.gauss_beam(self.fwhm[iband], lmax=self.lmax)
            self.beams.append(bl)
            weight = self.map_ivar[iband] * self.masks[iband]
            weight = np.where(np.isfinite(weight), weight, 0.0)
            weights.append(weight)
            has_zero_weight |= np.any(weight == 0)
            has_positive_weight |= np.any(weight > 0)
            # Include masked pixels in the average; averaging only retained pixels overweights
            # partially observed bands. Each native grid contributes its own pixel count.
            avg_inv_noise_var = np.mean(weight)
            pix_factor = self.npix[iband] / (4.0 * np.pi)
            diag += cl * bl**2 * avg_inv_noise_var * pix_factor

        self._precond_ell = 1.0 / diag
        logger.debug(
            "Preconditioner dynamic range: %.3e  (min/max of M_ell)",
            self._precond_ell.min() / self._precond_ell.max(),
        )

        if rebuild_lowell:
            self._lowell_factor = None
            if self.precond_lmax > 0 and has_zero_weight and has_positive_weight:
                self._build_lowell_preconditioner(weights, min(self.precond_lmax, self.lmax))

    def _build_lowell_preconditioner(self, weights: Sequence[NDArray], lmax: int) -> None:
        """Factor a real-harmonic low-ell block, following C3's coupled CMB preconditioner.

        Only the preconditioner uses a coarser pixel grid. Summing inverse variances into its
        pixels preserves their total weight. The likelihood continues to use the original maps.
        Real coordinates are a_l0, sqrt(2)*Re(a_lm), sqrt(2)*Im(a_lm), giving a Euclidean inner
        product equal to dot_alm and a symmetric positive-definite matrix for Cholesky.
        """
        # Round up to a HEALPix power of two, but never upgrade a band's native grid.
        target_nside = 2**int(np.ceil(np.log2(max(1, lmax))))
        coarse_nsides = []
        coarse_weights = []
        for band, weight in enumerate(weights):
            nside = min(self.nsides[band], target_nside)
            coarse_nsides.append(nside)
            # power=-2 sums weights into coarse pixels instead of averaging them.
            coarse_weights.append(hp.ud_grade(weight, nside, power=-2))
        n_m0 = lmax + 1
        nalm = hp.Alm.getsize(lmax)
        nmodes = (lmax + 1)**2
        matrix = np.empty((nmodes, nmodes))
        sqrt_cl = self.Cl_sqrt[:lmax + 1]

        # Build the matrix one real basis vector at a time. The coordinate order is all m=0
        # modes, then real parts for m>0, then imaginary parts for m>0.
        for column in range(nmodes):
            basis = np.zeros(nalm, dtype=np.complex128)
            if column < n_m0:
                basis[column] = 1.0
            elif column < nalm:
                basis[column] = 1.0 / np.sqrt(2.0)
            else:
                basis[column - nalm + n_m0] = 1j / np.sqrt(2.0)
            result = basis.copy()  # Identity contribution from the renormalized prior.
            for band in range(self.nband):
                nside = coarse_nsides[band]
                response = sqrt_cl * self.beams[band][:lmax + 1]
                sky = alm2map(hp.almxfl(basis, response), nside, lmax)
                weighted_alm = alm2map_adjoint(sky * coarse_weights[band], nside, lmax)
                result += hp.almxfl(weighted_alm, response)
            matrix[:n_m0, column] = result[:n_m0].real
            matrix[n_m0:nalm, column] = np.sqrt(2.0) * result[n_m0:].real
            matrix[nalm:, column] = np.sqrt(2.0) * result[n_m0:].imag

        # Remove transform roundoff asymmetry before Cholesky factorization.
        matrix = 0.5 * (matrix + matrix.T)
        self._lowell_factor = cho_factor(matrix, lower=True, overwrite_a=True, check_finite=False)
        ell, m = hp.Alm.getlm(lmax)
        # Packed alm offsets depend on lmax: the low-ell modes are not a contiguous slice
        # of the full array once m>0, so map their (ell, m) indices explicitly.
        self._lowell_indices = hp.Alm.getidx(self.lmax, ell, m)
        self._lowell_n_m0 = n_m0
        logger.info(f"Built coupled CMB preconditioner through ell={lmax} "
                    f"({nmodes} real modes, band nsides={coarse_nsides}).")

    def preconditioner(self, x: NDArray) -> NDArray:
        """Apply the coupled low-ell block and the harmonic diagonal at higher ell."""
        result = hp.almxfl(x, self._precond_ell)
        if self._lowell_factor is not None:
            low = x[self._lowell_indices]  # Advanced indexing copies; x stays unchanged.
            n_m0 = self._lowell_n_m0
            nalm = low.size
            rhs = np.concatenate([low[:n_m0].real, np.sqrt(2.0) * low[n_m0:].real,
                                  np.sqrt(2.0) * low[n_m0:].imag])
            solved = cho_solve(self._lowell_factor, rhs, check_finite=False)
            low[:n_m0] = solved[:n_m0]
            low[n_m0:] = (solved[n_m0:nalm] + 1j * solved[nalm:]) / np.sqrt(2.0)
            # Replace the low-ell diagonal result, rather than adding a second correction.
            result[self._lowell_indices] = low
        return result

    def dot_alm(self, alm1: NDArray, alm2: NDArray) -> float:
        """Real inner product for healpy-packed alms, including the unstored negative-m modes."""
        # A real sky has real m=0 coefficients. Each stored m>0 coefficient represents a
        # conjugate pair, so its contribution to the full harmonic inner product counts twice.
        n_m0 = self.lmax + 1
        m0_product = np.sum((alm1[:n_m0] * alm2[:n_m0]).real)
        positive_m_product = np.sum((alm1[n_m0:] * np.conj(alm2[n_m0:])).real * 2)
        return float(m0_product + positive_m_product)

    def LHS_func(self, x_tilde: NDArray) -> NDArray:
        """Apply A = I + sqrt(C) sum_i B_i Y_i^T W_i Y_i B_i sqrt(C).

        W_i is the masked inverse variance. Y_i synthesizes on band i's native pixel grid;
        the Gaussian beam B_i and prior sqrt(C) are diagonal in harmonic space.
        """
        result = x_tilde.copy()  # The identity term is the renormalized prior precision.
        sky_alm = hp.almxfl(x_tilde, self.Cl_sqrt)
        for band in range(self.nband):
            # Project the same CMB onto each band's beam/grid, weight its pixels, then apply
            # the adjoint projection. Both sides need the beam and sqrt(C) for symmetry.
            beamed_alm = hp.almxfl(sky_alm, self.beams[band])
            sky_map = alm2map(beamed_alm, self.nsides[band], self.lmax)
            weighted_map = sky_map * self.map_ivar[band] * self.masks[band]
            weighted_alm = alm2map_adjoint(weighted_map, self.nsides[band], self.lmax)
            beamed_alm = hp.almxfl(weighted_alm, self.beams[band])
            result += hp.almxfl(beamed_alm, self.Cl_sqrt)
        return result

    def get_RHS_eqn_mean(self) -> NDArray:
        """Return sqrt(C) sum_i B_i Y_i^T W_i d_i for foreground-subtracted band maps d_i."""
        rhs = np.zeros(self.alm_len, dtype=np.complex128)
        for band in range(self.nband):
            weighted_map = self.map_sky[band] * self.map_ivar[band] * self.masks[band]
            weighted_alm = alm2map_adjoint(weighted_map, self.nsides[band], self.lmax)
            rhs += hp.almxfl(weighted_alm, self.beams[band])
        return hp.almxfl(rhs, self.Cl_sqrt)

    def get_RHS_eqn_fluct(self) -> NDArray:
        """Return omega_0 + sqrt(C) sum_i B_i Y_i^T sqrt(W_i) omega_i.

        Independent unit Gaussian harmonic and pixel draws give this RHS covariance A.
        Solving against A gives posterior covariance A^{-1} in the x_tilde coordinates.
        """
        # synalm gives variance 1 to m=0 and variance 1/2 to each real/imaginary part for m>0.
        # These are unit Gaussians in the same real coordinates used by the preconditioner.
        rhs = hp.synalm(np.ones(self.lmax + 1), self.lmax)
        for band in range(self.nband):
            pixel_noise = np.random.normal(0, 1, self.npix[band])
            # The mask multiplies inverse variance, so fractional apodization weights must
            # enter under the square root here to agree with the likelihood covariance.
            weighted_noise = pixel_noise * np.sqrt(self.map_ivar[band] * self.masks[band])
            noise_alm = alm2map_adjoint(weighted_noise, self.nsides[band], self.lmax)
            beamed_alm = hp.almxfl(noise_alm, self.beams[band])
            rhs += hp.almxfl(beamed_alm, self.Cl_sqrt)
        return rhs

    def solve_CG(self, LHS: Callable[[NDArray], NDArray], RHS: NDArray,
                 err_tol: float = 1e-6) -> NDArray:
        """Solve the renormalized system with CG and return physical CMB alms in uK_CMB.

        Args:
            LHS: Operator applying A to renormalized harmonic coefficients.
            RHS: Mean RHS, fluctuation RHS, or their sum in the same coordinates.
            err_tol: Pixell's squared relative preconditioned residual,
                (r^T M r) / (r_initial^T M r_initial), where M applies the preconditioner.

        Returns:
            CMB alms sqrt(C) x_tilde. Reaching maxiter logs a warning and returns the current
            iterate even if it has not converged. The stopping test uses CG's recursive residual.
        """
        cg = utils.CG(LHS, RHS, dot=self.dot_alm, M=self.preconditioner)
        iteration = 0
        while cg.err > err_tol:
            cg.step()
            iteration += 1
            logger.debug(f"CG iter {iteration:3d} - Residual {cg.err:.3e}")
            if iteration >= self.maxiter:
                logger.warning(f"Maximum number of iterations ({self.maxiter}) reached in CG "
                               f"at residual {cg.err:.3e} (tolerance {err_tol:.1e}).")
                break
        else:
            logger.info(f"CG converged after {iteration} iterations "
                        f"(residual {cg.err:.3e} < {err_tol:.1e}).")
        return hp.almxfl(cg.x, self.Cl_sqrt)


def _load_params_from_chain(run_dir: str) -> Bunch | None:
    """The parameter file the run was launched with, read back out of its chain files.

    Every chain file stores the parameter file verbatim under `metadata/parameter_file_as_string`,
    which is what lets this tool rebuild the run's components without being told anything about
    them. Returns None if no chain file in `run_dir` carries it.
    """
    patterns = [os.path.join(run_dir, paths.CHAINS_COMPSEP, "*.h5"),
                os.path.join(run_dir, paths.CHAINS_BANDS, "*.h5")]
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
    freqs: dict[str, float] = {}
    for experiment_name in params.experiments:
        experiment = params.experiments[experiment_name]
        if "bands" not in experiment:
            continue
        for band_name in experiment.bands:
            freqs[band_name] = float(experiment.bands[band_name].freq)
    return freqs


def _build_intensity_components(params: Bunch, compsep_path: str) -> list[DiffuseComponent]:
    """Restore intensity alms, SED parameters and amplitude beams from one saved sample.

    The parameter file defines the component types. Their saved SED state, including fixed
    parameters, defines the foregrounds to subtract. Joined IQU parameters use their I value.
    Missing state is an error: initial parameter values cannot replace a saved Gibbs sample.
    """
    comp_list = CompList.init_from_params(params.components, params)
    intensity_comps = comp_list.components_for_eval_pol("I")
    with h5py.File(compsep_path, "r") as handle:
        for comp in intensity_comps:
            alms = _read_view_alms_from_chain(comp, compsep_path)
            if alms is None:
                raise ValueError(f"Component {comp.comp_name!r} has no intensity alms in "
                                 f"{compsep_path!r}.")
            comp.alms = alms
            group_path = f"comps/{comp.shortname}"
            for param_name in comp.sed_param_names:
                dataset = f"{group_path}/sed/{param_name}"
                if dataset not in handle:
                    raise ValueError(f"Missing component state {dataset!r} in {compsep_path!r}.")
                # The component resolver expects a Python scalar or an [I, QU] list, whereas
                # HDF5 supplies numpy scalars/arrays. Select I before evaluating any SED.
                value = np.asarray(handle[dataset][()]).tolist()
                setattr(comp, param_name, comp._per_pol(value))
            dataset = f"{group_path}/amp_fwhm_arcmin"
            if dataset not in handle:
                raise ValueError(f"Missing component state {dataset!r} in {compsep_path!r}.")
            # SkyModel subtracts this beam in quadrature from the requested band beam.
            # Restoring it prevents smoothing the already-smoothed amplitudes a second time.
            comp.amp_fwhm_rad = np.radians(float(handle[dataset][()]) / 60.0)
    return intensity_comps


def _read_mask(mask_path: str, nside: int, smoothing_fwhm_deg: float) -> NDArray:
    """Read a RING inverse-variance weight mask, optionally tapered outside excluded pixels.

    Any masked child pixel excludes its parent when degrading resolution. Zero FWHM gives a
    binary mask; positive FWHM gives w = 1 - exp(-d**2 / (2*sigma**2)), where d is the angular
    distance to the nearest excluded pixel center. Excluded pixels retain exactly zero weight.
    """
    if not np.isfinite(smoothing_fwhm_deg) or smoothing_fwhm_deg < 0:
        raise ValueError("Mask apodization FWHM must be finite and nonnegative.")
    input_mask = hp.read_map(mask_path, field=0, dtype=np.float64)
    # ud_grade averages child pixels. Requiring full coverage prevents a partly masked parent
    # from reopening; thresholding the average at >0 would leak excluded sky into the fit.
    coverage = hp.ud_grade((input_mask > 0).astype(np.float64), nside)
    keep = coverage == 1.0
    weights = keep.astype(np.float64)
    if smoothing_fwhm_deg == 0 or np.all(keep) or not np.any(keep):
        return weights

    distance = enmap.distance_transform_healpix(keep)
    sigma = np.radians(smoothing_fwhm_deg) / np.sqrt(8.0 * np.log(2.0))
    # expm1 evaluates the small weights near the boundary without cancellation in 1 - exp(...).
    return -np.expm1(-0.5 * (distance / sigma)**2)


def _load_iteration(params: Bunch, iteration: int, compsep_path: str,
                    band_files: list[tuple[str, str]], band_freqs: dict[str, float]) -> Bunch:
    """Load one Gibbs sample's foreground-subtracted maps, noise, beams and CMB amplitudes.

    Band filenames are absolute paths paired with parameter-file band names. All brightness
    quantities returned here are in uK_CMB; the caller supplies the shared mask and C_l prior.
    Map lists retain each band's native pixel grid; beam_sizes contains FWHMs in radians.
    """
    components = _build_intensity_components(params, compsep_path)
    cmb_comps = []
    foreground_comps = []
    for component in components:
        if isinstance(component, CMB):
            cmb_comps.append(component)
        else:
            foreground_comps.append(component)
    if len(cmb_comps) != 1:
        raise ValueError(f"Expected exactly one CMB component, found {len(cmb_comps)}.")
    foreground_sky = SkyModel(foreground_comps)
    signal_maps: list[NDArray] = []
    ivar_maps: list[NDArray] = []
    beam_sizes: list[float] = []
    for band, filename in band_files:
        nu = band_freqs[band]
        with h5py.File(filename, "r") as handle:
            observed = handle["maps/observed_sky"][0].astype(np.float64)
            rms = handle["maps/rms"][0].astype(np.float64)
            stored_unit = handle["metadata/band_unit"][()]
            beam = np.radians(handle["metadata/map_fwhm_arcmin"][()] / 60.0)
        if isinstance(stored_unit, bytes):
            stored_unit = stored_unit.decode("utf-8")
        nside = hp.npix2nside(rms.size)

        # Band maps use stored_unit; the foreground model is uK_RJ at this band's frequency.
        rj_to_uK_CMB = rj_to_band_unit_factor(nu, "uK_CMB")
        band_to_uK_CMB = rj_to_uK_CMB / rj_to_band_unit_factor(nu, stored_unit)
        observed *= band_to_uK_CMB
        rms *= band_to_uK_CMB
        # Condition on this Gibbs sample's foregrounds. Foreground uncertainty is represented
        # by processing multiple Gibbs samples, not by perturbing foregrounds within a CMB draw.
        foreground = foreground_sky.get_sky_at_nu(nu, nside, "I", fwhm=beam)[0]
        observed -= foreground.astype(np.float64) * rj_to_uK_CMB
        ivar = np.zeros_like(rms)
        valid = rms > 0
        ivar[valid] = 1.0 / rms[valid]**2
        signal_maps.append(observed)
        ivar_maps.append(ivar)
        beam_sizes.append(beam)
        logger.info(f"iter {iteration}: read {band} ({nu:g} GHz, {stored_unit}) at nside {nside}.")

    # Chain CMB amplitudes are uK_RJ at their reference frequency, including for spectrum plots.
    cmb_alms = np.ascontiguousarray(cmb_comps[0].alms[0]).astype(np.complex128)
    cmb_alms *= rj_to_band_unit_factor(cmb_comps[0].nu_ref, "uK_CMB")
    return Bunch(signal_maps=signal_maps, ivar_maps=ivar_maps,
                 beam_sizes=np.array(beam_sizes), cmb_alms=cmb_alms, lmax=cmb_comps[0].lmax)


def main() -> int:
    """Select saved samples, prepare shared state, and write the requested CMB realizations."""
    parser = argparse.ArgumentParser(
        description="Draw constrained CMB realizations from a finished Commander4 run.")
    parser.add_argument(
        "run_dir",
        help=f"Path to a Commander4 run's output directory (its `output.dir`, containing "
             f"{paths.CHAINS_COMPSEP}/ and {paths.CHAINS_BANDS}/).")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for outputs. Defaults to <run_dir>/cmb_realizations.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--iter", type=int, nargs="+", default=None, dest="only_iters",
                           help="Gibbs iteration numbers to process (default: all found).")
    selection.add_argument("--burn-in", type=int, default=None, metavar="N",
                           help="Discard iterations numbered N or lower; process later samples.")
    parser.add_argument("--chain", type=int, default=1, help="Chain number to read (default 1).")
    parser.add_argument("--n-realizations", type=int, default=1,
                        help="Independent CMB realizations per Gibbs iteration (default 1).")
    parser.add_argument("--maxiter", type=int, default=1000,
                        help="Maximum CG iterations (default 1000).")
    parser.add_argument("--err-tol", type=float, default=1e-10,
                        help="Squared relative preconditioned residual tolerance (default 1e-10).")
    parser.add_argument("--precond-lmax", type=int, default=32,
                        help="Coupled low-ell preconditioner cutoff for masked runs (default 32). "
                             "Set 0 for diagonal only. Larger values cost more memory and setup.")
    parser.add_argument("--mask", default=None,
                        help="FITS binary mask (first field). Zero pixels are excluded; any masked "
                             "area excludes a pixel when reducing resolution. Optional.")
    parser.add_argument("--mask-fwhm-deg", type=float, default=0.0,
                        help="Outward Gaussian apodization FWHM in degrees. Default 0 disables "
                             "apodization. Retained pixels reach half weight at FWHM/2 from the "
                             "nearest excluded pixel center. Weights multiply inverse variance.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug-level logging.")
    args = parser.parse_args()
    if args.n_realizations < 1:
        parser.error("--n-realizations must be positive.")
    if args.burn_in is not None and args.burn_in < 0:
        parser.error("--burn-in must be nonnegative.")

    logger.handlers.clear()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("[cmb_real] %(levelname)s: %(message)s"))
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logger.propagate = False

    run_dir = os.path.abspath(args.run_dir)
    compsep_dir = os.path.join(run_dir, paths.CHAINS_COMPSEP)
    bands_dir = os.path.join(run_dir, paths.CHAINS_BANDS)
    if not os.path.isdir(compsep_dir) or not os.path.isdir(bands_dir):
        logger.error(f"Run output directory not found: {compsep_dir} or {bands_dir}")
        return 1

    params = _load_params_from_chain(run_dir)
    if params is None:
        logger.error(f"No chain file in {run_dir} carries the parameter file, so the run's "
                     f"components cannot be reconstructed.")
        return 1
    band_freqs = _band_frequencies(params)

    output_dir = args.output_dir or os.path.join(run_dir, "cmb_realizations")
    os.makedirs(output_dir, exist_ok=True)

    # Which (band, iteration) pairs the run actually wrote maps for. A band file exists on every
    # written iteration, but its `maps/` group is thinned separately, so check for the group too.
    bands_by_iter: dict[int, list[tuple[str, str]]] = {}
    for filename in sorted(os.listdir(bands_dir)):
        band, chain, iteration = _extract_band_chain_iter(filename)
        if band is None or chain != args.chain:
            continue
        with h5py.File(os.path.join(bands_dir, filename), "r") as f:
            if "maps/observed_sky" not in f:
                continue
        bands_by_iter.setdefault(iteration, []).append((band, filename))
    iterations = sorted(bands_by_iter)
    if args.only_iters is not None:
        iterations = [it for it in iterations if it in args.only_iters]
    if args.burn_in is not None:
        iterations = [it for it in iterations if it > args.burn_in]

    # Discover usable samples before preparing anything expensive. The first selected sample
    # supplies the reference noise weights and beams for the shared low-ell preconditioner.
    jobs: list[tuple[int, str, list[tuple[str, str]]]] = []
    for iteration in iterations:
        compsep_path = os.path.join(compsep_dir, f"chain{args.chain:02d}_iter{iteration:04d}.h5")
        if not os.path.isfile(compsep_path):
            logger.warning(f"No compsep chain for iteration {iteration}; skipping.")
            continue
        band_files = []
        for band, filename in bands_by_iter[iteration]:
            band_name = band.split("_")[-1]
            if band_name not in band_freqs:
                logger.warning(f"Band {band_name!r} is not in the parameter file; skipping.")
                continue
            band_files.append((band_name, os.path.join(bands_dir, filename)))
        if not band_files:
            logger.warning(f"No usable bands for iteration {iteration}; skipping.")
            continue
        jobs.append((iteration, compsep_path, band_files))
    if not jobs:
        logger.error(f"No usable Gibbs samples found for chain {args.chain} and this selection.")
        return 1

    iteration, compsep_path, band_files = jobs[0]
    data = _load_iteration(params, iteration, compsep_path, band_files, band_freqs)
    reference_bands = []
    for band, _ in band_files:
        reference_bands.append(band)
    # Equal-resolution bands share one mask. Build every resolution before either sampling loop.
    masks_by_nside: dict[int, NDArray] = {}
    masks = []
    for sky in data.signal_maps:
        nside = hp.npix2nside(sky.size)
        if nside not in masks_by_nside:
            if args.mask is None:
                masks_by_nside[nside] = np.ones(sky.size)
            else:
                masks_by_nside[nside] = _read_mask(args.mask, nside, args.mask_fwhm_deg)
        masks.append(masks_by_nside[nside])
    output_nside = max(masks_by_nside)

    # This fixed thermodynamic theory prior is shared by all Gibbs samples and realizations.
    import camb
    camb_params = camb.set_params(ombh2=0.022, omch2=0.122, H0=67.5, ns=0.96, As=2e-9,
                                 tau=0.06, omk=0, mnu=0.06, lmax=data.lmax + 100)
    camb_results = camb.get_results(camb_params)
    # raw_cl returns C_l, not D_l = l(l+1) C_l / (2 pi); column 0 is the temperature spectrum.
    spectra = camb_results.get_cmb_power_spectra(camb_params, CMB_unit="muK", raw_cl=True)["total"]
    cmb_cell_prior = spectra[:data.lmax + 1, 0]
    # CAMB supplies no monopole/dipole power. Give these nuisance modes a broad, finite prior.
    cmb_cell_prior[:2] = 1e6

    logger.info(f"Preparing shared preconditioner from Gibbs iteration {iteration} for "
                f"{len(jobs)} iterations, {args.n_realizations} realizations each.")
    solver = ConstrainedCMB(data.signal_maps, data.ivar_maps, cmb_cell_prior,
                            masks=masks, beam_fwhm=data.beam_sizes, maxiter=args.maxiter,
                            precond_lmax=args.precond_lmax)

    for job_index, (iteration, compsep_path, band_files) in enumerate(jobs):
        if job_index > 0:
            current_bands = []
            for band, _ in band_files:
                current_bands.append(band)
            if current_bands != reference_bands:
                raise ValueError("Selected samples must have the same bands in the same order.")
            data = _load_iteration(params, iteration, compsep_path, band_files, band_freqs)
            if data.lmax != solver.lmax:
                raise ValueError("Selected Gibbs samples must have the same CMB lmax.")
            solver.update_data(data.signal_maps, data.ivar_maps, data.beam_sizes)
        # The deterministic RHS stays fixed within a Gibbs sample. Only the random RHS changes
        # between realizations; reusing the solver must never reuse a previous random draw.
        rhs_mean = solver.get_RHS_eqn_mean()
        cmb_cell_in = hp.alm2cl(data.cmb_alms)

        for realization in range(1, args.n_realizations + 1):
            logger.info(f"iter {iteration}: realization {realization}/{args.n_realizations}.")
            rhs = rhs_mean + solver.get_RHS_eqn_fluct()
            cmb_alms = solver.solve_CG(solver.LHS_func, rhs, err_tol=args.err_tol)
            cmb_cell = hp.alm2cl(cmb_alms)
            cmb_map = hp.alm2map(cmb_alms, output_nside)
            suffix = f"_real{realization:04d}" if args.n_realizations > 1 else ""
            filename = f"chain{args.chain:02d}_iter{iteration:04d}{suffix}"
            out_base = os.path.join(output_dir, filename)
            hp.write_map(f"{out_base}_cmb_realization.fits", cmb_map, overwrite=True,
                         column_units="uK_CMB", extra_header=[("BUNIT", "uK_CMB"),
                         ("CHAIN", args.chain), ("ITER", iteration), ("REALIZ", realization)])

            # Plot D_l for readability; all three spectra use the shared CMB harmonic range.
            ell = np.arange(solver.lmax + 1)
            plt.figure()
            plt.loglog(ell, cmb_cell_in * ell * (ell + 1.) / (2. * np.pi), label="compsep CMB")
            plt.loglog(ell, cmb_cell * ell * (ell + 1.) / (2. * np.pi),
                       label="constrained realization")
            plt.loglog(ell, cmb_cell_prior * ell * (ell + 1.) / (2. * np.pi), c="k", label="Prior")
            plt.xlabel("multipole $\\ell$")
            plt.ylabel("$\\mathcal{D}_\\ell$ [$\\mu K_\\mathrm{CMB}^2$]")
            plt.legend()
            plt.savefig(f"{out_base}_Cell.png", dpi=120, bbox_inches="tight")
            plt.close()

            hp.mollview(cmb_map, cmap="RdBu_r",
                        title=f"Constrained CMB, iter {iteration}, realization {realization}",
                        min=-350., max=350., unit="uK_CMB")
            plt.savefig(f"{out_base}_cmb_realization.png", dpi=120, bbox_inches="tight")
            plt.close()
            logger.info(f"iter {iteration}: wrote {out_base}_cmb_realization.fits (+ 2 figures) "
                        f"from {len(band_files)} bands.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
