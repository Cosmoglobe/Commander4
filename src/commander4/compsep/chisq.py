"""How well the current sky model fits the band maps, and the record of it the chain keeps.

Commander4's analogue of Commander3's `comm_chisq_mod.f90`. Every CompSep rank owns one
`(band, polarization)` view, so the fit is evaluated locally and then combined over all bands:
after each sampling step into the all-band chi-squared the log shows (`log_fit`), and once per
Gibbs iteration into the per-band terms and maps the chain stores (`collect_fit_diagnostics`).

Commander3 spreads the same information over three outputs -- the chisq columns of
`fg_ind_mean_c<CCCC>.dat`, the `chisq_<postfix>.fits` map, and `res_<band>_<postfix>.fits`. All of
it comes from one quantity, the whitened residual

    z = (d - s(theta)) / sigma,

evaluated over the pixels a band actually observed.
"""
import logging

import healpy as hp
import numpy as np
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.data_models.detector_map import DetectorMap
from commander4.sky.comp_list import CompList

logger = logging.getLogger(__name__)


def log_fit(mpi_info: Bunch, band_map: DetectorMap, comp_list: CompList, label: str) -> None:
    """Log how well the current sky model fits this band and all bands, after the step `label`.

    Called collectively by every CompSep rank.
    """
    compsep = mpi_info.compsep
    sky_at_band = comp_list.get_sky_at_band(band_map)
    pol_names = ["Q", "U"] if band_map.pol == "QU" else ["I"]
    chi2_band, ndof_band = 0.0, 0
    for ipol in range(band_map.npol):
        # An unobserved pixel has zero inverse-noise weight, and contributes no degree of freedom.
        observed = band_map.map_inv_var[ipol] > 0
        z = ((band_map.map_sky[ipol] - sky_at_band[ipol])[observed]
             *np.sqrt(band_map.map_inv_var[ipol][observed]))
        chi2_band += np.sum(z**2, dtype=np.float64)
        ndof_band += z.size
        logger.verbose(f"Fit after {label} on rank {compsep.rank} for pol={pol_names[ipol]} "
                       f"({band_map.nu}GHz): mean|z|={np.mean(np.abs(z)):.3f}, "
                       f"red.chi2={np.mean(z**2):.3f} (ndof={z.size}).")

    total = compsep.comm.allreduce(chi2_band, op=MPI.SUM)
    ndof = compsep.comm.allreduce(ndof_band, op=MPI.SUM)
    if compsep.is_master:
        logger.info(f"Fit after {label}, all bands: chi2={total:.6e}, ndof={ndof}, "
                    f"red.chi2={total/ndof:.4f}")


def collect_fit_diagnostics(mpi_info: Bunch, params: Bunch, band_map: DetectorMap,
                            comp_list: CompList) -> tuple[dict, dict[str, float]]:
    """Gather the fit of the sky model to each band onto the CompSep master, for the chain file.

    Called collectively by every CompSep rank, once per Gibbs iteration. Each rank evaluates its
    own band. One small Python `gather` collects the per-band numbers, and one `Reduce` the
    low-resolution chi-squared map (resolution `output.chains.nside_chisq`, C3's `NSIDE_CHISQ`).
    With `output.chains.include.compsep_residual_maps`, the full-resolution `data - model` maps go
    in the same `gather`, which is the expensive case.

    Returns:
        `(fit_tree, band_frequencies)` on the master, and two empty dicts elsewhere. `fit_tree`
        holds the `chi2` group and, when kept, the `residuals` group; `band_frequencies` maps each
        band name to its centre frequency in GHz, which is what lets the chain writer record a
        mixing coefficient per band.
    """
    compsep = mpi_info.compsep
    # The chi-squared map resolution should be at or below the coarsest band nside, since the map
    # only ever sums z^2 into its pixels.
    chains = params.output.chains
    nside_chisq = int(getattr(chains, "nside_chisq", 256))
    include_chisq_map = bool(getattr(chains.include, "chisq_map", False))
    include_residual_maps = bool(getattr(chains.include, "compsep_residual_maps", False))

    residual = band_map.map_sky - comp_list.get_sky_at_band(band_map)
    npix, npix_chisq = band_map.map_sky.shape[-1], hp.nside2npix(nside_chisq)
    # ud_grade averages, so scaling by how many band pixels fall in a chi-squared pixel turns the
    # average into the sum over them that Commander3's chisq map accumulates (comm_output_mod.f90).
    # For a band coarser than the chi-squared map the factor is below 1: each band pixel's z^2 is
    # spread evenly over the chi-squared pixels inside it, so the map's total is kept either way.
    subpixels_per_chisq_pixel = npix/npix_chisq

    # z^2 in `nside_chisq` pixels. Row 0 is intensity and rows 1-2 are Q/U, so summing an I view
    # and a QU view of the same sky cannot make them collide.
    chi2_map = np.zeros((3, npix_chisq))
    chi2_band, ndof_band = 0.0, 0
    for ipol in range(band_map.npol):
        observed = band_map.map_inv_var[ipol] > 0
        z = residual[ipol][observed]*np.sqrt(band_map.map_inv_var[ipol][observed])
        chi2_band += np.sum(z**2, dtype=np.float64)
        ndof_band += z.size
        z2_full = np.zeros(npix)
        z2_full[observed] = z**2
        chi2_row = 1 + ipol if band_map.pol == "QU" else 0
        chi2_map[chi2_row] += hp.ud_grade(z2_full, nside_chisq)*subpixels_per_chisq_pixel

    # Commander3's chisq_<postfix>.fits: z^2 from every band added into one low-resolution map.
    chi2_map_total = np.zeros_like(chi2_map) if compsep.is_master else None
    compsep.comm.Reduce(chi2_map, chi2_map_total, op=MPI.SUM, root=0)
    band_fit = Bunch(name=mpi_info.band.name, pol=mpi_info.band.pol,
                     nu=float(band_map.nu), chi2=float(chi2_band), ndof=ndof_band,
                     residual=residual if include_residual_maps else None)
    band_fits = compsep.comm.gather(band_fit, root=0)
    if not compsep.is_master:
        return {}, {}

    total = sum(band_fit.chi2 for band_fit in band_fits)
    ndof = sum(band_fit.ndof for band_fit in band_fits)
    chi2: dict = {"total": total, "ndof": ndof, "bands": {}}
    chi2["reduced"] = total/ndof if ndof > 0 else float("nan")
    # Commander3's fg_ind_mean.dat also reports the chi-squared as a z-score, which is what says
    # whether a fit is good rather than merely large: chi-squared grows with the map size, its
    # z-score does not.
    chi2["z"] = (total - ndof)/np.sqrt(2.0*ndof) if ndof > 0 else float("nan")
    if include_chisq_map:
        chi2["map"] = chi2_map_total.astype(np.float32)

    residuals: dict = {}
    band_frequencies: dict[str, float] = {}
    for band_fit in band_fits:
        view_name = f"{band_fit.name}_{band_fit.pol}"
        chi2["bands"][view_name] = {
            "chi2": band_fit.chi2, "ndof": band_fit.ndof, "nu": band_fit.nu,
            "reduced": band_fit.chi2/band_fit.ndof if band_fit.ndof > 0 else float("nan")}
        band_frequencies[band_fit.name] = band_fit.nu
        if band_fit.residual is not None:
            residuals[view_name] = band_fit.residual.astype(np.float32)

    fit_tree: dict = {"chi2": chi2}
    if residuals:
        fit_tree["residuals"] = residuals
    return fit_tree, band_frequencies
