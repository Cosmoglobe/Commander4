"""How well the current sky model fits the band maps, for the log and for the chain file.

Each CompSep rank evaluates its own polband, and the results are combined over all ranks. The fit
is measured by the whitened residual z = (d - s) / sigma, over the pixels a band observed.
"""
import logging

import healpy as hp
import numpy as np
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.data_models.detector_map import DetectorMap
from commander4.diagnostics.performance import benchmark
from commander4.mpi.setup import MPIInfo
from commander4.sky.comp_list import CompList

logger = logging.getLogger(__name__)


def log_fit(mpi_info: MPIInfo, band_map: DetectorMap, comp_list: CompList, label: str) -> None:
    """Log how well the current sky model fits this band and all bands, after the step `label`.

    Every CompSep rank must call it.
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
    if compsep.rank == 0:
        logger.info(f"Fit after {label}, all bands: chi2={total:.6e}, ndof={ndof}, "
                    f"red.chi2={total/ndof:.4f}")


@benchmark("chain-gather")
def collect_fit_diagnostics(mpi_info: MPIInfo, params: Bunch, band_map: DetectorMap,
                            comp_list: CompList) -> tuple[dict, dict[str, float]]:
    """Gather the fit of the sky model to each band onto the CompSep master, for the chain file.

    Every CompSep rank must call it. The chi-squared map has the resolution
    `output.chains.nside_chisq`. With `output.chains.include.compsep_residual_maps`, the
    full-resolution `data - model` maps are gathered too, which is expensive.

    Returns:
        `(fit_tree, band_frequencies)` on the master, and two empty dicts elsewhere. `fit_tree`
        holds the `chi2` group and, when kept, the `residuals` group. `band_frequencies` maps each
        band name to its frequency in GHz.
    """
    compsep = mpi_info.compsep
    chains = params.output.chains
    nside_chisq = int(getattr(chains, "nside_chisq", 256))
    include_chisq_map = bool(getattr(chains.include, "chisq_map", False))
    include_residual_maps = bool(getattr(chains.include, "compsep_residual_maps", False))

    residual = band_map.map_sky - comp_list.get_sky_at_band(band_map)
    npix, npix_chisq = band_map.map_sky.shape[-1], hp.nside2npix(nside_chisq)
    # `ud_grade` averages, so scaling by the number of band pixels per chi-squared pixel turns the
    # average into a sum. For a band coarser than the map the factor is below 1, which spreads each
    # band pixel's z^2 over the map pixels inside it. The map's total is kept either way.
    subpixels_per_chisq_pixel = npix/npix_chisq

    # z^2 in `nside_chisq` pixels. Row 0 is I and rows 1-2 are Q and U, so the I and QU ranks of one
    # band do not collide in the sum over ranks.
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

    # z^2 of every band, added into one low-resolution map.
    chi2_map_total = np.zeros_like(chi2_map) if compsep.rank == 0 else None
    compsep.comm.Reduce(chi2_map, chi2_map_total, op=MPI.SUM, root=0)
    band_fit = Bunch(name=compsep.band_name, pol=compsep.pol,
                     nu=float(band_map.nu), chi2=float(chi2_band), ndof=ndof_band,
                     residual=residual if include_residual_maps else None)
    band_fits = compsep.comm.gather(band_fit, root=0)
    if compsep.rank != 0:
        return {}, {}

    total = sum(band_fit.chi2 for band_fit in band_fits)
    ndof = sum(band_fit.ndof for band_fit in band_fits)
    chi2: dict = {"total": total, "ndof": ndof, "bands": {}}
    chi2["reduced"] = total/ndof if ndof > 0 else float("nan")
    # The z-score says whether the fit is good: chi-squared grows with the map size, its z-score
    # does not.
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
