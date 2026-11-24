"""How well the current sky model fits the band maps, for the log and for the chain file.

Each CompSep rank evaluates its own polband, and the results are combined over all ranks. The fit
is measured by the whitened residual (d - s)/sigma, over the pixels a band observed.
"""
import logging

import healpy as hp
import numpy as np
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.data_models.band_map import BandMap
from commander4.diagnostics.performance import benchmark
from commander4.mpi.setup import MPIInfo
from commander4.sky.comp_list import CompList

logger = logging.getLogger(__name__)


def log_fit(mpi_info: MPIInfo, band_map: BandMap, comp_list: CompList, label: str) -> None:
    """Log how well the current sky model fits this polband and all polbands, after the step
    `label`.

    Every CompSep rank must call it.
    """
    compsep = mpi_info.compsep
    sky_at_band = comp_list.get_sky_at_band(band_map)
    pol_names = ["Q", "U"] if band_map.pol == "QU" else ["I"]
    chisq_polband, ndof_polband = 0.0, 0
    for ipol in range(band_map.npol):
        # An unobserved pixel has zero inverse-noise weight, and contributes no degree of freedom.
        observed = band_map.map_inv_var[ipol] > 0
        whitened_residual = ((band_map.map_sky[ipol] - sky_at_band[ipol])[observed]
                             *np.sqrt(band_map.map_inv_var[ipol][observed]))
        chisq_polband += np.sum(whitened_residual**2, dtype=np.float64)
        ndof_polband += whitened_residual.size
        logger.verbose(f"Fit after {label} on rank {compsep.rank} for pol={pol_names[ipol]} "
                       f"({band_map.nu}GHz): mean|(d-s)/sigma|="
                       f"{np.mean(np.abs(whitened_residual)):.3f}, "
                       f"red.chisq={np.mean(whitened_residual**2):.3f} "
                       f"(ndof={whitened_residual.size}).")

    total = compsep.comm.allreduce(chisq_polband, op=MPI.SUM)
    ndof = compsep.comm.allreduce(ndof_polband, op=MPI.SUM)
    if compsep.rank == 0:
        logger.info(f"Fit after {label}, all bands: chisq={total:.6e}, ndof={ndof}, "
                    f"red.chisq={total/ndof:.4f}")


@benchmark("chain-gather")
def collect_fit_diagnostics(mpi_info: MPIInfo, params: Bunch, band_map: BandMap,
                            comp_list: CompList) -> tuple[dict, dict[str, float]]:
    """Gather the fit of the sky model to each polband onto the CompSep master, for the chain file.

    Every CompSep rank must call it. The chi-squared map has the resolution
    `output.chains.nside_chisq`. With `output.chains.include.compsep_residual_maps`, the
    full-resolution `data - model` maps are gathered too, which is expensive.

    Returns:
        `(fit_tree, band_frequencies)` on the master, and two empty dicts elsewhere. `fit_tree`
        holds the `chisq` group and, when kept, the `residuals` group. `band_frequencies` maps each
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
    # band pixel's chi^2 over the map pixels inside it. The map's total is kept either way.
    subpixels_per_chisq_pixel = npix/npix_chisq

    # chi^2 in `nside_chisq` pixels. Row 0 is I and rows 1-2 are Q and U, so the I and QU ranks of
    # one band do not collide in the sum over ranks.
    chisq_map = np.zeros((3, npix_chisq))
    chisq_polband, ndof_polband = 0.0, 0
    for ipol in range(band_map.npol):
        observed = band_map.map_inv_var[ipol] > 0
        whitened_residual = residual[ipol][observed]*np.sqrt(band_map.map_inv_var[ipol][observed])
        chisq_polband += np.sum(whitened_residual**2, dtype=np.float64)
        ndof_polband += whitened_residual.size
        chisq_full = np.zeros(npix)
        chisq_full[observed] = whitened_residual**2
        chisq_row = 1 + ipol if band_map.pol == "QU" else 0
        chisq_map[chisq_row] += hp.ud_grade(chisq_full, nside_chisq)*subpixels_per_chisq_pixel

    # The chi^2 of every polband, added into one low-resolution map.
    chisq_map_total = np.zeros_like(chisq_map) if compsep.rank == 0 else None
    compsep.comm.Reduce(chisq_map, chisq_map_total, op=MPI.SUM, root=0)
    polband_fit = Bunch(band_name=compsep.band_name, pol=compsep.pol,
                        nu=float(band_map.nu), chisq=float(chisq_polband), ndof=ndof_polband,
                        residual=residual if include_residual_maps else None)
    polband_fits = compsep.comm.gather(polband_fit, root=0)
    if compsep.rank != 0:
        return {}, {}

    total = sum(polband_fit.chisq for polband_fit in polband_fits)
    ndof = sum(polband_fit.ndof for polband_fit in polband_fits)
    chisq: dict = {"total": total, "ndof": ndof, "bands": {}}
    chisq["reduced"] = total/ndof if ndof > 0 else float("nan")
    # The z-score says whether the fit is good: chi-squared grows with the map size, its z-score
    # does not.
    chisq["z"] = (total - ndof)/np.sqrt(2.0*ndof) if ndof > 0 else float("nan")
    if include_chisq_map:
        chisq["map"] = chisq_map_total.astype(np.float32)

    residuals: dict = {}
    band_frequencies: dict[str, float] = {}
    for polband_fit in polband_fits:
        polband_name = f"{polband_fit.band_name}_{polband_fit.pol}"
        chisq["bands"][polband_name] = {
            "chisq": polband_fit.chisq, "ndof": polband_fit.ndof, "nu": polband_fit.nu,
            "reduced": polband_fit.chisq/polband_fit.ndof if polband_fit.ndof > 0 else float("nan")}
        band_frequencies[polband_fit.band_name] = polband_fit.nu
        if polband_fit.residual is not None:
            residuals[polband_name] = polband_fit.residual.astype(np.float32)

    fit_tree: dict = {"chisq": chisq}
    if residuals:
        fit_tree["residuals"] = residuals
    return fit_tree, band_frequencies
