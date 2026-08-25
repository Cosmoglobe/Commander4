"""How well the current sky model fits the band maps, and the record of it the chain keeps.

Commander4's analogue of Commander3's `comm_chisq_mod.f90`. Every CompSep rank owns one
`(band, polarization)` view, so the fit is evaluated locally and then reduced two ways: to the
all-band chi-squared the Gibbs loop logs, and to the per-band terms and maps the chain stores.

Commander3 spreads the same information over three outputs -- the chisq columns of
`fg_ind_mean_c<CCCC>.dat`, the `chisq_<postfix>.fits` map, and `res_<band>_<postfix>.fits`. All of
it comes from one quantity, the whitened residual

    z = (d - s(theta)) / sigma,

evaluated over the pixels a band actually observed.
"""
import logging
from dataclasses import dataclass
from typing import NamedTuple

import healpy as hp
import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.data_models.detector_map import DetectorMap
from commander4.sky.sky_model import SkyModel

logger = logging.getLogger(__name__)


@dataclass
class BandFit:
    """One rank's own `(band, polarization)` contribution to the all-band chi-squared.

    Attributes:
        band_name: The band this view belongs to, e.g. ``Band30GHz``.
        pol: The execution polarization of the view, ``I`` or ``QU``.
        nu: Band centre frequency in GHz.
        chi2: Sum of z^2 over this view's observed pixels.
        ndof: How many observed pixels went into `chi2`.
        chi2_map: z^2 summed into `nside_chisq` pixels, shape (3, npix_chisq). Row 0 is intensity
            and rows 1-2 are Q/U, so summing an I view and a QU view of the same sky cannot make
            them collide.
        residual: Full-resolution `data - model`, shape (npol, npix), or None when the run did not
            ask for the residual maps.
    """

    band_name: str
    pol: str
    nu: float
    chi2: float
    ndof: int
    chi2_map: NDArray
    residual: NDArray | None


class ChisqResult(NamedTuple):
    """The all-band chi-squared, plus the local contribution that went into it.

    `total` and `ndof` are allreduced, so they are identical on every rank; `band` is this rank's
    own view and differs everywhere.
    """

    total: float
    ndof: int
    band: BandFit


def evaluate_chi2(compsep: Bunch, detector_data: DetectorMap, sky_model: SkyModel,
                  band_name: str, pol: str, nside_chisq: int, label: str | None = None,
                  keep_residual: bool = False) -> ChisqResult:
    """Whiten this band's residual against the current sky model and reduce it over all bands.

    Called collectively by every CompSep rank: the two allreduces at the end make `total` and `ndof`
    the all-band values on all of them.

    Args:
        compsep: The CompSep MPI info (`comm`, `rank`, `master`).
        detector_data: This rank's band map, its inverse-noise map and its beam.
        sky_model: The model to fit, realized at the band's frequency and resolution.
        band_name: Name of this rank's band, recorded in the returned `BandFit`.
        pol: Execution polarization of this rank's view, ``I`` or ``QU``.
        nside_chisq: Resolution of the chi-squared map (Commander3's `NSIDE_CHISQ`). It should be
            at or below the coarsest band nside, since the map only ever sums z^2 into its pixels.
        label: When given, the name of the sampling step just taken; logs a per-band and an all-band
            fit summary under it. `None` evaluates silently.
        keep_residual: Also keep the full-resolution `data - model` map, which is what the chain's
            optional per-band residual maps are built from.
    """
    band_pol = "QU" if detector_data.pol else "I"
    sky_at_band = sky_model.get_sky_at_nu(
        detector_data.nu, detector_data.nside, band_pol, fwhm=detector_data.fwhm_rad)
    pol_names = ["Q", "U"] if detector_data.pol else ["I"]
    npix, npix_chisq = detector_data.map_sky.shape[-1], hp.nside2npix(nside_chisq)
    # ud_grade averages, so scaling by how many fine pixels fall in a coarse one turns the average
    # into the sum over them that Commander3's chisq map accumulates (comm_output_mod.f90).
    subpixels_per_chisq_pixel = npix // npix_chisq

    chi2_map = np.zeros((3, npix_chisq))
    residual = np.zeros_like(detector_data.map_sky) if keep_residual else None
    chi2_local, ndof_local = 0.0, 0
    for ipol in range(detector_data.npol):
        # An unobserved pixel has zero inverse-noise weight, and contributes no degree of freedom.
        observed = detector_data.inv_n_map[ipol] > 0
        full_residual = detector_data.map_sky[ipol] - sky_at_band[ipol]
        z = full_residual[observed]*np.sqrt(detector_data.inv_n_map[ipol][observed])
        chi2_local += np.sum(z**2, dtype=np.float64)
        ndof_local += z.size
        if residual is not None:
            residual[ipol] = full_residual

        z2_full = np.zeros(npix)
        z2_full[observed] = z**2
        chi2_row = 0 if band_pol == "I" else 1 + ipol
        chi2_map[chi2_row] += hp.ud_grade(z2_full, nside_chisq)*subpixels_per_chisq_pixel

        if label is not None:
            logger.verbose(f"Fit after {label} on rank {compsep.rank} for pol={pol_names[ipol]} "
                           f"({detector_data.nu}GHz): mean|z|={np.mean(np.abs(z)):.3f}, "
                           f"red.chi2={np.mean(z**2):.3f} (ndof={z.size}).")

    total = float(compsep.comm.allreduce(chi2_local, op=MPI.SUM))
    ndof = int(compsep.comm.allreduce(ndof_local, op=MPI.SUM))
    if label is not None and compsep.rank == compsep.master:
        logger.info(f"Fit after {label}, all bands: chi2={total:.6e}, ndof={ndof}, "
                    f"red.chi2={total/ndof:.4f}")
    band = BandFit(band_name=band_name, pol=pol, nu=float(detector_data.nu),
                   chi2=float(chi2_local), ndof=int(ndof_local), chi2_map=chi2_map,
                   residual=residual)
    return ChisqResult(total=total, ndof=ndof, band=band)


def collect_fit_diagnostics(compsep: Bunch, result: ChisqResult, include_chisq_map: bool
                            ) -> tuple[dict, dict[str, float]]:
    """Gather the fit onto the CompSep master as the tree of datasets the chain writes.

    Each rank holds only its own band, so the per-band numbers and maps have to be collected before
    they can be written: one small Python `gather` of the scalars, and one `Reduce` of the
    low-resolution chi-squared map. The full-resolution residual maps ride along in the same
    `gather` when the run asked for them, which is the expensive case.

    Called collectively by every CompSep rank.

    Returns:
        `(fit_tree, band_frequencies)` on the master, and two empty dicts elsewhere. `fit_tree`
        holds the `chi2` group and, when kept, the `residuals` group; `band_frequencies` maps each
        band name to its centre frequency in GHz, which is what lets the chain writer record a
        mixing coefficient per band.
    """
    is_master = compsep.rank == compsep.master

    # Commander3's chisq_<postfix>.fits: z^2 from every band added into one low-resolution map.
    chi2_map_total = np.zeros_like(result.band.chi2_map) if is_master else None
    compsep.comm.Reduce(np.ascontiguousarray(result.band.chi2_map), chi2_map_total, op=MPI.SUM,
                        root=compsep.master)
    per_band = compsep.comm.gather(result.band, root=compsep.master)
    if not is_master:
        return {}, {}

    chi2: dict = {"total": result.total, "ndof": result.ndof, "bands": {}}
    chi2["reduced"] = result.total/result.ndof if result.ndof > 0 else float("nan")
    # Commander3's fg_ind_mean.dat also reports the chi-squared as a z-score, which is what says
    # whether a fit is good rather than merely large: chi-squared grows with the map size, its
    # z-score does not.
    chi2["z"] = ((result.total - result.ndof)/np.sqrt(2.0*result.ndof)
                 if result.ndof > 0 else float("nan"))
    if include_chisq_map:
        chi2["map"] = chi2_map_total.astype(np.float32)

    residuals: dict = {}
    band_frequencies: dict[str, float] = {}
    for band in per_band:
        view_name = f"{band.band_name}_{band.pol}"
        chi2["bands"][view_name] = {
            "chi2": band.chi2, "ndof": band.ndof, "nu": band.nu,
            "reduced": band.chi2/band.ndof if band.ndof > 0 else float("nan")}
        band_frequencies[band.band_name] = band.nu
        if band.residual is not None:
            residuals[view_name] = band.residual.astype(np.float32)

    fit_tree: dict = {"chi2": chi2}
    if residuals:
        fit_tree["residuals"] = residuals
    return fit_tree, band_frequencies
