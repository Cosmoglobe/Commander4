"""Turning a finished mapmaking solution into the products the rest of the run consumes.

Both mapmakers end the same way: split the solved maps into the I and QU `DetectorMap`s compsep
receives, optionally degrade them to a common analysis resolution, and collect the maps written to
the `maps/` group of the band chain file.
"""
import numpy as np
from numpy.typing import NDArray

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.detector_map import DetectorMap
from commander4.data_models.tod_samples import TODSamples
from commander4.tod.mapmaking.config import MapmakingConfig


def finalize_band_maps(map_signal: NDArray, map_rms: NDArray, pols: str,
                       experiment_data: DetectorGroupTOD, mapmaking: MapmakingConfig,
                       tod_samples: TODSamples, compsep_output: NDArray | None,
                       map_orbdipole: NDArray | None = None,
                       map_corrnoise: NDArray | None = None,
                       map_sidelobe: NDArray | None = None,
                       map_residual: NDArray | None = None,
                       map_nhit: NDArray | None = None,
                       map_cov: NDArray | None = None) -> tuple[dict, dict]:
    """Split the solved band maps into `DetectorMap`s, and collect what goes to the chain file.

    Args:
        map_signal: Solved sky map, shape (3, npix), rows I, Q, U.
        map_rms: Per-pixel white-noise rms, same shape.
        pols: Which polarizations this band carries, e.g. "I", "QU" or "IQU".
        compsep_output: The current sky model for this band, written as `skymodel`.
        map_sidelobe: Binned far-sidelobe pickup, in uK_RJ. Commander3's `tod_<freq>_sl` map.
        map_residual: Binned noise residual (data minus sky model, orbital dipole and correlated
            noise), in uK_RJ. Commander3's `tod_<freq>_res` map.
        map_nhit: Per-pixel count of accumulated good samples, shape (npix,).
        map_cov: The six unique elements of the per-pixel `P^T N^-1 P`, shape (6, npix). Only its
            inverse diagonal survives as `map_rms`, so this is the only place the QU off-diagonals
            are recorded.

    Returns:
        `(detmap_dict, maps_to_file)`. The detector maps are what compsep receives; `maps_to_file`
        is keyed by the name each map takes inside the chain file's `maps/` group. When
        `mapmaking.common_res_fwhm` is set, `observed_sky` and `rms` are the smoothed maps that
        compsep actually used, and `map_fwhm_arcmin` records the beam they are at.
    """
    detmap_dict_out = {}
    # Degrading to a common analysis resolution happens after mapmaking; 0 leaves the native beam.
    common_res_fwhm = mapmaking.common_res_fwhm
    if "I" in pols:
        detmap_I = DetectorMap(map_signal[0,:], map_rms[0,:], experiment_data.nu,
                               experiment_data.fwhm, experiment_data.nside,
                               lmax=mapmaking.band_lmax)
        detmap_I.g0 = tod_samples.abs_gain
        if common_res_fwhm:
            detmap_I.smooth_to_resolution(common_res_fwhm)
        detmap_dict_out["I"] = detmap_I
    if "QU" in pols:
        detmap_QU = DetectorMap(map_signal[1:3,:], map_rms[1:3,:], experiment_data.nu,
                                experiment_data.fwhm, experiment_data.nside,
                                lmax=mapmaking.band_lmax)
        detmap_QU.g0 = tod_samples.abs_gain
        if common_res_fwhm:
            detmap_QU.smooth_to_resolution(common_res_fwhm)
        detmap_dict_out["QU"] = detmap_QU

    maps_to_file = {}
    if common_res_fwhm:
        # Reuse the smoothing compsep already paid for rather than repeating it, filling the
        # (3, npix) layout back in so the written datasets keep their shape.
        sky_out = np.zeros_like(map_signal)
        rms_out = np.zeros_like(map_rms)
        if "I" in pols:
            sky_out[0,:] = detmap_dict_out["I"].map_sky[0]
            rms_out[0,:] = detmap_dict_out["I"].map_rms[0]
        if "QU" in pols:
            sky_out[1:3,:] = detmap_dict_out["QU"].map_sky
            rms_out[1:3,:] = detmap_dict_out["QU"].map_rms
        maps_to_file["observed_sky"] = sky_out
        maps_to_file["rms"] = rms_out
        maps_to_file["map_fwhm_arcmin"] = float(common_res_fwhm)
    else:
        maps_to_file["observed_sky"] = map_signal
        maps_to_file["rms"] = map_rms
        maps_to_file["map_fwhm_arcmin"] = float(experiment_data.fwhm)
    # The aux maps are debug output binned straight from the TODs, and stay at the native beam even
    # when the two above are smoothed. The mapmaker only builds one the run asked for, so being
    # present is the whole gate; the two below come from elsewhere and are gated here instead.
    for name, aux_map in (("orbdipole", map_orbdipole), ("corrnoise", map_corrnoise),
                          ("sidelobe", map_sidelobe), ("res", map_residual),
                          ("nhit", map_nhit)):
        if aux_map is not None:
            maps_to_file[name] = aux_map
    if mapmaking.include_sky_model_maps:
        maps_to_file["skymodel"] = compsep_output
    if mapmaking.include_cov_maps:
        maps_to_file["cov"] = map_cov
    return detmap_dict_out, maps_to_file
