"""Writing the per-iteration chain files: band samples and maps on the TOD side, components on the
CompSep side.

What a file holds about each sampled object is decided by that object: each TOD sampling step
gives its arrays in `chain_arrays`, which `TODSamples.gather_chain_arrays` gathers over the band's
ranks, and each component writes its own group in `write_chain`. The functions here own the file:
its name, its metadata, the band maps.

A file is written as `<name>.tmp` and renamed when complete, so a restart never reads a file that
a crash left half-written.
"""
import os
import h5py
import numpy as np
import healpy as hp
import datetime
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.diagnostics.performance import benchmark
from commander4.file_io import paths
from commander4.sky.comp_list import CompList

# What `output.chains.interval` can thin. `bands` and `compsep` gate a whole file; `maps` gates
# only the `maps/` group inside the band file.
CHAIN_KINDS = ("bands", "compsep", "maps")

# How each `maps/` dataset behaves under the two transforms `write_band_chain_to_file` applies:
# degrading to `output.chains.maps_nside`, and converting uK_RJ to the band's own unit.
#
#   "brightness"  a sky brightness: averages when degraded, and picks up one factor of D.
#   "rms"         a brightness uncertainty: degrades in inverse variance, one factor of D.
#   "weight"      a summed inverse variance (uK_RJ^-2): sums when degraded, D^-2.
#   "count"       a pure sample count: sums when degraded, unit-free.
#
# Anything not listed is treated as "brightness", which every sky map here is.
MAP_KINDS = {"rms": "rms", "cov": "weight", "nhit": "count"}


def _degrade_map(value: NDArray, kind: str, nside_out: int) -> NDArray:
    """Downgrade one `maps/` dataset to `nside_out`.

    A brightness is averaged. A weight, a count and an rms (through its inverse variance) are
    summed, and for them a finer `nside_out` raises `ValueError`.
    """
    if kind == "brightness":
        return hp.ud_grade(value, nside_out, dtype=np.float32)
    # `ud_grade` averages, so scaling by the number of sub-pixels turns the average into a sum.
    additive = 1.0/value**2 if kind == "rms" else value
    subpixels = value.shape[-1] // hp.nside2npix(nside_out)
    if subpixels < 1:
        raise ValueError(f"_degrade_map cannot upgrade a '{kind}' map to nside {nside_out}.")
    summed = hp.ud_grade(additive.astype(np.float64), nside_out)*subpixels
    if kind == "rms":  # A pixel with no observed sub-pixel keeps an infinite rms.
        rms = np.divide(1.0, np.sqrt(summed), out=np.full_like(summed, np.inf), where=summed > 0)
        return rms.astype(np.float32)
    elif kind == "count":
        return np.round(summed).astype(np.int64)
    elif kind == "weight":  # A summed inverse variance, left as float64.
        return summed
    else:
        raise ValueError(f"Unknown map kind '{kind}' in _degrade_map.")


def _to_band_unit(value: NDArray, kind: str, band_unit_factor: float) -> NDArray:
    """Return one `maps/` dataset converted from uK_RJ to the band's own unit."""
    # Out of place: the caller's arrays are shared with the uK_RJ `DetectorMap` sent to CompSep.
    if kind == "count":  # A sample count carries no thermodynamic unit.
        return value
    if kind == "weight":  # An inverse variance, i.e. uK_RJ^-2.
        return value/band_unit_factor**2
    return value*band_unit_factor


def should_write_chain(params: Bunch, kind: str, iter: int) -> bool:
    """Whether the chain output `kind` (one of `CHAIN_KINDS`) is written on iteration `iter`.

    `iter` counts from 1. The interval is `output.chains.interval.<kind>`, by default 1 (every
    iteration). The maps appear only on iterations where both `bands` and `maps` fire, so keep
    `maps` a multiple of `bands`.
    """
    if kind not in CHAIN_KINDS:
        raise ValueError(f"Unknown chain kind {kind!r}; expected one of {list(CHAIN_KINDS)}.")
    if "interval" not in params.output.chains:
        return True
    interval = params.output.chains.interval
    if not isinstance(interval, Bunch):
        raise ValueError(f"'output.chains.interval' must be a block of per-output entries, not a "
                         f"single number (got {interval!r}). Legal keys are {list(CHAIN_KINDS)}; "
                         f"any may be omitted, defaulting to 1 (every iteration).")
    unknown = [key for key in interval if key not in CHAIN_KINDS]
    if unknown:
        raise ValueError(f"'output.chains.interval' has unknown entries {unknown}. Legal keys are "
                         f"{list(CHAIN_KINDS)}: 'bands' and 'compsep' thin a whole file, 'maps' "
                         f"thins the maps group inside the band file.")
    return (iter - 1) % int(interval[kind]) == 0 if kind in interval else True


@benchmark("filewrite-band")
def write_band_chain_to_file(params: Bunch, chain: int, iter: int, exp_name: str, band_name: str,
                             tod_arrays: dict[str, NDArray], maps_to_file: dict,
                             band_unit_factor: float = 1.0) -> None:
    """Write one band's Gibbs sample: the per-scan TOD samples and the output maps, in one file.

    The caller decides whether the file is written at all; this function only gates the `maps/`
    group. `tod_arrays` is keyed by dataset path, and its gains are already in the band's unit.
    The maps arrive in uK_RJ and are written in the band's `band_unit` (`band_unit_factor` D, see
    `MAP_KINDS`). The scalar `map_fwhm_arcmin` in `maps_to_file` becomes file metadata: the beam of
    `observed_sky` and `rms` (native, or `compsep.common_res_fwhm`).
    """
    chains = params.output.chains
    nside_out = getattr(chains, "maps_nside", "native")
    chain_file = paths.band_chain_file(paths.resolve_output_dir(params.output), exp_name,
                                       band_name, chain, iter)

    write_maps = should_write_chain(params, "maps", iter)
    maps_to_file = dict(maps_to_file)
    map_fwhm_arcmin = maps_to_file.pop("map_fwhm_arcmin", None)

    with h5py.File(chain_file + ".tmp", "w") as file:
        file["metadata/datetime"] = datetime.datetime.now().isoformat()
        file["metadata/parameter_file_as_string"] = params.parameter_file_as_string
        if write_maps and map_fwhm_arcmin is not None:
            file["metadata/map_fwhm_arcmin"] = map_fwhm_arcmin
        for key, value in tod_arrays.items():
            file[key] = value
        if write_maps:
            for key, value in maps_to_file.items():
                kind = MAP_KINDS.get(key, "brightness")
                # `maps_nside` is a cap: it thins maps that are finer than it, and leaves anything
                # already coarser alone.
                if nside_out != "native" and nside_out < hp.npix2nside(value.shape[-1]):
                    value = _degrade_map(value, kind, nside_out)
                if band_unit_factor != 1.0:
                    value = _to_band_unit(value, kind, band_unit_factor)
                file[f"maps/{key}"] = value
    os.replace(chain_file + ".tmp", chain_file)


def _write_nested(group: h5py.Group, tree: dict) -> None:
    """Write a nested dict of arrays and scalars, one HDF5 group per dict level.

    `None` values and empty dicts are skipped, so an output the run did not produce is absent from
    the file rather than an empty group.
    """
    for key, value in tree.items():
        if isinstance(value, dict):
            if value:
                _write_nested(group.require_group(str(key)), value)
        elif value is not None:
            group[str(key)] = value


@benchmark("filewrite-compsep")
def write_compsep_chain_to_file(comp_list: CompList, params: Bunch, chain: int, iter: int,
                                diagnostics: dict | None = None,
                                band_frequencies: dict[str, float] | None = None) -> None:
    """Write one component-separation sample: the components, and how well they fit the data.

    Does nothing if this chain or iteration is not written (`output.chains.write`,
    `output.chains.interval.compsep`). `diagnostics` is the fit and sampler record of
    `process_compsep` (see `chisq.collect_fit_diagnostics`). `band_frequencies` maps each band name
    to its frequency in GHz, for the mixing coefficients.
    """
    if chain not in params.output.chains.write or not should_write_chain(params, "compsep", iter):
        return
    chain_file = paths.compsep_chain_file(paths.resolve_output_dir(params.output), chain, iter)
    with h5py.File(chain_file + ".tmp", "w") as file:
        file["metadata/datetime"] = datetime.datetime.now().isoformat()
        file["metadata/parameter_file_as_string"] = params.parameter_file_as_string
        _write_nested(file, diagnostics or {})
        comp_list.write_chain(file, band_frequencies or {})
    os.replace(chain_file + ".tmp", chain_file)
