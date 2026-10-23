"""Writing the per-iteration chain files: band samples and maps on the TOD side, components on the
CompSep side.

What a file holds about each sampled object is decided by that object: each TOD sampling step
gives its arrays in `chain_arrays`, which `TODSamples.gather_chain_arrays` gathers over the band's
ranks, and each component writes its own group in `write_chain`. The
functions here own the file: its name, its metadata, the band maps, and the rule that a file only
ever appears under its final name once it is complete (it is written as `<name>.tmp` and renamed),
so a restart never reads a file a crash left half-written.
"""
import os
import h5py
import numpy as np
import healpy as hp
import datetime
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.file_io import paths
from commander4.sky.comp_list import CompList

# What `output.chains.interval` can thin. `bands` and `compsep` gate a whole file; `maps` gates
# only the `maps/` group inside the band file, because the maps are orders of magnitude larger
# than the per-scan samples next to them and are usually the only output worth thinning.
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
    """Bring one `maps/` dataset down to `nside_out`, in whatever quantity it is additive in.

    `hp.ud_grade` averages the sub-pixels it merges, which is right for a brightness but wrong for
    the other kinds: a weight, a count, and an rms (through its inverse variance) all add.

    Only downgrading is defined here: the sub-pixel count below is what turns the average back
    into a sum, and it is zero (silently collapsing rms to inf and the counts to zero) for a finer
    `nside_out`. The caller is responsible for not asking for one.
    """
    if kind == "brightness":
        return hp.ud_grade(value, nside_out, dtype=np.float32)
    # The rest are additive, so turn ud_grade's average back into a sum by scaling with how many
    # sub-pixels fell in each new pixel. An rms adds as its inverse variance, 1/sigma^2.
    additive = 1.0/value**2 if kind == "rms" else value
    subpixels = value.shape[-1] // hp.nside2npix(nside_out)
    if subpixels < 1:
        raise ValueError(f"_degrade_map cannot upgrade a '{kind}' map to nside {nside_out}.")
    summed = hp.ud_grade(additive.astype(np.float64), nside_out)*subpixels
    if kind == "rms":
        return (1.0/np.sqrt(summed)).astype(np.float32)
    elif kind == "count":
        return np.round(summed).astype(np.int64)
    elif kind == "weight":  # A summed inverse variance, left as float64.
        return summed
    else:
        raise ValueError(f"Unknown map kind '{kind}' in _degrade_map.")


def _to_band_unit(value: NDArray, kind: str, band_unit_factor: float) -> NDArray:
    """Convert one `maps/` dataset from uK_RJ to the band's own unit.

    Out-of-place, so the caller's arrays -- shared with the uK_RJ `DetectorMap` sent to compsep --
    stay untouched. D is a Python float, so the dtype survives.
    """
    if kind == "count":  # A sample count carries no thermodynamic unit.
        return value
    if kind == "weight":  # An inverse variance, i.e. uK_RJ^-2.
        return value/band_unit_factor**2
    return value*band_unit_factor


def should_write_chain(params: Bunch, kind: str, iter: int) -> bool:
    """Whether iteration `iter` (1-indexed) is one this `kind` of chain output is written on.

    The interval comes from `output.chains.interval.<kind>`, and defaults to 1 (write every
    iteration) when that entry, or the whole block, is absent.

    Note that `maps` gates a group inside the band file, so the maps appear only on iterations
    where *both* `bands` and `maps` fire; keep `maps` a multiple of `bands`.

    Args:
        kind: One of `CHAIN_KINDS`, naming which chain output is being written.
        iter: Gibbs iteration number, counted from 1.

    Raises:
        ValueError: If `output.chains.interval` is not a block of `CHAIN_KINDS` entries. Silently
            ignoring a key nobody reads would thin the wrong outputs, or nothing at all.
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


def write_band_chain_to_file(params: Bunch, chain: int, iter: int, exp_name: str, band_name: str,
                             tod_arrays: dict[str, NDArray], maps_to_file: dict,
                             band_unit_factor: float = 1.0) -> None:
    """Write one band's Gibbs sample: the per-scan TOD samples and the output maps, in one file.

    `tod_arrays` is what `TODSamples.gather_chain_arrays` collected: the sampled state that
    `TODSamples.read_chain` reads back on a restart, plus diagnostics and debug TODs. Its keys are
    dataset paths; h5py creates their groups. The caller decides if the file is written at all —
    `TODSamples.gather_chain_arrays` owns that gate, because it has to be applied before its
    collective gathers — so this function only gates the `maps/` group.

    The maps arrive in uK_RJ and are written in the band's `band_unit`, each converted according to
    its `MAP_KINDS` entry by `band_unit_factor` D (=1 for uK_RJ). The gains in `tod_arrays` were
    already divided by D by the gather, gain having brightness in its denominator.

    `maps_to_file` also carries the scalar `map_fwhm_arcmin` (see `mapmaking.output`), which is the
    beam `observed_sky` and `rms` are at. It becomes file metadata rather than a dataset, and is
    what says whether those maps sit at the band's native beam or at `compsep.common_res_fwhm`,
    which is the beam `skymodel` is always at.
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

    `None` values and empty dicts are skipped rather than written, so a diagnostic the run did not
    produce (a prior that is switched off, a sampler that never ran) is simply absent from the file
    instead of appearing as an empty group.
    """
    for key, value in tree.items():
        if isinstance(value, dict):
            if value:
                _write_nested(group.require_group(str(key)), value)
        elif value is not None:
            group[str(key)] = value


def write_compsep_chain_to_file(comp_list: CompList, params: Bunch, chain: int, iter: int,
                                diagnostics: dict | None = None,
                                band_frequencies: dict[str, float] | None = None) -> None:
    """Write one component-separation sample: the components, and how well they fit the data.

    Each component writes its own `comps/<shortname>/` group (`CompList.write_chain`), which is
    what `CompList.load_initial_state` reads back on a restart.

    `diagnostics` is the goodness-of-fit and sampler bookkeeping `process_compsep` collected for
    this iteration -- see `compsep.chisq.collect_fit_diagnostics` for its shape. C3 spreads the same
    information over `chisq_<postfix>.fits`, `fg_ind_mean_c<CCCC>.dat` and `nonlin-samples_*.dat`;
    here it lives next to the components it describes.

    `band_frequencies` maps each band name to its centre frequency in GHz, and is what lets the
    mixing coefficients be written per band.
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
