"""Writing the per-iteration chain files: band samples and maps on the TOD side, components on the
CompSep side."""
import os
import h5py
import numpy as np
import healpy as hp
import datetime
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.file_io import paths
from commander4.sky.comp_list import CompList
from commander4.sky.component import Component

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
_MAP_KINDS = {"rms": "rms", "cov": "weight", "nhit": "count"}


def _degrade_map(value: NDArray, kind: str, nside_out: int) -> NDArray:
    """Bring one `maps/` dataset to `nside_out`, in whatever quantity it is additive in.

    `hp.ud_grade` averages the sub-pixels it merges, which is right for a brightness but wrong for
    the other kinds: an rms has to average in inverse variance, and a weight or a count has to add.
    """
    if kind == "brightness":
        return hp.ud_grade(value, nside_out, dtype=np.float32)
    if kind == "rms":
        return 1.0/np.sqrt(hp.ud_grade(1.0/value**2, nside_out, dtype=np.float32))
    # Turn the average back into a sum by scaling with how many sub-pixels fell in each new pixel.
    subpixels = value.shape[-1] // hp.nside2npix(nside_out)
    summed = hp.ud_grade(value.astype(np.float64), nside_out)*subpixels
    if kind == "count":
        return np.round(summed).astype(np.int64)
    return summed


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
                             band_unit_factor: float = 1.0, band_unit: str = "uK_RJ") -> None:
    """Write one band's Gibbs sample: the per-scan TOD samples and the output maps, in one file.

    The TOD samples land at the top level (which is where `TODSamples` reads them back from for
    `gibbs.init_from_chain`) and the maps under `maps/`. The caller decides whether the file is
    written at all — `TODSamples.gather_chain_arrays` owns that gate, because it has to be applied
    before its collective gathers — so this function only gates the `maps/` group.

    The maps arrive in uK_RJ and are written in the band's `band_unit`, each converted according to
    its `_MAP_KINDS` entry by `band_unit_factor` D (=1 for uK_RJ). The gains in `tod_arrays` were
    already divided by D by the gather, gain having brightness in its denominator.

    `maps_to_file` also carries the scalar `map_fwhm_arcmin` (see `mapmaking.output`), which is the
    beam `observed_sky` and `rms` are at. It becomes file metadata rather than a dataset, and is
    what says whether those maps sit at the band's native beam or at `compsep.common_res_fwhm`,
    which is the beam `skymodel` is always at.
    """
    chains = params.output.chains
    nside_out = chains.maps_nside
    chain_dir = paths.subdir(params, paths.CHAINS_BANDS)
    filename = f"{exp_name}_{band_name}_chain{chain:02d}_iter{iter:04d}.h5"
    chain_file = os.path.join(chain_dir, filename)

    write_maps = should_write_chain(params, "maps", iter)
    maps_to_file = dict(maps_to_file)
    map_fwhm_arcmin = maps_to_file.pop("map_fwhm_arcmin", None)

    with h5py.File(chain_file, "w") as file:
        file["metadata/datetime"] = datetime.datetime.now().isoformat()
        file["metadata/parameter_file_as_string"] = params.parameter_file_as_string
        # Thermodynamic unit the written maps and gains are expressed in (maps are brightnesses in
        # band_unit; gain is [detector units]/band_unit).
        file["metadata/band_unit"] = band_unit
        if write_maps and map_fwhm_arcmin is not None:
            file["metadata/map_fwhm_arcmin"] = map_fwhm_arcmin
        for key, value in tod_arrays.items():
            file[key] = value
        if not write_maps:
            return
        for key, value in maps_to_file.items():
            kind = _MAP_KINDS.get(key, "brightness")
            if nside_out != "native" and hp.npix2nside(value.shape[-1]) != nside_out:
                value = _degrade_map(value, kind, nside_out)
            if band_unit_factor != 1.0:
                value = _to_band_unit(value, kind, band_unit_factor)
            file[f"maps/{key}"] = value


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


def write_compsep_chain_to_file(comp_list: list[Component] | CompList, params: Bunch,
                                chain: int, iter: int, diagnostics: dict | None = None,
                                band_frequencies: dict[str, float] | None = None):
    """Write one component-separation sample: the components, and how well they fit the data.

    `diagnostics` is the goodness-of-fit and sampler bookkeeping `process_compsep` collected for
    this iteration -- see `compsep.chisq.collect_fit_diagnostics` for its shape. C3 spreads the same
    information over `chisq_<postfix>.fits`, `fg_ind_mean_c<CCCC>.dat` and `nonlin-samples_*.dat`;
    here it lives next to the components it describes.

    `band_frequencies` maps each band name to its centre frequency in GHz, and is what lets the
    mixing coefficients be written per band.
    """
    if chain not in params.output.chains.write or not should_write_chain(params, "compsep", iter):
        return
    chain_dir = paths.subdir(params, paths.CHAINS_COMPSEP)
    chain_file = os.path.join(chain_dir, f"chain{chain:02d}_iter{iter:04d}.h5")
    components = comp_list.components if isinstance(comp_list, CompList) else comp_list
    with h5py.File(chain_file, "w") as file:
        file["metadata/datetime"] = datetime.datetime.now().isoformat()
        file["metadata/parameter_file_as_string"] = params.parameter_file_as_string
        _write_nested(file, diagnostics or {})
        seen_shortnames = set()
        for comp in components:
            if comp.shortname in seen_shortnames:
                raise ValueError(f"Duplicate component shortname '{comp.shortname}' in compsep chain.")
            seen_shortnames.add(comp.shortname)
            # Diffuse components carry their amplitudes as alms; point sources carry one amplitude
            # per source instead and have no `alms` at all, so each is written under the name that
            # says what it is rather than forcing the point sources into an alm-shaped dataset.
            if hasattr(comp, "alms"):
                file[f"comps/{comp.shortname}/alms"] = comp.alms
                # The realized power spectrum of those alms (Commander3's `sigma_l`) and the lmax
                # they run to. Free -- alm2cl needs no transform -- and it saves every reader the
                # alm2cl themselves.
                file[f"comps/{comp.shortname}/sigma_l"] = comp.sigma_l
                file[f"comps/{comp.shortname}/lmax"] = comp.lmax
            else:
                file[f"comps/{comp.shortname}/source_amps"] = comp._data
            file[f"comps/{comp.shortname}/comp_name"] = comp.comp_name
            file[f"comps/{comp.shortname}/shortname"] = comp.shortname
            # The beam these amplitudes carry, in arcmin. This is 0.0 if the CG compsep was used,
            # as it solves for the intrinsic sky. However, if the per-pix compsep solver was used,
            # the sky components are smoothed to this resolution.
            file[f"comps/{comp.shortname}/amp_fwhm_arcmin"] = np.degrees(comp.amp_fwhm_rad)*60.0
            # The SED parameters this component type declares (see `Component.sed_param_names`),
            # so the chain records the sampled spectral indices and everything else needed to
            # evaluate the SED. `nu_ref` may be a per-polarization array, hence no float() cast.
            for param_name in comp.sed_param_names:
                file[f"comps/{comp.shortname}/sed/{param_name}"] = getattr(comp, param_name)
            # The C(l) prior's model parameters (Commander3's Dl_amp/Dl_beta/Dl_theta), not the
            # evaluated spectrum: the evaluated one is reconstructible, and `sigma_l` above already
            # records the realized power. A `None` amplitude means the prior is off (C3 CL_TYPE
            # none), which writes nothing at all, exactly as C3 does.
            for param_name in comp.Cl_prior_param_names:
                value = getattr(comp, param_name)
                if value is not None:
                    file[f"comps/{comp.shortname}/Cl_prior/{param_name}"] = value
            # The mixing "matrix", C3's mixmat_<comp>_<band>. C4 indices are scalar, so a
            # component's whole mixing matrix at a band is the single number `get_sed(nu)`.
            for band_name, nu in (band_frequencies or {}).items():
                file[f"comps/{comp.shortname}/mixing/{band_name}"] = comp.get_sed(nu)
            if comp.defined_pol is not None:
                file[f"comps/{comp.shortname}/defined_pol"] = comp.defined_pol
            if comp.eval_pol is not None:
                file[f"comps/{comp.shortname}/eval_pol"] = comp.eval_pol
