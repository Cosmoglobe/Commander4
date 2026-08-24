"""Writing the per-iteration chain files: band maps on the TOD side, components on the CompSep
side."""
import os
import h5py
import numpy as np
import healpy as hp
import datetime
from pixell.bunch import Bunch

from commander4.file_io import paths
from commander4.sky.comp_list import CompList
from commander4.sky.component import Component

# The three chain outputs, each thinned by its own `output.chains.interval` entry. They are
# separate because they differ by orders of magnitude in size: a datamaps file holds several
# full-sky maps per band, while a compsep file holds alms and a TOD file per-scan scalars, so
# the maps are usually the only output worth thinning.
CHAIN_KINDS = ("tod", "compsep", "datamaps")


def should_write_chain(params: Bunch, kind: str, iter: int) -> bool:
    """Whether iteration `iter` (1-indexed) is one this `kind` of chain output is written on.

    The interval comes from `output.chains.interval.<kind>`, and defaults to 1 (write every
    iteration) when that entry, or the whole block, is absent.

    Args:
        kind: One of `CHAIN_KINDS`, naming which of the three chain outputs is being written.
        iter: Gibbs iteration number, counted from 1.

    Raises:
        ValueError: If `output.chains.interval` is a bare number rather than a per-kind block,
            which is the pre-2026 schema and would otherwise silently thin the wrong outputs.
    """
    if kind not in CHAIN_KINDS:
        raise ValueError(f"Unknown chain kind {kind!r}; expected one of {list(CHAIN_KINDS)}.")
    if "interval" not in params.output.chains:
        return True
    interval = params.output.chains.interval
    if not isinstance(interval, Bunch):
        raise ValueError(
            f"'output.chains.interval' is now one entry per chain output, not a single number. "
            f"Replace 'interval: {interval}' with a block naming the outputs to thin, e.g.\n"
            f"    interval:\n      tod: 1\n      compsep: 1\n      datamaps: {interval}\n"
            f"Any of {list(CHAIN_KINDS)} may be omitted, defaulting to 1 (every iteration).")
    return (iter - 1) % int(interval[kind]) == 0 if kind in interval else True


def write_map_chain_to_file(params: Bunch, chain: int, iter: int, exp_name:str,
                            band_name: str, maps_to_file: dict, band_unit_factor: float = 1.0,
                            band_unit: str = "uK_RJ") -> None:
    """Write a band's per-iteration output maps to the datamaps chain file.

    Maps in `maps_to_file` are uK_RJ brightnesses; they are written in the band's `band_unit` by
    multiplying by `band_unit_factor` D (=1 for uK_RJ). The multiply is out-of-place so the caller's
    arrays (shared with the uK_RJ DetectorMap sent to compsep) stay untouched.

    `maps_to_file` also carries the scalar `map_fwhm_arcmin` (see `mapmaking.output`), which is the
    beam `map_observed_sky` and `map_rms` are at. It becomes file metadata rather than a dataset,
    and is what says whether those maps sit at the band's native beam or at
    `compsep.common_res_fwhm`, which is the beam `map_skymodel` is always at.
    """
    chains = params.output.chains
    if chain not in chains.write or not should_write_chain(params, "datamaps", iter):
        return
    nside_out = chains.maps_nside
    chain_dir = paths.subdir(params, paths.CHAINS_DATAMAPS)
    filename = f"{exp_name}_{band_name}_chain{chain:02d}_iter{iter:04d}.h5"
    chain_file = os.path.join(chain_dir, filename)

    maps_to_file = dict(maps_to_file)
    map_fwhm_arcmin = maps_to_file.pop("map_fwhm_arcmin", None)

    with h5py.File(chain_file, "w") as file:
        file["metadata/datetime"] = datetime.datetime.now().isoformat()
        file["metadata/parameter_file_as_string"] = params.parameter_file_as_string
        # Thermodynamic unit the written maps are expressed in (all maps are brightnesses in band_unit).
        file["metadata/band_unit"] = band_unit
        if map_fwhm_arcmin is not None:
            file["metadata/map_fwhm_arcmin"] = map_fwhm_arcmin
        for key, value, in maps_to_file.items():
            if nside_out != "native" and hp.npix2nside(value.shape[-1]) != nside_out:
                if "rms" in key:
                    value = 1.0 / np.sqrt(
                        hp.ud_grade(1.0 / value**2, nside_out, dtype=np.float32)
                    )
                else:
                    value = hp.ud_grade(value, nside_out, dtype=np.float32)
            # uK_RJ -> band_unit (out-of-place copy; D is a Python float, so dtype is preserved).
            if band_unit_factor != 1.0:
                value = value * band_unit_factor
            file[key] = value


def write_compsep_chain_to_file(comp_list: list[Component] | CompList, params: Bunch,
                                chain: int, iter: int):
    if chain not in params.output.chains.write or not should_write_chain(params, "compsep", iter):
        return
    chain_dir = paths.subdir(params, paths.CHAINS_COMPSEP)
    chain_file = os.path.join(chain_dir, f"chain{chain:02d}_iter{iter:04d}.h5")
    components = comp_list.components if isinstance(comp_list, CompList) else comp_list
    with h5py.File(chain_file, "w") as file:
        file["metadata/datetime"] = datetime.datetime.now().isoformat()
        file["metadata/parameter_file_as_string"] = params.parameter_file_as_string
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
            if comp.defined_pol is not None:
                file[f"comps/{comp.shortname}/defined_pol"] = comp.defined_pol
            if comp.eval_pol is not None:
                file[f"comps/{comp.shortname}/eval_pol"] = comp.eval_pol
