import os
import h5py
import numpy as np
import healpy as hp
import datetime
from pixell.bunch import Bunch

from commander4.file_io import paths
from commander4.sky.comp_list import CompList
from commander4.sky.component import Component


def write_map_chain_to_file(params: Bunch, chain: int, iter: int, exp_name:str,
                            band_name: str, maps_to_file: dict, band_unit_factor: float = 1.0,
                            band_unit: str = "uK_RJ") -> None:
    """Write a band's per-iteration output maps to the datamaps chain file.

    Maps in `maps_to_file` are uK_RJ brightnesses; they are written in the band's `band_unit` by
    multiplying by `band_unit_factor` D (=1 for uK_RJ). The multiply is out-of-place so the caller's
    arrays -- shared with the uK_RJ DetectorMap sent to compsep -- stay untouched.
    """
    chains = params.output.chains
    if chain not in chains.write or (iter-1) % chains.interval != 0:
        return
    nside_out = chains.maps_nside
    chain_dir = paths.subdir(params, paths.CHAINS_DATAMAPS)
    filename = f"{exp_name}_{band_name}_chain{chain:02d}_iter{iter:04d}.h5"
    chain_file = os.path.join(chain_dir, filename)

    with h5py.File(chain_file, "w") as file:
        file["metadata/datetime"] = datetime.datetime.now().isoformat()
        file["metadata/parameter_file_as_string"] = params.parameter_file_as_string
        # Thermodynamic unit the written maps are expressed in (all maps are brightnesses in band_unit).
        file["metadata/band_unit"] = band_unit
        for key, value, in maps_to_file.items():
            if nside_out != "native" and hp.npix2nside(value.shape[-1]) != nside_out:
                if "rms" in key:
                    value = 1.0/hp.ud_grade(1.0/value**2, nside_out, dtype=np.float32)**2
                else:
                    value = hp.ud_grade(value, nside_out, dtype=np.float32)
            # uK_RJ -> band_unit (out-of-place copy; D is a Python float, so dtype is preserved).
            if band_unit_factor != 1.0:
                value = value * band_unit_factor
            print("key", key, np.min(value), np.max(value), flush=True)
            file[key] = value


def write_compsep_chain_to_file(comp_list: list[Component] | CompList, params: Bunch,
                                chain: int, iter: int):
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
            file[f"comps/{comp.shortname}/alms"] = comp.alms
            file[f"comps/{comp.shortname}/comp_name"] = comp.comp_name
            file[f"comps/{comp.shortname}/shortname"] = comp.shortname
            if comp.defined_pol is not None:
                file[f"comps/{comp.shortname}/defined_pol"] = comp.defined_pol
            if comp.eval_pol is not None:
                file[f"comps/{comp.shortname}/eval_pol"] = comp.eval_pol