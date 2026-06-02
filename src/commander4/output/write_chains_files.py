import os
import h5py
import numpy as np
import healpy as hp
import datetime
from pixell.bunch import Bunch

from commander4.sky_models.component import Component, CompList


def write_map_chain_to_file(params: Bunch, chain: int, iter: int, exp_name:str,
                            band_name: str, maps_to_file: dict) -> None:
    if chain not in params.general.chains_to_write or (iter-1)%params.general.chain_maps_interval != 0:
        return
    nside_out = params.general.chain_maps_nside
    chain_dir = os.path.join(params.general.output_paths.chains, "datamaps")
    filename = f"{exp_name}_{band_name}_chain{chain:02d}_iter{iter:04d}.h5"
    chain_file = os.path.join(chain_dir, filename)

    with h5py.File(chain_file, "w") as file:
        file["metadata/datetime"] = datetime.datetime.now().isoformat()
        file["metadata/parameter_file_as_string"] = params.parameter_file_as_string
        for key, value, in maps_to_file.items():
            if nside_out != "native" and hp.npix2nside(value.shape[-1]) != nside_out:
                if "rms" in key:
                    value = 1.0/hp.ud_grade(1.0/value**2, nside_out, dtype=np.float32)**2
                else:
                    value = hp.ud_grade(value, nside_out, dtype=np.float32)
            print("key", key, np.min(value), np.max(value), flush=True)
            file[key] = value


def write_compsep_chain_to_file(comp_list: list[Component] | CompList, params: Bunch,
                                chain: int, iter: int):
    chain_dir = os.path.join(params.general.output_paths.chains, "compsep")
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
            file[f"comps/{comp.shortname}/longname"] = comp.longname
            file[f"comps/{comp.shortname}/shortname"] = comp.shortname
            if comp.defined_pol is not None:
                file[f"comps/{comp.shortname}/defined_pol"] = comp.defined_pol
            if comp.eval_pol is not None:
                file[f"comps/{comp.shortname}/eval_pol"] = comp.eval_pol