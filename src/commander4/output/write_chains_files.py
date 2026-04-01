import os
import h5py
import numpy as np
import healpy as hp
import datetime
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.data_models.TOD_samples import TODSamples
from commander4.sky_models.component import Component


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
                value = hp.ud_grade(value, nside_out, dtype=np.float32)
            file[key] = value


def write_compsep_chain_to_file(comp_list: list[Component], params: Bunch, chain: int, iter: int):
    chain_dir = os.path.join(params.general.output_paths.chains, "compsep")
    chain_file = os.path.join(chain_dir, f"chain{chain:02d}_iter{iter:04d}.h5")
    with h5py.File(chain_file, "w") as file:
        file["metadata/datetime"] = datetime.datetime.now().isoformat()
        file["metadata/parameter_file_as_string"] = params.parameter_file_as_string
        for comp in comp_list:
            file[f"comps/{comp.shortname}/alms"] = comp.alms
            file[f"comps/{comp.shortname}/longname"] = comp.longname
            file[f"comps/{comp.shortname}/shortname"] = comp.shortname