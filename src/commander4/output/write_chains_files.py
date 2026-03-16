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
        file["metadata/parameter_file_as_binary_yaml"] = params.parameter_file_binary_yaml
        for key, value, in maps_to_file.items():
            if nside_out != "native" and hp.npix2nside(value.shape[-1]) != nside_out:
                value = hp.ud_grade(value, nside_out, dtype=np.float32)
            file[key] = value


def write_tod_chain_to_file(band_comm: MPI.Comm, tod_samples: TODSamples,
                            params: Bunch, chain: int, iter: int) -> None:
    tod_samples_batches = band_comm.gather(tod_samples, root=0)
    if band_comm.Get_rank() == 0:
        exp_name = tod_samples.experiment_name
        band_name = tod_samples.band_name
        chain_dir = os.path.join(params.general.output_paths.chains, "tod")
        # filename = f"{exp_name}_{det_name}_chain{chain:02d}_iter{iter:04d}.h5"
        filename = f"{exp_name}_{band_name}_chain{chain:02d}_iter{iter:04d}.h5"
        chain_file = os.path.join(chain_dir, filename)

        write_dict = {}
        for key, value in vars(tod_samples).items():
            write_dict[key] = []

        already_seen_bands = []
        for tod_samples_other in tod_samples_batches:
            for key, value in vars(tod_samples_other).items():
                if np.ndim(value) == 0 and tod_samples_other.band_name not in already_seen_bands:
                    write_dict[key].append(value)
                else:
                    write_dict[key].append(value)
            already_seen_bands.append(tod_samples_other.band_name)
        
        with h5py.File(chain_file, "w") as file:
            file["metadata/datetime"] = datetime.datetime.now().isoformat()
            file["metadata/parameter_file_as_string"] = params.parameter_file_as_string
            file["metadata/parameter_file_as_binary_yaml"] = params.parameter_file_binary_yaml
            for key in write_dict.keys():
                # If each entry is 0-dimensional, concatenate will crash, so use regular np.array.
                if key in ["g0_est", "rel_gain_est", "time_dep_rel_gain_est", "alpha_est", "fknee_est"]:
                    if np.ndim(write_dict[key][0]) == 0:
                        arr = np.array(write_dict[key])
                    else:
                        arr = np.concatenate(write_dict[key], axis=-1)
                    file[key] = arr


def write_compsep_chain_to_file(comp_list: list[Component], params: Bunch, chain: int, iter: int):
    chain_dir = os.path.join(params.general.output_paths.chains, "compsep")
    chain_file = os.path.join(chain_dir, f"chain{chain:02d}_iter{iter:04d}.h5")
    with h5py.File(chain_file, "w") as file:
        file["metadata/datetime"] = datetime.datetime.now().isoformat()
        file["metadata/parameter_file_as_string"] = params.parameter_file_as_string
        file["metadata/parameter_file_as_binary_yaml"] = params.parameter_file_binary_yaml
        for comp in comp_list:
            file[f"comps/{comp.shortname}/alms"] = comp.alms
            file[f"comps/{comp.shortname}/longname"] = comp.longname
            file[f"comps/{comp.shortname}/shortname"] = comp.shortname