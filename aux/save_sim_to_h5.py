import numpy as np
from src.python.parse_params import params, params_dict
from commander_tod import commander_tod
from tqdm import trange

def save_to_h5_file(ds, theta, phi, psi, orb_dir=None, fname=None):
    nside = params.NSIDE
    chunk_size = int(3600*params.SAMP_FREQ)  # Chunk size = 1 hour
    ntod = params.NTOD
    n_chunks = params.NTOD // chunk_size
    print(f'Number of scans is {n_chunks}')

    output_path = params.OUTPUT_FOLDER
    version = 'v1'
    comm_tod = commander_tod(output_path, "", version, overwrite=True)

    if fname is None:
        fname = f'tod_sim_{params.NSIDE}_s{params.SIGMA_SCALE}_b{params.FWHM[0]:.0f}'

    COMMON_GROUP = "/common"
    HUFFMAN_COMPRESSION = ["huffman", {"dictNum": 1}]

    comm_tod.init_file(freq=fname, od="", mode="w")
    comm_tod.add_field(COMMON_GROUP + "/nside", [nside])

    for pid in trange(n_chunks):
        pid_label = f'{pid+1:06}'
        pid_common_group = pid_label + "/common"
        for i, freq in enumerate(params.FREQ):
            pid_data_group = f'{pid_label}/{freq:04}'

            comm_tod.add_field(pid_common_group + "/ntod", [chunk_size])

            tod_chunk_i = ds[i][pid*chunk_size : (pid+1)*chunk_size]
            theta_chunk_i = theta[pid*chunk_size : (pid+1)*chunk_size]
            phi_chunk_i =   phi[pid*chunk_size : (pid+1)*chunk_size]
            psi_chunk_i =   psi[pid*chunk_size : (pid+1)*chunk_size]

            comm_tod.add_field(pid_data_group + "/tod", tod_chunk_i)
            comm_tod.add_field(pid_data_group + "/theta", theta_chunk_i)
            comm_tod.add_field(pid_data_group + "/phi", phi_chunk_i)
            comm_tod.add_field(pid_data_group + "/psi", psi_chunk_i)

            if orb_dir is not None:
                orb_dir_chunk_i =   orb_dir[pid*chunk_size : (pid+1)*chunk_size]
                comm_tod.add_field(pid_data_group + "/orbital_dir_vec", orb_dir_chunk_i)

        comm_tod.finalize_chunk(pid+1)

    if (ntod//chunk_size != ntod/chunk_size):
        pid = n_chunks
        pid_label = f'{pid+1:06}'
        pid_common_group = pid_label + "/common"
        for i, freq in enumerate(params.FREQ):
            pid_data_group = f'{pid_label}/{freq:04}'

            comm_tod.add_field(pid_common_group + "/ntod", [len(tod_chunk_i)])

            tod_chunk_i = ds[i][pid*chunk_size : ]
            theta_chunk_i = theta[pid*chunk_size : ]
            phi_chunk_i =   phi[pid*chunk_size : ]
            psi_chunk_i = psi[pid*chunk_size : ]
            comm_tod.add_field(pid_data_group + "/tod", tod_chunk_i)
            comm_tod.add_field(pid_data_group + "/theta", theta_chunk_i)
            comm_tod.add_field(pid_data_group + "/phi", phi_chunk_i)
            comm_tod.add_field(pid_data_group + "/psi", psi_chunk_i)

            if orb_dir is not None:
                orb_dir_chunk_i =   orb_dir[pid*chunk_size : ]
                comm_tod.add_field(pid_data_group + "/orbital_dir_vec", orb_dir_chunk_i)

        comm_tod.finalize_chunk(pid+1)


    comm_tod.finalize_file()
