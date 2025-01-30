from commander_tod import commander_tod
import numpy as np
import paramfile_sim as param

def save_to_h5_file(ds, pix, psi, fname=None):
    nside = param.NSIDE
    npix = 12*nside**2
    chunk_size = npix//40
    ntod = param.NTOD
    n_chunks = param.NTOD // chunk_size
    print(f'Number of scans is {n_chunks}')

    output_path = param.OUTPUT_FOLDER
    version = 'v1'
    comm_tod = commander_tod(output_path, "", version, overwrite=True)

    if fname is None:
        hdf_filename = f'tod_sim_{param.NSIDE}_s{param.SIGMA_SCALE}_b{param.FWHM[0]:.0f}'

    COMMON_GROUP = "/common"
    HUFFMAN_COMPRESSION = ["huffman", {"dictNum": 1}]

    comm_tod.init_file(freq=hdf_filename, od="", mode="w")
    comm_tod.add_field(COMMON_GROUP + "/nside", [nside])

    for pid in range(n_chunks):
        pid_label = f'{pid+1:06}'
        pid_common_group = pid_label + "/common"
        for i, freq in enumerate(param.FREQ):
            pid_data_group = f'{pid_label}/{freq:04}'

            comm_tod.add_field(pid_common_group + "/ntod", [chunk_size])


            tod_chunk_i = ds[i][pid*chunk_size : (pid+1)*chunk_size]
            pix_chunk_i =   pix[pid*chunk_size : (pid+1)*chunk_size]
            psi_chunk_i =   psi[pid*chunk_size : (pid+1)*chunk_size]

            comm_tod.add_field(pid_data_group + "/tod", tod_chunk_i)
            comm_tod.add_field(pid_data_group + "/pix", pix_chunk_i)
            comm_tod.add_field(pid_data_group + "/psi", psi_chunk_i)
        comm_tod.finalize_chunk(pid+1)

    if (ntod//chunk_size != ntod/chunk_size):
        pid = n_chunks
        pid_label = f'{pid+1:06}'
        pid_common_group = pid_label + "/common"
        for i, freq in enumerate(param.FREQ):
            pid_data_group = f'{pid_label}/{freq:04}'

            tod_chunk_i = ds[i][pid*chunk_size : ]
            pix_chunk_i = pix[pid*chunk_size : ]
            psi_chunk_i = psi[pid*chunk_size : ]
            comm_tod.add_field(pid_common_group + "/ntod", [len(tod_chunk_i)])
            comm_tod.add_field(pid_data_group + "/tod", tod_chunk_i)
            comm_tod.add_field(pid_data_group + "/pix", pix_chunk_i)
            comm_tod.add_field(pid_data_group + "/psi", psi_chunk_i)
        comm_tod.finalize_chunk(pid+1)


    comm_tod.finalize_file()
