import h5py
import logging
import numpy as np
import healpy as hp
from output import log
from src.python.data_models.detector_TOD import DetectorTOD
from src.python.data_models.scan_TOD import ScanTOD
import sys
from src.python.utils.commander_tod import commander_tod


def read_Planck_TOD_data(database_filename: str, band: int, scan_idx_start: int, scan_idx_stop: int, nside: int, fwhm: float) -> DetectorTOD:
    logger = logging.getLogger(__name__)

    oids = []
    pids = []
    filenames = []
    # with open("/mn/stornext/d16/cmbco/bp/mathew/test/filelist_30.txt") as infile:
    with open(database_filename + f"filelist_{band:02d}.txt") as infile:
        infile.readline()
        for line in infile:
            pid, filename, _, _, _ = line.split()
            pids.append(f"{int(pid):06d}")
            filenames.append(filename[1:-1])
            oids.append(filename.split(".")[0].split("_")[-1])
    com_tod = commander_tod(database_filename, "LFI")
    scanlist = []
    num_included = 0
    if band == 30:
        detname = "27"
        local_nside = 512
    elif band == 44:
        detname = "24"
        local_nside = 512
    elif band == 70:
        detname = "18"
        local_nside = 1024
    if band == 30 and scan_idx_start == 0:
        from tqdm import trange
        myrange = range #trange
    else:
        myrange = range

    for i_pid in myrange(scan_idx_start, scan_idx_stop):
        if (i_pid-scan_idx_start) % 100 == 0:
            print(band, scan_idx_start, i_pid-scan_idx_start, "/", scan_idx_stop-scan_idx_start)
        pid = pids[i_pid]
        oid = oids[i_pid]
        com_tod.init_file(f"{band:03d}", oid)
        flag_27M = com_tod.decompress(f"/{pid}/{detname}M/flag/", compression="huffman")
        flag_27S = com_tod.decompress(f"/{pid}/{detname}S/flag/", compression="huffman")
        pix_27M = com_tod.decompress(f"/{pid}/{detname}M/pix/", compression="huffman")
        tod_27M = com_tod.decompress(f"/{pid}/{detname}M/tod/")[()].astype(np.float32)
        tod_27S = com_tod.decompress(f"/{pid}/{detname}S/tod/")[()].astype(np.float32)
        vsun = com_tod.decompress(f"/{pid}/common/vsun/")[()]
        fsamp = com_tod.decompress(f"/common/fsamp/")[()]
        if local_nside != nside:
            pix_27M = hp.ang2pix(nside, *hp.pix2ang(local_nside, pix_27M))
        # vsun = vsun[None,:]*np.ones_like(pix_27M)[:,None]  #TODO: This doesn't need to be recast. Will save memory.
        mask = ((flag_27M & flag_27S) & 6111232) == 0
        # LOS_vec = hp.pix2vec(512, pix_27M)  #TODO: This can be inferred later when needed, saving memory.
        if mask.all():
            theta, phi = hp.pix2ang(nside, pix_27M)
            scanlist.append(ScanTOD((tod_27M + tod_27S)/2.0, theta, phi, np.zeros_like(theta), 0., i_pid))
            # scanlist[-1].LOS_vec = LOS_vec
            scanlist[-1].orb_dir_vec = vsun
            scanlist[-1].fsamp = fsamp
            num_included += 1
    print(num_included/(scan_idx_stop-scan_idx_start))
    det = DetectorTOD(scanlist, float(band), fwhm)
    return det
