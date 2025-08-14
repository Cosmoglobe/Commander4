import logging
import numpy as np
import healpy as hp
from pixell.bunch import Bunch
from src.python.data_models.detector_TOD import DetectorTOD
from src.python.data_models.scan_TOD import ScanTOD
from src.python.utils.commander_tod import commander_tod
from src.python.data_models.detector_samples import DetectorSamples
from src.python.data_models.scan_samples import ScanSamples


def read_Planck_TOD_data(database_filename: str, my_band: Bunch, params: Bunch, scan_idx_start: int, scan_idx_stop: int) -> DetectorTOD:
    logger = logging.getLogger(__name__)

    oids = []
    pids = []
    filenames = []
    with open(database_filename + f"filelist_{my_band.freq_identifier:02d}.txt") as infile:
        infile.readline()
        for line in infile:
            pid, filename, _, _, _ = line.split()
            pids.append(f"{int(pid):06d}")
            filenames.append(filename[1:-1])
            oids.append(filename.split(".")[0].split("_")[-1])
    com_tod = commander_tod(database_filename, "LFI")
    scanlist = []
    num_included = 0
    if my_band.freq_identifier == 30:
        detname = "27"
        local_nside = 512
    elif my_band.freq_identifier == 44:
        detname = "24"
        local_nside = 512
    elif my_band.freq_identifier == 70:
        detname = "18"
        local_nside = 1024
    if my_band.freq_identifier == 70 and scan_idx_stop >= 45850:
        from tqdm import trange
        myrange = trange
    else:
        myrange = range

    bad_PIDs = np.load("/mn/stornext/d23/cmbco/jonas/c4_testing/Commander4/badPIDs.npy")
    previous_oid = -999999
    for i_pid in myrange(scan_idx_start, scan_idx_stop):
        pid = pids[i_pid]
        oid = oids[i_pid]
        if pid in bad_PIDs:
            continue
        if oid != previous_oid:  # Only open file if it's not the same file as the last PID.
            com_tod.init_file(f"{my_band.freq_identifier:03d}", oid)
        flag_M = com_tod.decompress(f"/{pid}/{detname}M/flag/", compression="huffman")
        flag_S = com_tod.decompress(f"/{pid}/{detname}S/flag/", compression="huffman")
        pix_M = com_tod.decompress(f"/{pid}/{detname}M/pix/", compression="huffman").astype(np.uint32)
        tod_M = com_tod.decompress(f"/{pid}/{detname}M/tod/")[()].astype(np.float32)
        tod_S = com_tod.decompress(f"/{pid}/{detname}S/tod/")[()].astype(np.float32)
        vsun = com_tod.decompress(f"/{pid}/common/vsun/")[()]
        fsamp = com_tod.decompress("/common/fsamp/")[()]
        if local_nside != my_band.nside:
            pix_M = hp.ang2pix(my_band.nside, *hp.pix2ang(local_nside, pix_M)).astype(np.uint32)
        mask = ((flag_M & flag_S) & 6111232) == 0
        if mask.all():
            tod = (tod_M + tod_S)/2.0
            if np.mean(np.abs(tod)) < 0.001 and np.std(tod) < 0.001:  # Check for crazy data.
                theta, phi = hp.pix2ang(my_band.nside, pix_M)
                scanlist.append(ScanTOD(tod, theta.astype(np.float32), phi.astype(np.float32), np.zeros_like(theta, dtype=np.float32), 0., i_pid))
                scanlist[-1].orb_dir_vec = vsun
                scanlist[-1].fsamp = fsamp
                scanlist[-1].PID = pid
                num_included += 1
    logger.info(f"Fraction of scans included for {my_band.freq_identifier}: "
                f"{num_included/(scan_idx_stop-scan_idx_start)*100:.1f} %")
    det_static = DetectorTOD(scanlist, float(my_band.freq), my_band.fwhm, my_band.nside)
    det_static.detector_id = my_band.detector_id

    scansample_list = []
    for iscan in range(num_included):
        scansample_list.append(ScanSamples())
        scansample_list[-1].time_dep_rel_gain_est = 0.0
        scansample_list[-1].rel_gain_est = my_band.rel_gain_est
        scansample_list[-1].gain_est = my_band.rel_gain_est + params.initial_g0
    det_samples = DetectorSamples(scansample_list)
    det_samples.detector_id = my_band.detector_id
    det_samples.g0_est = params.initial_g0

    return det_static, det_samples