import logging
import numpy as np
import healpy as hp
import os
import h5py
import gc
from numpy.typing import NDArray
from astropy.io import fits
from pixell.bunch import Bunch
from commander4.cmdr4_support import utils as cpp_utils
from commander4.data_models.detector_TOD import DetectorTOD
from commander4.data_models.scan_TOD import ScanTOD
from commander4.data_models.detector_samples import DetectorSamples


def find_good_Fourier_time(Fourier_times:NDArray, ntod:int) -> int:
    if ntod <= 10_000 or ntod >= 400_000:
        return ntod
    search_start = int(0.99*ntod)  # Consider sizes up to 1% smaller than ntod.
    best_ntod = np.argmin(Fourier_times[search_start:ntod+1])
    best_ntod += search_start
    assert(best_ntod <= ntod)
    return best_ntod


def tod_reader(my_experiment: Bunch, my_band: Bunch, my_det: Bunch, params: Bunch, my_det_id: int,
               scan_idx_start: int, scan_idx_stop: int) -> DetectorTOD:
    logger = logging.getLogger(__name__)
    oids = []
    pids = []
    filenames = []
    detname = str(my_det)
    expname = str(my_experiment)
    with open(my_band.filelist) as infile:
        infile.readline()
        for line in infile:
            pid, filename, _, _, _ = line.split()
            pids.append(f"{int(pid):06d}")
            filenames.append(filename[1:-1])
            oids.append(filename.split(".")[0].split("_")[-1])
    scanlist = []
    num_included = 0
    
    processing_mask_map = np.ones(12*my_band.eval_nside**2, dtype=bool)
    if "bad_PIDs_path" in my_experiment:
        bad_PIDs = np.load(my_experiment.bad_PIDs_path)
    else:
        bad_PIDs = np.array([])

    Fourier_times = np.load(my_experiment.Fourier_times_path)

    # Attempting to reduce fragmentation by allocating buffers.
    ntod_upper_bound = int(my_band.fsamp*100*3600)  # 10 hour scan.
    flag_buffer = np.zeros(ntod_upper_bound, dtype=np.int64)
    tod_buffer = np.zeros(ntod_upper_bound, dtype=np.float32)
    
    ntod_sum_original = 0
    ntod_sum_final = 0
    for i_pid in range(scan_idx_start, scan_idx_stop):
        pid = pids[i_pid]
        oid = oids[i_pid]
        if pid in bad_PIDs:
            continue

        filename = f"LB_{my_band.freq_identifier:03d}_M2_{oid.zfill(6)}.h5"
        filepath = os.path.join(my_experiment.data_path, filename)
        with h5py.File(filepath, "r") as f:
            ntod = int(f[f"/{pid}/common/ntod"][()])
            ntod_optimal = find_good_Fourier_time(Fourier_times, ntod)
            tod = f[f"/{pid}/{detname}/tod/"][:ntod_optimal].astype(np.float32)
            huffman_tree = f[f"/{pid}/common/hufftree"][()]
            huffman_symbols = f[f"/{pid}/common/huffsymb"][()]
            pix_encoded = f[f"/{pid}/{detname}/pix/"][()]
            psi_encoded = f[f"/{pid}/{detname}/psi/"][()]
            vsun = f[f"/{pid}/common/vsun/"][()]
            fsamp = float(f["/common/fsamp/"][()])
            npsi = int(f["/common/npsi/"][()])
            flag_encoded = f[f"/{pid}/{detname}/flag/"][()]
        if ntod > ntod_upper_bound:
            raise ValueError(f"{ntod_upper_bound} {ntod}")
        flag_buffer[:ntod] = 0.0
        flag_buffer[:ntod] = cpp_utils.huffman_decode(np.frombuffer(flag_encoded, dtype=np.uint8),
                                                      huffman_tree, huffman_symbols,
                                                      flag_buffer[:ntod])
        flag_buffer[:ntod_optimal] = np.cumsum(flag_buffer[:ntod_optimal])
        flag_buffer[:ntod_optimal] &= 6111232
        if np.sum(flag_buffer[:ntod_optimal]) == 0:
            tod_buffer[:ntod_optimal] = np.abs(tod)
            scanlist.append(ScanTOD(tod, pix_encoded, psi_encoded, 0., pid, my_band.eval_nside,
                                    my_band.data_nside, fsamp, vsun, huffman_tree, huffman_symbols,
                                    npsi,processing_mask_map, ntod,
                                    pix_is_compressed=my_experiment.pix_is_compressed,
                                    psi_is_compressed=my_experiment.psi_is_compressed))
            num_included += 1
            ntod_sum_original += ntod
            ntod_sum_final += ntod_optimal
        if i_pid % 10 == 0:
            gc.collect()

    logger.info(f"Fraction of scans included for {my_band.freq_identifier} {detname}: "
                f"{num_included/(scan_idx_stop-scan_idx_start)*100:.1f} %")
    logger.info(f"Avg scan size remaining after Fourier cut {my_band.freq_identifier} {detname}: "
                f"{ntod_sum_final/ntod_sum_original*100:.1f} %")

    det_static = DetectorTOD(scanlist, float(my_band.freq)+float(my_det.bandpass_shift),
                             my_band.fwhm, my_band.eval_nside, my_band.data_nside, detname, expname)
    det_static.detector_id = my_det_id

    return det_static