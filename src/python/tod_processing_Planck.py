import logging
import numpy as np
import healpy as hp
import os
import h5py
import gc
from numpy.typing import NDArray
from cmdr4_support.utils import huffman_decode
from astropy.io import fits
from pixell.bunch import Bunch
from src.python.data_models.detector_TOD import DetectorTOD
from src.python.data_models.scan_TOD import ScanTOD
from src.python.data_models.detector_samples import DetectorSamples
from src.python.data_models.scan_samples import ScanSamples

def get_processing_mask(my_band: Bunch) -> DetectorTOD:
    """Subtracts the sky model from the TOD data.
    Input:
        experiment_data (DetectorTOD): The experiment TOD object.
        params (Bunch): The parameters from the input parameter file.
    Output:
        experiment_data (DetectorTOD): The experiment TOD with the estimated white noise level added to each scan.
    """
    hdul = fits.open(my_band.processing_mask)
    mask = hdul[1].data["TEMPERATURE"].flatten().astype(bool)
    nside = np.sqrt(mask.size//12)
    if nside != my_band.eval_nside:
        mask = hp.ud_grade(mask.astype(np.float64), my_band.eval_nside) == 1
    return mask

def find_good_Fourier_time(Fourier_times:NDArray, ntod:int) -> int:
    if ntod <= 10_000 or ntod >= 400_000:
        return ntod
    search_start = int(0.99*ntod)  # Consider sizes up to 1% smaller than ntod.
    best_ntod = np.argmin(Fourier_times[search_start:ntod+1])
    best_ntod += search_start
    assert(best_ntod <= ntod)
    return best_ntod


def read_Planck_TOD_data(my_experiment: str, my_band: Bunch, my_det: Bunch, params: Bunch, my_detector_id: int, scan_idx_start: int, scan_idx_stop: int, bad_PIDs_path:str=None) -> tuple[DetectorTOD, DetectorSamples]:
    logger = logging.getLogger(__name__)
    oids = []
    pids = []
    filenames = []
    detname = my_det.name
    with open(my_experiment.data_path + f"filelist_{my_band.freq_identifier:02d}.txt") as infile:
        infile.readline()
        for line in infile:
            pid, filename, _, _, _ = line.split()
            pids.append(f"{int(pid):06d}")
            filenames.append(filename[1:-1])
            oids.append(filename.split(".")[0].split("_")[-1])
    scanlist = []
    num_included = 0
    
    processing_mask_map = get_processing_mask(my_band)
    if bad_PIDs_path is not None:
        bad_PIDs = np.load(bad_PIDs_path)
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

        filename = f"LFI_{my_band.freq_identifier:03d}_{oid.zfill(6)}.h5"
        filepath = os.path.join(my_experiment.data_path, filename)
        with h5py.File(filepath, "r") as f:
            ntod = int(f[f"/{pid}/common/ntod"][0])  # ntod is a size-1 array for some reason.
            ntod_optimal = find_good_Fourier_time(Fourier_times, ntod)
            tod = f[f"/{pid}/{detname}/tod/"][:ntod_optimal]
            huffman_tree = f[f"/{pid}/common/hufftree"][()]
            huffman_symbols = f[f"/{pid}/common/huffsymb"][()]
            pix_encoded = f[f"/{pid}/{detname}/pix/"][()]
            psi_encoded = f[f"/{pid}/{detname}/psi/"][()]
            vsun = f[f"/{pid}/common/vsun/"][()]
            fsamp = f["/common/fsamp/"][()]
            npsi = int(f["/common/npsi/"][0])
            flag_encoded = f[f"/{pid}/{detname}/flag/"][()]
        if ntod > ntod_upper_bound:
            raise ValueError(f"{ntod_upper_bound} {ntod}")
        flag_buffer[:ntod] = 0.0
        flag_buffer[:ntod] = huffman_decode(np.frombuffer(flag_encoded, dtype=np.uint8), huffman_tree, huffman_symbols, flag_buffer[:ntod])
        flag_buffer[:ntod_optimal] = np.cumsum(flag_buffer[:ntod_optimal])
        flag_buffer[:ntod_optimal] &= 6111232
        if np.sum(flag_buffer[:ntod_optimal]) == 0:
            tod_buffer[:ntod_optimal] = np.abs(tod)
            if np.mean(tod_buffer[:ntod_optimal]) < 0.001 and np.std(tod) < 0.001:  # Check for crazy data.
                scanlist.append(ScanTOD(tod, pix_encoded, psi_encoded, 0., pid, my_band.eval_nside, my_band.data_nside,
                                        fsamp, vsun, huffman_tree, huffman_symbols, npsi, processing_mask_map, ntod))
                num_included += 1
            ntod_sum_original += ntod
            ntod_sum_final += ntod_optimal
        if i_pid % 10 == 0:
            gc.collect()

    logger.info(f"Fraction of scans included for {my_band.freq_identifier} {my_det.name}: "
                f"{num_included/(scan_idx_stop-scan_idx_start)*100:.1f} %")
    logger.info(f"Avg scan size remaining after Fourier cut {my_band.freq_identifier} {my_det.name}: "
                f"{ntod_sum_final/ntod_sum_original*100:.1f} %")

    det_static = DetectorTOD(scanlist, float(my_band.freq), my_band.fwhm, my_band.eval_nside, my_band.data_nside, my_det.name)
    det_static.detector_id = my_detector_id

    scansample_list = []
    for iscan in range(num_included):
        scansample_list.append(ScanSamples())
        scansample_list[-1].time_dep_rel_gain_est = 0.0
        scansample_list[-1].rel_gain_est = my_det.rel_gain_est
        scansample_list[-1].gain_est = my_det.rel_gain_est + params.initial_g0
    det_samples = DetectorSamples(scansample_list)
    det_samples.detector_id = my_detector_id
    det_samples.g0_est = params.initial_g0
    det_samples.detname = detname

    return det_static, det_samples