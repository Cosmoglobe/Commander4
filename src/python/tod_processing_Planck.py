import logging
import numpy as np
import healpy as hp
import os
import h5py
from astropy.io import fits
from pixell.bunch import Bunch
from src.python.data_models.detector_TOD import DetectorTOD
from src.python.data_models.scan_TOD import ScanTOD
from src.python.data_models.detector_samples import DetectorSamples
from src.python.data_models.scan_samples import ScanSamples
from cmdr4_support.utils import huffman_decode

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
        mask = hp.ud_grade(mask.astype(np.float64), 512) == 1
    return mask

def read_Planck_TOD_data(database_filename: str, my_band: Bunch, my_det: Bunch, params: Bunch, my_detector_id: int, scan_idx_start: int, scan_idx_stop: int, bad_PIDs_path:str=None) -> tuple[DetectorTOD, DetectorSamples]:
    logger = logging.getLogger(__name__)
    oids = []
    pids = []
    filenames = []
    detname = my_det.name
    with open(database_filename + f"filelist_{my_band.freq_identifier:02d}.txt") as infile:
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
    for i_pid in range(scan_idx_start, scan_idx_stop):
        pid = pids[i_pid]
        oid = oids[i_pid]
        if pid in bad_PIDs:
            continue

        filename = f"LFI_{my_band.freq_identifier:03d}_{oid.zfill(6)}.h5"
        filepath = os.path.join(database_filename, filename)
        with h5py.File(filepath, "r") as f:
            ntod = f[f"/{pid}/common/ntod"][()]
            npsi = f["/common/npsi/"][()]
            huffman_tree = f[f"/{pid}/common/hufftree"][()]
            huffman_symbols = f[f"/{pid}/common/huffsymb"][()]
            flag_encoded = f[f"/{pid}/{detname}/flag/"][()]
            pix_encoded = f[f"/{pid}/{detname}/pix/"][()]
            psi_encoded = f[f"/{pid}/{detname}/psi/"][()]
            tod = f[f"/{pid}/{detname}/tod/"][()]
            vsun = f[f"/{pid}/common/vsun/"][()]
            fsamp = f["/common/fsamp/"][()]

        flag = np.zeros(ntod, dtype=np.int64)
        flag = huffman_decode(np.frombuffer(flag_encoded, dtype=np.uint8), huffman_tree, huffman_symbols, flag)
        flag = np.cumsum(flag)
        del(flag_encoded)
        mask = (flag & 6111232) == 0
        if mask.all():
            if np.mean(np.abs(tod)) < 0.001 and np.std(tod) < 0.001:  # Check for crazy data.
                scanlist.append(ScanTOD(tod, pix_encoded, psi_encoded, 0., pid, my_band.eval_nside, my_band.data_nside,
                                        fsamp, vsun, huffman_tree, huffman_symbols, npsi, processing_mask_map))
                num_included += 1
        del(tod)
        del(mask)
        del(flag)

    logger.info(f"Fraction of scans included for {my_band.freq_identifier} {my_det.name}: "
                f"{num_included/(scan_idx_stop-scan_idx_start)*100:.1f} %")

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