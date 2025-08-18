import logging
import numpy as np
import healpy as hp
from pixell.bunch import Bunch
from src.python.data_models.detector_TOD import DetectorTOD
from src.python.data_models.scan_TOD import ScanTOD
from src.python.utils.commander_tod import commander_tod
from src.python.data_models.detector_samples import DetectorSamples
from src.python.data_models.scan_samples import ScanSamples
from astropy.io import fits

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
    if nside != my_band.nside:
        mask = hp.ud_grade(mask.astype(np.float64), 512) == 1
    return mask


def read_Planck_TOD_data(database_filename: str, my_band: Bunch, my_det: Bunch, params: Bunch, my_detector_id: int, scan_idx_start: int, scan_idx_stop: int) -> tuple[DetectorTOD, DetectorSamples]:
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
    com_tod = commander_tod(database_filename, "LFI")
    scanlist = []
    num_included = 0
    # if my_band.freq_identifier == 30:
    #     detname = "27"
    #     local_nside = 512
    # elif my_band.freq_identifier == 44:
    #     detname = "24"
    #     local_nside = 512
    # elif my_band.freq_identifier == 70:
    #     detname = "18"
    #     local_nside = 1024
    if my_band.freq_identifier == 70:
        local_nside = 1024
    else:
        local_nside = 512
        
    if my_band.freq_identifier == 70 and scan_idx_stop >= 45850:
        print("HELLO", detname)
        from tqdm import trange
        myrange = trange
    else:
        myrange = range
    print(my_band.freq, detname, scan_idx_start, scan_idx_stop)

    bad_PIDs = np.load("/mn/stornext/d23/cmbco/jonas/c4_testing/Commander4/badPIDs.npy")
    previous_oid = -999999
    for i_pid in myrange(scan_idx_start, scan_idx_stop):
        pid = pids[i_pid]
        oid = oids[i_pid]
        if pid in bad_PIDs:
            continue
        if oid != previous_oid:  # Only open file if it's not the same file as the last PID.
            com_tod.init_file(f"{my_band.freq_identifier:03d}", oid)
        flag = com_tod.decompress(f"/{pid}/{detname}/flag/", compression="huffman")
        npsi = com_tod.decompress("/common/npsi/")[()]
        pix = com_tod.decompress(f"/{pid}/{detname}/pix/", compression="huffman").astype(np.uint32)
        psi = com_tod.decompress(f"/{pid}/{detname}/psi/", compression="huffman").astype(np.float32)
        psi = 2*np.pi * psi/npsi  # Convert
        tod = com_tod.decompress(f"/{pid}/{detname}/tod/")[()].astype(np.float32)
        # tod_S = com_tod.decompress(f"/{pid}/{detname}S/tod/")[()].astype(np.float32)
        # This is the orbital velocity relative to the sun, in galactic coordinates:
        vsun = com_tod.decompress(f"/{pid}/common/vsun/")[()]
        fsamp = com_tod.decompress("/common/fsamp/")[()]
        if local_nside != my_band.nside:
            pix = hp.ang2pix(my_band.nside, *hp.pix2ang(local_nside, pix)).astype(np.uint32)
        mask = ((flag) & 6111232) == 0
        if mask.all():
            if np.mean(np.abs(tod)) < 0.001 and np.std(tod) < 0.001:  # Check for crazy data.
                scanlist.append(ScanTOD(tod, pix, psi, 0., pid, my_band.nside, fsamp, vsun))
                num_included += 1
    logger.info(f"Fraction of scans included for {my_band.freq_identifier}: "
                f"{num_included/(scan_idx_stop-scan_idx_start)*100:.1f} %")

    processing_mask_map = get_processing_mask(my_band)
    det_static = DetectorTOD(scanlist, float(my_band.freq), my_band.fwhm, my_band.nside, processing_mask_map)
    det_static.detector_id = my_detector_id

    scansample_list = []
    for iscan in range(num_included):
        scansample_list.append(ScanSamples())
        scansample_list[-1].time_dep_rel_gain_est = 0.0
        scansample_list[-1].rel_gain_est = my_band.rel_gain_est
        scansample_list[-1].gain_est = my_band.rel_gain_est + params.initial_g0
    det_samples = DetectorSamples(scansample_list)
    det_samples.detector_id = my_detector_id
    det_samples.g0_est = params.initial_g0
    det_samples.detname = detname

    return det_static, det_samples