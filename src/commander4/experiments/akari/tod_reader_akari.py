import logging
import numpy as np
import healpy as hp
import os
import h5py
import gc
from pixell.bunch import Bunch
from numpy.typing import NDArray
from astropy.io import fits
from mpi4py import MPI
from commander4.cmdr4_support import utils as cpp_utils
from commander4.data_models.detector_TOD import DetectorTOD
from commander4.data_models.scan_TOD import ScanTOD
from commander4.data_models.detector_group_TOD import DetGroupTOD


def get_processing_mask(my_band: Bunch) -> DetectorTOD:
    """ Finds and returns the processing mask for the relevant band.
    """
    hdul = fits.open(my_band.processing_mask)
    mask = hdul[1].data['I_Stokes'].flatten().astype(bool)
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


def tod_reader(band_comm: MPI.Comm, my_experiment: str, my_band: Bunch, det_names: list[str],
               params: Bunch, scan_idx_start: int,
               scan_idx_stop: int) -> DetGroupTOD:
    logger = logging.getLogger(__name__)
    oids = []
    pids = []
    filenames = []
    bandname = my_band._name
    expname = my_experiment._name

    with open(my_band.filelist) as infile:
        infile.readline()
        for line in infile:
            pid, filename, _, _, _ = line.split()
            pids.append(f"{int(pid):06d}")
            filenames.append(filename[1:-1])
            oids.append(filename.split(".")[0].split("_")[-1])

    processing_mask_map = get_processing_mask(my_band)
    if "bad_PIDs_path" in my_experiment:
        bad_PIDs = np.load(my_experiment.bad_PIDs_path)
    else:
        bad_PIDs = np.array([])

    Fourier_times = np.load(my_experiment.Fourier_times_path)

    # Attempting to reduce fragmentation by allocating buffers.
    ntod_upper_bound = int(my_band.fsamp*100*3600)  # 10 hour scan.
    flag_buffer = np.zeros(ntod_upper_bound, dtype=np.int64)
    tod_buffer = np.zeros(ntod_upper_bound, dtype=np.float32)

    scan_list = []
    num_included = 0
    ntod_sum_original = 0
    ntod_sum_final = 0
    for i_pid in range(scan_idx_start, scan_idx_stop):
        pid = pids[i_pid]
        if pid in bad_PIDs:
            continue

        filepath = filenames[i_pid]
        with h5py.File(filepath, "r") as f:
            ntod = int(f[f"/{pid}/common/ntod"][()].item())
            ntod_optimal = find_good_Fourier_time(Fourier_times, ntod)
            huffman_tree = f[f"/{pid}/common/hufftree"][()]
            huffman_symbols = f[f"/{pid}/common/huffsymb"][()]
            fsamp = float(f["/common/fsamp/"][()].item())
            npsi = int(f["/common/npsi/"][()].item())
            if ntod > ntod_upper_bound:
                raise ValueError(f"{ntod_upper_bound} {ntod}")
            vsun = np.ones(3)  # dummy, we don't have that in Akari.
            detector_list = []
            for det_name in det_names:
                tod = f[f"/{pid}/{det_name}/tod/"][:ntod_optimal].astype(np.float32)
                pix_encoded = f[f"/{pid}/{det_name}/pix/"][()]
                flag_encoded = f[f"/{pid}/{det_name}/flag/"][()]
                detector = DetectorTOD(tod, pix_encoded, [], my_band.eval_nside,
                                       my_band.data_nside, fsamp, vsun, huffman_tree,
                                       huffman_symbols, npsi, processing_mask_map, ntod,
                                       flag_encoded=flag_encoded,
                                       flag_bitmask=my_experiment.flag_bitmaks,
                                       pix_is_compressed=my_experiment.pix_is_compressed,
                                       psi_is_compressed=False)
                detector_list.append(detector)
                ntod_sum_original += ntod
                ntod_sum_final += ntod_optimal
        scanID = int(pid)
        scan = ScanTOD(detector_list, 0., scanID, scan_idx_start, scan_idx_stop)
        scan_list.append(scan)
        num_included += 1
        if i_pid % 10 == 0:
            gc.collect()

    ndet = len(det_names)
    band_tod = DetGroupTOD(scan_list, expname, bandname, my_band.eval_nside, my_band.freq,
                           my_band.fwhm, ndet, my_band.polarizations)

    ### Collect some info on master rank of each band and print it ###
    local_tot_scans = scan_idx_stop - scan_idx_start
    local_stats = np.array([num_included, local_tot_scans, ntod_sum_final, ntod_sum_original])
    global_stats = np.zeros_like(local_stats)
    band_comm.Reduce(local_stats, global_stats, op=MPI.SUM, root=0)
    if band_comm.Get_rank() == 0:
        total_included, total_scans, total_ntod_final, total_ntod_original = global_stats
        frac_included = 0.0
        if total_scans > 0:
            frac_included = total_included / total_scans * 100.0
        avg_scan_remaining = 0.0
        if total_ntod_original > 0:
            avg_scan_remaining = total_ntod_final / total_ntod_original * 100.0
        logger.info(f"Band {bandname} finished reading TODs from file.")
        logger.info(f"Fraction of scans included for {bandname}: {frac_included:.1f} %")
        logger.info(f"Fraction of TODs left after Fourier cut for {bandname}: "
                    f"{avg_scan_remaining:.1f} %")

    return band_tod