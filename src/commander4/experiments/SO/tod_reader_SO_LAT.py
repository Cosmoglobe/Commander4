import logging
import numpy as np
import healpy as hp
from astropy.io import fits
import h5py
import gc
from numpy.typing import NDArray
from pixell.bunch import Bunch
from mpi4py import MPI

from commander4.cmdr4_support import utils as cpp_utils
from commander4.data_models.detector_TOD import DetectorTOD
from commander4.data_models.detector_group_TOD import DetGroupTOD
from commander4.data_models.scan_TOD import ScanTOD
from commander4.simulations.inplace_litebird_sim import replace_tod_with_sim
from commander4.output.log import logassert
from commander4.noise_sampling.noise_psd import NoisePSD, NoisePSDOof
from commander4.data_models.pointing import PixelPointing
from commander4.experiments.tod_read_utils import read_processing_masks, find_good_Fourier_time
from commander4.logging.performance_logger import benchmark, bench_summary, start_bench,\
                                            stop_bench, log_memory, increment_count, bench_reset
logger = logging.getLogger(__name__)


def find_good_Fourier_time(Fourier_times:NDArray, ntod:int) -> int:
    if ntod <= 10_000 or ntod >= 400_000:
        return ntod
    search_start = int(0.99*ntod)  # Consider sizes up to 1% smaller than ntod.
    best_ntod = np.argmin(Fourier_times[search_start:ntod+1])
    best_ntod += search_start
    assert(best_ntod <= ntod)
    return best_ntod


def tod_reader(band_comm: MPI.Comm, my_experiment: str, my_band: Bunch, all_det_names: list[str],
               params: Bunch, scan_idx_start: int,
               scan_idx_stop: int) -> DetGroupTOD:
    start_bench("reader-startup")
    oids = []
    pids = []
    filepaths = []
    bandname = my_band._name
    expname = my_experiment._name

    with open(my_band.filelist) as infile:
        infile.readline()
        for line in infile:
            pid, filepath, _, _, _ = line.split()
            pids.append(f"{int(pid):06d}")
            filepaths.append(filepath[1:-1])
            oids.append(filepath.split(".")[0].split("_")[-1])
    default_mask, specific_masks = read_processing_masks(band_comm, my_band)

    if "bad_PIDs_path" in my_experiment:
        bad_PIDs = np.load(my_experiment.bad_PIDs_path)
    else:
        bad_PIDs = np.array([])

    Fourier_times = np.load(my_experiment.Fourier_times_path)

    # Attempting to reduce fragmentation by allocating buffers.
    ntod_upper_bound = int(100*100*3600)  # 10 hour scan.
    ndet = len(all_det_names)
    
    ntod_sum_original = 0
    ntod_sum_final = 0
    scan_list = []
    included_detector_scans = 0
    det_init_scalars = np.zeros((ndet, 4)) + np.nan
    stop_bench("reader-startup")
    for i_pid in range(scan_idx_start, scan_idx_stop+1):
        pid = pids[i_pid]
        filepath = filepaths[i_pid]
        if pid in bad_PIDs:
            continue
        start_bench("fileread")
        good_scan = True
        with h5py.File(filepath, "r") as f:
            data_nside = int(f["common/nside"][()].item())
            ntod = int(f[f"/{pid}/common/ntod"][()].item())
            ntod_optimal = find_good_Fourier_time(Fourier_times, ntod)
            huffman_tree = f[f"/{pid}/common/hufftree"][()]
            huffman_symbols = f[f"/{pid}/common/huffsymb"][()]
            # Second Huffman set might not exist: .get() returns None if dataset is absent.
            if f"/{pid}/common/hufftree2" in f and f"/{pid}/common/huffsymb2" in f:
                huffman_tree2 = f[f"/{pid}/common/hufftree2"][()]
                huffman_symbols2 = f[f"/{pid}/common/huffsymb2"][()]
            else:
                huffman_tree2 = None
                huffman_symbols2 = None
            vsun = f[f"/{pid}/common/vsun/"][()]
            fsamp = float(f["/common/fsamp/"][()].item())
            npsi = int(f["/common/npsi/"][()].item())

            if ntod > ntod_upper_bound:
                raise ValueError(f"{ntod_upper_bound} {ntod}")

            detector_list = []
            detector_names = []
            idet_accepted = 0
            for idet, det_name in enumerate(all_det_names):
                if my_experiment.tod_is_compressed:
                    tod = f[f"/{pid}/{det_name}/ztod/"][()]
                else:
                    tod = f[f"/{pid}/{det_name}/tod/"][:ntod_optimal].astype(np.float32)
                pix_encoded = f[f"/{pid}/{det_name}/pix/"][()]
                psi_encoded = f[f"/{pid}/{det_name}/psi/"][()]
                flag_encoded = f[f"/{pid}/{det_name}/flag/"][()]
                # gain_init, sigma0_init, fknee_init, alpha_init:
                init_scalars = f[f"/{pid}/{det_name}/scalars"][()]
                if init_scalars[0] > 100.0:
                    continue

                det_init_scalars[idet] = init_scalars
                det_pointing = PixelPointing(pix_encoded, psi_encoded, huffman_tree,
                                             huffman_symbols, npsi, my_band.eval_nside, data_nside,
                                             ntod, ntod_optimal)

                detector = DetectorTOD(det_name, idet, idet_accepted, tod, det_pointing, fsamp, vsun, huffman_tree,
                                       huffman_symbols, default_mask, specific_masks, ntod,
                                       ntod_optimal,
                                       huffman_tree2=huffman_tree2,
                                       huffman_symbols2=huffman_symbols2,
                                       flag_encoded=flag_encoded,
                                       bad_data_bitmask=my_experiment.bad_data_bitmask,
                                       init_scalars=init_scalars,
                                       tod_is_compressed=my_experiment.tod_is_compressed)
                unmasked_fraction = np.sum(detector.good_data_mask)/detector.good_data_mask.size
                if unmasked_fraction < 0.9:
                    continue
                if(detector.tod == 0).all():
                    continue
                
                detector_list.append(detector)
                detector_names.append(det_name)
                ntod_sum_original += ntod
                ntod_sum_final += ntod_optimal
                idet_accepted += 1

            included_detector_scans += idet_accepted
        stop_bench("fileread")
        if len(detector_list) == 0:
            good_scan = False
        if good_scan:
            scanID = int(pid)
            scan = ScanTOD(detector_list, 0., scanID)
            scan_list.append(scan)
        if i_pid % 10 == 0:
            gc.collect()

    noise_model = NoisePSDOof(P_active_mean = [np.nan, 10.0, -2.7],
                              P_active_rms = [np.nan, np.inf, np.inf],
                              P_uni = [[np.nan, np.nan], [0.03, 30.0], [-4.0, -1.5]],
                              nu_fit = [[np.nan, np.nan], [0, 10.0], [0, 10.0]])
    band_tod = DetGroupTOD(scan_list, expname, bandname, my_band.eval_nside, my_band.freq,
                           my_band.fwhm, fsamp, ndet, my_band.polarization, noise_model)

    ### Summarize detector-scan inclusion and Fourier-cut retention ###
    local_tot_scans = (scan_idx_stop + 1) - scan_idx_start
    local_tot_detector_scans = ndet * local_tot_scans
    local_stats = np.array([
        included_detector_scans,
        local_tot_detector_scans,
        ntod_sum_final,
        ntod_sum_original,
    ], dtype=np.int64)
    global_stats = np.zeros_like(local_stats)
    band_comm.Reduce(local_stats, global_stats, op=MPI.SUM, root=0)
    if band_comm.Get_rank() == 0:
        total_included_detector_scans, total_detector_scans, total_ntod_final, total_ntod_original = global_stats
        frac_included = 0.0
        if total_detector_scans > 0:
            frac_included = total_included_detector_scans / total_detector_scans * 100.0
        avg_scan_remaining = 0.0
        if total_ntod_original > 0:
            avg_scan_remaining = total_ntod_final / total_ntod_original * 100.0
        logger.info(f"Band {bandname} finished reading TODs from file.")
        logger.info(f"Fraction of detector-scans included for {bandname}: {frac_included:.1f} %")
        logger.info(f"Fraction of TODs left after Fourier cut for {bandname}: "\
                    f"{avg_scan_remaining:.1f} %")

    return band_tod