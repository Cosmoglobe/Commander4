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
from commander4.logging.performance_logger import benchmark, bench_summary, start_bench,\
                                            stop_bench, log_memory, increment_count, bench_reset
from commander4.data_models.pointing import DetectorBoresightPointing, ScanBoresightPointing
logger = logging.getLogger(__name__)

def get_processing_mask(my_band: Bunch) -> DetectorTOD:
    """ Finds and returns the processing mask for the relevant band.
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


def tod_reader(band_comm: MPI.Comm, my_experiment: str, my_band: Bunch, det_names: list[str],
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
    if "processing_mask" in my_band:
        processing_mask_map = np.ones(12*my_band.eval_nside**2, dtype=bool)
        if band_comm.Get_rank() == 0:
            processing_mask_map[:] = get_processing_mask(my_band)        
        band_comm.Bcast(processing_mask_map, root=0)
    else:
        processing_mask_map = np.ones(12*my_band.eval_nside**2, dtype=bool)

    if "bad_PIDs_path" in my_experiment:
        bad_PIDs = np.load(my_experiment.bad_PIDs_path)
    else:
        bad_PIDs = np.array([])

    Fourier_times = np.load(my_experiment.Fourier_times_path)

    # Attempting to reduce fragmentation by allocating buffers.
    ntod_upper_bound = int(100*100*3600)  # 10 hour scan.
    
    ntod_sum_original = 0
    ntod_sum_final = 0
    scan_list = []
    num_included = 0
    stop_bench("reader-startup")
    for i_pid in range(scan_idx_start, scan_idx_stop+1):
        pid = pids[i_pid]
        filepath = filepaths[i_pid]
        start_bench("fileread")
        if pid in bad_PIDs:
            continue
        good_scan = True
        with h5py.File(filepath, "r") as f:
            ntod = int(f[f"/{pid}/common/ntod"][()].item())
            ntod_optimal = find_good_Fourier_time(Fourier_times, ntod)
            huffman_tree = f[f"/{pid}/common/hufftree"][()]
            huffman_symbols = f[f"/{pid}/common/huffsymb"][()]
            # Second Huffman set might not exist.
            if f"/{pid}/common/hufftree2" in f and f"/{pid}/common/huffsymb2" in f:
                huffman_tree2 = f[f"/{pid}/common/hufftree2"][()]
                huffman_symbols2 = f[f"/{pid}/common/huffsymb2"][()]
            else:
                huffman_tree2 = None
                huffman_symbols2 = None
            fsamp = float(f["/common/fsamp/"][()].item())
            det_responses = f["/common/resp/"][()]

            # The detector names are stored as a single "Bytes-like" string, formatted like a
            # Python list. We extract the string from the Bytes, and then re-create the list with .split(",").
            det_names_file = f["/common/det"].asstr()[()].split(",")
            det_names_file = [det.strip() for det in det_names_file]

            processing_mask_nside = hp.npix2nside(processing_mask_map.size)
            logassert(my_band.eval_nside == processing_mask_nside,
                      f"Processing mask (band {bandname}) "
                      f"has nside {processing_mask_nside} while eval_nside = {my_band.eval_nside} "
                      "(NB: eval_nside can be set different from native data nside)", logger)

            if ntod > ntod_upper_bound:
                raise ValueError(f"{ntod_upper_bound} {ntod}")

            all_detector_offsets = f["/common/detoff/"][()]
            all_polarization_angles = f["/common/polang/"][()]
            site_location = f["/common/site/"][()]
            boresight = f[f"/{pid}/common/bore/"][()]
            time_start_mjd = f[f"/{pid}/common/time/"][0]
            time_end_mjd = f[f"/{pid}/common/time_end/"][0]

            scan_pointing = ScanBoresightPointing(time_start_mjd, time_end_mjd, ntod, site_location,
                                            boresight, all_detector_offsets, all_polarization_angles,
                                            my_band.eval_nside, ntod_optimal)

            detector_list = []
            for idet, det_name in enumerate(det_names):
                # Find the index of the current detector in the file order of detectors.
                det_file_idx = det_names_file.index(det_name)

                if my_experiment.tod_is_compressed:
                    tod = f[f"/{pid}/{det_name}/ztod/"][()]
                else:
                    tod = f[f"/{pid}/{det_name}/tod/"][:ntod_optimal].astype(np.float32)

                pointing = DetectorBoresightPointing(scan_pointing, det_file_idx)
                det_response = det_responses[det_file_idx]

                flag_encoded = f[f"/{pid}/{det_name}/flag/"][()]
                gain_init, sigma0_init, fknee_init, alpha_init = f[f"/{pid}/{det_name}/scalars"][()]

                detector = DetectorTOD(tod, pointing, fsamp, np.zeros(3), huffman_tree, huffman_symbols,
                                       processing_mask_map, ntod, ntod_optimal,
                                       huffman_tree2=huffman_tree2,
                                       huffman_symbols2=huffman_symbols2,
                                       flag_encoded=flag_encoded,
                                       flag_bitmask=my_experiment.flag_bitmask,
                                       tod_is_compressed=my_experiment.tod_is_compressed,
                                       det_response=det_response)
                if np.sum(detector.full_mask) == 0 or (detector.tod == 0).all():
                    continue
                detector_list.append(detector)
                ntod_sum_original += ntod
                ntod_sum_final += ntod_optimal

        if good_scan:
            scanID = int(pid)
            scan = ScanTOD(detector_list, 0., scanID)
            scan_list.append(scan)
            num_included += 1
        if i_pid % 10 == 0:
            gc.collect()

    ndet = len(det_names)
    noise_model = NoisePSDOof()
    band_tod = DetGroupTOD(scan_list, expname, bandname, my_band.eval_nside, my_band.freq,
                           my_band.fwhm, fsamp, ndet, my_band.polarization, noise_model)

    ### Collect some info on master rank of each detector and print it ###
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
        logger.info(f"Fraction of TODs left after Fourier cut for {bandname}: "\
                    f"{avg_scan_remaining:.1f} %")

    return band_tod