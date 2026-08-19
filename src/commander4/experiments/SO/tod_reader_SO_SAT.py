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
from commander4.experiments.tod_read_utils import read_processing_masks, find_good_Fourier_time
logger = logging.getLogger(__name__)


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
    default_mask, specific_masks = read_processing_masks(band_comm, my_band)

    if "bad_PIDs_path" in my_experiment:
        bad_PIDs = np.load(my_experiment.bad_PIDs_path)
    else:
        bad_PIDs = np.array([])

    Fourier_times = np.load(my_experiment.fourier_times_path)

    # Attempting to reduce fragmentation by allocating buffers.
    ntod_upper_bound = int(100*100*3600)  # 10 hour scan.

    # Drop detector-scans that are mostly flagged, before they reach the samplers. A detector-scan
    # with too few *adjacent* unflagged samples has no measurable white-noise level: the sigma0
    # estimator returns inf for it, which makes the correlated-noise CG divide by zero and yields a
    # NaN n_corr, and since the gain solve reduces s^T N^-1 s across the whole band a few of those
    # turn the band-wide sum into NaN. The default 0.0 keeps every detector-scan that has at least
    # one good sample (the historical behaviour); the SO LAT reader uses the equivalent of 0.9.
    min_unmasked_fraction = float(getattr(my_experiment, "min_unmasked_fraction", 0.0))
    ndet = len(det_names)

    ntod_sum_original = 0
    ntod_sum_final = 0
    scan_list = []
    included_detector_scans = 0
    stop_bench("reader-startup")
    for i_pid in range(scan_idx_start, scan_idx_stop+1):
        pid = pids[i_pid]
        filepath = filepaths[i_pid]
        if pid in bad_PIDs:
            continue
        start_bench("fileread")
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
            # idet is the detector's full-band column (its position in ``det_names``); idet_accepted
            # is its position among the detectors actually kept in this scan (det_idx_local), and is
            # only advanced when a detector passes the cuts below.
            idet_accepted = 0
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
                # gain_init, sigma0_init, fknee_init, alpha_init:
                init_scalars = f[f"/{pid}/{det_name}/scalars"][()]

                detector = DetectorTOD(det_name, idet, idet_accepted, tod, pointing, fsamp,
                                       np.zeros(3), huffman_tree, huffman_symbols,
                                       default_mask, specific_masks, ntod, ntod_optimal,
                                       huffman_tree2=huffman_tree2,
                                       huffman_symbols2=huffman_symbols2,
                                       flag_encoded=flag_encoded,
                                       bad_data_bitmask=my_experiment.bad_data_bitmask,
                                       init_scalars=init_scalars,
                                       tod_is_compressed=my_experiment.tod_is_compressed,
                                       det_response=det_response)
                # `<=` so the 0.0 default still drops fully-flagged detector-scans.
                if np.mean(detector.good_data_mask) <= min_unmasked_fraction:
                    continue
                if (detector.tod == 0).all():
                    continue
                detector_list.append(detector)
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

    noise_model = NoisePSDOof()
    band_tod = DetGroupTOD(scan_list, expname, bandname, my_band.eval_nside, my_band.freq,
                           my_band.fwhm, fsamp, ndet, my_band.polarization, noise_model)

    ### Summarize detector-scan inclusion and Fourier-cut retention ###
    # The scan loop is inclusive of scan_idx_stop, so the local count is stop+1-start; and the
    # fraction is reported per detector-scan rather than per scan, since a scan survives as long as
    # any one of its detectors does and would otherwise hide how many detectors were dropped.
    local_tot_scans = (scan_idx_stop + 1) - scan_idx_start
    local_tot_detector_scans = ndet * local_tot_scans
    local_stats = np.array([included_detector_scans, local_tot_detector_scans,
                            ntod_sum_final, ntod_sum_original], dtype=np.int64)
    global_stats = np.zeros_like(local_stats)
    band_comm.Reduce(local_stats, global_stats, op=MPI.SUM, root=0)
    if band_comm.Get_rank() == 0:
        total_included, total_detector_scans, total_ntod_final, total_ntod_original = global_stats
        frac_included = 0.0
        if total_detector_scans > 0:
            frac_included = total_included / total_detector_scans * 100.0
        avg_scan_remaining = 0.0
        if total_ntod_original > 0:
            avg_scan_remaining = total_ntod_final / total_ntod_original * 100.0
        logger.info(f"Band {bandname} finished reading TODs from file.")
        logger.info(f"Fraction of detector-scans included for {bandname}: {frac_included:.1f} % "
                    f"({total_included}/{total_detector_scans})")
        logger.info(f"Fraction of TODs left after Fourier cut for {bandname}: "\
                    f"{avg_scan_remaining:.1f} %")

    return band_tod