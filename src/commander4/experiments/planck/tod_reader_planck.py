import logging
import numpy as np
import healpy as hp
import h5py
import gc
import time
from numpy.typing import NDArray
from astropy.io import fits
from mpi4py import MPI
from pixell.bunch import Bunch
from commander4.data_models.detector_TOD import DetectorTOD
from commander4.data_models.scan_TOD import ScanTOD
from commander4.data_models.detector_group_TOD import DetGroupTOD
from commander4.noise_sampling.noise_psd import NoisePSD, NoisePSDOof
from commander4.data_models.pointing import PixelPointing
from commander4.logging.performance_logger import benchmark, bench_summary, start_bench,\
                                            stop_bench, log_memory, increment_count, bench_reset

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


def tod_reader(band_comm: MPI.Comm, my_experiment: str, my_band: Bunch, all_det_names: list[str],
               params: Bunch, scan_idx_start: int,
               scan_idx_stop: int) -> DetGroupTOD:
    oids = []
    pids = []
    filepaths = []
    # detname = my_det._name
    bandname = my_band._name
    expname = my_experiment._name

    with open(my_band.filelist) as infile:
        infile.readline()
        for line in infile:
            pid, filename, _, _, _ = line.split()
            pids.append(f"{int(pid):06d}")
            filepaths.append(filename[1:-1])
            oids.append(filename.split(".")[0].split("_")[-1])

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

    scan_list = []
    nscans = scan_idx_stop - scan_idx_start
    num_included = 0
    ntod_sum_original = 0
    ntod_sum_final = 0
    ndet = len(all_det_names)
    det_init_scalars = np.zeros((ndet, 4)) + np.nan
    # Small de-sycnronization sleep.
    # time.sleep(5.0 * (band_comm.Get_rank() / band_comm.Get_size()))
    for i_pid in range(scan_idx_start, scan_idx_stop):
        # Evenly distributed small sleep-interupts, distributed across scans, totalling 60 seconds.
        # time.sleep(60.0 / (nscans * band_comm.Get_size()))
        pid = pids[i_pid]
        scanID = int(pid)
        filepath = filepaths[i_pid]
        if pid in bad_PIDs:
            continue
        good_scan = True
        with h5py.File(filepath, "r") as f:
            data_nside = int(f["common/nside"][()].item())
            ntod = int(f[f"/{pid}/common/ntod"][()].item())
            ntod_optimal = find_good_Fourier_time(Fourier_times, ntod)
            huffman_tree = f[f"/{pid}/common/hufftree"][()]
            huffman_symbols = f[f"/{pid}/common/huffsymb"][()]
            vsun = f[f"/{pid}/common/vsun/"][()]
            fsamp = float(f["/common/fsamp/"][()].item())
            npsi = int(f["/common/npsi/"][()].item())
            detector_list = []
            # idet is the detector's full-band column (its position in ``all_det_names``);
            # idet_accepted (det_idx_local) advances only when a detector survives the cuts below.
            idet_accepted = 0
            for idet, det_name in enumerate(all_det_names):
                tod = f[f"/{pid}/{det_name}/tod/"][:ntod_optimal].astype(np.float32, copy=False)
                pix_encoded = f[f"/{pid}/{det_name}/pix/"][()]
                # Intensity-only bands have no psi in the files; feed a zero psi (unused by I-only
                # mapmaking, but PixelPointing requires a length-matched array).
                if "QU" in my_band.polarization:
                    psi_encoded = f[f"/{pid}/{det_name}/psi/"][()]
                else:
                    psi_encoded = np.zeros(ntod_optimal, dtype=np.float32)
                flag_encoded = f[f"/{pid}/{det_name}/flag/"][()]
                init_scalars = f[f"/{pid}/{det_name}/scalars/"][()]
                # Data format has this weird thing were gain seems to be in "micro-gain"...
                init_scalars[0] *= 1e-6

                det_init_scalars[idet] = init_scalars
                det_pointing = PixelPointing(pix_encoded, psi_encoded, huffman_tree,
                                             huffman_symbols, npsi, my_band.eval_nside, data_nside,
                                             ntod, ntod_optimal)

                detector = DetectorTOD(det_name, idet, idet_accepted, tod, det_pointing, fsamp,
                                       vsun, huffman_tree, huffman_symbols, processing_mask_map,
                                       ntod, ntod_optimal,
                                       flag_encoded=flag_encoded,
                                       bad_data_bitmask = 6111232,
                                       init_scalars = init_scalars)
                if (detector.tod == 0).all():
                    continue
                if np.mean(np.abs(detector.tod)) > 0.001 or np.std(detector.tod) > 0.001:
                    continue
                if not np.isfinite(detector.tod).all():
                    continue
                if detector.good_data_mask.mean() < 0.75:
                    continue
                detector_list.append(detector)
                ntod_sum_original += ntod
                ntod_sum_final += ntod_optimal
                idet_accepted += 1
        if len(detector_list) == 0:
            good_scan = False
        if good_scan:
            scan = ScanTOD(detector_list, 0., scanID)
            scan_list.append(scan)
            num_included += 1
        if band_comm.Get_rank() == 0 and (i_pid-scan_idx_start) % (nscans // 5) == 0:
            logger.debug(f"Reading scans from disk, progress on master rank of band {bandname}: "\
                         f"{i_pid-scan_idx_start}/{nscans}")
        if i_pid % 10 == 0:
            gc.collect()

    # Initialize noise model with defaults and uniform priors suited for LFI.
    noise_model = NoisePSDOof(P_active_mean = [np.nan, 0.1, -1.0],
                              P_active_rms = [np.nan, np.inf, np.inf],
                              P_uni = [[np.nan, np.nan], [0.01, 0.5], [-2.5, -0.25]],
                              nu_fit = [[np.nan, np.nan], [0, 3.0], [0, 3.0]])

    band_tod = DetGroupTOD(scan_list, expname, bandname, my_band.eval_nside, my_band.freq,
                           my_band.fwhm, fsamp, ndet, my_band.polarization, noise_model)
    # my_det_central_freq = my_band.freq

    # TODO: Re-implement bandpass shift.
    # if "bandpass_shift" in my_det:
    #     my_det_central_freq += my_det.bandpass_shift
    # det_static = DetectorTOD(scanlist, my_det_central_freq, my_band.fwhm, my_band.eval_nside,
    #                          data_nside, expname, bandname, detname)
    # det_static.detector_id = my_det_id

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