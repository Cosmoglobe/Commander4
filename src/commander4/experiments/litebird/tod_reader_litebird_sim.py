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
from commander4.data_models.scan_TOD import ScanTOD
from commander4.data_models.detector_group_TOD import DetGroupTOD
from commander4.noise_sampling.noise_psd import NoisePSD, NoisePSDOof
from commander4.simulations.inplace_litebird_sim import replace_tod_with_sim
from commander4.output.log import logassert

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
        processing_mask_map = get_processing_mask(my_band)
    else:
        processing_mask_map = np.ones(12*my_band.eval_nside**2, dtype=bool)

    if "bad_PIDs_path" in my_experiment:
        bad_PIDs = np.load(my_experiment.bad_PIDs_path)
    else:
        bad_PIDs = np.array([])

    Fourier_times = np.load(my_experiment.Fourier_times_path)

    # Attempting to reduce fragmentation by allocating buffers.
    ntod_upper_bound = int(my_band.fsamp*100*3600)  # 10 hour scan.
    flag_buffer = np.zeros(ntod_upper_bound, dtype=np.int64)

    ntod_sum_original = 0
    ntod_sum_final = 0
    scan_list = []
    num_included = 0
    for i_pid in range(scan_idx_start, scan_idx_stop):
        pid = pids[i_pid]
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

            processing_mask_nside = hp.npix2nside(processing_mask_map.size)
            logassert(my_band.eval_nside == processing_mask_nside,
                      f"Processing mask (band {bandname}) "
                      f"has nside {processing_mask_nside} while eval_nside = {my_band.eval_nside} "
                      "(NB: eval_nside can be set different from native data nside)", logger)

            if ntod > ntod_upper_bound:
                raise ValueError(f"{ntod_upper_bound} {ntod}")

            detector_list = []
            for det_name in det_names:
                tod = f[f"/{pid}/{det_name}/tod/"][:ntod_optimal].astype(np.float32, copy=False)
                pix_encoded = f[f"/{pid}/{det_name}/pix/"][()]
                psi_encoded = f[f"/{pid}/{det_name}/psi/"][()]
                flag_encoded = f[f"/{pid}/{det_name}/flag/"][()]

                # Some simulations have a (1,N) shape for pixels; remove leading dimension.
                if pix_encoded.ndim == 2 and pix_encoded.shape[0] == 1:
                    pix_encoded = pix_encoded[0]
                if psi_encoded.ndim == 2 and psi_encoded.shape[0] == 1:
                    psi_encoded = psi_encoded[0]

                flag_buffer[:ntod] = 0.0
                flag_buffer[:ntod] = cpp_utils.huffman_decode(
                    np.frombuffer(flag_encoded, dtype=np.uint8),
                    huffman_tree, huffman_symbols, flag_buffer[:ntod])
                flag_buffer[:ntod_optimal] = np.cumsum(flag_buffer[:ntod_optimal])
                flag_buffer[:ntod_optimal] &= 6111232
                if np.sum(flag_buffer[:ntod_optimal]) != 0:
                    good_scan = False

                detector = DetectorTOD(tod, pix_encoded, psi_encoded, my_band.eval_nside,
                                       data_nside, fsamp, vsun, huffman_tree, huffman_symbols,
                                       npsi, processing_mask_map, ntod,
                                       pix_is_compressed=my_experiment.pix_is_compressed,
                                       psi_is_compressed=my_experiment.psi_is_compressed)
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

    if my_experiment.replace_tod_with_sim:
        replace_tod_with_sim(band_comm, band_tod, my_band, params, my_experiment.sim_params)

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