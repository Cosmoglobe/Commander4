"""TOD reader for AKARI far-infrared survey data (``experiment_id: akari``)."""
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
from commander4.backend import utils as cpp_utils
from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.scan_tod import ScanTOD
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.pointing import PixelPointing
from commander4.tod.noise.psd import NoisePSDOof
from commander4.file_io.experiments.read_utils import (
    apply_noise_priors,
    find_good_fourier_size,
    read_processing_masks,
)
from commander4.diagnostics.performance import benchmark, bench_summary, start_bench,\
                                               stop_bench, log_memory, increment_count, bench_reset

logger = logging.getLogger(__name__)

def tod_reader(band_comm: MPI.Comm, my_experiment: Bunch, my_band: Bunch, 
               all_det_names: list[str],
               params: Bunch, scan_idx_start: int,
               scan_idx_stop: int) -> DetectorGroupTOD:
    """Read this rank's scans for one AKARI band from its HDF5 scan files.

    Args:
        band_comm: The band's MPI communicator.
        my_experiment, my_band: The experiment and band parameter blocks.
        all_det_names: Ordered per-band detector names; a detector's position here is its
            ``det_idx_fullband`` column in the dense per-detector sample arrays.
        params: The full parameter file.
        scan_idx_start, scan_idx_stop: This rank's slice of the band's scan list.

    Returns:
        The band's `DetectorGroupTOD`, holding only the scans and detectors this rank read.
    """
    oids = []
    pids = []
    filepaths = []
    bandname = my_band._name
    expname = my_experiment._name

    with open(my_band.filelist) as infile:
        infile.readline()
        for line in infile:
            pid, filename, _, _, _ = line.split()
            pids.append(f"{int(pid):06d}")
            filepaths.append(filename[1:-1])
            oids.append(filename.split(".")[0].split("_")[-1])

    default_mask, specific_masks = read_processing_masks(band_comm, my_band)

    if "bad_PIDs_path" in my_experiment:
        bad_PIDs = np.load(my_experiment.bad_PIDs_path)
    else:
        bad_PIDs = np.array([])


    # # Attempting to reduce fragmentation by allocating buffers.
    # ntod_upper_bound = int(my_band.fsamp*100*3600)  # 10 hour scan.
    # flag_buffer = np.zeros(ntod_upper_bound, dtype=np.int64)
    # tod_buffer = np.zeros(ntod_upper_bound, dtype=np.float32)

    scan_list = []
    nscans = scan_idx_stop - scan_idx_start
    num_included = 0
    ntod_sum_original = 0
    ntod_sum_final = 0
    ndet = len(all_det_names)
    det_init_scalars = np.zeros((ndet, 4)) + np.nan

    for i_pid in range(scan_idx_start, scan_idx_stop):
        pid = pids[i_pid]
        scanID = int(pid)
        filepath = filepaths[i_pid]
        if pid in bad_PIDs:
            continue
        good_scan = True
        with h5py.File(filepath, "r") as f:
            data_nside = int(f["common/nside"][()].item())
            ntod = int(f[f"/{pid}/common/ntod"][()].item())
            ntod_optimal = find_good_fourier_size(ntod)
            huffman_tree = f[f"/{pid}/common/hufftree"][()]
            huffman_symbols = f[f"/{pid}/common/huffsymb"][()]
            fsamp = float(f["/common/fsamp/"][()].item())
            npsi = int(f["/common/npsi/"][()].item())
            detector_list = []
            # if ntod > ntod_upper_bound:
            #     raise ValueError(f"{ntod_upper_bound} {ntod}")
            vsun = np.ones(3)  # dummy, we don't have that in Akari.
            # Akari is intensity-only: the files carry no psi, so we hand PixelPointing a zero psi of
            # the right length (psi is unused by I-only mapmaking, but PixelPointing requires one).
            psi_zeros = np.zeros(ntod_optimal, dtype=np.float32)
            detector_list = []
            for idet, det_name in enumerate(all_det_names):
                # logger.warning(f"Reading detector {det_name} for scan {pid} from file {filepath}")
                tod = f[f"/{pid}/{det_name}/tod/"][:ntod_optimal].astype(np.float32)
                pix_encoded = f[f"/{pid}/{det_name}/pix/"][()]
                flag_encoded = f[f"/{pid}/{det_name}/flag/"][()]
                init_scalars = f[f"/{pid}/{det_name}/scalars"][()]
                                # Data format has this weird thing were gain seems to be in "micro-gain"...
                init_scalars[0] *= 1e-6

                det_init_scalars[idet] = init_scalars
                det_pointing = PixelPointing(pix_encoded, psi_zeros, huffman_tree, huffman_symbols,
                                             npsi, my_band.eval_nside, data_nside, ntod,
                                             ntod_optimal)
                detector = DetectorTOD(
                    name=det_name,
                    det_idx_fullband=idet,
                    tod=tod,
                    pointing=det_pointing,
                    sampling_rate_hz=fsamp,
                    orbital_velocity_m_per_s=vsun,
                    huffman_tree=huffman_tree,
                    huffman_symbols=huffman_symbols,
                    default_proc_mask=default_mask,
                    specific_proc_masks=specific_masks,
                    flag_encoded=flag_encoded,
                    bad_data_bitmask=my_experiment.bad_data_bitmask,
                    init_scalars=init_scalars,
                )
                if (detector.tod == 0).all():
                    logger.warning(f"Detector {detector.name} has all-zero TOD for scan {pid}. Skipping.")
                    continue
                if not np.isfinite(detector.tod).all():
                    logger.warning(f"Detector {detector.name} has non-finite TOD for scan {pid}. Skipping.")
                    continue
                if detector.good_data_mask.mean() < 0.50:
                    logger.warning(f" Flag: {detector.flag} ")
                    logger.warning(f"Detector {detector.name} has less than 50% good data for scan {pid}. Skipping.")
                    continue
                detector_list.append(detector)
                ntod_sum_original += ntod
                ntod_sum_final += ntod_optimal
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

    noise_model = NoisePSDOof()
    apply_noise_priors(noise_model, params, expname, bandname)
    band_tod = DetectorGroupTOD(scan_list, expname, bandname, my_band.eval_nside, my_band.freq,
                           my_band.fwhm, fsamp, ndet, my_band.polarization, noise_model)

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
        logger.info(f"Band {bandname}: read TODs with {frac_included:.1f}% of scans included and "
                    f"{avg_scan_remaining:.1f}% of samples retained after the Fourier cut.")

    return band_tod
