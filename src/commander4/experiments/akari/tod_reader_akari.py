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
from commander4.data_models.pointing import PixelPointing
from commander4.noise_sampling.noise_psd import NoisePSDOof
from commander4.experiments.tod_read_utils import read_processing_masks, find_good_Fourier_time

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

    default_mask, specific_masks = read_processing_masks(band_comm, my_band)
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
            # Akari is intensity-only: the files carry no psi, so we hand PixelPointing a zero psi of
            # the right length (psi is unused by I-only mapmaking, but PixelPointing requires one).
            psi_zeros = np.zeros(ntod_optimal, dtype=np.float32)
            detector_list = []
            # All detectors are kept, so the full-band column idet and the per-scan column
            # idet_accepted advance together here.
            idet_accepted = 0
            for idet, det_name in enumerate(det_names):
                tod = f[f"/{pid}/{det_name}/tod/"][:ntod_optimal].astype(np.float32)
                pix_encoded = f[f"/{pid}/{det_name}/pix/"][()]
                flag_encoded = f[f"/{pid}/{det_name}/flag/"][()]
                # gain_init, sigma0_init, fknee_init, alpha_init:
                init_scalars = f[f"/{pid}/{det_name}/scalars"][()]
                det_pointing = PixelPointing(pix_encoded, psi_zeros, huffman_tree, huffman_symbols,
                                             npsi, my_band.eval_nside, my_band.data_nside, ntod,
                                             ntod_optimal)
                detector = DetectorTOD(det_name, idet, idet_accepted, tod, det_pointing, fsamp,
                                       vsun, huffman_tree, huffman_symbols, default_mask,
                                       specific_masks, ntod, ntod_optimal,
                                       flag_encoded=flag_encoded,
                                       bad_data_bitmask=my_experiment.bad_data_bitmask,
                                       init_scalars=init_scalars)
                if (detector.tod == 0).all():
                    continue
                if not np.isfinite(detector.tod).all():
                    continue
                if detector.good_data_mask.mean() < 0.5:
                    continue
                detector_list.append(detector)
                ntod_sum_original += ntod
                ntod_sum_final += ntod_optimal
                idet_accepted += 1
        if len(detector_list) > 0:
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