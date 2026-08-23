"""TOD reader for the standard Commander4 HDF5 scan format (``experiment_id: general``).

Reads the format and applies no instrument policy of its own: no data-quality cuts, no tuned noise
priors, no format quirks. Everything it needs comes from the file (pointing, flags, and the
per-detector ``scalars`` that seed gain and noise) or from the parameter file.

Use this for any dataset already in the standard layout, including everything ``sims/simgen``
writes. An instrument only needs its own reader once it needs behaviour this one deliberately has
none of, which in practice means detector-scan quality cuts in the instrument's own units, or noise
priors tuned to it (compare ``planck_lfi.py``, which is this reader plus exactly those two things).
"""
import logging
import gc

import numpy as np
import h5py
from pixell.bunch import Bunch
from mpi4py import MPI

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


logger = logging.getLogger(__name__)

# Bits marking unusable samples in the cumulative flag stream. Matches `GOOD_SCAN_BITMASK` in
# sims/simgen/writers.py, which documents the contract a writer of this format must satisfy.
GOOD_DATA_BITMASK = 6111232


def tod_reader(band_comm: MPI.Comm, my_experiment: str, my_band: Bunch, det_names: list[str],
               params: Bunch, scan_idx_start: int,
               scan_idx_stop: int) -> DetectorGroupTOD:
    """Read this rank's scans for one band from its HDF5 scan files.

    Each file holds one scan, with its own pointing per detector.

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

    default_mask, specific_masks = read_processing_masks(band_comm, my_band)

    if "bad_PIDs_path" in my_experiment:
        bad_PIDs = np.load(my_experiment.bad_PIDs_path)
    else:
        bad_PIDs = np.array([])


    # Attempting to reduce fragmentation by allocating buffers.
    ntod_upper_bound = int(my_band.fsamp*100*3600)  # 10 hour scan.

    ntod_sum_original = 0
    ntod_sum_final = 0
    scan_list = []
    num_included = 0
    for i_pid in range(scan_idx_start, scan_idx_stop):
        pid = pids[i_pid]
        filepath = filepaths[i_pid]
        if pid in bad_PIDs:
            continue
        with h5py.File(filepath, "r") as f:
            data_nside = int(f["common/nside"][()].item())
            ntod = int(f[f"/{pid}/common/ntod"][()].item())
            ntod_optimal = find_good_fourier_size(ntod)
            huffman_tree = f[f"/{pid}/common/hufftree"][()]
            huffman_symbols = f[f"/{pid}/common/huffsymb"][()]
            vsun = f[f"/{pid}/common/vsun/"][()]
            fsamp = float(f["/common/fsamp/"][()].item())
            npsi = int(f["/common/npsi/"][()].item())

            if ntod > ntod_upper_bound:
                raise ValueError(f"Scan {pid} of band {bandname} has {ntod} samples, above the "
                                 f"{ntod_upper_bound}-sample ceiling this reader allocates for.")

            detector_list = []
            # No detector-scan is rejected here, so the full-band column idet and the per-scan
            # column idet_accepted advance together. A reader that does cut detectors (see
            # `planck_lfi.py`) must advance idet_accepted only for the ones it keeps.
            idet_accepted = 0
            for idet, det_name in enumerate(det_names):
                tod = f[f"/{pid}/{det_name}/tod/"][:ntod_optimal].astype(np.float32, copy=False)
                pix_encoded = f[f"/{pid}/{det_name}/pix/"][()]
                psi_encoded = f[f"/{pid}/{det_name}/psi/"][()]
                flag_encoded = f[f"/{pid}/{det_name}/flag/"][()]
                # [gain, sigma0, fknee, alpha] per detector, with the gain in "micro-gain" (the
                # Planck convention this format inherits; simgen writes det.gain*1e6). These seed
                # the chain, so a parameter file need not repeat values the file already carries.
                init_scalars = f[f"/{pid}/{det_name}/scalars/"][()]
                init_scalars[0] *= 1e-6

                # Some simulations have a (1,N) shape for pixels; remove leading dimension.
                if pix_encoded.ndim == 2 and pix_encoded.shape[0] == 1:
                    pix_encoded = pix_encoded[0]
                if psi_encoded.ndim == 2 and psi_encoded.shape[0] == 1:
                    psi_encoded = psi_encoded[0]

                det_pointing = PixelPointing(pix_encoded, psi_encoded, huffman_tree,
                                             huffman_symbols, npsi, my_band.eval_nside, data_nside,
                                             ntod, ntod_optimal)
                # Flags are kept per sample rather than used to reject the whole scan, so a
                # partly flagged scan still contributes its good samples.
                detector = DetectorTOD(det_name, idet, idet_accepted, tod, det_pointing, fsamp,
                                       vsun, huffman_tree, huffman_symbols, default_mask,
                                       specific_masks, ntod, ntod_optimal,
                                       flag_encoded=flag_encoded,
                                       bad_data_bitmask=GOOD_DATA_BITMASK,
                                       init_scalars=init_scalars)
                detector_list.append(detector)
                ntod_sum_original += ntod
                ntod_sum_final += ntod_optimal
                idet_accepted += 1
        scan_list.append(ScanTOD(detector_list, 0., int(pid)))
        num_included += 1
        if i_pid % 10 == 0:
            gc.collect()
    ndet = len(det_names)

    noise_model = NoisePSDOof()
    apply_noise_priors(noise_model, params, expname, bandname)

    band_tod = DetectorGroupTOD(scan_list, expname, bandname, my_band.eval_nside, my_band.freq,
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
        logger.info(f"Band {bandname}: read TODs with {frac_included:.1f}% of scans included and "
                    f"{avg_scan_remaining:.1f}% of samples retained after the Fourier cut.")

    return band_tod
