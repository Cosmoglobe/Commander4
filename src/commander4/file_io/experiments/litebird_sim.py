"""TOD reader for ``litebird_sim`` simulated data (``experiment_id: litebird_sim``).

Keeps the in-place-simulation hook (``replace_tod_with_sim``), which regenerates the TOD from a
sky model while reusing the file's pointing. Data written by ``simgen`` shares this HDF5
layout but should be read with ``experiment_id: general``, which has no such hook.
"""
import logging
import numpy as np
import healpy as hp
from astropy.io import fits
import h5py
import gc
from numpy.typing import NDArray
from pixell.bunch import Bunch
from mpi4py import MPI

from commander4.backend import utils as cpp_utils
from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.scan_tod import ScanTOD
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.pointing import PixelPointing
from commander4.tod.noise.psd import NoisePSD, NoisePSDOof
from commander4.file_io.experiments.read_utils import (
    apply_noise_priors,
    find_good_fourier_size,
    read_processing_masks,
)
from commander4.simulations.inplace_litebird_sim import replace_tod_with_sim


logger = logging.getLogger(__name__)


def tod_reader(band_comm: MPI.Comm, my_experiment: str, my_band: Bunch, det_names: list[str],
               params: Bunch, scan_idx_start: int,
               scan_idx_stop: int) -> DetectorGroupTOD:
    """Read this rank's scans for one LiteBIRD band from its litebird_sim HDF5 files.

    Each file holds one scan with its own pointing per detector.

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
            pid, filepath, _, _, _ = line.split()
            pids.append(f"{int(pid):06d}")
            filepaths.append(filepath[1:-1])
            oids.append(filepath.split(".")[0].split("_")[-1])

    default_mask, specific_masks = read_processing_masks(band_comm, my_band)

    if "bad_PIDs_path" in my_experiment:
        bad_PIDs = np.load(my_experiment.bad_PIDs_path)
    else:
        bad_PIDs = np.array([])


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
            ntod_optimal = find_good_fourier_size(ntod)
            huffman_tree = f[f"/{pid}/common/hufftree"][()]
            huffman_symbols = f[f"/{pid}/common/huffsymb"][()]
            vsun = f[f"/{pid}/common/vsun/"][()]
            fsamp = float(f["/common/fsamp/"][()].item())
            npsi = int(f["/common/npsi/"][()].item())

            if ntod > ntod_upper_bound:
                raise ValueError(f"{ntod_upper_bound} {ntod}")

            detector_list = []
            for idet, det_name in enumerate(det_names):
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

                det_pointing = PixelPointing(pix_encoded, psi_encoded, huffman_tree,
                                             huffman_symbols, npsi, my_band.eval_nside, data_nside,
                                             ntod, ntod_optimal)
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
                )
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
    apply_noise_priors(noise_model, params, expname, bandname)

    band_tod = DetectorGroupTOD(scan_list, expname, bandname, my_band.eval_nside, my_band.freq,
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
        logger.info(f"Band {bandname}: read TODs with {frac_included:.1f}% of scans included and "
                    f"{avg_scan_remaining:.1f}% of samples retained after the Fourier cut.")

    return band_tod
