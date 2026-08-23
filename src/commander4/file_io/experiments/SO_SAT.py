"""TOD reader for the Simons Observatory Small Aperture Telescopes (``experiment_id: SO_SAT``)."""
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
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.scan_tod import ScanTOD
from commander4.simulations.inplace_litebird_sim import replace_tod_with_sim
from commander4.tod.noise.psd import NoisePSD, NoisePSDOof
from commander4.diagnostics.performance import benchmark, bench_summary, start_bench,\
                                               stop_bench, log_memory, increment_count, bench_reset
from commander4.data_models.pointing import DetectorBoresightPointing, ScanBoresightPointing
from commander4.file_io.experiments.read_utils import (
    apply_noise_priors,
    find_good_fourier_size,
    read_processing_masks,
)
logger = logging.getLogger(__name__)


def tod_reader(band_comm: MPI.Comm, my_experiment: str, my_band: Bunch, det_names: list[str],
               params: Bunch, scan_idx_start: int,
               scan_idx_stop: int) -> DetectorGroupTOD:
    """Read this rank's scans for one SO SAT band from its HDF5 scan files.

    Differs from the LAT reader in how pointing arrives: the SAT files store the boresight path
    plus per-detector focal-plane offsets and polarization angles, from which
    `ScanBoresightPointing` reconstructs each detector's pixels and angles. The LAT files instead
    carry each detector's pointing directly, Huffman-compressed.

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
    for i_pid in range(scan_idx_start, scan_idx_stop):
        pid = pids[i_pid]
        filepath = filepaths[i_pid]
        if pid in bad_PIDs:
            continue
        start_bench("fileread")
        good_scan = True
        with h5py.File(filepath, "r") as f:
            ntod = int(f[f"/{pid}/common/ntod"][()].item())
            ntod_optimal = find_good_fourier_size(ntod)
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

                detector = DetectorTOD(
                    name=det_name,
                    det_idx_fullband=idet,
                    tod=tod,
                    pointing=pointing,
                    sampling_rate_hz=fsamp,
                    orbital_velocity_m_per_s=np.zeros(3),
                    huffman_tree=huffman_tree,
                    huffman_symbols=huffman_symbols,
                    default_proc_mask=default_mask,
                    specific_proc_masks=specific_masks,
                    huffman_tree2=huffman_tree2,
                    huffman_symbols2=huffman_symbols2,
                    flag_encoded=flag_encoded,
                    bad_data_bitmask=my_experiment.bad_data_bitmask,
                    init_scalars=init_scalars,
                    tod_is_compressed=my_experiment.tod_is_compressed,
                    det_response=det_response,
                )
                # `<=` so the 0.0 default still drops fully-flagged detector-scans.
                if np.mean(detector.good_data_mask) <= min_unmasked_fraction:
                    continue
                if (detector.tod == 0).all():
                    continue
                detector_list.append(detector)
                ntod_sum_original += ntod
                ntod_sum_final += ntod_optimal
            included_detector_scans += len(detector_list)
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
    apply_noise_priors(noise_model, params, expname, bandname)
    band_tod = DetectorGroupTOD(scan_list, expname, bandname, my_band.eval_nside, my_band.freq,
                           my_band.fwhm, fsamp, ndet, my_band.polarization, noise_model)

    ### Summarize detector-scan inclusion and Fourier-cut retention ###
    # The fraction is reported per detector-scan rather than per scan, since a scan survives as
    # long as any one of its detectors does and would otherwise hide dropped detectors.
    local_tot_scans = scan_idx_stop - scan_idx_start
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
        logger.info(f"Band {bandname}: read TODs with {frac_included:.1f}% of detector-scans "
                    f"included ({total_included}/{total_detector_scans}) and "
                    f"{avg_scan_remaining:.1f}% of samples retained after the Fourier cut.")

    return band_tod
