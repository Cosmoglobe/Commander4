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
from commander4.simulations.inplace_litebird_sim import replace_tod_with_sim
from commander4.output.log import logassert
import commander4.compression.huffman as huffman
from commander4.logging.performance_logger import benchmark, bench_summary, start_bench,\
                                            stop_bench, log_memory, increment_count, bench_reset

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
    logger = logging.getLogger(__name__)
    ndet = len(det_names)
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

    Fourier_times = np.load(my_experiment.Fourier_times_path)

    # Attempting to reduce fragmentation by allocating buffers.
    ntod_upper_bound = int(my_band.fsamp*100*3600)  # 10 hour scan.

    ntod_sum_original = 0
    ntod_sum_final = 0
    scan_list = []
    num_included = 0
    stop_bench("reader-startup")
    for i_pid in range(scan_idx_start, scan_idx_stop):
        pid = pids[i_pid]
        filepath = filepaths[i_pid]
        start_bench("fileread")
        with h5py.File(filepath, "r") as f:
            data_nside = int(f["common/nside"][()].item())
            ntod = int(f[f"/{pid}/common/ntod"][()].item())
            ntod_optimal = find_good_Fourier_time(Fourier_times, ntod)
            vsun = f[f"/{pid}/common/vsun/"][()]
            fsamp = float(f["/common/fsamp/"][()].item())

            if my_experiment.pix_is_compressed or my_experiment.psi_is_compressed:
                raise NotImplementedError("Compressed data not yet implemented in litebird injection sims.")
                huffman_tree = f[f"/{pid}/common/hufftree"][()]
                huffman_symbols = f[f"/{pid}/common/huffsymb"][()]
                npsi = int(f["/common/npsi/"][()].item())

            # Since we spawn all detectors from the same pointing, we keep the pix and psi reading
            # outside the detector-loop, to avoid loading the disk system too much.
            det_name_Synne = "001_000_002_60A_166_T" # Temporary hard-coded solution.
            # tod = np.zeros(ntod_optimal, dtype=np.float32)
            default_pix = f[f"/{pid}/{det_name_Synne}/pix/"][:ntod_optimal].astype(np.int32)
            default_psi = f[f"/{pid}/{det_name_Synne}/psi/"][:ntod_optimal].astype(np.float32)
        stop_bench("fileread")

        start_bench("compress")
        processing_mask_nside = hp.npix2nside(processing_mask_map.size)
        logassert(my_band.eval_nside == processing_mask_nside,
                    f"Processing mask (band {bandname}) "
                    f"has nside {processing_mask_nside} while eval_nside = {my_band.eval_nside} "
                    "(NB: eval_nside can be set different from native data nside)", logger)

        if ntod > ntod_upper_bound:
            raise ValueError(f"{ntod_upper_bound} {ntod}")
        
        # Add some artificial rotations of the polarization angles.
        psi_offsets = np.deg2rad(np.array([0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]))

        # Some simulations have a (1,N) shape for pixels; remove leading dimension.
        if default_pix.ndim == 2 and default_pix.shape[0] == 1:
            default_pix = default_pix[0]
        if default_psi.ndim == 2 and default_psi.shape[0] == 1:
            default_psi = default_psi[0]
        
        npsi = 4096
        # Add these four rotation options to the read-in psi angles.
        detector_list = []
        tod_alldet = np.zeros((ndet, ntod_optimal), dtype=np.float32)
        for idet in range(ndet):
            tod = tod_alldet[idet]

            # Ideally we would do the huffman compression outside the detector-loop, but that's
            # tricky if we want to add psi-offsets. Also, it's not that expensive anyway
            psi = default_psi.copy()
            psi += psi_offsets[idet%psi_offsets.size]
            psi = huffman.preproc_digitize_and_diff(psi, npsi)
            pix = huffman.preproc_diff(default_pix)
            huffman_tree, huffman_symbols, sym_codes, sym_lengths = huffman.build_huffman_tree([pix, psi])
            psi_encoded = huffman.huffman_compress_array(psi, sym_codes, sym_lengths)
            pix_encoded = huffman.huffman_compress_array(pix, sym_codes, sym_lengths)

            detector = DetectorTOD(tod, pix_encoded, psi_encoded, my_band.eval_nside,
                                    data_nside, fsamp, vsun, huffman_tree, huffman_symbols,
                                    npsi, processing_mask_map, ntod_optimal,
                                    pix_is_compressed=True, # Hard-coded to true since we compress manually.
                                    psi_is_compressed=True)
            detector_list.append(detector)
            ntod_sum_original += ntod
            ntod_sum_final += ntod_optimal
            gc.collect()
        stop_bench("compress")
        scanID = int(pid)
        scan = ScanTOD(detector_list, 0., scanID, scan_idx_start, scan_idx_stop)
        scan_list.append(scan)
        num_included += 1
    ndet = len(det_names)

    band_tod = DetGroupTOD(scan_list, expname, bandname, my_band.eval_nside, my_band.freq,
                           my_band.fwhm, ndet, my_band.polarization)

    start_bench("skysim")
    if my_experiment.replace_tod_with_sim:
        replace_tod_with_sim(band_comm, band_tod, my_band, params, my_experiment.sim_params)
    stop_bench("skysim")

    bench_summary(band_comm)
    bench_reset()

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