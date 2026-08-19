"""MPI orchestration: build the sky, scan it into per-detector TODs, write the files.

Pipeline:
  1. Rank 0 builds each band's beam-smoothed sky map and broadcasts it.
  2. ``(band, scan)`` work items are split across ranks.
  3. Each rank, for its scans: gets the band's shared-boresight pointing, projects the sky for each
     detector (+ orbital dipole), adds per-detector noise, applies TOD modifiers (cross-talk), and
     writes the scan file in the litebird format.
  4. Rank 0 gathers the written files and writes a per-band ``filelist.txt``.

By default a "scan" is a fixed wall-clock duration (``scan_duration_sec``); its sample count is per
band (``round(duration * fsamp)``, made even), so bands with different sampling rates stay
time-aligned. The scan length is owned by the pointing strategy, so a strategy can override it (e.g.
RasterScan makes one scan exactly one full fill of the patch).
"""
import logging
import os
import healpy as hp
import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray
from pixell.bunch import Bunch

from simgen.config import bget, load_params
from simgen.diagnostics import hit_map, noise_map_rhs, white_noise_normal_matrix, write_band_diagnostics
from simgen.instrument import Band, build_bands
from simgen.modifiers import build_modifiers
from simgen.noise import make_noise_model
from simgen.pointing import make_pointing
from simgen.sky import build_band_sky_maps, compute_orbital_dipole
from simgen.writers import write_scan_file, write_filelist, DEFAULT_NPSI

logger = logging.getLogger(__name__)


def _eval_and_data_pix(theta: NDArray, phi: NDArray, data_nside: int,
                       eval_nside: int) -> tuple[NDArray, NDArray]:
    """Pixel indices stored on disk (data_nside) and used for sky lookup (eval_nside).

    The eval pixels are derived from the stored pixels exactly as ``PixelPointing.get_pix`` remaps
    them on read, so the baked signal matches what the reader reconstructs.
    """
    pix_data = hp.ang2pix(data_nside, theta, phi)
    return pix_data, _remap_data_pix(pix_data, data_nside, eval_nside)


def _remap_data_pix(pix_data: NDArray, data_nside: int, eval_nside: int) -> NDArray:
    """Remap stored data-resolution pixels exactly as ``PixelPointing.get_pix`` does."""
    if eval_nside == data_nside:
        return pix_data
    th, ph = hp.pix2ang(data_nside, pix_data)
    return hp.ang2pix(eval_nside, th, ph)


def _project_signal(skymap: NDArray, pix: NDArray, psi: NDArray, polarization: str) -> NDArray:
    """Sky signal seen by a detector: I + Q cos2psi + U sin2psi (rows present per polarization)."""
    if polarization == "I":
        return skymap[0][pix]
    if polarization == "QU":
        return skymap[0][pix] * np.cos(2 * psi) + skymap[1][pix] * np.sin(2 * psi)
    return skymap[0][pix] + skymap[1][pix] * np.cos(2 * psi) + skymap[2][pix] * np.sin(2 * psi)


def _broadcast_sky_maps(comm: MPI.Comm, bands: list[Band], params: Bunch,
                        truth_map_dir: str | None = None) -> dict[str, NDArray[np.floating]]:
    """Build all band sky maps on rank 0 (writing the truth maps there too) and broadcast them."""
    rank = comm.Get_rank()
    band_maps = build_band_sky_maps(params, bands, truth_map_dir) if rank == 0 else {
        b.name: np.empty((b.npol, 12 * b.eval_nside**2), dtype=np.float32) for b in bands}
    for band in bands:
        comm.Bcast(band_maps[band.name], root=0)
    return band_maps


def _det_signal(band: Band, chunk, det, skymap: NDArray,
                include_orbdip: bool) -> tuple[NDArray, NDArray, NDArray]:
    """Sky signal (+ orbital dipole) plus the pixel and psi arrays for one detector's pointing."""
    pix_data, pix_eval = _eval_and_data_pix(chunk.theta, chunk.phi, band.data_nside, band.eval_nside)
    psi = chunk.psi + det.psi_offset
    signal = _project_signal(skymap, pix_eval, psi, band.polarization)
    if include_orbdip and np.any(chunk.vsun != 0.0):
        signal = signal + compute_orbital_dipole(chunk.vsun, pix_eval, band.eval_nside,
                                                 band.freq, band.units)
    return signal, pix_data, psi


def _simulate_scan(band: Band, strategy, sample_offset: int, skymap: NDArray, ntod: int,
                   scan_idx: int, band_idx: int, noise_model, modifiers,
                   seed: int, include_orbdip: bool) -> tuple[dict, dict, dict, dict, NDArray]:
    """Build one scan's per-detector pix / psi / TOD dicts (signal + noise + modifiers) and vsun.

    For shared-boresight strategies the pointing is computed once and reused (only the per-detector
    polarization-angle offset differs). For ``per_detector_pointing`` strategies (e.g. RasterScan)
    the pointing is recomputed per detector with that detector's focal-plane offset, so detectors
    are spatially offset. ``vsun`` is always taken from the boresight pointing (det_offset = 0).
    """
    bore = strategy.compute(sample_offset, ntod)
    shared = None if strategy.per_detector_pointing else _det_signal_shared(band, bore, skymap,
                                                                            include_orbdip)
    det_pix, det_psi = {}, {}
    tod_matrix = np.zeros((band.ndet, ntod), dtype=np.float32)
    noise_matrix = np.zeros((band.ndet, ntod), dtype=np.float32)
    for det in band.detectors:
        if strategy.per_detector_pointing:
            chunk = strategy.compute(sample_offset, ntod, det_offset=det.fp_offset)
            signal, pix_data, psi = _det_signal(band, chunk, det, skymap, include_orbdip)
        else:
            shared_orbdip, pix_data, shared_pix_eval = shared
            psi = bore.psi + det.psi_offset
            # Shared boresight: same pixels for every detector, signal differs only via psi.
            signal = _project_signal(skymap, shared_pix_eval, psi, band.polarization)
            if shared_orbdip is not None:  # add the (shared) orbital dipole, computed once
                signal = signal + shared_orbdip
        rng = np.random.default_rng([seed, band_idx, scan_idx, det.idx])
        noise = noise_model.realize(ntod, band.fsamp, det.sigma0, rng)
        # Bolometer transfer function acts on the optical signal only (d = T(g*s) + n); detector
        # noise is added after the thermal/readout response, matching the mapmaker model T P m + n.
        clean = det.gain * signal
        if det.transfer is not None:
            clean = det.transfer.apply(clean, band.fsamp)
        tod_matrix[det.idx] = clean + noise
        noise_matrix[det.idx] = noise
        det_pix[det.name] = pix_data
        det_psi[det.name] = psi

    ctx = Bunch(scan_idx=scan_idx, band_idx=band_idx)
    for mod in modifiers:
        tod_matrix = mod.apply(tod_matrix, band, ctx)
        noise_matrix = mod.apply(noise_matrix, band, ctx)

    det_tod = {det.name: tod_matrix[det.idx] for det in band.detectors}
    det_noise = {det.name: noise_matrix[det.idx] for det in band.detectors}
    return det_pix, det_psi, det_tod, det_noise, bore.vsun


def _det_signal_shared(band: Band, bore, skymap: NDArray, include_orbdip: bool):
    """Precompute the shared-boresight pixels and orbital dipole reused across detectors."""
    pix_data, pix_eval = _eval_and_data_pix(bore.theta, bore.phi, band.data_nside, band.eval_nside)
    orbdip = None
    if include_orbdip and np.any(bore.vsun != 0.0):
        orbdip = compute_orbital_dipole(bore.vsun, pix_eval, band.eval_nside, band.freq, band.units)
    return orbdip, pix_data, pix_eval


def run(parameter_file: str) -> int:
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    params, params_dict = load_params(parameter_file)

    sim = params.simulation
    bands = build_bands(params)
    nscans = int(sim.nscans)
    scan_duration = float(bget(sim, "scan_duration_sec", 3600.0))
    seed = int(bget(params.general, "seed", 0))
    npsi = int(bget(sim, "npsi", DEFAULT_NPSI))
    include_orbdip = bool(bget(sim, "orbital_dipole", True))
    compress = bool(bget(sim, "compress", True))  # Huffman-compress pix/psi (flag is always compressed).
    output_dir = params.general.output_dir
    debug_output_dir = bget(sim, "debug_output_dir", None)

    # Create per-band output directories before any rank writes into them.
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        if debug_output_dir:
            os.makedirs(debug_output_dir, exist_ok=True)
        for band in bands:
            os.makedirs(os.path.join(output_dir, band.name), exist_ok=True)
        with open(os.path.join(output_dir, "simgen_params.yml"), "w") as f:
            import yaml
            f.write(yaml.dump(params_dict, sort_keys=False))
    comm.Barrier()

    # Per-component input amplitude maps: written alongside the other diagnostics, and usable as
    # `init_from` / `amp_prior_mean_map` in the matching Commander4 parameter file.
    write_truth = debug_output_dir and bool(bget(sim, "write_component_truth_maps", True))
    band_maps = _broadcast_sky_maps(comm, bands, params,
                                    debug_output_dir if write_truth else None)
    modifiers = build_modifiers(params)

    # Per-band shared pointing strategy (FilePointing opens its file once here).
    pointing = {b.name: make_pointing(sim.pointing, b.fsamp) for b in bands}
    noise_models = {b.name: make_noise_model(b) for b in bands}

    # Split (band, scan) work across ranks.
    work = [(bi, si) for bi in range(len(bands)) for si in range(nscans)]
    my_work = np.array_split(np.array(work), size)[rank] if work else np.empty((0, 2), int)
    if rank == 0:
        logger.info("simgen: %d bands x %d scans = %d scan-files across %d ranks.",
                    len(bands), nscans, len(work), size)

    my_entries: dict[str, list[tuple[int, str]]] = {b.name: [] for b in bands}
    my_hits = {b.name: np.zeros(12 * b.eval_nside**2, dtype=np.int64) for b in bands} if debug_output_dir else {}
    my_normal = {
        b.name: np.zeros(({"I": 1, "QU": 3, "IQU": 6}[b.polarization], 12 * b.eval_nside**2),
                         dtype=np.float64)
        for b in bands
    } if debug_output_dir else {}
    my_noise_rhs = {
        b.name: np.zeros((b.npol, 12 * b.eval_nside**2), dtype=np.float64)
        for b in bands
    } if debug_output_dir else {}
    for bi, si in my_work:
        band = bands[int(bi)]
        ntod = pointing[band.name].samples_per_scan(scan_duration, band.fsamp)
        det_pix, det_psi, det_tod, det_noise, vsun = _simulate_scan(
            band, pointing[band.name], int(si) * ntod, band_maps[band.name], ntod, int(si), int(bi),
            noise_models[band.name], modifiers, seed, include_orbdip)
        oof = band.noise.oof
        fknee = float(bget(oof, "fknee", 0.0))
        alpha = float(bget(oof, "alpha", 0.0))
        det_scalars = {
            det.name: np.array([det.gain * 1e6, det.sigma0, fknee, alpha], dtype=np.float64)
            for det in band.detectors
        }
        pid = int(si) + 1
        path = os.path.join(output_dir, band.name, f"scan_{pid:06d}.h5")
        write_scan_file(path, pid, band.data_nside, band.fsamp, npsi, ntod, vsun,
                        band.det_names, det_pix, det_psi, det_tod, det_scalars, compress=compress)
        if debug_output_dir:
            det_eval_pix = {
                name: _remap_data_pix(pix, band.data_nside, band.eval_nside)
                for name, pix in det_pix.items()
            }
            my_hits[band.name] += hit_map(band, det_eval_pix)
            my_normal[band.name] += white_noise_normal_matrix(band, det_eval_pix, det_psi)
            my_noise_rhs[band.name] += noise_map_rhs(band, det_eval_pix, det_psi, det_noise)
        my_entries[band.name].append((pid, path))

    # Gather written files on rank 0 and write per-band filelists.
    all_entries = comm.gather(my_entries, root=0)
    debug_hits, debug_normal, debug_noise_rhs = {}, {}, {}
    if debug_output_dir:
        for band in bands:
            hits = np.empty_like(my_hits[band.name]) if rank == 0 else None
            normal = np.empty_like(my_normal[band.name]) if rank == 0 else None
            noise_rhs = np.empty_like(my_noise_rhs[band.name]) if rank == 0 else None
            comm.Reduce(my_hits[band.name], hits, op=MPI.SUM, root=0)
            comm.Reduce(my_normal[band.name], normal, op=MPI.SUM, root=0)
            comm.Reduce(my_noise_rhs[band.name], noise_rhs, op=MPI.SUM, root=0)
            if rank == 0:
                debug_hits[band.name] = hits
                debug_normal[band.name] = normal
                debug_noise_rhs[band.name] = noise_rhs
    if rank == 0:
        for band in bands:
            merged = [e for part in all_entries for e in part[band.name]]
            write_filelist(os.path.join(output_dir, band.name, "filelist.txt"), merged)
            logger.info("Band %s: wrote %d scan files + filelist.", band.name, len(merged))
            if debug_output_dir:
                write_band_diagnostics(debug_output_dir, band, band_maps[band.name],
                                       debug_hits[band.name], debug_normal[band.name],
                                       debug_noise_rhs[band.name])
                logger.info("Band %s: wrote diagnostic maps and plots to %s.", band.name,
                            debug_output_dir)
        logger.info("simgen finished. Output in %s", output_dir)
    return 0
