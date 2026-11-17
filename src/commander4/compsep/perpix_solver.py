"""The per-pixel amplitude solver: an independent least-squares fit in each pixel.

Alternative to the global CG solve for the case where every band has been brought to a common
resolution. The mixing matrix is then the same at every multipole, so the amplitudes follow from a
small (nband x ncomp) normal-equation solve per pixel, done in C++. Unlike the CG solver, the
amplitudes it produces carry the common beam rather than being deconvolved.
"""
import time
import logging
import healpy as hp
import numpy as np
from mpi4py import MPI
from pixell import curvedsky

from commander4.sky.comp_list import CompList
from commander4.backend import compsep as cpp_compsep
from commander4.data_models.detector_map import DetectorMap
from commander4.diagnostics.performance import benchmark

logger = logging.getLogger(__name__)


def solve_compsep_perpix(solver_comm: MPI.Comm, band_map: DetectorMap, comp_list: CompList,
                         double_precision: bool, nthreads: int,
                         rng: np.random.Generator) -> None:
    """ A pixel-by-pixel solver for the component separation problem, which leaves the solution in
        the components of `comp_list`, on every rank of `solver_comm`. Requires uniform nside and a
        common beam across all bands (smoothing to a common resolution is done at the data sources,
        controlled by ``compsep.common_res_fwhm``). This solver ignores beams entirely, so bands
        arriving at differing resolutions are silently mixed; a warning is logged below.
        The per-pixel solves run on the master rank only, with ``nthreads`` threads, and draw
        their fluctuation terms from the master's ``rng``. All components must be diffuse
        (`read_sampling_groups` checks this at start-up).
    """
    is_master = solver_comm.Get_rank() == 0
    if is_master:
        logger.verbose("Starting pixel-by-pixel component separation.")
    complex_dtype = np.complex128 if double_precision else np.complex64
    real_dtype = np.float64 if double_precision else np.float32

    # This solver has no beam model, so all bands should already share one resolution (smoothed at
    # ingest via compsep.common_res_fwhm); differing FWHMs silently mix resolutions, so warn.
    all_fwhm = solver_comm.allgather(band_map.fwhm)
    if not np.allclose(all_fwhm, all_fwhm[0]):
        logger.warning(f"Per-pixel solver received bands at differing resolutions {all_fwhm}; the "
                       "result mixes resolutions. Set compsep.common_res_fwhm to a common beam.")

    ncomp = len(comp_list)
    with benchmark("perpix-gather"):
        freqs = np.array(solver_comm.gather(band_map.nu, root=0))
        all_map_sky = solver_comm.gather(band_map.map_sky, root=0)
        all_map_inv_var = solver_comm.gather(band_map.map_inv_var, root=0)  # 0 where unobserved.

    npix = hp.nside2npix(band_map.nside)
    comp_maps = [None]*band_map.npol
    if is_master:  # Unfortunately, only master rank does the work.
        map_shapes = np.array([_map.shape for _map in all_map_sky])
        if not np.all(map_shapes == map_shapes[0]):
            raise ValueError("Per-pixel solver requires all maps to have the same shape, but "
                             f"received {map_shapes}.")
        # The mixing matrix: one row per band, one column per component.
        M = np.empty((len(freqs), ncomp))
        for icomp in range(ncomp):
            M[:, icomp] = comp_list[icomp].get_sed(freqs)
        for ipol in range(band_map.npol):
            t0 = time.time()
            # This Stokes row of every band, shape (nband, npix).
            maps_sky = np.array([band_sky[ipol] for band_sky in all_map_sky], dtype=np.float64)
            maps_inv_var = np.array([band_inv_var[ipol] for band_inv_var in all_map_inv_var],
                                    dtype=np.float64)
            rand = rng.standard_normal((npix, len(freqs)))
            # TODO: Should scale M to make solution more well-conditioned, and then adjust
            # solution with the scaling factor used.
            with benchmark("perpix-solve"):
                comp_maps[ipol] = cpp_compsep.solve_perpix(maps_sky, maps_inv_var, M, rand,
                                                           nthreads)
            logger.info(f"Finished pixel-by-pixel component separation in {time.time()-t0:.2f}s "
                        f"for polarization {ipol+1} of {band_map.npol}.")

    comp_maps = solver_comm.bcast(comp_maps, root=0)
    with benchmark("perpix-alm"):
        for icomp in range(ncomp):
            input_map = np.array([comp_maps[ipol][icomp] for ipol in range(band_map.npol)],
                                 dtype=real_dtype)
            comp_alms = curvedsky.map2alm_healpix(input_map, niter=3, spin=band_map.spin,
                                                  lmax=comp_list[icomp].lmax)
            comp_list[icomp].amps = comp_alms.astype(complex_dtype, copy=False)
            # These amplitudes are component maps at the data resolution; record that beam so the
            # forward model applies no extra smoothing when predicting a band at this resolution.
            comp_list[icomp].amp_fwhm_rad = band_map.fwhm_rad
