import time
import ctypes
import logging
import numpy as np
from mpi4py import MPI
from pixell import curvedsky
from pixell.bunch import Bunch

from commander4.output.log import logassert
from commander4.sky_models.component import Component
from commander4.utils.ctypes_lib import load_cmdr4_ctypes_lib
from commander4.data_models.detector_map import DetectorMap


def solve_compsep_perpix(proc_comm: MPI.Comm, detector_data: DetectorMap,
                         comp_list: list[Component], params: Bunch) -> list[Component]:
    """ A pixel-by-pixel solver for the component separation problem. Requires uniform nside and a
        common beam across all bands (smoothing to a common resolution is done at the data sources,
        controlled by ``general.common_res_fwhm``); this solver ignores beams entirely, so mixed
        resolutions silently mix resolutions and trigger a warning below.
    """
    # TODO: Add support for non-Diffuse components (point sources, templates).
    logger = logging.getLogger(__name__)
    if proc_comm.Get_rank() == 0:
        logger.info("Starting pixel-by-pixel component separation.")
    if params.general.CG_float_precision == "double":  # FIXME: bad parameter name.
        complex_dtype = np.complex128
        real_dtype = np.float64
    else:
        complex_dtype = np.complex64
        real_dtype = np.float32

    npol = detector_data.npol
    pol = detector_data.pol
    spin = 2 if pol else 0
    map_sky = detector_data.map_sky
    band_freq = detector_data.nu
    map_rms = detector_data.map_rms
    ctypes_lib = load_cmdr4_ctypes_lib()
    ctypes_lib.solve_compsep.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), # map_sky
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), # map_rms
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), # M
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), # rnd
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), # map_out
    ]

    # This solver has no beam model, so all bands should already share one resolution (smoothed at
    # ingest via general.common_res_fwhm); differing FWHMs silently mix resolutions, so warn.
    all_fwhm = proc_comm.allgather(detector_data.fwhm)
    if not np.allclose(all_fwhm, all_fwhm[0]):
        logger.warning(f"Per-pixel solver received bands at differing resolutions {all_fwhm}; the "
                       "result mixes resolutions. Set general.common_res_fwhm to a common beam.")

    ncomp = len(comp_list)
    all_freq = proc_comm.gather(band_freq, root=0)
    all_map_sky = proc_comm.gather(map_sky, root=0)
    all_map_rms = proc_comm.gather(map_rms, root=0)

    nside = detector_data.nside
    npix = 12*nside**2
    comp_maps = [None, None] if pol else [None]
    if proc_comm.Get_rank() == 0:  # Unfortunately, only master rank does the work.
        map_shapes = np.array([_map.shape for _map in all_map_sky])
        logassert(np.all(map_shapes == map_shapes[0]), "Per-pixel solver requires all maps to have"\
                  f" the same nside, but received nsides: {map_shapes}", logger)
        for ipol in range(npol):
            t0 = time.time()
            freqs = []
            maps_sky = []
            maps_rms = []
            for iband in range(len(all_freq)):
                if all_map_sky[iband][ipol] is not None:
                    freqs.append(all_freq[iband])
                    maps_sky.append(all_map_sky[iband][ipol])
                    maps_rms.append(all_map_rms[iband][ipol])
            freqs = np.array(freqs)
            maps_sky = np.array(maps_sky)
            maps_rms = np.array(maps_rms)
            nband = len(freqs)
            comp_maps[ipol] = np.zeros((ncomp, npix))
            M = np.empty((nband, ncomp))
            for icomp in range(ncomp):
                M[:, icomp] = comp_list[icomp].get_sed(freqs)
            rand = np.random.randn(npix, nband)
            # TODO: Write unit tests that confirm Python and C gives same answers.
            # TODO: Should scale M to make solution more well-conditioned, and then adjust
            # solution with the scaling factor used.
            ctypes_lib.solve_compsep(npix, nband, ncomp, maps_sky.astype(np.float64, copy=False),
                                  maps_rms.astype(np.float64, copy=False), M, rand, comp_maps[ipol])
            logger.info(f"Finished pixel-by-pixel component separation in {time.time()-t0:.2f}s "\
                        f"for polarization {ipol+1} of {npol}.")

    comp_maps = proc_comm.bcast(comp_maps, root=0)
    for icomp in range(ncomp):
        if pol:
            input_map = np.array([comp_maps[0][icomp], comp_maps[1][icomp]], dtype=real_dtype)
        else:
            input_map = np.array([comp_maps[0][icomp]], dtype=real_dtype)
        comp_alms = curvedsky.map2alm_healpix(input_map, niter=3, spin=spin,
                                              lmax=comp_list[icomp].lmax)
        comp_list[icomp].alms = comp_alms.astype(complex_dtype, copy=False)
        # These amplitudes are component maps at the data resolution; record that beam so the
        # forward model applies no extra smoothing when predicting a band at this resolution.
        comp_list[icomp].amp_fwhm_rad = detector_data.fwhm_rad

    return comp_list
