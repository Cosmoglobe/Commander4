import time
import healpy as hp
import ctypes
import logging
import numpy as np
from mpi4py import MPI
from pixell import curvedsky
from pixell.bunch import Bunch
from numpy.typing import NDArray

from commander4.output.log import logassert
from commander4.sky_models.component import Component
from commander4.utils.ctypes_lib import load_cmdr4_ctypes_lib
from commander4.data_models.detector_map import DetectorMap


def smooth_signal_map_noiseweighted(map_signal: NDArray, map_rms: NDArray, fwhm_rad: float):
    """ Smooths a signal map with noise weighting.
    """
    # Weight map is 1 / variance
    map_inv_var = 1.0 / (map_rms**2) 
    
    # Normalize the weight:
    smoothed_weight = hp.smoothing(map_inv_var, fwhm=fwhm_rad)
    
    # Multiply signal by weight, smooth, and divide by smoothed weight
    unnormalized_smooth_signal = hp.smoothing(map_signal * map_inv_var, fwhm=fwhm_rad)
    
    return unnormalized_smooth_signal / smoothed_weight


def smooth_rms_map_noiseweighted(rms_map: NDArray, fwhm_rad: float):
    """ Calculates what the per-pixel RMS is for any signal map that has beem smoothed by a
        Gaussian beam using inverse-variance noise weighted smoothing. I.e. produces the correct RMS
        for the function `smooth_signal_map_noiseweighted`.
    """
    npix = rms_map.shape[0]
    nside = hp.npix2nside(npix)
    smoothed_inv_var = hp.smoothing(1.0/rms_map**2, fwhm=fwhm_rad)

    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1)

    # Retrieve the beam and pixel window harmonic coefficients
    b_ell = hp.gauss_beam(fwhm_rad, lmax=lmax)
    p_ell = hp.pixwin(nside, lmax=lmax)

    # 2. Calculate the solid angle of a single pixel (C_ell for unit white noise)
    omega_pix = hp.nside2resol(nside)**2 

    # 3. Compute the exact harmonic variance of the band-limited, windowed noise
    true_empirical_norm = np.sum((2 * ell + 1) / (4 * np.pi) * omega_pix * (p_ell**2) * (b_ell**2))

    # Scale FWHM for the squared Gaussian kernel
    fwhm_rad_sq = fwhm_rad / np.sqrt(2.0)

    # Noise-Weighted Analytical: smooth weight map with squared beam, divide by (smoothed_weight)^2
    numerator = hp.smoothing(1.0/rms_map**2, fwhm=fwhm_rad_sq) * true_empirical_norm
    analytical_variance = numerator / (smoothed_inv_var**2)

    return np.sqrt(analytical_variance)


def solve_compsep_perpix(proc_comm: MPI.Comm, detector_data: DetectorMap,
                         comp_list: list[Component], params: Bunch) -> list[Component]:
    """ A pixel-by-pixel solver for the component separation problem. Requires uniform nside, unlike
        the CG solver. Also requires common beam smoothing, but handles this by smoothing all maps
        to the lowest resolution map.
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
    map_sky = detector_data.map_sky.copy()  # Make copy so we don't overwrite if we are smoothing.
    band_freq = detector_data.nu
    map_rms = detector_data.map_rms.copy() 
    ctypes_lib = load_cmdr4_ctypes_lib()
    ctypes_lib.solve_compsep.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), # map_sky
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), # map_rms
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), # M
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), # rnd
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), # map_out
    ]

    if params.general.smooth_to_common_res:
        fwhm = detector_data.fwhm
        all_fwhm = proc_comm.allgather(fwhm)
        max_fwhm = np.max(all_fwhm)
        my_smoothing_fwhm = np.sqrt(max_fwhm**2 - fwhm**2)
        logger.info(f"{detector_data.nu} GHz map with FWHM = {fwhm:.1f} arcmin will be smoothed by"\
                    f" {my_smoothing_fwhm:.1f} arcmin to reach {max_fwhm:.1f} arcmin.")
        if params.general.smooth_to_common_res:
            fwhm_rad = np.deg2rad(my_smoothing_fwhm/60.0)
            for ipol in range(map_sky.shape[0]):  # Loop over 1 or 2 polarizations.
                map_sky[ipol] = smooth_signal_map_noiseweighted(map_sky[ipol], map_rms[ipol],
                                                                fwhm_rad)
                map_rms[ipol] = smooth_rms_map_noiseweighted(map_rms[ipol], fwhm_rad)

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
            idx = 0
            for i in range(ncomp):
                # if ipol == 0 or comp_list[i].polarized:
                M[:,idx] = comp_list[i].get_sed(freqs)
                idx += 1
            rand = np.random.randn(npix,nband)
            # TODO: Write unit tests that confirm Python and C gives same answers.
            # TODO: Should scale M to make solution more well-conditioned, and then adjust
            # solution with the scaling factor used.
            ctypes_lib.solve_compsep(npix, nband, ncomp, maps_sky.astype(np.float64, copy=False),
                                  maps_rms.astype(np.float64, copy=False), M, rand, comp_maps[ipol])
            logger.info(f"Finished pixel-by-pixel component separation in {time.time()-t0:.2f}s "\
                        f"for polarization {ipol+1} of 3.")

    comp_maps = proc_comm.bcast(comp_maps, root=0)
    for icomp in range(ncomp):
        if pol:
            input_map = np.array([comp_maps[0][icomp], comp_maps[1][icomp]], dtype=real_dtype)
        else:
            input_map = np.array([comp_maps[0][icomp]], dtype=real_dtype)
        comp_alms = curvedsky.map2alm_healpix(input_map, niter=3, spin=spin,
                                              lmax=comp_list[icomp].lmax)
        comp_list[icomp].alms = comp_alms.astype(complex_dtype, copy=False)

    return comp_list
