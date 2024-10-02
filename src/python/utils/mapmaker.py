import numpy as np
import os
import healpy as hp
import ctypes as ct
from data_models import DetectorTOD, DetectorMap

current_dir_path = os.path.dirname(os.path.realpath(__file__))
src_dir_path = os.path.abspath(os.path.join(os.path.join(current_dir_path, os.pardir), os.pardir))

def single_det_mapmaker_python(det_static: DetectorTOD, det_cs_map: np.array) -> tuple[np.array, np.array]:
    """ From a single detector object, which contains a list of Scans, calculate signal and rms map.
    """
    npix = det_cs_map.shape[-1]
    nside = hp.npix2nside(npix)
    detmap_signal = np.zeros(npix)
    detmap_inv_var = np.zeros(npix)
    for scan in det_static.scans:
        scan_map, theta, phi, psi = scan.data
        pix = hp.ang2pix(nside, theta, phi)
        sky_subtracted_tod = det_cs_map[pix] - scan_map
        sigma0 = np.std(sky_subtracted_tod[1:] - sky_subtracted_tod[:-1])/np.sqrt(2)
        detmap_signal += np.bincount(pix, weights=scan_map/sigma0**2, minlength=npix)
        detmap_inv_var += np.bincount(pix, minlength=npix)/sigma0**2
    detmap_rms = np.zeros(npix) + np.inf
    detmap_rms[detmap_signal != 0] = 1.0/np.sqrt(detmap_inv_var[detmap_signal != 0])
    detmap_signal[detmap_signal != 0] /= detmap_inv_var[detmap_signal != 0]
    return detmap_signal, detmap_rms


def single_det_mapmaker(det_static: DetectorTOD, det_cs_map: np.array) -> tuple[np.array, np.array]:
    """ From a single detector object, which contains a list of Scans, calculate signal and rms map.
    """
    npix = det_cs_map.shape[-1]
    nside = hp.npix2nside(npix)
    detmap_signal = np.zeros(npix)
    detmap_inv_var = np.zeros(npix)

    maplib = ct.cdll.LoadLibrary(os.path.join(src_dir_path, "cpp/mapmaker.so"))
    ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_int64, ndim=1, flags="contiguous")
    ct_f64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=1, flags="contiguous")
    maplib.map_weight_accumulator.argtypes = [ct_f64_dim1, ct.c_double, ct_i64_dim1, ct.c_int64, ct.c_int64]
    maplib.map_accumulator.argtypes = [ct_f64_dim1, ct_f64_dim1, ct.c_double, ct_i64_dim1, ct.c_int64, ct.c_int64]

    for scan in det_static.scans:
        scan_map, theta, phi, psi = scan.data
        ntod = scan_map.shape[0]
        pix = hp.ang2pix(nside, theta, phi)
        sky_subtracted_tod = det_cs_map[pix] - scan_map
        sigma0 = np.std(sky_subtracted_tod[1:] - sky_subtracted_tod[:-1])/np.sqrt(2)
        inv_var = 1.0/sigma0**2
        # detmap_signal += np.bincount(pix, weights=scan_map/sigma0**2, minlength=npix)
        # detmap_inv_var += np.bincount(pix, minlength=npix)/sigma0**2
        maplib.map_weight_accumulator(detmap_inv_var, inv_var, pix, ntod, npix)
        maplib.map_accumulator(detmap_signal, scan_map, inv_var, pix, ntod, npix)

    detmap_rms = np.zeros(npix) + np.inf
    detmap_rms[detmap_signal != 0] = 1.0/np.sqrt(detmap_inv_var[detmap_signal != 0])
    detmap_signal[detmap_signal != 0] /= detmap_inv_var[detmap_signal != 0]
    return detmap_signal, detmap_rms