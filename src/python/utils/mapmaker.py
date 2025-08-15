import numpy as np
import os
import healpy as hp
import ctypes as ct
from pixell import bunch
import pysm3.units as pysm3_u

from src.python.data_models.detector_TOD import DetectorTOD
from src.python.utils.map_utils import get_sky_model_TOD, calculate_s_orb

current_dir_path = os.path.dirname(os.path.realpath(__file__))
src_dir_path = os.path.abspath(os.path.join(os.path.join(current_dir_path, os.pardir), os.pardir))

def single_det_mapmaker_python(det_static: DetectorTOD, det_cs_map: np.array) -> tuple[np.array, np.array]:
    """ From a single detector object, which contains a list of Scans, calculate signal and rms map.
    """
    npix = det_cs_map.shape[-1]
    detmap_signal = np.zeros(npix)
    detmap_inv_var = np.zeros(npix)
    for scan in det_static.scans:
        raw_tod = scan.tod
        pix = scan.pix
        sky_subtracted_tod = det_cs_map[pix] - raw_tod
        sigma0 = np.std(sky_subtracted_tod[1:] - sky_subtracted_tod[:-1])/np.sqrt(2)
        detmap_signal += np.bincount(pix, weights=raw_tod/sigma0**2, minlength=npix)
        detmap_inv_var += np.bincount(pix, minlength=npix)/sigma0**2
    detmap_rms = np.zeros(npix) + np.inf
    detmap_rms[detmap_signal != 0] = 1.0/np.sqrt(detmap_inv_var[detmap_signal != 0])
    detmap_signal[detmap_signal != 0] /= detmap_inv_var[detmap_signal != 0]
    return detmap_signal, detmap_rms


def single_det_mapmaker(det_static: DetectorTOD, det_cs_map: np.array) -> tuple[np.array, np.array]:
    """ From a single detector object, which contains a list of Scans, calculate signal and rms map.
    """
    npix = det_cs_map.shape[-1]
    detmap_signal = np.zeros(npix)
    detmap_inv_var = np.zeros(npix)

    maplib = ct.cdll.LoadLibrary(os.path.join(src_dir_path, "cpp/mapmaker.so"))
    ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_int64, ndim=1, flags="contiguous")
    ct_f64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=1, flags="contiguous")
    maplib.map_weight_accumulator.argtypes = [ct_f64_dim1, ct.c_double, ct_i64_dim1, ct.c_int64, ct.c_int64]
    maplib.map_accumulator.argtypes = [ct_f64_dim1, ct_f64_dim1, ct.c_double, ct_i64_dim1, ct.c_int64, ct.c_int64]

    for scan in det_static.scans:
        raw_tod = scan.tod
        pix = scan.pix
        ntod = raw_tod.shape[0]
        sky_subtracted_tod = det_cs_map[pix] - raw_tod
        sigma0 = np.std(sky_subtracted_tod[1:] - sky_subtracted_tod[:-1])/np.sqrt(2)
        inv_var = 1.0/sigma0**2
        # detmap_signal += np.bincount(pix, weights=scan_map/sigma0**2, minlength=npix)
        # detmap_inv_var += np.bincount(pix, minlength=npix)/sigma0**2
        maplib.map_weight_accumulator(detmap_inv_var, inv_var, pix.astype(np.int64), ntod, npix)
        maplib.map_accumulator(detmap_signal, raw_tod, inv_var, pix.astype(np.int64), ntod, npix)

    detmap_rms = np.zeros(npix) + np.inf
    detmap_rms[detmap_signal != 0] = 1.0/np.sqrt(detmap_inv_var[detmap_signal != 0])
    detmap_signal[detmap_signal != 0] /= detmap_inv_var[detmap_signal != 0]
    return detmap_signal, detmap_rms


def single_det_map_accumulator(det_static: DetectorTOD, det_cs_map: np.array, sample_params, params: bunch) -> tuple[np.array, np.array]:
    """ From a single detector object, which contains a list of Scans, calculate a weighted (BUT UNNORMALIZED) signal map, and an inverse variance map.
        The purpose of this function is to be called multiple times, such that both the unnormalized signal map and inv-var maps can be further accumulated and normalized later.

        This is where we will be writing the polarization mapmaker.
    """
    npix = det_cs_map.shape[-1]
    detmap_corr_noise = np.zeros(npix)  # Healpix map holding the accumulated correlated noise realizations.
    detmap_rawobs = np.zeros(npix)  # Healpix map holding the accumulated sky signal map.
    detmap_orbdipole = np.zeros(npix)  # Healpix map holding the accumulated sky signal map.
    detmap_skysub = np.zeros(npix)  # Healpix map holding the accumulated sky signal map.
    detmap_signal = np.zeros(npix)  # Healpix map holding the accumulated sky signal map.
    detmap_inv_var = np.zeros(npix)  # Healpix map holding the accumulated inverse variance.
    detmap_hits = np.zeros(npix)  # Healpix map holding the accumulated inverse variance.

    maplib = ct.cdll.LoadLibrary(os.path.join(src_dir_path, "cpp/mapmaker.so"))
    ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_int64, ndim=1, flags="contiguous")
    ct_f64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=1, flags="contiguous")
    maplib.map_weight_accumulator.argtypes = [ct_f64_dim1, ct.c_double, ct_i64_dim1, ct.c_int64, ct.c_int64]
    maplib.map_accumulator.argtypes = [ct_f64_dim1, ct_f64_dim1, ct.c_double, ct_i64_dim1, ct.c_int64, ct.c_int64]

    for scan, scanparams in zip(det_static.scans, sample_params.scans):
        raw_tod = scan.tod
        pix = scan.pix
        ntod = raw_tod.shape[0]
        s_orb = calculate_s_orb(scan, det_static)
        inv_var = 1.0/scanparams.sigma0**2
        gain = scanparams.gain_est
        sky_subtracted_TOD = raw_tod - gain*get_sky_model_TOD(scan, det_cs_map)
        # detmap_signal += np.bincount(pix, weights=scan_map/sigma0**2, minlength=npix)
        # detmap_inv_var += np.bincount(pix, minlength=npix)/sigma0**2
        maplib.map_weight_accumulator(detmap_hits, 1.0, pix.astype(np.int64), ntod, npix)
        maplib.map_weight_accumulator(detmap_inv_var, (inv_var).astype(np.float64), pix.astype(np.int64), ntod, npix)
        maplib.map_accumulator(detmap_rawobs, (raw_tod/gain).astype(np.float64), inv_var, pix.astype(np.int64), ntod, npix)
        maplib.map_accumulator(detmap_signal, ((raw_tod - scanparams.n_corr_est)/gain - s_orb).astype(np.float64), inv_var, pix.astype(np.int64), ntod, npix)
        maplib.map_accumulator(detmap_orbdipole, s_orb.astype(np.float64), 1.0, pix.astype(np.int64), ntod, npix)
        maplib.map_accumulator(detmap_skysub, (sky_subtracted_TOD/gain).astype(np.float64), inv_var, pix.astype(np.int64), ntod, npix)
        if params.sample_corr_noise:
            maplib.map_accumulator(detmap_corr_noise, (scanparams.n_corr_est/gain).astype(np.float64), 1.0, pix.astype(np.int64), ntod, npix)

    return detmap_rawobs, detmap_signal, detmap_orbdipole, detmap_skysub, detmap_corr_noise, detmap_inv_var, detmap_hits

def single_det_map_accumulator_IQU(det_static: DetectorTOD, det_cs_map: np.array, params: bunch) -> tuple[np.array, np.array]:
    """ From a single detector object, which contains a list of Scans, calculate a weighted (BUT UNNORMALIZED) signal map, and an inverse variance map.
        The purpose of this function is to be called multiple times, such that both the unnormalized signal map and inv-var maps can be further accumulated and normalized later.

        This is where we will be writing the polarization mapmaker.
    """
    test_polang_coverage = False

    npix = det_cs_map.shape[-1]
    detmap_corr_noise = np.zeros((3,npix))  # Healpix map holding the accumulated correlated noise realizations.
    detmap_rawobs = np.zeros((3,npix))  # Healpix map holding the accumulated sky signal map.
    detmap_orbdipole = np.zeros((3,npix))  # Healpix map holding the accumulated sky signal map.
    detmap_skysub = np.zeros((3,npix))  # Healpix map holding the accumulated sky signal map.
    detmap_signal = np.zeros((3,npix))  # Healpix map holding the accumulated sky signal map.
    detmap_inv_var = np.zeros((6,npix))  # Healpix map holding the accumulated inverse variance.

    maplib = ct.cdll.LoadLibrary(os.path.join(src_dir_path, "cpp/mapmaker.so"))
    ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_int64, ndim=1, flags="contiguous")
    ct_f64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=1, flags="contiguous")
    ct_f64_dim2 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=2, flags="contiguous")
    maplib.map_weight_accumulator_IQU.argtypes = [ct_f64_dim2, ct.c_double,
            ct_i64_dim1, ct_f64_dim1, ct.c_int64, ct.c_int64]
    maplib.map_accumulator_IQU.argtypes = [ct_f64_dim2, ct_f64_dim1,
            ct.c_double, ct_i64_dim1, ct_f64_dim1, ct.c_int64, ct.c_int64]

    if test_polang_coverage:
        sum_cos2psi = np.zeros(npix)
        sum_sin2psi = np.zeros(npix)
        hits_map = np.zeros(npix)

    for scan in det_static.scans:
        raw_tod = scan.tod
        pix = scan.pix
        psi = scan.psi

        ntod = raw_tod.shape[0]
        inv_var = 1.0/scan.sigma0**2
        maplib.map_weight_accumulator_IQU(detmap_inv_var,
                (inv_var).astype(np.float64), pix.astype(np.int64), psi.astype(np.float64), ntod, npix)
        maplib.map_accumulator_IQU(detmap_rawobs,
                (raw_tod/scan.gain_est).astype(np.float64), inv_var, pix.astype(np.int64),
                psi.astype(np.float64), ntod, npix)
        maplib.map_accumulator_IQU(detmap_signal, ((raw_tod - scan.n_corr_est -
            scan.orbital_dipole)/scan.gain_est).astype(np.float64), inv_var, pix.astype(np.int64),
            psi.astype(np.float64), ntod, npix)
        maplib.map_accumulator_IQU(detmap_orbdipole,
                (scan.orbital_dipole/scan.gain_est).astype(np.float64), inv_var,
                pix.astype(np.int64), psi.astype(np.float64), ntod, npix)
        maplib.map_accumulator_IQU(detmap_skysub,
                (scan.sky_subtracted_tod/scan.gain_est).astype(np.float64),
                inv_var, pix.astype(np.int64), psi.astype(np.float64), ntod, npix)
        if params.sample_corr_noise:
            maplib.map_accumulator_IQU(detmap_corr_noise,
                    (scan.n_corr_est).astype(np.float64), inv_var, pix.astype(np.int64),
                    psi.astype(np.float64), ntod, npix)

        
        if test_polang_coverage:
            sum_cos2psi += np.bincount(pix, weights=np.cos(2*psi), minlength=npix)
            sum_sin2psi += np.bincount(pix, weights=np.sin(2*psi), minlength=npix)
            hits_map   += np.bincount(pix, minlength=npix)

    if test_polang_coverage:
        R = (sum_cos2psi/hits_map)**2 + (sum_sin2psi/hits_map)**2
        import matplotlib.pyplot as plt
        hp.mollview(R, norm='hist', title='R')
        plt.show()

    return detmap_rawobs, detmap_signal, detmap_orbdipole, detmap_skysub, detmap_corr_noise, detmap_inv_var
