import numpy as np
import os
import healpy as hp
import ctypes as ct
from data_models import DetectorTOD, DetectorMap
import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft

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


def single_det_map_accumulator(det_static: DetectorTOD, det_cs_map: np.array) -> tuple[np.array, np.array]:
    """ From a single detector object, which contains a list of Scans, calculate a weighted (BUT UNNORMALIZED) signal map, and an inverse variance map.
        The purpose of this function is to be called multiple times, such that both the unnormalized signal map and inv-var maps can be further accumulated and normalized later.
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

        f_samp = 180.0
        freq = np.fft.rfftfreq(ntod, d = 1/f_samp)
        fknee = 1.0
        alpha = -1.0
        N = freq.shape[0]

        C_wn = sigma0*np.ones(N)
        C_1f_inv = np.zeros(N)  # 1.0/C_1f
        C_1f_inv[1:] = freq[1:]/sigma0

        const = 1.0  # This is the normalization constant for FFT, which I'm unsure what is for scipys FFT, might be wrong!
        w1 = (np.random.normal(0, 1, N) + 1.j*np.random.normal(0, 1, N))/np.sqrt(2)
        w2 = (np.random.normal(0, 1, N) + 1.j*np.random.normal(0, 1, N))/np.sqrt(2)
        # I'm always a bit confused about when it's fine to use rfft as opposed to full fft, so might want to double check this:
        n_corr_est_fft_WF = rfft(sky_subtracted_tod)/(1 + C_wn*C_1f_inv)
        n_corr_est_fft_fluct = (const*(np.sqrt(C_wn)*w1 + C_wn*np.sqrt(C_1f_inv)*w2))/(1 + C_wn*C_1f_inv)
        n_corr_est_fft = n_corr_est_fft_WF + n_corr_est_fft_fluct
        n_corr_est_fft[0] = 0.0
        n_corr_est_fft_WF[0] = 0.0
        n_corr_est_WF = irfft(n_corr_est_fft_WF, n=sky_subtracted_tod.shape[0])
        n_corr_est = irfft(n_corr_est_fft, n=sky_subtracted_tod.shape[0])


        # if scan.scanID < 10:
        #     plt.figure(figsize=(20,10))
        #     plt.plot(det_cs_map[pix], label="observed sky")
        #     # plt.savefig(f"../../plots/scan_{scan.scanID}_skymap.png")
        #     # plt.close()
        #     # plt.figure()
        #     plt.plot(sky_subtracted_tod, label="sky subtracted TOD")
        #     # plt.savefig(f"../../plots/scan_{scan.scanID}_skysub.png")
        #     # plt.close()
        #     # plt.figure()
        #     plt.plot(n_corr_est, label="Ncorr realization")
        #     plt.plot(n_corr_est_WF, label="Ncorr WF")
        #     plt.legend()
        #     plt.savefig(f"../../plots/scan_{scan.scanID}.png")
        #     plt.close()

        sky_subtracted_tod -= n_corr_est

        inv_var = 1.0/sigma0**2
        # detmap_signal += np.bincount(pix, weights=scan_map/sigma0**2, minlength=npix)
        # detmap_inv_var += np.bincount(pix, minlength=npix)/sigma0**2
        maplib.map_weight_accumulator(detmap_inv_var, inv_var, pix, ntod, npix)
        maplib.map_accumulator(detmap_signal, scan_map, inv_var, pix, ntod, npix)

    return detmap_signal, detmap_inv_var
