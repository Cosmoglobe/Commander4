import numpy as np
import os, sys
import healpy as hp
sys.path.append("../")
from data_models import DetectorTOD, DetectorMap

def single_det_mapmaker_python(det_static: DetectorTOD, det_cs_map: np.array) -> DetectorMap:
    """ From a single detector object, which contains a list of Scans, calculate signal and rms map.s
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
    detmap_rms =  1.0/np.sqrt(detmap_inv_var)
    detmap_signal /= detmap_inv_var
    return detmap_signal, detmap_rms