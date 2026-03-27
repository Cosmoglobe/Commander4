import numpy as np
import pysm3.units as pysm3_u
from commander4.data_models.scan_TOD import ScanTOD
from commander4.data_models.detector_TOD import DetectorTOD
from commander4.data_models.detector_group_TOD import DetGroupTOD
from commander4.output import log
from numpy.typing import NDArray
import ducc0
import logging
import os
from numba import njit

POLS_DICT = {"I":1, "QU":2, "IQU":3} #more allowed in the future.
T_CMB = 2.725 * 1e6  # CMB temperature in uK_CMB units.
C = 299792458  # m/s (Speed of light)
T_CMB_div_C = T_CMB / C
# Precomputing the conversion factor from 1 uK_CMB to 1 uK_RJ
uK_CMB_to_uK_RJ_dict = {}

def get_static_sky_TOD(det_compsep_map: NDArray[np.floating], pix: NDArray[np.integer],
                       psi: NDArray[np.floating]|None = None) -> NDArray[np.floating]:
    """ Projects the current sky-model at our band frequency (in uK_RJ, without gain) into the
        specified scan pointing. The sky model does not include the orbital dipole.
    """
    if psi is None:
        return _get_static_sky_TOD_I(det_compsep_map, pix)
    elif det_compsep_map.shape[0] == 2:
        return _get_static_sky_TOD_QU(det_compsep_map, pix, psi)
    elif det_compsep_map.shape[0] == 3:
        return _get_static_sky_TOD_IQU(det_compsep_map, pix, psi)
    else:
        raise ValueError("Input compsep map has mismatching dimensions.")

@njit(fastmath=True)
def _get_static_sky_TOD_IQU(det_compsep_map: NDArray[np.floating], pix: NDArray[np.integer],
                       psi: NDArray[np.floating]) -> NDArray[np.floating]:
    """ Projects the current sky-model at our band frequency (in uK_RJ, without gain) into the
        specified scan pointing. The sky model does not include the orbital dipole.
    """
    sky = det_compsep_map[0, pix] + np.cos(2*psi)*det_compsep_map[1, pix] \
    + np.sin(2*psi)*det_compsep_map[2, pix]
    return sky.astype(np.float32, copy=False)

@njit(fastmath=True)
def _get_static_sky_TOD_QU(det_compsep_map: NDArray[np.floating], pix: NDArray[np.integer],
                       psi: NDArray[np.floating]) -> NDArray[np.floating]:
    """ Projects the current sky-model at our band frequency (in uK_RJ, without gain) into the
        specified scan pointing. The sky model does not include the orbital dipole.
    """
    sky = np.cos(2*psi)*det_compsep_map[0, pix] \
    + np.sin(2*psi)*det_compsep_map[1, pix]
    return sky.astype(np.float32, copy=False)

@njit(fastmath=True)
def _get_static_sky_TOD_I(det_compsep_map: NDArray[np.floating], pix: NDArray[np.integer]
                          ) -> NDArray[np.floating]:
    """ Projects the current sky-model at our band frequency (in uK_RJ, without gain) into the
        specified scan pointing. The sky model does not include the orbital dipole.
    """
    sky = det_compsep_map[0, pix]
    return sky.astype(np.float32, copy=False)

def get_s_orb_TOD(det: DetectorTOD, experiment: DetGroupTOD, pix: NDArray[np.integer],
                  nthreads:int = None) -> NDArray:
    """ Compute the orbital dipole contribution to the TOD for a single detector.

    Projects the CMB dipole induced by the satellite's orbital motion into the
    detector pointing, returning a TOD-length array in uK_RJ units.

    Args:
        det (DetectorTOD): Single-detector TOD data (provides orbital velocity direction).
        experiment (DetGroupTOD): Experiment-level data (provides nu and nside).
        pix (NDArray[np.integer]): Decompressed pixel indices for this detector.
        nthreads (int, optional): Number of threads for HEALPix operations.
            Defaults to the OMP_NUM_THREADS environment variable.

    Returns:
        NDArray: Orbital dipole signal in uK_RJ, shape ``(npix,)``.
    """
    # If nthreads is not set, put it to how many threads OMP has.
    nthreads = int(os.environ["OMP_NUM_THREADS"]) if nthreads is None else nthreads
    if experiment.nu not in uK_CMB_to_uK_RJ_dict:
        uK_CMB_to_uK_RJ_dict[experiment.nu] = (1*pysm3_u.uK_CMB).to(pysm3_u.uK_RJ,
                        equivalencies=pysm3_u.cmb_equivalencies(experiment.nu*pysm3_u.GHz)).value
    geom = ducc0.healpix.Healpix_Base(experiment.nside, "RING")
    LOS_vec = geom.pix2vec(pix, nthreads=nthreads)
    if det.orb_dir_vec is not None:
        LOS_vec *= det.orb_dir_vec
    # How much do the LOS and orbital velocity align?
    s_orb = np.sum(LOS_vec, axis=-1, dtype=np.float32)
    s_orb *= T_CMB_div_C
    s_orb *= uK_CMB_to_uK_RJ_dict[experiment.nu]  # Converting to uK_RJ units.
    return s_orb.astype(np.float32, copy=False)

def fwhm2sigma(fwhm):
    return fwhm/(2*np.sqrt(2*np.log(2)))

def gauss_beam(x, fwhm):
    """
    Gaussian integral-normalized beam in map space.
    Be CAREFUL giving `x` and `fwhm` in the same units of measure. 
    """
    sigma = fwhm2sigma(fwhm)
    return 1/(2*np.pi*sigma**2)*np.exp(-(x)**2 / (2*sigma**2))

def get_gauss_beam_radius(fwhm, frac=1e-4):
    """
    Finds the distance from the center of a gaussian beam corresponding to a fraction 'frac'
    of the peak intensity.
    """
    sigma = fwhm2sigma(fwhm)
    return sigma * np.sqrt( - 2* np.log(frac))

def get_npol(pols:str):
    """
    Return the number of map polarizaiton components given the polarization string `pols`.
    """
    logger = logging.getLogger(__name__)
    log.logassert(pols in POLS_DICT, "Unrecognised polarization string", logger)
    return POLS_DICT[pols]
    
def is_pol_supported(pols:str):
    """
    Checks if the given polarization string `pols` is matching one of the supported pol configs.
    """
    if pols in POLS_DICT.keys():
        return True
    else:
        return False

def assert_pol_supported(pols:str):
    """
    Asserts if the given polarization string `pols` is matching one of the supported pol configs.
    """
    log.logassert(is_pol_supported(pols), 
                  f"Unsupported polarization string {pols}", 
                  logging.getLogger(__name__))