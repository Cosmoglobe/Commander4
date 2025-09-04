import numpy as np
import healpy as hp
import pysm3.units as pysm3_u
from src.python.data_models.scan_TOD import ScanTOD
from src.python.data_models.detector_TOD import DetectorTOD
from numpy.typing import NDArray


def get_static_sky_TOD(det_compsep_map: NDArray[np.floating], pix: NDArray[np.integer], psi: NDArray[np.floating]) -> DetectorTOD:
    """ Projects the current sky-model at our band frequency (in uK_RJ, without gain) into the
        specified scan pointing. The sky model does not include the orbital dipole.
    """
    sky = det_compsep_map[0, pix] + np.cos(2*psi)*det_compsep_map[1, pix]\
        + np.sin(2*psi)*det_compsep_map[2, pix]
    return sky


def get_s_orb_TOD(scan: ScanTOD, experiment: DetectorTOD, pix: NDArray[np.integer]) -> NDArray:
    T_CMB = 2.725 * 1e6  # CMB temperature in uK_CMB units.
    C = 299792458  # m/s (Speed of light)
    # Precomputing the conversion factor from 1 uK_CMB to 1 uK_RJ (not that this conversion is only valid for temperatures close to the CMB).
    uK_CMB_to_uK_RJ = (1*pysm3_u.uK_CMB).to(pysm3_u.uK_RJ, equivalencies=pysm3_u.cmb_equivalencies(experiment.nu*pysm3_u.GHz)).value

    theta, phi = hp.pix2ang(experiment.nside, pix)
    LOS_vec = hp.ang2vec(theta, phi).astype(np.float32)
    dot_product = np.sum(scan.orb_dir_vec * LOS_vec, axis=-1, dtype=np.float32)  # How much do the LOS and orbital velocity align?
    s_orb = T_CMB * dot_product / C  # The orbital dipole in units of uK_CMB.
    s_orb = s_orb * uK_CMB_to_uK_RJ  # Converting to uK_RJ units.
    return s_orb