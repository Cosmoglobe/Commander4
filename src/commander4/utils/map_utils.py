import numpy as np
import pysm3.units as pysm3_u
from commander4.data_models.scan_TOD import ScanTOD
from commander4.data_models.detector_TOD import DetectorTOD
from numpy.typing import NDArray
import ducc0
import os
from numba import njit

T_CMB = 2.725 * 1e6  # CMB temperature in uK_CMB units.
C = 299792458  # m/s (Speed of light)
T_CMB_div_C = T_CMB / C
# Precomputing the conversion factor from 1 uK_CMB to 1 uK_RJ
uK_CMB_to_uK_RJ_dict = {}


@njit(fastmath=True)
def get_static_sky_TOD(det_compsep_map: NDArray[np.floating], pix: NDArray[np.integer], psi: NDArray[np.floating]) -> DetectorTOD:
    """ Projects the current sky-model at our band frequency (in uK_RJ, without gain) into the
        specified scan pointing. The sky model does not include the orbital dipole.
    """
    sky = det_compsep_map[0, pix] + np.cos(2*psi)*det_compsep_map[1, pix]\
        + np.sin(2*psi)*det_compsep_map[2, pix]
    return sky.astype(np.float32)


def get_s_orb_TOD(scan: ScanTOD, experiment: DetectorTOD, pix: NDArray[np.integer],
                  nthreads:int = None) -> NDArray:
    # If nthreads is not set, put it to how many threads OMP has.
    nthreads = int(os.environ["OMP_NUM_THREADS"]) if nthreads is None else nthreads
    if experiment.nu not in uK_CMB_to_uK_RJ_dict:
        uK_CMB_to_uK_RJ_dict[experiment.nu] = (1*pysm3_u.uK_CMB).to(pysm3_u.uK_RJ,
                        equivalencies=pysm3_u.cmb_equivalencies(experiment.nu*pysm3_u.GHz)).value
    geom = ducc0.healpix.Healpix_Base(experiment.nside, "RING")
    LOS_vec = geom.pix2vec(pix, nthreads=nthreads)
    LOS_vec *= scan.orb_dir_vec
    s_orb = np.sum(LOS_vec, axis=-1, dtype=np.float32)  # How much do the LOS and orbital velocity align?
    s_orb *= T_CMB_div_C
    s_orb *= uK_CMB_to_uK_RJ_dict[experiment.nu]  # Converting to uK_RJ units.
    return s_orb.astype(np.float32)