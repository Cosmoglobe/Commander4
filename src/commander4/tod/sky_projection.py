"""Evaluating the modelled signal along a detector-scan's pointing.

Two signals are projected here: the static sky model (C3's `s_sky`) and the orbital dipole induced
by the observer's motion (C3's `s_orb`). Both turn a sky-domain quantity into a TOD, and both are
numba kernels because they run once per detector-scan per Gibbs iteration.
"""
import logging
import os

import ducc0
import numpy as np
import pysm3.units as pysm3_u
from numba import njit
from numpy.typing import NDArray

from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.detector_group_tod import DetectorGroupTOD

logger = logging.getLogger(__name__)

T_CMB = 2.725 * 1e6  # CMB temperature in uK_CMB units.
C = 299792458  # m/s (Speed of light)
T_CMB_div_C = T_CMB / C
# Precomputing the conversion factor from 1 uK_CMB to 1 uK_RJ
uK_CMB_to_uK_RJ_dict = {}


def get_static_sky_tod(det_compsep_map: NDArray[np.floating], pix: NDArray[np.integer],
                       psi: NDArray[np.floating]|None = None) -> NDArray[np.floating]:
    """ Projects the current sky-model at our band frequency (in uK_RJ, without gain) into the
        specified scan pointing. The sky model does not include the orbital dipole.
    """
    if psi is None:
        return _get_static_sky_tod_I(det_compsep_map, pix)
    elif det_compsep_map.shape[0] == 2:
        return _get_static_sky_tod_QU(det_compsep_map, pix, psi)
    elif det_compsep_map.shape[0] == 3:
        return _get_static_sky_tod_IQU(det_compsep_map, pix, psi)
    else:
        raise ValueError("Input compsep map has mismatching dimensions.")

@njit(fastmath=True)
def _get_static_sky_tod_IQU(det_compsep_map: NDArray[np.floating], pix: NDArray[np.integer],
                       psi: NDArray[np.floating]) -> NDArray[np.float32]:
    sky = np.empty(pix.shape[0], dtype=np.float32)
    for i in range(pix.shape[0]):
        p = pix[i]
        angle = 2.0 * psi[i]
        sky[i] = det_compsep_map[0, p] + np.cos(angle)*det_compsep_map[1, p]\
               + np.sin(angle)*det_compsep_map[2, p]                 
    return sky

@njit(fastmath=True)
def _get_static_sky_tod_QU(det_compsep_map: NDArray[np.floating], pix: NDArray[np.integer],
                       psi: NDArray[np.floating]) -> NDArray[np.float32]:
    sky = np.empty(pix.shape[0], dtype=np.float32)
    for i in range(pix.shape[0]):
        p = pix[i]
        angle = 2.0 * psi[i]
        sky[i] = np.cos(angle)*det_compsep_map[0, p] + np.sin(angle)*det_compsep_map[1, p]                 
    return sky

@njit(fastmath=True)
def _get_static_sky_tod_I(det_compsep_map: NDArray[np.floating], pix: NDArray[np.integer]
                          ) -> NDArray[np.float32]:
    sky = np.empty(pix.shape[0], dtype=np.float32)
    for i in range(pix.shape[0]):
        sky[i] = det_compsep_map[0, pix[i]]
    return sky


def get_s_orb_tod(det: DetectorTOD, experiment: DetectorGroupTOD, pix: NDArray[np.integer],
                  nthreads:int = None) -> NDArray:
    """ Compute the orbital dipole contribution to the TOD for a single detector.

    Projects the CMB dipole induced by the satellite's orbital motion into the
    detector pointing, returning a TOD-length array in uK_RJ units.

    Args:
        det (DetectorTOD): Single-detector TOD data (provides orbital velocity direction).
        experiment (DetectorGroupTOD): Experiment-level data (provides nu and nside).
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
