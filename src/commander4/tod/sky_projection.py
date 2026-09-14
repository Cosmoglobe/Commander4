"""Evaluating the modelled signal along a detector-scan's pointing.

Two signals are projected here: the static sky model (C3's `s_sky`) and the orbital dipole induced
by the observer's motion (C3's `s_orb`). Both turn a sky-domain quantity into a TOD, and both are
numba kernels because they run once per detector-scan per Gibbs iteration. The sky-projection
kernels are threaded over samples (`prange`): the map is read-only, so each sample is independent,
and every gain step plus the mapmaker re-evaluates them for every detector-scan.
"""
import logging
import os

import ducc0
import numpy as np
import pysm3.units as pysm3_u
from numba import njit, prange
from numpy.typing import NDArray

from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.detector_group_tod import DetectorGroupTOD

logger = logging.getLogger(__name__)

#TODO: Units should be handled in a more robust way.
T_CMB = 2.725 * 1e6  # CMB temperature in uK_CMB units.
C = 299792458  # m/s (Speed of light)
T_CMB_div_C = T_CMB / C
# Precomputing the conversion factor from 1 uK_CMB to 1 uK_RJ
uK_CMB_to_uK_RJ_dict = {}


def project_sky_to_tod(det_compsep_map: NDArray[np.floating], pix: NDArray[np.integer],
                       psi: NDArray[np.floating] | None = None,
                       response_I_P: tuple[float, float] = (1.0, 1.0)) -> NDArray[np.floating]:
    """Project the current sky model into one detector's pointing.

    Args:
        det_compsep_map: The current sky model as seen by this detector.
        pix: The healpix pixel indices of the TOD.
        psi: The polarization angles of the TOD.
        response_I_P: How sensitive the detector is to intensity and polarization, as
            ``DetectorTOD.response_I_P`` returns it. Almost always the [1, 1] default.
    Returns:
        A TOD of the sky projected onto the pointing of the detector.
    """
    resp_I, resp_P = response_I_P
    # An I-only band's sky map has a single component, and psi is irrelevant to it. TODView always
    # passes psi (it does not know the band's polarization), so the component count, not psi, is
    # what selects the intensity-only kernel.
    if psi is None or det_compsep_map.shape[0] == 1:
        return _project_sky_to_tod_I(det_compsep_map, pix, resp_I)
    elif det_compsep_map.shape[0] == 2:
        return _project_sky_to_tod_QU(det_compsep_map, pix, psi, resp_P)
    elif det_compsep_map.shape[0] == 3:
        if resp_I == 0.0:
            return _project_sky_to_tod_QU(det_compsep_map[1:3], pix, psi, resp_P)
        if resp_P == 0.0:
            return _project_sky_to_tod_I(det_compsep_map, pix, resp_I)
        return _project_sky_to_tod_IQU(det_compsep_map, pix, psi, resp_I, resp_P)
    else:
        raise ValueError("Input compsep map has mismatching dimensions.")

@njit(fastmath=True, parallel=True)
def _project_sky_to_tod_IQU(det_compsep_map: NDArray[np.floating], pix: NDArray[np.integer],
                            psi: NDArray[np.floating],
                            resp_I: float, resp_P: float) -> NDArray[np.float32]:
    sky = np.empty(pix.shape[0], dtype=np.float32)
    if resp_I == 1.0 and resp_P == 1.0:
        for i in prange(pix.shape[0]):
            p = pix[i]
            angle = 2.0 * psi[i]
            sky[i] = (det_compsep_map[0, p] + np.cos(angle) * det_compsep_map[1, p]
                      + np.sin(angle) * det_compsep_map[2, p])
    else:
        for i in prange(pix.shape[0]):
            p = pix[i]
            angle = 2.0 * psi[i]
            sky[i] = resp_I * det_compsep_map[0, p] + resp_P * (
                np.cos(angle) * det_compsep_map[1, p] + np.sin(angle) * det_compsep_map[2, p])
    return sky

@njit(fastmath=True, parallel=True)
def _project_sky_to_tod_QU(det_compsep_map: NDArray[np.floating], pix: NDArray[np.integer],
                           psi: NDArray[np.floating],
                           resp_P: float) -> NDArray[np.float32]:
    sky = np.empty(pix.shape[0], dtype=np.float32)
    if resp_P == 0.0:
        sky[:] = 0.0
    elif resp_P == 1.0:
        for i in prange(pix.shape[0]):
            p = pix[i]
            angle = 2.0 * psi[i]
            sky[i] = (np.cos(angle) * det_compsep_map[0, p]
                      + np.sin(angle) * det_compsep_map[1, p])
    else:
        for i in prange(pix.shape[0]):
            p = pix[i]
            angle = 2.0 * psi[i]
            sky[i] = resp_P * (
                np.cos(angle) * det_compsep_map[0, p] + np.sin(angle) * det_compsep_map[1, p]
            )
    return sky

@njit(fastmath=True, parallel=True)
def _project_sky_to_tod_I(det_compsep_map: NDArray[np.floating], pix: NDArray[np.integer],
                          resp_I: float) -> NDArray[np.float32]:
    sky = np.empty(pix.shape[0], dtype=np.float32)
    if resp_I == 0.0:
        sky[:] = 0.0
    elif resp_I == 1.0:
        for i in prange(pix.shape[0]):
            sky[i] = det_compsep_map[0, pix[i]]
    else:
        for i in prange(pix.shape[0]):
            sky[i] = resp_I * det_compsep_map[0, pix[i]]
    return sky


def get_s_orb_tod(det: DetectorTOD, experiment: DetectorGroupTOD, pix: NDArray[np.integer],
                  nthreads: int | None = None) -> NDArray:
    """ Compute the orbital dipole contribution to the TOD for a single detector.

    Projects the CMB dipole induced by the satellite's orbital motion into the
    detector pointing, returning a TOD-length array in uK_RJ units.

    Args:
        det (DetectorTOD): Single-detector TOD data (provides orbital velocity in metres/second).
        experiment (DetectorGroupTOD): Experiment-level data (provides nu and nside).
        pix (NDArray[np.integer]): Decompressed pixel indices for this detector.
        nthreads (int, optional): Number of threads for HEALPix operations.
            Defaults to the OMP_NUM_THREADS environment variable.

    Returns:
        NDArray: Orbital dipole signal in uK_RJ, shape ``(npix,)``.
    """
    orbital_velocity = det.orbital_velocity_m_per_s
    if orbital_velocity is None:
        return np.zeros(pix.shape, dtype=np.float32)
    # The orbital dipole is unpolarized, so only the intensity response matters.
    resp_I, _ = det.response_I_P
    if resp_I == 0.0:  # Polarization-only detectors short-curcuit to no dipole.
        return np.zeros(pix.shape, dtype=np.float32)

    # If nthreads is not set, put it to how many threads OMP has.
    nthreads = int(os.environ["OMP_NUM_THREADS"]) if nthreads is None else nthreads
    if experiment.nu not in uK_CMB_to_uK_RJ_dict:
        uK_CMB_to_uK_RJ_dict[experiment.nu] = (1*pysm3_u.uK_CMB).to(pysm3_u.uK_RJ,
                        equivalencies=pysm3_u.cmb_equivalencies(experiment.nu*pysm3_u.GHz)).value
    geom = ducc0.healpix.Healpix_Base(experiment.nside, "RING")
    LOS_vec = geom.pix2vec(pix, nthreads=nthreads)
    LOS_vec *= orbital_velocity
    # How much do the LOS and orbital velocity align?
    s_orb = np.sum(LOS_vec, axis=-1, dtype=np.float32)
    s_orb *= T_CMB_div_C
    s_orb *= uK_CMB_to_uK_RJ_dict[experiment.nu]  # Converting to uK_RJ units.
    if resp_I != 1.0:
        s_orb *= resp_I  # Multiply with the intensity efficiency of the detector.
    return s_orb.astype(np.float32, copy=False)
