import numpy as np
import healpy as hp
from pixell.bunch import Bunch
from src.python.data_models.detector_map import DetectorMap
from astropy.io import fits
import logging
from src.python.output.log import logassert
import pysm3.units as pysm3_u


def read_sim_map_from_file(my_band: Bunch) -> DetectorMap:
    """To be run once before starting TOD processing.

    Determines whether the process is TOD master, creates the band communicator
    and determines whether the process is the band master. Also reads the
    experiment data.

    Input:
        my_band (Bunch): The section of the parameter file corresponding to this CompSep band, as a "Bunch" type.

    Output:
        data (list of DetectorMaps): nbands (DetectorMap)
    """
    logger = logging.getLogger(__name__)
    map_signal = hp.read_map(my_band.path_signal_map)
    map_rms = hp.read_map(my_band.path_rms_map)
    nside = np.sqrt(map_signal.size//12)
    logassert(nside.is_integer(), f"Npix dimension of map ({map_signal.size}) resulting in a non-integer nside ({nside}).", logger)
    nside = int(nside)
    return DetectorMap(map_signal, None, map_rms, my_band.freq, my_band.fwhm, my_band.nside)


def read_Planck_map_from_file(my_band: Bunch) -> DetectorMap:
    logger = logging.getLogger(__name__)
    if "WMAP" in str(my_band):
        # WMAP maps are in mK_CMB
        map_signal = 1e3*fits.open(my_band.path_signal_map)[1].data["I_Stokes"].flatten().astype(np.float32)
        map_rms = 1e3*fits.open(my_band.path_rms_map)[1].data["II_Stokes"].flatten().astype(np.float32)  
        map_rms = np.sqrt(map_rms)
    elif "857" in str(my_band):
        # I think HFI maps are in uK_CMB
        map_signal = fits.open(my_band.path_signal_map)[1].data["TEMPERATURE"].flatten().astype(np.float32)
        map_rms = fits.open(my_band.path_rms_map)[1].data["TEMPERATURE"].flatten().astype(np.float32)
        map_signal = hp.reorder(map_signal, inp="NEST", out="RING")
        map_rms = hp.reorder(map_rms, inp="NEST", out="RING")
    else:
        # I think Haslam is in uK_CMB
        map_signal = fits.open(my_band.path_signal_map)[1].data["TEMPERATURE"].flatten().astype(np.float32)
        map_rms = fits.open(my_band.path_rms_map)[1].data["TEMPERATURE"].flatten().astype(np.float32)

    map_signal = map_signal * pysm3_u.uK_CMB
    map_signal = map_signal.to(pysm3_u.uK_RJ, equivalencies=pysm3_u.cmb_equivalencies(my_band.freq*pysm3_u.GHz)).value
    map_rms = map_rms * pysm3_u.uK_CMB
    map_rms = map_rms.to(pysm3_u.uK_RJ, equivalencies=pysm3_u.cmb_equivalencies(my_band.freq*pysm3_u.GHz)).value

    nside = np.sqrt(map_signal.size//12)
    logassert(nside.is_integer(), f"Npix dimension of map ({map_signal.size}) resulting in a non-integer nside ({nside}).", logger)
    nside = int(nside)

    if nside != 512:
        map_signal = hp.ud_grade(map_signal, 512)
        map_rms = hp.ud_grade(map_rms, 512)
        nside = 512

    detmap = DetectorMap(map_signal, np.zeros_like(map_signal, dtype=np.float32), map_rms, my_band.freq, my_band.fwhm, nside)
    detmap.g0 = 0.0
    detmap.gain = 0.0
    detmap.skysub_map = np.zeros_like(map_signal, dtype=np.float32)
    detmap.rawobs_map = np.zeros_like(map_signal, dtype=np.float32)
    detmap.orbdipole_map = np.zeros_like(map_signal, dtype=np.float32)
    return detmap