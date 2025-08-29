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
    map_signal = [None, None, None]
    map_rms = [None, None, None]
    if my_band.file_convention == "WMAP":
        # WMAP maps are in mK_CMB and need to be multiplied by 1000. Also, the uncertainty is in variance units,
        # and needs to be square-rooted.
        data_names = ["I_Stokes", "Q_Stokes", "U_Stokes"]
        rms_names = ["II_Stokes", "Q_Stokes", "U_Stokes"]
        for ipol in range(3):
            if my_band.polarizations[ipol]:
                map_signal[ipol] = 1e3*fits.open(my_band.path_signal_map)[1].data[data_names[ipol]].flatten().astype(np.float32)
                map_rms[ipol] = np.sqrt(1e3*fits.open(my_band.path_rms_map)[1].data[rms_names[ipol]].flatten().astype(np.float32))
    elif my_band.file_convention == "HFI":
        # I think HFI maps are in uK_CMB. They are also in nested healpix ordering, and need to be converted.
        data_names = ["TEMPERATURE", "Q-POLARISATION", "U-POLARISATION"]
        rms_names = ["TEMPERATURE", "Q-POLARISATION", "U-POLARISATION"]
        for ipol in range(3):
            if my_band.polarizations[ipol]:
                map_signal[ipol] = fits.open(my_band.path_signal_map)[1].data[data_names[ipol]].flatten().astype(np.float32)
                map_rms[ipol] = fits.open(my_band.path_rms_map)[1].data[rms_names[ipol]].flatten().astype(np.float32)
                map_signal[ipol] = hp.reorder(map_signal[ipol], inp="NEST", out="RING")
                map_rms[ipol] = hp.reorder(map_rms[ipol], inp="NEST", out="RING")
    elif my_band.file_convention == "Haslam":
        # I think Haslam maps are in uK_CMB.
        data_names = ["TEMPERATURE", "Q-POLARISATION", "U-POLARISATION"]
        rms_names = ["TEMPERATURE", "Q-POLARISATION", "U-POLARISATION"]
        for ipol in range(3):
            if my_band.polarizations[ipol]:
                map_signal[ipol] = fits.open(my_band.path_signal_map)[1].data[data_names[ipol]].flatten().astype(np.float32)
                map_rms[ipol] = fits.open(my_band.path_rms_map)[1].data[rms_names[ipol]].flatten().astype(np.float32)

    # Convert from input units (uK_CMB) to Commander processing units (uK_RJ)
    n_corr = [None, None, None]
    for ipol in range(3):
        if my_band.polarizations[ipol]:
            map_signal[ipol] = map_signal[ipol] * pysm3_u.uK_CMB
            map_signal[ipol] = map_signal[ipol].to(pysm3_u.uK_RJ, equivalencies=pysm3_u.cmb_equivalencies(my_band.freq*pysm3_u.GHz)).value
            map_rms[ipol] = map_rms[ipol] * pysm3_u.uK_CMB
            map_rms[ipol] = map_rms[ipol].to(pysm3_u.uK_RJ, equivalencies=pysm3_u.cmb_equivalencies(my_band.freq*pysm3_u.GHz)).value

            nside = np.sqrt(map_signal[ipol].size//12)
            logassert(nside.is_integer(), f"Npix dimension of map ({map_signal[ipol].size}) resulting in a non-integer nside ({nside}).", logger)
            nside = int(nside)

            if nside != 512:
                map_signal[ipol] = hp.ud_grade(map_signal[ipol], 512)
                map_rms[ipol] = 1.0/np.sqrt(hp.ud_grade(1.0/map_rms[ipol]**2, 512))
                nside = 512
            n_corr[ipol] = np.zeros_like(map_signal[ipol], dtype=np.float32)

    detmap = DetectorMap(map_signal, n_corr ,map_rms, my_band.freq, my_band.fwhm, nside)
    detmap.g0 = 0.0
    detmap.gain = 0.0
    detmap.skysub_map = n_corr
    detmap.rawobs_map = n_corr
    detmap.orbdipole_map = n_corr
    return detmap