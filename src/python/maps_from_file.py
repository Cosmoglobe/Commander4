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
    return DetectorMap(map_signal, map_rms, my_band.freq, my_band.fwhm, my_band.nside)


def read_data_map_from_file(my_band: Bunch) -> DetectorMap:
    logger = logging.getLogger(__name__)
    #polarizations relevant for the current compsep band (either I or QU).
    logassert(my_band.identifier.endswith("_I") or my_band.identifier.endswith("_QU"), f'band identifier {my_band.identifier} has wrong or missing polarization ending, _I or _QU expected.', logger)
    pols_to_read = [True, False, False] if my_band.identifier.endswith("_I") else [False, True, True]
    npol = np.count_nonzero(pols_to_read) #Polarizations to be stored in myband
    map_signal = []
    map_rms = []
    map_cov = None
    if my_band.file_convention == "WMAP":
        # WMAP maps are in mK_CMB and need to be multiplied by 1000. Also, the uncertainty is in variance units,
        # and needs to be square-rooted.
        data_names = ["I_Stokes", "Q_Stokes", "U_Stokes"]
        rms_names = ["II_Stokes", "Q_Stokes", "U_Stokes"]
        for ipol in range(3):
            if pols_to_read[ipol]:
                map_signal.append(1e3*fits.open(my_band.path_signal_map)[1].data[data_names[ipol]].flatten().astype(np.float32))
                map_rms.append(1e3*np.sqrt(fits.open(my_band.path_rms_map)[1].data[rms_names[ipol]].flatten().astype(np.float32)))
    elif my_band.file_convention == "WMAP_pol":
        # WMAP maps are in mK_CMB and need to be multiplied by 1000. Also, the uncertainty is in variance units,
        # and needs to be square-rooted.
        logassert(pols_to_read == [False, True, True], 'File convention "WMAP_pol can not be used for Intensity maps', logger)
        nside = 16
        indices_ring = np.arange(0, 12*nside**2, dtype=int)
        indices_nest = hp.ring2nest(nside, indices_ring)
        data_names = ["TEMPERATURE", "Q-POLARISATION", "U-POLARISATION"]
        rms_names = ["TEMPERATURE", "Q-POLARISATION", "U-POLARISATION"]
        for ipol in range(1,3):
                map_signal[ipol] = 1e3*fits.open(my_band.path_signal_map)[1].data[data_names[ipol]].flatten().astype(np.float32)
                map_signal[ipol] = hp.reorder(map_signal[ipol], inp="NEST", out="RING")
        _map_cov = 1e6*fits.open(my_band.path_cov_map)[0].data.astype(np.float32)
        map_cov = np.zeros_like(_map_cov)
        map_cov[indices_ring,:] = _map_cov[indices_nest,:]
        map_cov[12*nside**2+indices_ring,:] = _map_cov[12*nside**2+indices_nest,:]
        map_cov[:,12*nside**2+indices_ring] = _map_cov[:,12*nside**2+indices_nest]
        map_cov[:,indices_ring] = _map_cov[:,indices_nest]
        
    elif my_band.file_convention == "HFI":
        # I think HFI maps are in uK_CMB. They are also in nested healpix ordering, and need to be converted.
        data_names = ["TEMPERATURE", "Q-POLARISATION", "U-POLARISATION"]
        rms_names = ["TEMPERATURE", "Q-POLARISATION", "U-POLARISATION"]
        for ipol in range(3):
            if pols_to_read[ipol]:
                aux_map = fits.open(my_band.path_signal_map)[1].data[data_names[ipol]].flatten().astype(np.float32)
                map_signal.append(hp.reorder(aux_map, inp="NEST", out="RING"))
                aux_map = fits.open(my_band.path_rms_map)[1].data[rms_names[ipol]].flatten().astype(np.float32)
                map_rms.append(hp.reorder(aux_map, inp="NEST", out="RING"))
    elif my_band.file_convention == "Haslam":
        # I think Haslam maps are in uK_CMB.
        data_names = ["TEMPERATURE", "Q-POLARISATION", "U-POLARISATION"]
        rms_names = ["TEMPERATURE", "Q-POLARISATION", "U-POLARISATION"]
        append_idx = 0
        for ipol in range(3):
            if pols_to_read[ipol]:
                map_signal.append(fits.open(my_band.path_signal_map)[1].data[data_names[ipol]].flatten().astype(np.float32))
                aux_map = fits.open(my_band.path_rms_map)[1].data[rms_names[ipol]].flatten().astype(np.float32)
                map_rms.append(np.sqrt(aux_map**2 + (0.01*map_signal[append_idx])**2))
                append_idx += 1

    logassert(len(map_signal) == npol, f"Shape of loaded signal map {my_band.path_signal_map} does not match polarization count.", logger)
    logassert(len(map_rms) == len(map_signal), f"Shape of loaded rms map {my_band.path_rms_map} does not match signal map's one.", logger)
    # Convert from input units (uK_CMB) to Commander processing units (uK_RJ)
    n_corr = []
    for ipol in range(npol):
        map_signal[ipol] = map_signal[ipol] * pysm3_u.uK_CMB
        map_signal[ipol] = map_signal[ipol].to(pysm3_u.uK_RJ, equivalencies=pysm3_u.cmb_equivalencies(my_band.freq*pysm3_u.GHz)).value
        map_rms[ipol] = map_rms[ipol] * pysm3_u.uK_CMB
        map_rms[ipol] = map_rms[ipol].to(pysm3_u.uK_RJ, equivalencies=pysm3_u.cmb_equivalencies(my_band.freq*pysm3_u.GHz)).value

        nside = np.sqrt(map_signal[ipol].size//12)
        logassert(nside.is_integer(), f"Npix dimension of map ({map_signal[ipol].size}) resulting in a non-integer nside ({nside}).", logger)
        nside = int(nside)

        if my_band.data_nside != my_band.eval_nside:
            map_signal[ipol] = hp.ud_grade(map_signal[ipol], my_band.eval_nside)
            map_rms[ipol] = 1.0/np.sqrt(hp.ud_grade(1.0/map_rms[ipol]**2, my_band.eval_nside))
        n_corr.append(np.zeros_like(map_signal[ipol], dtype=np.float32))

    logassert(len(map_rms) == len(map_signal), f"Shape of correlated noise map does not match shape of signal map {my_band.path_signal_map}.", logger)

    detmap = DetectorMap(np.array(map_signal), np.array(map_rms), my_band.freq, my_band.fwhm, my_band.eval_nside)
    detmap.g0 = 0.0
    detmap.gain = 0.0

    # TODO: no plots support for now, will be readded through chain files afterwards.
    # detmap.skysub_map = n_corr
    # detmap.rawobs_map = n_corr
    # detmap.orbdipole_map = n_corr

    return detmap