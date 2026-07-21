import numpy as np
import healpy as hp
from astropy.io import fits
import logging
import pysm3.units as pysm3_u
from pixell.bunch import Bunch

from commander4.output.log import logassert
from commander4.data_models.detector_map import DetectorMap

logger = logging.getLogger(__name__)

POLARIZATION_INDEX = {"I": 0, "Q": 1, "U": 2}


def _get_map_info(band: Bunch, maptype: str):
    if maptype not in {"signal", "rms"}:
        raise ValueError(f"Unknown maptype {maptype}.")

    map_key = f"{maptype}_map"
    if map_key not in band:
        raise ValueError(f"Band {band._name} must specify {map_key}.")

    map_config = band[map_key]
    map_data_type = map_config.type if "type" in map_config else maptype
    if "path" not in map_config and not (maptype == "rms" and map_data_type != "rms"):
        raise ValueError(f"Band {band._name} must specify a path for {map_key}.")
    if "dataset_names" not in map_config or len(map_config.dataset_names) != 3:
        raise ValueError(f"Band {band._name} must specify exactly three dataset_names for "\
                         f"{map_key}, even if all three do not exist (use None or '').")
    filename = map_config.path if "path" in map_config else None
    return filename, map_config.dataset_names, map_data_type


def retrieve_map_from_fits_file(band: Bunch, pol: str, maptype: str):
    filename, dataset_names, _ = _get_map_info(band, maptype)
    dataset_name = dataset_names[POLARIZATION_INDEX[pol]]

    with fits.open(filename) as hdul:
        if dataset_name not in hdul[1].data.columns.names:
            raise ValueError(f"Could not find dataset {dataset_name} for polarization {pol} "\
                             f"in file {filename}.")

        ### FIND DATA ###
        data = hdul[1].data[dataset_name].flatten().astype(np.float32, copy=False)

        ### FIND UNITS ###
        units_param = band.units if "units" in band else None
        column_idx = hdul[1].data.columns.names.index(dataset_name)
        units_file = hdul[1].data.columns.units[column_idx]
        if isinstance(units_param, str):
            units_param = units_param.strip() or None
        if isinstance(units_file, str):
            units_file = units_file.strip() or None
        if units_file is not None and units_file.lower() == "unknown":
            units_file = None
        if units_file is None and units_param is None:
            logging.warning(f"No units specified for {band._name}. Assuming uK_CMB!")
            units = "uK_CMB"
        elif units_file is not None and units_param is not None:
            if units_file != units_param:
                logging.warning(
                    f"Both data-file ({units_file}) and param-file ({units_param}) specify "
                    f"map units for {band._name}; using param-file value."
                )
            units = units_param
        else:
            units = units_file or units_param

        ### FIND ORDERING ###
        ordering_file = hdul[1].header.get("ORDERING")
        ordering_param = band.ordering.upper() if "ordering" in band else None
        if ordering_param == "NESTED":
            ordering_param = "NEST"
        if ordering_file == "NESTED":
            ordering_file = "NEST"
        if ordering_file is not None:
            ordering_file = ordering_file.upper()
        if ordering_file == "UNKNOWN":
            ordering_file = None
        if ordering_file is None and ordering_param is None:
            logging.warning(f"No ordering specified for {band._name}. Assuming RING ordering!")
            ordering = "RING"
        elif ordering_file is not None and ordering_param is not None:
            if ordering_file != ordering_param:
                logging.warning(f"Both map-file ({ordering_file}) and param-file ({band.ordering}) "
                                f"specify healpix ordering for {band._name}; using the latter.")
            ordering = ordering_param
        else:
            ordering = ordering_file or ordering_param
        if ordering not in {"RING", "NEST"}:
            raise ValueError(
                f"Unrecognized healpix ordering {ordering} for band {band._name}. "
                "Expected RING or NEST."
            )

        ### REORDER ###
        if ordering == "NEST":
            data = hp.reorder(data, inp="NEST", out="RING")

        ### UNIT CONVERSION ###
        if units != "uK_RJ":
            data = (data * pysm3_u.Unit(units)).to(
                pysm3_u.uK_RJ,
                equivalencies=pysm3_u.cmb_equivalencies(band.freq * pysm3_u.GHz),
            ).value

    return data



def read_data_map_from_file(my_band: Bunch, params: Bunch) -> DetectorMap:
    """ Reads the map data for a given band from disk. Used for bands that do not have a TOD
        processing component. The map is smoothed to the common resolution on read (when enabled).
    Args:
        my_band (Bunch): A parameter file subset for the band to read from file.
        params (Bunch): Full parameter file, for the common-resolution smoothing settings.
    Returns:
        detector_map (DetectorMap): Object holding signal map and other relevant data (rms, nu...)
    """
    pols = my_band.polarization
    logassert(pols in ["I", "QU", "IQU"], f"Specified polarization {pols} not recognized.", logger)
    maps_sky = []
    maps_rms = []
    rms_map_type = _get_map_info(my_band, "rms")[2]

    for pol in pols:
        map_sky = retrieve_map_from_fits_file(my_band, pol, "signal")
        if rms_map_type == "debug_uniform":
            logger.warning(f"Band {my_band._name} is using uniform rms maps meant for debugging.")
            map_rms = np.zeros_like(map_sky) + np.nanmean(np.abs(map_sky))
        elif rms_map_type == "rms":
            map_rms = retrieve_map_from_fits_file(my_band, pol, "rms")
        else:
            raise ValueError(f"Unrecognized rms map type {rms_map_type} for band {my_band._name}.")

        if "add_signal_fraction_to_rms" in my_band:
            map_rms = np.sqrt(map_rms**2 \
                              + (my_band.add_signal_fraction_to_rms*np.nanmean(np.abs(map_sky)))**2)
        # TODO: Figure out how to read covariance maps as opposed to RMS maps.

        nside = np.sqrt(map_sky.size//12)
        logassert(nside.is_integer(), f"Npix dimension of map ({map_sky.size}) "
                  f"resulting in a non-integer nside ({nside}).", logger)
        nside = int(nside)

        if "eval_nside" in my_band and nside != my_band.eval_nside:
            logger.info(f"Converting map {my_band._name} from nside {nside} to "\
                        f"{my_band.eval_nside}.")
            map_sky = hp.ud_grade(map_sky, my_band.eval_nside)
            map_rms = 1.0/np.sqrt(hp.ud_grade(1.0/map_rms**2, my_band.eval_nside))
            nside = my_band.eval_nside

        maps_sky.append(map_sky)
        maps_rms.append(map_rms)

    detmap = DetectorMap(np.array(maps_sky), np.array(maps_rms), my_band.freq, my_band.fwhm, nside)
    detmap.g0 = 0.0
    detmap.gain = 0.0
    # Smooth to the common analysis resolution on read (single switch: general.common_res_fwhm; a
    # missing or falsy value leaves the band at its native beam).
    if "common_res_fwhm" in params.general and params.general.common_res_fwhm:
        detmap.smooth_to_resolution(float(params.general.common_res_fwhm))

    return detmap