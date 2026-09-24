from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING
import zodipy
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from commander4.tod.config import ZodiConfig
from commander4.data_models.detector_group_tod import DetectorGroupTOD

class Zodi:
    """ Precompute the Zodi model object"""

    def __init__(self, experiment_data: DetectorGroupTOD, zodi_cfg: ZodiConfig):
        self.zodi_cfg = zodi_cfg
        self.nu = experiment_data.nu #TODO: when implementing bandpass remember to propagate here
        self.data_nside = experiment_data.nside
        self.zodi_model = zodipy.Model(self.nu * u.GHz)
        #update the default zodipy params
        self.update_zodipy_params(self.zodi_cfg.update_params)

    def _update_dict_strict(self, base_dict: dict, update_dict: dict) -> dict:
        """Recursively updates base_dict with values from update_dict.

        Raises:
            KeyError: If a key in update_dict does not exist in base_dict.
            TypeError: If a nested structure in update_dict does not match base_dict.

        Returns:
            The updated base_dict (modified in-place).
        """
        for key, value in update_dict.items():
            if key not in base_dict:
                raise KeyError(
                    f"Key '{key}' does not exist in the target configuration."
                )

            if isinstance(value, dict):
                if not isinstance(base_dict[key], dict):
                    raise TypeError(
                        f"Structure mismatch at '{key}': expected a dictionary in target configuration."
                    )
                update_dict_strict(base_dict[key], value)
            else:
                base_dict[key] = value

        return base_dict

    def update_zodipy_params(self, update_dict: dict):
        """Updates the Zodi model parameters in-place with values from update_dict.
        
        Raises:
            KeyError: If a key in update_dict does not exist in the current model params dict.
            TypeError: If a nested structure in update_dict does not match the current model params dict.
        """
        model_d = self.zodi_model.get_parameters()
        model_d = self._update_dict_strict(model_d, update_dict)
        self.zodi_model.update_parameters(model_d)
        return self.zodi_model


    def evaluate(self, pix: NDArray, scan_time: float) -> NDArray:
        #I consider the time constant through the whole scan
        lon, lat = hp.pix2ang(self.data_nside, pix, lonlat=True) * u.deg
        obstime = Time(scan_time) #TODO: WHAT TIME FORMATS ARE THE TODS STAMPS??
        skycoord = SkyCoord(lon, lat, obstime=time, frame="galactic")
        # Evaluate the zodiacal light model
        emissions = model.evaluate(skycoord)
        return emissions