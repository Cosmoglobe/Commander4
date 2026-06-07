import numpy as np
from pixell.bunch import Bunch
from commander4.sky_models.component import CompList


def build_initial_sky_model(params: Bunch) -> "SkyModel":
    """Construct the initial sky model directly from the parameter file.

    Builds the full component list, loads each component's initial alms (from its ``init_from`` or
    the global ``init_chain_path``, else zeros), and wraps it in a SkyModel. This is rank-agnostic
    and performs no MPI, so it is used both by CompSep ranks and -- when no CompSep ranks exist --
    by the TOD band masters to construct the initial sky locally.
    """
    comp_list = CompList.init_from_params(params.components, params)
    comp_list.load_initial_alms(params)
    return SkyModel(comp_list)


class SkyModel:
    def __init__(self, components:CompList):
        # components = list of Component objects
        self._components = components

    def get_sky(self, band):
        """ Get sky from a bandpass.
        """
        raise NotImplementedError

    def get_sky_at_nu(self, nu, nside, pols_required, fwhm=None):
        """Get the realized sky at one frequency.

        The component list may be either the split execution list used during CompSep (`I` and
        `QU` views) or a joined logical list containing `IQU` components.
        """
        npix = 12*nside**2

        if pols_required == "I":
            skymap = np.zeros((1, npix), dtype=np.float32)
        elif pols_required == "QU":
            skymap = np.zeros((2, npix), dtype=np.float32)
        elif pols_required == "IQU":
            skymap = np.zeros((3, npix), dtype=np.float32)
        else:
            raise ValueError("Unrecognized polarization string")

        for component in self._components:
            if component.eval_pol == "I":
                if pols_required in ("I", "IQU"):
                    skymap[0] += component.get_sky(nu, nside, fwhm)[0]
            elif component.eval_pol == "QU":
                if pols_required == "QU":
                    skymap += component.get_sky(nu, nside, fwhm)
                elif pols_required == "IQU":
                    skymap[1:] += component.get_sky(nu, nside, fwhm)
            elif component.eval_pol == "IQU":
                component_sky = component.get_sky(nu, nside, fwhm)
                if pols_required == "I":
                    skymap[0] += component_sky[0]
                elif pols_required == "QU":
                    skymap += component_sky[1:]
                else:
                    skymap += component_sky
            else:
                raise ValueError(f"Unsupported component polarization '{component.eval_pol}'.")
        return skymap
