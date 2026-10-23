"""Reading a component's amplitudes from a FITS map, and picking a view's rows of stored data.

`CompList` uses the FITS reader for a component's `init_from` map and its `amp_prior_mean_map`;
each component reads its chain group itself (`read_chain`). Both need the rows of a stored
(npol, ...) array that belong to a polarization view, which `_pol_row_indices` gives.
"""
from __future__ import annotations

import logging
import typing

import healpy as hp
import numpy as np
from numpy.typing import NDArray

from commander4.math_utils.alm import project_alms
from commander4.math_utils.sht import pseudo_alm_to_map_inverse

if typing.TYPE_CHECKING:
    from commander4.sky.diffuse_components import DiffuseComponent

logger = logging.getLogger(__name__)


# Stokes channels stored, in order, for each polarization mode. Used to map the rows of a stored
# (npol, ...) array (whose layout follows its polarization mode) onto the rows an execution view
# needs. Applies equally to chain alm arrays and FITS maps, since both are laid out by polarization.
POL_CHANNELS = {"I": ("I",), "QU": ("Q", "U"), "IQU": ("I", "Q", "U")}


def _pol_row_indices(data: NDArray, eval_pol: str, shortname: str, source_path: str):
    """Row indices in a stored (npol, ...) array for `eval_pol`'s Stokes channels.

    The stored polarization mode is inferred from the number of rows (1=I, 2=QU, 3=IQU). Returns
    None if the stored data does not contain all channels `eval_pol` needs, so the caller can leave
    those alms at zero. Raises only if the row count is not a recognized polarization mode.
    """
    nrows = data.shape[0]
    stored_pol = {1: "I", 2: "QU", 3: "IQU"}.get(nrows)
    if stored_pol is None:
        raise ValueError(
            f"Initial data for component {shortname!r} in {source_path!r} has an unexpected first "
            f"dimension ({nrows}); expected 1 (I), 2 (QU) or 3 (IQU).")
    row_of = {channel: row for row, channel in enumerate(POL_CHANNELS[stored_pol])}
    if any(channel not in row_of for channel in POL_CHANNELS[eval_pol]):
        return None
    return [row_of[channel] for channel in POL_CHANNELS[eval_pol]]


def _read_view_alms_from_fits(comp: DiffuseComponent, fits_path: str) -> NDArray | None:
    """This view's alms from a FITS sky map (transformed), or None if its polarization isn't present.

    The map's polarization content is inferred purely from its shape (npol, npix), so the column
    names do not matter. The map is converted from its ``units`` to the component's amplitude unit
    (at the component's reference frequency) before being transformed to alms.

    Shared by the ``init_from`` initial guess and the ``amp_prior_mean_map`` prior mean, since both
    are sky maps living in the component's own amplitude space and must be read identically.
    """
    sky_map = np.atleast_2d(hp.read_map(fits_path, field=None))
    rows = _pol_row_indices(sky_map, comp.eval_pol, comp.shortname, fits_path)
    if rows is None:
        logger.debug(f"Map {fits_path!r} has no {comp.eval_pol!r} data for component "
                     f"{comp.comp_name!r}; leaving those alms at zero.")
        return None
    view_map = np.ascontiguousarray(sky_map[rows], dtype=np.float64)
    view_map = comp.init_map_to_amplitude(view_map)
    nside = hp.npix2nside(view_map.shape[-1])

    # Only perform map2alm up to ell = 3*map_nside - 1.
    # If the component lmax exceeds this, truncate remaining alms to zero.
    effective_lmax = min(comp.lmax, 3*nside-1)
    alm_temp = pseudo_alm_to_map_inverse(view_map, nside, effective_lmax,
                            spin = 0 if view_map.shape[0] == 1 else 2, epsilon = 1e-8, maxiter = 5)
    return project_alms(alm_temp, comp.lmax)
