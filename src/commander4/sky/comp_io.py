"""Reading a component's initial amplitudes from a chain file or a FITS map.

Used by `DiffuseComponent` and `CompList` to initialize from an earlier run (`init_chain_path`) or
from an external map (`init_from`). The fiddly part these helpers own is deciding which rows of a
stored (npol, ...) array correspond to the polarization view being initialized.
"""
from __future__ import annotations

import logging
import typing

import h5py
import healpy as hp
import numpy as np
from numpy.typing import NDArray

from commander4.diagnostics import log
from commander4.math_utils.alm import project_alms
from commander4.math_utils.sht import pseudo_alm_to_map_inverse

if typing.TYPE_CHECKING:
    from commander4.sky.diffuse_components import DiffuseComponent

logger = logging.getLogger(__name__)


# Stokes channels stored, in order, for each polarization mode. Used to map the rows of a stored
# (npol, ...) array (whose layout follows its polarization mode) onto the rows an execution view
# needs. Applies equally to chain alm arrays and FITS maps, since both are laid out by polarization.
_POL_CHANNELS = {"I": ("I",), "QU": ("Q", "U"), "IQU": ("I", "Q", "U")}


def _pol_row_indices(data: NDArray, eval_pol: str, shortname: str, source_path: str):
    """Row indices in a stored (npol, ...) array for `eval_pol`'s Stokes channels.

    The stored polarization mode is inferred from the number of rows (1=I, 2=QU, 3=IQU). Returns
    None if the stored data does not contain all channels `eval_pol` needs, so the caller can leave
    those alms at zero. Raises only if the row count is not a recognized polarization mode.
    """
    nrows = data.shape[0]
    stored_pol = {1: "I", 2: "QU", 3: "IQU"}.get(nrows)
    log.logassert(stored_pol is not None,
                  f"Initial data for component {shortname!r} in {source_path!r} has an unexpected "
                  f"first dimension ({nrows}); expected 1 (I), 2 (QU) or 3 (IQU).", logger)
    row_of = {channel: row for row, channel in enumerate(_POL_CHANNELS[stored_pol])}
    if any(channel not in row_of for channel in _POL_CHANNELS[eval_pol]):
        return None
    return [row_of[channel] for channel in _POL_CHANNELS[eval_pol]]


def _read_view_alms_from_chain(comp: DiffuseComponent, chain_path: str) -> NDArray | None:
    """This view's alms from a compsep chain (``comps/<shortname>/alms``), or None if not present.

    A missing component is logged as an error (but not fatal); a component present without this
    view's polarization is a benign partial initialization and only debug-logged.
    """
    with h5py.File(chain_path, "r") as f:
        group_path = f"comps/{comp.shortname}"
        if group_path not in f or "alms" not in f[group_path]:
            logger.error(f"Component {comp.comp_name!r} (shortname {comp.shortname!r}) not found in "
                         f"init chain {chain_path!r}; leaving its alms at zero.")
            return None
        stored_alms = f[f"{group_path}/alms"][()]
    rows = _pol_row_indices(stored_alms, comp.eval_pol, comp.shortname, chain_path)
    if rows is None:
        logger.debug(f"Init chain {chain_path!r} has no {comp.eval_pol!r} data for component "
                     f"{comp.comp_name!r}; leaving those alms at zero.")
        return None
    return project_alms(np.ascontiguousarray(stored_alms[rows]), comp.lmax)


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


def _restore_sampled_sed_params_from_chain(comp: DiffuseComponent, chain_path: str) -> None:
    """Set `comp`'s *sampled* SED parameters from a chain's ``comps/<shortname>/sed/`` group.

    Only parameters the run is configured to *sample* are restored. Fixed ones (``nu_ref``, ``T``)
    stay under the parameter file's control, so changing one there is not silently overridden by an
    older chain. Parameters absent from the chain (written before this group existed) are left as
    the parameter file set them.
    """
    if "sample_spectral_index" not in comp.comp_params \
            or not bool(comp.comp_params.sample_spectral_index):
        return
    with h5py.File(chain_path, "r") as f:
        sed_group = f.get(f"comps/{comp.shortname}/sed")
        if sed_group is None:
            return
        # `beta` is currently the only sampled SED parameter, so it is the only one restored.
        for param_name in comp.sed_param_names:
            if param_name != "beta" or param_name not in sed_group:
                continue
            value = sed_group[param_name][()]
            logger.info(f"Component {comp.comp_name!r} ({comp.eval_pol}): restored sampled "
                        f"{param_name} = {value} from {chain_path!r} (parameter file said "
                        f"{getattr(comp, param_name)}).")
            setattr(comp, param_name, float(value))


def _load_component_alms(comp: DiffuseComponent, source_path: str) -> None:
    """Set `comp`'s initial alms from `source_path`, dispatching on its file type.

    ``.h5``/``.hd5`` files are read as compsep chains (alms *and* the sampled SED parameters taken
    directly); ``.fits`` files are read as sky maps and transformed to alms, and carry no SED
    information. If the source does not contain this component or its polarization, the alms are
    left at their initial value (zeros).
    """
    lower_path = str(source_path).lower()
    if lower_path.endswith((".h5", ".hd5")):
        view_alms = _read_view_alms_from_chain(comp, source_path)
        _restore_sampled_sed_params_from_chain(comp, source_path)
    elif lower_path.endswith(".fits"):
        view_alms = _read_view_alms_from_fits(comp, source_path)
    else:
        log.logassert(False,
                      f"Unsupported init file {source_path!r} for component {comp.comp_name!r}: "
                      f"expected a .h5/.hd5 chain or a .fits map.", logger)
    if view_alms is not None:
        comp.alms = view_alms.astype(comp.dtype, copy=False)
