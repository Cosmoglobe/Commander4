"""Read initial sky state from chain files, or diffuse amplitudes from FITS maps."""
from __future__ import annotations

import logging
import typing

import h5py
import healpy as hp
import numpy as np
from numpy.typing import NDArray

from commander4.math_utils.alm import project_alms
from commander4.math_utils.sht import pseudo_alm_to_map_inverse

if typing.TYPE_CHECKING:
    from commander4.sky.component import Component
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
    if stored_pol is None:
        raise ValueError(
            f"Initial data for component {shortname!r} in {source_path!r} has an unexpected first "
            f"dimension ({nrows}); expected 1 (I), 2 (QU) or 3 (IQU).")
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


def _load_component_state(comp: Component, source_path: str, *, amplitudes: bool = True,
                          spectral_parameters: bool = True) -> None:
    """Load the requested amplitudes and spectral values, independently of subsequent sampling.

    Chain files must contain the requested component and polarization. Reference frequencies
    define the amplitude convention and must match; other spectral values may be restored.
    FITS maps carry only amplitudes and retain the existing partial-polarization behaviour.
    """
    from commander4.sky.diffuse_components import DiffuseComponent
    from commander4.sky.point_sources import RadioSources

    lower_path = str(source_path).lower()
    if lower_path.endswith(".fits"):
        if not isinstance(comp, DiffuseComponent):
            raise ValueError("FITS initialization is only supported for diffuse components.")
        if amplitudes:
            view_alms = _read_view_alms_from_fits(comp, source_path)
            if view_alms is not None:
                comp.alms = view_alms.astype(comp.dtype, copy=False)
        return
    if not lower_path.endswith((".h5", ".hd5", ".hdf5")):
        raise ValueError(f"Unsupported init file {source_path!r} for component "
                         f"{comp.comp_name!r}: expected a .h5/.hd5 chain or a .fits map.")

    with h5py.File(source_path, "r") as f:
        if "metadata/complete" in f and not bool(f["metadata/complete"][()]):
            raise ValueError(f"Sky init file {source_path!r} is incomplete.")
        group_path = f"comps/{comp.shortname}"
        if group_path not in f:
            raise ValueError(f"Component {comp.shortname!r} not found in {source_path!r}; "
                             "set its 'init_from' to null to keep its configured initialization.")
        group = f[group_path]
        for key, expected in (("comp_name", comp.comp_name),
                              ("component_class", type(comp).__name__),
                              ("amplitude_unit", comp.amplitude_unit)):
            # Class and amplitude-unit metadata were added with separate TOD/sky initialization.
            if key in group and group[key].asstr()[()] != expected:
                raise ValueError(f"Component {comp.shortname!r}: {key} differs from "
                                 f"{source_path!r}.")

        required = []
        reference_params = ("nu_ref", "nu_0", "nu_peak_ref")
        for name in comp.sed_param_names:
            if spectral_parameters or name in reference_params:
                required.append(f"sed/{name}")
        if amplitudes:
            required.extend(["alms" if isinstance(comp, DiffuseComponent) else "source_amps",
                             "amp_fwhm_arcmin"])
        if isinstance(comp, RadioSources):
            required.append("source_lonlat")
            if spectral_parameters:
                required.append("source_spectral_indices")
        missing = []
        for name in required:
            if name not in group:
                missing.append(name)
        if missing:
            raise ValueError(f"Component {comp.shortname!r} in {source_path!r} "
                             f"is missing {missing}.")

        for name in comp.sed_param_names:
            if not spectral_parameters and name not in reference_params:
                continue
            value = np.asarray(group[f"sed/{name}"][()])
            # Joined IQU components may store different reference frequencies or spectral
            # values for I and QU. Select the same view used when constructing the component.
            if value.shape == (2,) and comp.eval_pol in ("I", "QU"):
                value = value[0 if comp.eval_pol == "I" else 1]
            if name in reference_params:
                if not np.allclose(value, getattr(comp, name), rtol=1e-10, atol=0):
                    raise ValueError(f"Component {comp.shortname!r}: reference parameter {name} "
                                     f"differs from {source_path!r}.")
            else:
                setattr(comp, name, float(value))

        if isinstance(comp, RadioSources):
            if not np.array_equal(group["source_lonlat"][:], comp.lonlat_arr):
                raise ValueError(f"Point-source positions or ordering differ from {source_path!r}.")
            if spectral_parameters:
                indices = group["source_spectral_indices"][:]
                if indices.shape != comp.alpha_arr.shape:
                    raise ValueError(f"Point-source index shape differs in {source_path!r}.")
                comp.alpha_arr = indices.astype(comp.alpha_arr.dtype)
        if amplitudes:
            if isinstance(comp, DiffuseComponent):
                stored_alms = group["alms"][:]
                rows = _pol_row_indices(stored_alms, comp.eval_pol, comp.shortname, source_path)
                if rows is None:
                    raise ValueError(f"Component {comp.shortname!r} in {source_path!r} has no "
                                     f"{comp.eval_pol} amplitudes.")
                comp.alms = project_alms(np.ascontiguousarray(stored_alms[rows]), comp.lmax)
                comp.alms = comp.alms.astype(comp.dtype, copy=False)
            else:
                values = group["source_amps"][:]
                if values.shape != comp._data.shape:
                    raise ValueError(f"Point-source amplitude shape differs in {source_path!r}.")
                comp._data = values.astype(comp._data.dtype)
            comp.amp_fwhm_rad = np.radians(float(group["amp_fwhm_arcmin"][()]) / 60.0)
    logger.info(f"Component {comp.comp_name!r} ({comp.eval_pol}): initialized from "
                f"{source_path!r} (amplitudes={amplitudes}, "
                f"spectral_parameters={spectral_parameters}).")
