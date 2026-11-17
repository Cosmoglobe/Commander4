"""`CompList`: an ordered list of component views, which the CG can use as a vector.

A component defined for IQU is stored as two views, one for I and one for QU, because the two are
solved on separate MPI ranks. `CompList` holds five kinds of method, and nothing else:

- building the views from the parameter file (`init_from_params`),
- reorganizing them (`split_for_pol`, `joined`),
- the sky that all the components together give at a band (`get_sky_at_band`, `get_sky_at_nu`),
- the vector arithmetic of the CG (`zeros_like`, `copy`, `dot`, `add_scaled`, `scale_and_add`),
- writing the chain and keeping the MPI ranks in step (`write_chain`, `broadcast_pol_views`).

Work on one component at a time (`comp2band`, the C_l prior, loading its starting state, ...) is
not passed down through the list: the code that needs it loops over the components itself.
"""
from __future__ import annotations

import copy

import h5py
import healpy as hp
import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray
from pixell.bunch import Bunch

import commander4.sky as sky
from commander4.data_models.band import Band
from commander4.data_models.detector_map import DetectorMap
from commander4.sky.component import Component
from commander4.sky.diffuse_components import DiffuseComponent
from commander4.parameters.schema import EXECUTION_POLS
from commander4.math_utils.arithmetic import inplace_scale_add, inplace_add_scaled_vec
from commander4.math_utils.sht import alm_to_map


class CompList:
    """An ordered list of component views, which the CG can use as a vector.

    It behaves like a list: `len(comp_list)`, `comp_list[i]` and `for comp in comp_list` all act
    on `comp_list.components`.
    """

    def __init__(self, components: list[Component]):
        self.components = components

    def __len__(self):
        return len(self.components)

    def __getitem__(self, index):
        return self.components[index]

    def __iter__(self):
        return iter(self.components)

    # ----- Building -----

    @classmethod
    def init_from_params(cls, params: Bunch) -> CompList:
        """The views of every enabled component under `components` in the parameter file.

        A component has one view per polarization it is defined for: I, QU, or both for IQU. The
        list does not depend on the MPI layout: a view whose polarization the run does not solve
        stays at its starting value.
        """
        comp_list = []
        comp_name_of_shortname = {}
        for comp_name in params.components:
            component = params.components[comp_name]
            if not component.enabled:
                continue
            component_cls = getattr(sky, component.component_class, None)
            if not isinstance(component_cls, type) or not issubclass(component_cls, Component):
                raise ValueError(f"Unknown component_class {component.component_class!r} for "
                                 f"component {comp_name!r}.")
            if ("polarization" not in component.params
                    or component.params.polarization not in EXECUTION_POLS):
                raise ValueError(f"'components.{comp_name}.params.polarization' must be "
                                 f"one of {list(EXECUTION_POLS)}.")
            for pol in EXECUTION_POLS[component.params.polarization]:
                comp = component_cls(params, comp_name, pol)
                comp_list.append(comp)
            # The chain stores each component under its shortname, so two components cannot
            # share one.
            if comp.shortname in comp_name_of_shortname:
                raise ValueError(f"Shortname {comp.shortname!r} is used for both "
                                 f"{comp_name_of_shortname[comp.shortname]!r} and "
                                 f"{comp.comp_name!r}.")
            comp_name_of_shortname[comp.shortname] = comp.comp_name
        return cls(comp_list)

    # ----- Reorganizing -----

    def split_for_pol(self, pol: str) -> CompList:
        """Return the views evaluated for one polarization, "I" or "QU".

        The returned list holds the same component objects as this one, not copies.
        """
        return CompList([comp for comp in self.components if comp.pol == pol])

    def joined(self) -> CompList:
        """Return one component per component name, with the I and QU views of each joined.

        The joined components are copies, with all polarization rows in `amps`.
        """
        views_of_name = {}  # Keeps the order in which the names first appear.
        for comp in self.components:
            if comp.comp_name not in views_of_name:
                views_of_name[comp.comp_name] = []
            views_of_name[comp.comp_name].append(comp)
        joined_components = []
        for views in views_of_name.values():
            if len(views) == 1:
                joined_components.append(copy.deepcopy(views[0]))
            else:
                joined_components.append(views[0].join_split_views(views[1]))
        return CompList(joined_components)

    # ----- The sky at a band -----

    def get_sky_at_band(self, band_map: DetectorMap) -> NDArray:
        """The sky as one band map sees it: at its frequency, resolution, beam and lmax.

        The result has the shape of `band_map.map_sky`, so `band_map.map_sky` minus it is the
        residual of the band.
        """
        return self.get_sky_at_nu(band_map.nu, band_map.nside, band_map.pol,
                                  fwhm=band_map.fwhm_rad, lmax=band_map.lmax)

    def get_sky_at_nu(self, nu: float, nside: int, pols_required: str, fwhm: float | None = None,
                      lmax: int | None = None) -> NDArray:
        """The sky of all components at one frequency, resolution, beam and lmax.

        Diffuse components go through `comp2band`, the CG solver's own operator (cut at the band
        lmax, mixing matrix, beam), and are then synthesized at `nside`, as C3's `evalBand` does:
        the sky compared with a band's data is exactly the model the amplitudes were solved for.
        Point sources are painted in pixel space and added, as in C3's `compute_residual`.

        The list may be the split views used during CompSep (`I` and `QU`) or a joined list
        with `IQU` components.

        Args:
            pols_required: The Stokes rows of the result: "I", "QU" or "IQU".
            fwhm: Beam to realize the sky through, in radians. `None` asks for the sharpest the
                amplitudes can give: the largest `amp_fwhm_rad` of the components (normally 0.0).
            lmax: The band lmax (`DetectorMap.lmax`). `None` means `3*nside - 1`, the default band
                lmax.

        Raises:
            ValueError: If `fwhm` is finer than the beam the amplitudes already carry.
        """
        # The beam the amplitudes carry: 0.0 after the CG solver, which gives a deconvolved sky,
        # and `compsep.common_res_fwhm` after the per-pixel solver.
        amp_fwhm = max((comp.amp_fwhm_rad for comp in self.components), default=0.0)
        if fwhm is None:
            fwhm = amp_fwhm
        elif fwhm < amp_fwhm and not np.isclose(fwhm, amp_fwhm):
            raise ValueError(
                f"Asked for the sky at a beam of {np.degrees(fwhm)*60:.4g} arcmin, but these "
                f"amplitudes already carry {np.degrees(amp_fwhm)*60:.4g} arcmin of smoothing and "
                f"cannot be realized any sharper. Pass fwhm=None for the sharpest available.")
        lmax = 3*nside - 1 if lmax is None else lmax
        nrows = len(pols_required)
        skymap = np.zeros((nrows, 12*nside**2), dtype=np.float32)
        diffuse = [comp for comp in self.components if isinstance(comp, DiffuseComponent)]
        # One band per view: intensity (map row 0, spin 0) and polarization (the last two rows,
        # spin 2). A joined IQU component contributes to both.
        for view, rows, spin in (("I", slice(0, 1), 0), ("QU", slice(nrows - 2, nrows), 2)):
            if view[0] not in pols_required:
                continue
            # Single precision, like `skymap`, which the result goes into.
            band = Band(np.zeros((len(view), hp.Alm.getsize(lmax)), dtype=np.complex64), nu,
                        np.degrees(fwhm)*60.0, nside)
            for comp in diffuse:
                if comp.pol in (view, "IQU"):
                    comp.comp2band(band)
            skymap[rows] += alm_to_map(band.alms, nside, lmax, spin=spin)
        if "I" in pols_required:
            for comp in self.components:  # Point sources.
                if not isinstance(comp, DiffuseComponent) and comp.pol == "I":
                    skymap[0] += comp.get_sky(nu, nside, fwhm)[0]
        return skymap

    # ----- The vector arithmetic of the CG -----
    # Two lists that are combined must hold the same views in the same order. Every CG vector is
    # made from the right-hand side with `copy` or `zeros_like`, which guarantees it.

    def copy(self) -> CompList:
        """Return a copy with its own amplitude arrays.

        Only `amps` is copied. Everything else in a component (settings, the C_l prior, the pixel
        discs of point sources) is shared with the original, and must not be changed in place.
        """
        new_components = []
        for comp in self.components:
            new_comp = copy.copy(comp)  # A new object whose attributes are the original's.
            new_comp.amps = comp.amps.copy()
            new_components.append(new_comp)
        return CompList(new_components)

    def zeros_like(self) -> CompList:
        """As `copy`, but with all amplitudes set to zero."""
        zeros = self.copy()
        for comp in zeros:
            comp.amps[:] = 0
        return zeros

    def dot(self, other: CompList) -> float:
        """The dot product of two lists: the sum of the dot products of their components.

        Each component class defines its own dot product (`Component.dot`), because complex alms
        and point-source amplitudes need different ones.
        """
        res = 0.0
        for comp, other_comp in zip(self.components, other.components):
            res += float(comp.dot(other_comp))
        return res

    def add_scaled(self, other: CompList, scalar: float) -> None:
        """`self += scalar*other`, in place."""
        for comp, other_comp in zip(self.components, other.components):
            inplace_add_scaled_vec(comp.amps, other_comp.amps, scalar)

    def scale_and_add(self, other: CompList, scalar: float) -> None:
        """`self = scalar*self + other`, in place."""
        for comp, other_comp in zip(self.components, other.components):
            inplace_scale_add(comp.amps, other_comp.amps, scalar)

    # ----- The chain file and MPI -----

    def write_chain(self, file: h5py.File, band_frequencies: dict[str, float]) -> None:
        """Write every component's sample into `file`, as one `comps/<shortname>/` group each.

        Split I and QU views are joined first, so each component is written once with all its
        polarization rows. `band_frequencies` maps each band name to its centre frequency in GHz.
        """
        for comp in self.joined():
            comp.write_chain(file.create_group(f"comps/{comp.shortname}"), band_frequencies)

    def broadcast_pol_views(self, comm: MPI.Comm, *, pol: str, source: int) -> None:
        """Broadcast all execution views of `pol` from `source` to every rank in `comm`.

        Used after a sampling step: only the ranks that actually solved a given polarization hold
        the updated component data, so broadcasting that polarization's views from one authoritative
        `source` rank restores a globally consistent component list.
        """
        for comp in self.split_for_pol(pol):
            comp.bcast_data(comm, root=source)
            # The amplitudes' resolution travels with them: the solving rank's value is authoritative.
            comp.amp_fwhm_rad = comm.bcast(comp.amp_fwhm_rad, root=source)
            # FIXME: The above feels fragile: if we add more attributes that could change during
            # the amplitude solve, they would have to be added here. We could either introduce an
            # amplitude object that holds related information as well as the alms, or at least make
            # things more visible by specifying what must be transferred in __init__:
            #     _amp_metadata_attrs: tuple[str, ...] = ("amp_fwhm_rad",)
