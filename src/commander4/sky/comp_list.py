"""`CompList`: the list of components the samplers treat as a single vector.

Component separation solves for all component amplitudes at once, so the CG driver needs to add,
scale and dot whole lists of components as if they were vectors. `CompList` provides that view,
plus construction of the components named in the parameter file and the per-polarization execution
views. The free functions at the end are the vector operations the CG driver calls.
"""
from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.data_models.band import Band
import commander4.sky as sky
from commander4.sky.comp_io import _load_component_alms, _read_view_alms_from_fits
from commander4.sky.component import Component
from commander4.sky.diffuse_components import DiffuseComponent
from commander4.polarization import EXECUTION_POLS
from commander4.polarization import assert_pol_supported
from commander4.math_utils.arithmetic import inplace_scale_add, inplace_add_scaled_vec

logger = logging.getLogger(__name__)


class CompList:
    """A list of component execution views, usable as a single vector.

    Supports the arithmetic the CG driver needs (add, scale, dot) by applying it componentwise, and
    owns construction of the components named in the parameter file.
    """

    def __init__(self, comp_list:list[Component]):
        self._validate_comp_list(comp_list)
        self.comp_list = comp_list

    @staticmethod
    def _group_by_logical_key(
        comp_list: list[Component],
    ) -> list[tuple[tuple[type["Component"], str], list[Component]]]:
        grouped_components = {}
        logical_order = []
        for comp in comp_list:
            if comp.logical_key not in grouped_components:
                grouped_components[comp.logical_key] = []
                logical_order.append(comp.logical_key)
            grouped_components[comp.logical_key].append(comp)
        return [(logical_key, grouped_components[logical_key]) for logical_key in logical_order]

    @staticmethod
    def _partition_execution_views(
        group: list[Component],
    ) -> tuple[list[Component], list[Component]]:
        split_views = [comp for comp in group if comp.is_split_view]
        unsplit_views = [comp for comp in group if not comp.is_split_view]
        return split_views, unsplit_views

    @staticmethod
    def _validate_comp_list(comp_list: list[Component]) -> None:
        """Check that a component list has a coherent logical and execution-view layout."""
        if not isinstance(comp_list, list):
            raise TypeError("comp_list must be a list of Component objects.")

        shortname_to_comp_name = {}
        for idx, comp in enumerate(comp_list):
            if not isinstance(comp, Component):
                raise TypeError(f"comp_list[{idx}] must be a Component.")
            if comp.defined_pol is not None:
                assert_pol_supported(comp.defined_pol)
            if comp.eval_pol is not None:
                assert_pol_supported(comp.eval_pol)

            prev_comp_name = shortname_to_comp_name.get(comp.shortname)
            if prev_comp_name is not None and prev_comp_name != comp.comp_name:
                raise ValueError(
                    f"Shortname {comp.shortname!r} is used for both {prev_comp_name!r} and "
                    f"{comp.comp_name!r}."
                )
            shortname_to_comp_name[comp.shortname] = comp.comp_name

        for logical_key, group in CompList._group_by_logical_key(comp_list):
            comp_name = logical_key[1]
            component_types = {type(comp) for comp in group}
            if len(component_types) > 1:
                raise ValueError(
                    f"Component name {comp_name!r} is shared across multiple component classes."
                )

            shortnames = {comp.shortname for comp in group}
            if len(shortnames) > 1:
                raise ValueError(
                    f"Component name {comp_name!r} is associated with multiple shortnames: "
                    f"{sorted(shortnames)!r}."
                )

            split_views, unsplit_views = CompList._partition_execution_views(group)
            if split_views and unsplit_views:
                raise ValueError(
                    f"Component name {comp_name!r} mixes split and unsplit execution views."
                )
            if len(unsplit_views) > 1:
                raise ValueError(f"Duplicate logical component {comp_name!r}.")
            if len(split_views) > 2:
                raise ValueError(f"Component {comp_name!r} has too many split execution views.")
            split_pols = [comp.eval_pol for comp in split_views]
            if len(set(split_pols)) != len(split_pols):
                raise ValueError(f"Component {comp_name!r} repeats a split execution view.")

    @classmethod
    def init_from_params(cls, components:Bunch, params:Bunch):
        # Build the full logical component list: every enabled component contributes one execution
        # view per polarization it defines (I, QU, or both for an IQU component). Construction is
        # deliberately independent of the MPI/compsep layout: a view whose polarization is not
        # actually solved or used in a given run stays inert at its initial value.
        comp_list = []
        for component_str in components:
            component = components[component_str]
            if not component.enabled:
                continue
            component_cls = getattr(sky, component.component_class, None)
            if not isinstance(component_cls, type) or not issubclass(component_cls, Component):
                raise ValueError(f"Unknown component_class {component.component_class!r} for "
                                 f"component {component._name!r}.")
            if "lmax" in component.params and component.params.lmax == "full":
                # "The most a map at the compsep resolution can carry", which is also the default
                # band lmax (param_schema.resolve_band_lmax), so such a component is never solved
                # for modes no band can see.
                component.params.lmax = 3*params.compsep.nside - 1
            component_pol = component.params.polarization if "polarization" in component.params \
                else "I"
            if component_pol not in EXECUTION_POLS:
                raise ValueError(
                    f"Unrecognized polarization in parameter file for component {component_str}")
            for eval_pol in EXECUTION_POLS[component_pol]:
                comp_list.append(component_cls(component.params, params.compsep, eval_pol=eval_pol,
                                               comp_name=component._name, allocate_empty_alms=True))
        return cls(comp_list)

    def load_initial_alms(self, params: Bunch) -> None:
        """Populate each component's alms with an initial guess read from a file.

        For every component the source is its own ``init_from`` parameter (inside the component's
        ``params`` block) if present, otherwise the global ``params.gibbs.init_from_chain``. The
        source may be a compsep chain (``.h5``/``.hd5``, alms read directly) or a FITS sky map
        (``.fits``, transformed to alms); the type is decided by the file extension. If neither path
        is set the alms are left at their allocated value (zeros). Only diffuse (alm-based)
        components are supported for now.
        """
        global_path = params.gibbs.init_from_chain if "init_from_chain" in params.gibbs else None
        for comp in self.comp_list:
            has_explicit_path = "init_from" in comp.comp_params
            source_path = comp.comp_params.init_from if has_explicit_path else global_path
            if not source_path:
                continue  # No initial guess requested; leave the allocated zeros.
            if not isinstance(comp, DiffuseComponent):
                if has_explicit_path:
                    raise ValueError(
                        f"Component {comp.comp_name!r}: 'init_from' is currently only supported "
                        "for diffuse (alm-based) components.")
                continue
            _load_component_alms(comp, source_path)

    def load_amp_prior_means(self) -> None:
        """Populate each component's prior mean mu from its ``amp_prior_mean_map`` parameter.

        This is C3's ``COMP_AMP_PRIOR_MAP``: a FITS sky map giving the mean of the Gaussian
        amplitude prior a ~ N(mu, S). It goes through the same conversion as an ``init_from`` map
        (unit conversion at the component's reference frequency, truncation to l = 3*nside-1, and
        the iterative LSMR inverse SHT), because mu and the amplitudes live in the same space and
        must be treated identically. C3 instead uses a single non-iterative `YtW` analysis here,
        which is the less accurate of the two on a HEALPix grid.

        Read once and cached on the component (C3 likewise transforms its mu once, at component
        initialization). The S^{-1/2} that turns mu into a right-hand-side term is *not* applied
        here but per solve, since the prior S may change between Gibbs iterations while mu does not.
        """
        for comp in self.comp_list:
            source_path = comp.comp_params.amp_prior_mean_map \
                if "amp_prior_mean_map" in comp.comp_params else None
            if not source_path:
                continue  # Zero-mean prior.
            if not isinstance(comp, DiffuseComponent):
                raise ValueError(
                    f"Component {comp.comp_name!r}: 'amp_prior_mean_map' is only supported for "
                    "diffuse (alm-based) components.")
            if not str(source_path).lower().endswith(".fits"):
                raise ValueError(
                    f"Component {comp.comp_name!r}: 'amp_prior_mean_map' must be a .fits sky map, "
                    f"but got {source_path!r}.")
            view_alms = _read_view_alms_from_fits(comp, source_path)
            if view_alms is not None:
                comp.amp_prior_mean = view_alms.astype(comp.dtype, copy=False)
                logger.info(f"Component {comp.comp_name!r} ({comp.eval_pol}): prior mean read from "
                            f"{source_path!r}.")

    def _assert_consistent_comps(self, other: "CompList") -> None:
        if not isinstance(other, CompList):
            raise TypeError("Both operands must be CompList objects.")
        if len(self.comp_list) != len(other.comp_list):
            raise ValueError("Component lists must match in length.")
        self_keys = [comp.execution_key for comp in self.comp_list]
        other_keys = [comp.execution_key for comp in other.comp_list]
        if self_keys != other_keys:
            raise ValueError("Component lists must contain the same execution views in the same order.")

    def components_for_eval_pol(self, target_pol: str) -> list[Component]:
        assert_pol_supported(target_pol)
        return [comp for comp in self.comp_list if comp.eval_pol == target_pol]

    def split_for_eval_pol(self, target_pol: str) -> "CompList":
        """Return the execution-view subset evaluated for one polarization stream."""
        return CompList(self.components_for_eval_pol(target_pol))

    def copy_matching_data_from(self, other: "CompList") -> None:
        if not isinstance(other, CompList):
            raise TypeError("Input must be a CompList.")
        other_by_key = {}
        for comp in other.comp_list:
            if comp.execution_key in other_by_key:
                raise ValueError(f"Duplicate component execution key {comp.execution_key!r}.")
            other_by_key[comp.execution_key] = comp
        self_keys = {comp.execution_key for comp in self.comp_list}
        extra_keys = [key for key in other_by_key if key not in self_keys]
        if extra_keys:
            raise ValueError(f"Found unknown components in source CompList: {extra_keys!r}")
        for comp in self.comp_list:
            other_comp = other_by_key.get(comp.execution_key)
            if other_comp is None:
                continue
            comp._assert_consistent_comp(other_comp)
            np.copyto(comp._data, other_comp._data)
            comp.amp_fwhm_rad = other_comp.amp_fwhm_rad

    def broadcast_pol_views(self, comm: MPI.Comm, *, eval_pol: str, source: int) -> None:
        """Broadcast all execution views of `eval_pol` from `source` to every rank in `comm`.

        Used after a sampling step: only the ranks that actually solved a given polarization hold
        the updated component data, so broadcasting that polarization's views from one authoritative
        `source` rank restores a globally consistent component list.
        """
        for comp in self.components_for_eval_pol(eval_pol):
            comp.bcast_data_blocking(comm, root=source)
            # The amplitudes' resolution travels with them: the solving rank's value is authoritative.
            comp.amp_fwhm_rad = comm.bcast(comp.amp_fwhm_rad, root=source)
            # FIXME: The above feels fragile: if we add more attributes that could change during
            # the amplitude solve, they would have to be added here. We could either introduce an
            # amplitude object that holds related information as well as the alms, or at least make
            # things more visible by specifying what must be transferred in __init__:
            #     _amp_metadata_attrs: tuple[str, ...] = ("amp_fwhm_rad",)

    def joined(self) -> "CompList":
        """Collapse split execution views back to one logical component per `comp_name`."""
        joined_components = []
        for logical_key, group in self._group_by_logical_key(self.comp_list):
            split_views, unsplit_views = self._partition_execution_views(group)
            if unsplit_views and split_views:
                raise ValueError(
                    f"Logical component {logical_key[1]!r} mixes split and unsplit execution views."
                )
            if len(unsplit_views) > 1:
                raise ValueError(f"Duplicate unsplit component {logical_key[1]!r}.")
            if unsplit_views:
                joined_components.append(deepcopy(unsplit_views[0]))
                continue
            if len(split_views) == 1:
                joined_components.append(deepcopy(split_views[0]))
                continue
            if len(split_views) != 2:
                raise ValueError(
                    f"Expected one or two execution views for {logical_key[1]!r}, got {len(group)}."
                )
            joined_components.append(split_views[0].join_split_views(split_views[1]))

        return CompList(joined_components)
    
    @property
    def components(self):
        return self.comp_list
    
    def __len__(self):
        return len(self.comp_list)
    
    def __matmul__(self, other) -> float:
        """ `dot(comp_list1, comp_list2)`. Calculates the correct dot product between two lists of
            Component objects where the alms follow the Healpy complex storing convention, for
            components with alms. It will automatically handle the correct dot product definition for
            each type of Component.
        """
        self._assert_consistent_comps(other)
        res = 0.0
        for c1, c2 in zip(self.components, other.components):
            res += float(c1 @ c2)
        return res

    def _apply_componentwise_op(self, other: "CompList", component_op, *, inplace: bool) -> "CompList":
        self._assert_consistent_comps(other)
        target = self if inplace else deepcopy(self)
        for target_comp, other_comp in zip(target.components, other.components):
            component_op(target_comp, other_comp)
        return target
    
    def __add__(self, other):
        return self._apply_componentwise_op(other, Component.__iadd__, inplace=False)

    def __iadd__(self, other):
        return self._apply_componentwise_op(other, Component.__iadd__, inplace=True)

    def __sub__(self, other):
        return self._apply_componentwise_op(other, Component.__isub__, inplace=False)

    def __isub__(self, other):
        return self._apply_componentwise_op(other, Component.__isub__, inplace=True)

    def __mul__(self, other):
        return self._apply_componentwise_op(other, Component.__imul__, inplace=False)

    def __imul__(self, other):
        return self._apply_componentwise_op(other, Component.__imul__, inplace=True)

    def __truediv__(self, other):
        return self._apply_componentwise_op(other, Component.__itruediv__, inplace=False)

    def __itruediv__(self, other):
        return self._apply_componentwise_op(other, Component.__itruediv__, inplace=True)

    def __getitem__(self, index):
        return self.comp_list[index]

    def __iter__(self):
        for item in self.comp_list:
            yield item

    def __array_function__(self, func, types, args, kwargs):
        # Lets numpy functions such as np.zeros_like dispatch onto CompList.
        if not all(issubclass(t, CompList) for t in types):
            return NotImplemented

        if func is np.zeros_like:
            return self._zeros_like(*args, **kwargs)

        return NotImplemented

    def _zeros_like(self, other, dtype=None, order='K', subok=True, shape=None):
        out = deepcopy(other)
        out.comp_list = [
            np.zeros_like(c,
            dtype=dtype,
            order=order,
            subok=subok,
            shape=shape) for c in other
            ]
        return out

    # MPI functions
    def bcast_data_blocking(self, comm:MPI.Comm, root:int=0):
        for comp in self.comp_list:
            comp.bcast_data_blocking(comm, root=root)
    
    def accum_data_blocking(self, comm:MPI.Comm, root:int=0):
        for comp in self.comp_list:
            comp.accum_data_blocking(comm, root=root)

    def accum_data_non_blocking(self, comm:MPI.Comm, root:int=0) -> list[MPI.Request]:
        requests = []
        for comp in self.comp_list:
            req = comp.accum_data_non_blocking(comm, root=root)
            requests.append(req)
        return requests

    # CompSep solver functions
    def eval_comp_from_band(self, band_in:Band, nthreads:int=1):
        """ Evaluates the band_in's contribution to all the comp_list_out objects, and stores them
            in-place.
        """
        for comp in self.comp_list:
            comp.eval_comp_from_band(band_in, nthreads=nthreads)
    
    def project_comp_to_band(self, band_out:Band, nthreads:int=1) -> NDArray[np.complexfloating]:
        """ Projects all the components in this list, overwriting the `band_out` object's alms.
        """
        band_out.alms = np.zeros_like(band_out.alms)
        for comp in self.comp_list:
            comp.project_comp_to_band(band_out, nthreads=nthreads)

    def apply_Cl_prior_sqrt(self):
        """ Applies to each component its own C_l prior square root (S^{1/2}), acting on its alms.

        Takes no target argument, unlike the per-component method: each component has its own C_l
        prior and its own alms, so there is no single array a list-level call could scale.
        """
        for comp in self.comp_list:
            comp.apply_Cl_prior_sqrt(comp.alms)

    def inplace_add_scaled(self, list_other, scalar):
        """ `list_inplace += scalar*list_other`
        """
        self._assert_consistent_comps(list_other)
        
        for ci, co in zip(self.comp_list, list_other.comp_list):
            inplace_add_scaled_vec(ci._data, co._data, scalar)
    
    def inplace_scale_and_add(self, list_other, scalar):
        """ `list_inplace = scalar*list_inplace + list_other`
        """
        self._assert_consistent_comps(list_other)

        for ci, co in zip(self.comp_list, list_other.comp_list):
            inplace_scale_add(ci._data, co._data, scalar)


# These functions are common array operations, but made to work on the comp-lists, which are
# lists of Component objects, each containing component-specifically formatted data.

def inplace_complist_add_scaled_array(list_inplace:list[Component], list_other:list[Component],
                                      scalar):
    """ `list_inplace += scalar*list_other`
    """
    if len(list_inplace) != len(list_other):
        raise ValueError("Component lists must match in length.")
    
    for ci, co in zip(list_inplace, list_other):
        inplace_add_scaled_vec(ci._data, co._data, scalar)

def inplace_complist_scale_and_add(list_inplace:list[Component], list_other:list[Component], scalar):
    """ `list_inplace = scalar*list_inplace + list_other`
    """
    if len(list_inplace) != len(list_other):
        raise ValueError("Component lists must match in length.")

    for ci, co in zip(list_inplace, list_other):
        inplace_scale_add(ci._data, co._data, scalar)

def complist_dot(comp_list1:CompList, comp_list2:CompList) -> float:
    """ `dot(comp_list1, comp_list2)`. Calculates the correct dot product between two lists of
        Component objects where the alms follow the Healpy complex storing convention, for
        components with alms. It will automatically handle the correct dot product definition for
        each type of Component.
    """
    comp_list1._assert_consistent_comps(comp_list2)
    if len(comp_list1) == 0:
        logger.warning("Dot product between empty component lists; result is 0.")
    res = 0.0
    for c1, c2 in zip(comp_list1, comp_list2):
        res += float(c1 @ c2)
    return res

def complist_norm(comp_list:list[Component]) -> float:
    """ `norm(comp_list)`. The Euclidean (L2) norm of a list of Component objects, treating it as a
        single vector of values.
    """
    return float(np.sqrt(complist_dot(comp_list, comp_list)))
