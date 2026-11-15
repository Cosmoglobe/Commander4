"""`CompList`: an ordered list of component views, which the CG can use as a vector.

A component defined for IQU is stored as two views, one for I and one for QU, because the two are
solved on separate MPI ranks. `CompList` holds four kinds of method, and nothing else:

- building the views from the parameter file (`init_from_params`),
- reorganizing them (`split_for_eval_pol`, `joined`),
- the vector arithmetic of the CG (`zeros_like`, `copy`, `dot`, `add_scaled`, `scale_and_add`),
- files and MPI (`write_chain`, `load_initial_state`, `load_amp_prior_means`,
  `broadcast_pol_views`).

Physics that acts on one component at a time (`comp2band`, the C_l prior, ...) is not passed down
through the list: the code that needs it loops over the components itself.
"""
from __future__ import annotations

import copy
import logging

import h5py
from mpi4py import MPI
from pixell.bunch import Bunch

import commander4.sky as sky
from commander4.sky.comp_io import _read_view_alms_from_fits
from commander4.sky.component import Component
from commander4.sky.diffuse_components import DiffuseComponent
from commander4.polarization import EXECUTION_POLS
from commander4.polarization import assert_pol_supported
from commander4.math_utils.arithmetic import inplace_scale_add, inplace_add_scaled_vec
from commander4.parameters.schema import resolve_param

logger = logging.getLogger(__name__)


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
    def init_from_params(cls, components:Bunch, params:Bunch):
        # Build the full logical component list: every enabled component contributes one execution
        # view per polarization it defines (I, QU, or both for an IQU component). Construction is
        # deliberately independent of the MPI/compsep layout: a view whose polarization is not
        # actually solved or used in a given run stays inert at its initial value.
        resolve_param(params, "double_precision", ("compsep",), default=False, legal_types=bool)
        comp_list = []
        comp_name_of_shortname = {}
        for component_str in components:
            component = components[component_str]
            if not component.enabled:
                continue
            component_cls = getattr(sky, component.component_class, None)
            if not isinstance(component_cls, type) or not issubclass(component_cls, Component):
                raise ValueError(f"Unknown component_class {component.component_class!r} for "
                                 f"component {component._name!r}.")
            component_pol = component.params.polarization if "polarization" in component.params \
                else "I"
            if component_pol not in EXECUTION_POLS:
                raise ValueError(
                    f"Unrecognized polarization in parameter file for component {component_str}")
            for eval_pol in EXECUTION_POLS[component_pol]:
                comp = component_cls(component.params, params.compsep, eval_pol=eval_pol,
                                     comp_name=component._name, allocate_empty_amps=True)
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

    def split_for_eval_pol(self, target_pol: str) -> CompList:
        """Return the views evaluated for one polarization, "I" or "QU".

        The returned list holds the same component objects as this one, not copies.
        """
        assert_pol_supported(target_pol)
        return CompList([comp for comp in self.components if comp.eval_pol == target_pol])

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

    # ----- Files and MPI -----

    def write_chain(self, file: h5py.File, band_frequencies: dict[str, float]) -> None:
        """Write every component's sample into `file`, as one `comps/<shortname>/` group each.

        Split I and QU views are joined first, so each component is written once with all its
        polarization rows. `band_frequencies` maps each band name to its centre frequency in GHz.
        """
        for comp in self.joined():
            comp.write_chain(file.create_group(f"comps/{comp.shortname}"), band_frequencies)

    def load_initial_state(self, chain_file: str | None, apply_init_from: bool) -> None:
        """Set each component's starting state from a CompSep chain file or its own `init_from`.

        Per component, the source is its `init_from` parameter if `apply_init_from` is set and the
        component has one, and `chain_file` otherwise. So `init_from: null` keeps a component at its
        parameter-file initialization. The file type decides how it is read: a `.h5`/`.hd5` chain
        file goes through the component's own `read_chain`, and a `.fits` sky map is transformed to
        diffuse alms. Without a source, a component keeps its parameter-file values and zero
        amplitudes.

        Args:
            chain_file: The CompSep chain file this chain starts from, or None.
            apply_init_from: False when `chain_file` is this chain's own saved state (on
                `gibbs.chain_status: append`), which must win over the initial guesses.
        """
        for comp in self.components:
            source = chain_file
            if apply_init_from and "init_from" in comp.comp_params:
                source = comp.comp_params.init_from
            if not source:
                continue
            if str(source).lower().endswith(".fits"):
                if not isinstance(comp, DiffuseComponent):
                    raise ValueError(f"Component {comp.comp_name!r}: a FITS 'init_from' map only "
                                     "works for diffuse (alm-based) components.")
                view_alms = _read_view_alms_from_fits(comp, source)
                if view_alms is not None:
                    comp.amps = view_alms.astype(comp.amps.dtype, copy=False)
            elif str(source).lower().endswith((".h5", ".hd5")):
                with h5py.File(source, "r") as file:
                    group_path = f"comps/{comp.shortname}"
                    if group_path not in file:
                        logger.error(f"Component {comp.comp_name!r} (shortname "
                                     f"{comp.shortname!r}) not found in {source!r}; keeping its "
                                     "parameter-file initialization.")
                        continue
                    comp.read_chain(file[group_path])
            else:
                raise ValueError(f"Unsupported init file {source!r} for component "
                                 f"{comp.comp_name!r}: expected a .h5/.hd5 chain or a .fits map.")

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
        for comp in self.components:
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
            if comp.Cl_prior_amplitude is None:
                raise ValueError(
                    f"Component {comp.comp_name!r} ({comp.eval_pol}): 'amp_prior_mean_map' needs a "
                    "prior to be the mean of, but 'Cl_prior_amplitude' is null (no prior).")
            view_alms = _read_view_alms_from_fits(comp, source_path)
            if view_alms is not None:
                comp.amp_prior_mean = view_alms.astype(comp.amps.dtype, copy=False)
                logger.info(f"Component {comp.comp_name!r} ({comp.eval_pol}): prior mean read from "
                            f"{source_path!r}.")

    def broadcast_pol_views(self, comm: MPI.Comm, *, eval_pol: str, source: int) -> None:
        """Broadcast all execution views of `eval_pol` from `source` to every rank in `comm`.

        Used after a sampling step: only the ranks that actually solved a given polarization hold
        the updated component data, so broadcasting that polarization's views from one authoritative
        `source` rank restores a globally consistent component list.
        """
        for comp in self.split_for_eval_pol(eval_pol):
            comp.bcast_data_blocking(comm, root=source)
            # The amplitudes' resolution travels with them: the solving rank's value is authoritative.
            comp.amp_fwhm_rad = comm.bcast(comp.amp_fwhm_rad, root=source)
            # FIXME: The above feels fragile: if we add more attributes that could change during
            # the amplitude solve, they would have to be added here. We could either introduce an
            # amplitude object that holds related information as well as the alms, or at least make
            # things more visible by specifying what must be transferred in __init__:
            #     _amp_metadata_attrs: tuple[str, ...] = ("amp_fwhm_rad",)
