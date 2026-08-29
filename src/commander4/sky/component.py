"""The abstract base class every sky component inherits from.

`Component` owns the parts shared by every component regardless of how it is represented on the
sky: the amplitude buffer and its arithmetic, the polarization view, the parameter unpacking, and
the interface (`get_sed`, `realize_as_map`, ...) that the samplers call. The concrete families live
next to this file: `diffuse_components.py` and `point_sources.py`.
"""
from copy import deepcopy

import numpy as np
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.polarization import assert_pol_supported
from commander4.math_utils.arithmetic import dot, inplace_arr_add,\
        inplace_arr_sub, inplace_arr_prod, inplace_arr_truediv
from commander4.parameters.schema import resolve_param

class Component:
    """Abstract base class for every sky component.

    Holds the amplitude buffer and its arithmetic, the polarization view, and the interface the
    samplers call. Concrete families are defined in `diffuse_components.py` and `point_sources.py`.
    """

    default_shortname = "comp"
    legal_pols: tuple[str, ...] = ("I", "QU", "IQU")
    requires_defined_pol = False

    # Names of the instance attributes defining this component's SED. The compsep chain writer
    # stores each under `comps/<shortname>/sed/`, so a chain describes its own SED without the
    # parameter file. Fixed parameters (`nu_ref`, `T`) are listed alongside sampled ones, since
    # which are sampled is a property of the run rather than of the component.
    sed_param_names: tuple[str, ...] = ()

    # Names of the instance attributes defining this component's C(l) prior, stored under
    # `comps/<shortname>/Cl_prior/`. Commander3's `Dl_amp`/`Dl_beta`/`Dl_theta` analogue: the model
    # parameters, not the evaluated spectrum. Empty for components that carry no such prior.
    Cl_prior_param_names: tuple[str, ...] = ()

    @classmethod
    def _assert_legal_pol(cls, pol: str | None, *, role: str, required: bool = False) -> None:
        if pol is None:
            if required:
                raise ValueError(f"{cls.__name__} requires a defined polarization mode.")
            return
        assert_pol_supported(pol)
        if pol not in cls.legal_pols:
            raise ValueError(f"{cls.__name__} does not support {role} polarization {pol!r}. "
                             f"Allowed polarizations: {cls.legal_pols!r}.")

    def __init__(self, comp_params: Bunch, global_params: Bunch, *,
                 shortname: str | None = None, comp_name: str | None = None,
                 eval_pol: str | None = None, allocate_empty_alms: bool = False):
        self.comp_params = comp_params
        self.global_params = global_params
        self.shortname = (
            shortname
            if shortname is not None
            else comp_params.shortname if "shortname" in comp_params
            else self.default_shortname
        )
        self.comp_name = comp_params._name if comp_name is None else comp_name
        self.defined_pol = comp_params.polarization if "polarization" in comp_params else None
        type(self)._assert_legal_pol(
            self.defined_pol,
            role="defined",
            required=type(self).requires_defined_pol,
        )
        self.eval_pol = self.defined_pol if eval_pol is None else eval_pol
        type(self)._assert_legal_pol(self.eval_pol, role="evaluation")
        # Look for "double_precision" in the parameter Bunch given.
        self.double_prec = resolve_param(global_params, "double_precision", ("",),
                                         default=False, legal_types=bool)
        self._data = None
        # FWHM beam of the component. If the CG solver was used, this will be 0, as it solves for
        # deconvolved components. Only non-zero if the common-resolution per-pix solver was used.
        self.amp_fwhm_rad = 0.0

    @property
    def logical_id(self) -> str:
        return self.comp_name

    @property
    def logical_key(self) -> tuple[type["Component"], str]:
        return (type(self), self.logical_id)

    @property
    def execution_key(self) -> tuple[type["Component"], str, str | None]:
        return (type(self), self.logical_id, self.eval_pol)

    @property
    def is_split_view(self) -> bool:
        return self.defined_pol == "IQU" and self.eval_pol in ("I", "QU")

    @property
    def execution_label(self) -> str:
        if self.eval_pol is None or not self.is_split_view:
            return self.shortname
        return f"{self.shortname}[{self.eval_pol}]"

    def _assert_consistent_comp(self, other: "Component") -> None:
        if not isinstance(other, Component):
            raise TypeError("Both operands must be Component objects.")
        if type(self) is not type(other):
            raise TypeError("Both operands must be of the same Component type.")
        mismatched = [
            attr for attr in (
                "comp_name",
                "shortname",
                "defined_pol",
                "eval_pol",
            )
            if getattr(self, attr) != getattr(other, attr)
        ]
        if mismatched:
            raise ValueError(
                "Components must represent the same execution view. "
                f"Mismatched fields: {', '.join(mismatched)}"
            )
        if self._data is None or other._data is None:
            raise ValueError("Cannot operate on Components with no data.")
        if self._data.shape != other._data.shape:
            raise ValueError("Data arrays of the two Components must match in size.")

    def join_split_views(self, other: "Component") -> "Component":
        if not isinstance(other, Component):
            raise TypeError("Can only join Component objects.")
        if type(self) is not type(other):
            raise TypeError("Split views must be of the same Component type.")
        if self.defined_pol != "IQU" or other.defined_pol != "IQU":
            raise ValueError("Only IQU-defined components can be joined.")
        if not self.is_split_view or not other.is_split_view:
            raise ValueError("Only split component views can be joined.")
        if {self.eval_pol, other.eval_pol} != {"I", "QU"}:
            raise ValueError("Joining requires one intensity view and one QU view.")
        mismatched = [
            attr for attr in ("comp_name", "shortname", "defined_pol")
            if getattr(self, attr) != getattr(other, attr)
        ]
        if mismatched:
            raise ValueError(
                "Split views must refer to the same logical component. "
                f"Mismatched fields: {', '.join(mismatched)}"
            )
        if self._data is None or other._data is None:
            raise ValueError("Cannot join split views with no data.")
        intensity_comp, pol_comp = (self, other) if self.eval_pol == "I" else (other, self)
        if intensity_comp._data.shape[1:] != pol_comp._data.shape[1:]:
            raise ValueError("Split views must have compatible alm dimensions.")
        joined = deepcopy(intensity_comp)
        joined.eval_pol = joined.defined_pol
        joined._data = np.concatenate((intensity_comp._data, pol_comp._data), axis=0)

        # The joined view is deep-copied from the intensity view, so any SED or C(l)-prior parameter
        # given per polarization (`nu_ref: [I, QU]` is common) would otherwise silently keep only
        # the I value. Restore the [I, QU] pair, i.e. undo `_per_pol`, so the joined component still
        # describes both views. Joined components are only used on output paths (chain writing, name
        # and lmax reporting), never for SED or prior evaluation, which happen on the split views.
        for param_name in intensity_comp.sed_param_names + intensity_comp.Cl_prior_param_names:
            i_value, qu_value = getattr(intensity_comp, param_name), getattr(pol_comp, param_name)
            if not np.array_equal(i_value, qu_value):
                setattr(joined, param_name, np.array([i_value, qu_value]))
        return joined

    @property
    def amp_prior_mean(self):
        """The mean mu of the Gaussian prior on this component's amplitudes, or `None` if there is
        no prior.

        Defined on the base class, where it is always None; `DiffuseComponent` overrides it with a
        mean that can be read from a sky map.
        """
        return None

    def _apply_array_op(self, other: "Component", arr_op, *, inplace: bool) -> "Component":
        self._assert_consistent_comp(other)
        target = self if inplace else deepcopy(self)
        arr_op(target._data, other._data)
        return target

    def __add__(self, other):
        return self._apply_array_op(other, inplace_arr_add, inplace=False)
    
    def __iadd__(self, other):
        return self._apply_array_op(other, inplace_arr_add, inplace=True)
    
    def __sub__(self, other):
        return self._apply_array_op(other, inplace_arr_sub, inplace=False)
    
    def __isub__(self, other):
        return self._apply_array_op(other, inplace_arr_sub, inplace=True)
    
    def __mul__(self, other):
        return self._apply_array_op(other, inplace_arr_prod, inplace=False)
    
    def __imul__(self, other):
        return self._apply_array_op(other, inplace_arr_prod, inplace=True)
    
    def __truediv__(self, other):
        return self._apply_array_op(other, inplace_arr_truediv, inplace=False)
    
    def __itruediv__(self, other):
        return self._apply_array_op(other, inplace_arr_truediv, inplace=True)
    
    def __matmul__(self, other):
        self._assert_consistent_comp(other)
        return dot(self._data, other._data)

    def bcast_data_blocking(self, comm:MPI.Comm, root=0):
        """Broadcast the component's data array from the root MPI rank."""
        if not isinstance(self._data, np.ndarray):
            raise RuntimeError("Component data must be allocated as an array before broadcast.")
        comm.Bcast(self._data, root=root)

    def bcast_data_non_blocking(self, comm:MPI.Comm, root=0):
        """As `bcast_data_blocking`, but returns the MPI request instead of waiting on it."""
        if not isinstance(self._data, np.ndarray):
            raise RuntimeError("Component data must be allocated as an array before broadcast.")
        req = comm.Ibcast(self._data, root=root)
        return req

    def accum_data_blocking(self, comm:MPI.Comm, root=0):
        """Sum the component's data array across the communicator onto the root rank."""
        if not isinstance(self._data, np.ndarray):
            raise RuntimeError("Component data must be allocated as an array before accumulation.")
        myrank=comm.Get_rank()
        send, recv = (MPI.IN_PLACE, self._data) if myrank == root else (self._data, None)
        comm.Reduce(send, recv, op=MPI.SUM, root=root)

    def accum_data_non_blocking(self, comm:MPI.Comm, root=0):
        """As `accum_data_blocking`, but returns the MPI request instead of waiting on it."""
        if not isinstance(self._data, np.ndarray):
            raise RuntimeError("Component data must be allocated as an array before accumulation.")
        myrank=comm.Get_rank()
        send, recv = (MPI.IN_PLACE, self._data) if myrank == root else (self._data, None)
        req = comm.Ireduce(send, recv, op=MPI.SUM, root=root)
        return req

    def __array_function__(self, func, types, args, kwargs):
        # Lets numpy functions such as np.zeros_like dispatch onto Component.
        if not all(issubclass(t, Component) for t in types):
            return NotImplemented

        if func is np.zeros_like:
            return self._zeros_like(*args, **kwargs)

        return NotImplemented

    def _zeros_like(self, other, dtype=None, order='K', subok=True, shape=None):
        zeros = np.zeros_like(
            other._data,
            dtype=dtype,
            order=order,
            subok=subok,
            shape=shape,
        )
        out = deepcopy(other)
        out._data = zeros
        return out
