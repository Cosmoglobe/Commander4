"""The abstract base class every sky component inherits from.

`Component` owns the parts shared by every component regardless of how it is represented on the
sky: the amplitude buffer, the polarization view, the parameter unpacking, and the interface
(`get_sed`, `realize_as_map`, ...) that the samplers call. The concrete families live next to this
file: `diffuse_components.py` and `point_sources.py`.
"""
from copy import deepcopy
import logging

import h5py
import numpy as np
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.polarization import assert_pol_supported
from commander4.math_utils import arithmetic
from commander4.parameters.schema import resolve_param

logger = logging.getLogger(__name__)

class Component:
    """Abstract base class for every sky component.

    Holds the amplitude buffer, the polarization view, and the interface the samplers call.
    Concrete families are defined in `diffuse_components.py` and `point_sources.py`.
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

    # Amplitude of the C(l) prior. None means no prior on the amplitudes: the CG solver then leaves
    # out every prior term (C3's `CL_TYPE = none`). `DiffuseComponent` reads it from the parameter
    # file.
    Cl_prior_amplitude = None

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
                 eval_pol: str | None = None, allocate_empty_amps: bool = False):
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
        # The amplitudes the CompSep solves for, as one array: alms of shape (npol, nalm) for a
        # diffuse component, one value per source for point sources. None until allocated.
        self.amps = None
        # FWHM beam of the component. If the CG solver was used, this will be 0, as it solves for
        # deconvolved components. Only non-zero if the common-resolution per-pix solver was used.
        self.amp_fwhm_rad = 0.0

    def join_split_views(self, other: "Component") -> "Component":
        """Join the I view and the QU view of one IQU component into a new component.

        The result has `eval_pol` "IQU" and all three polarization rows in `amps`.
        """
        if {self.eval_pol, other.eval_pol} != {"I", "QU"}:
            raise ValueError("Joining requires one intensity view and one QU view.")
        intensity_comp, pol_comp = (self, other) if self.eval_pol == "I" else (other, self)
        joined = deepcopy(intensity_comp)
        joined.eval_pol = joined.defined_pol
        joined.amps = np.concatenate((intensity_comp.amps, pol_comp.amps), axis=0)

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

    def write_chain(self, group: h5py.Group, band_frequencies: dict[str, float]) -> None:
        """Write this component's sample into its chain group, `comps/<shortname>/`.

        Writes what every component shares; subclasses add their amplitudes. `band_frequencies`
        maps each band name to its centre frequency in GHz, for the per-band mixing coefficients.
        """
        group["comp_name"] = self.comp_name
        group["shortname"] = self.shortname
        # The beam these amplitudes carry, in arcmin. This is 0.0 if the CG compsep was used,
        # as it solves for the intrinsic sky. However, if the per-pix compsep solver was used,
        # the sky components are smoothed to this resolution.
        group["amp_fwhm_arcmin"] = np.degrees(self.amp_fwhm_rad)*60.0
        # The SED parameters this component type declares (see `sed_param_names`), so the chain
        # records the sampled spectral indices and everything else needed to evaluate the SED.
        # `nu_ref` may be a per-polarization array, hence no float() cast. A parameter the
        # component leaves unset (None) is not written.
        for param_name in self.sed_param_names:
            value = getattr(self, param_name)
            if value is not None:
                group[f"sed/{param_name}"] = value
        # The mixing "matrix", C3's mixmat_<comp>_<band>. C4 indices are scalar, so a
        # component's whole mixing matrix at a band is the single number `get_sed(nu)`.
        for band_name, nu in band_frequencies.items():
            group[f"mixing/{band_name}"] = self.get_sed(nu)
        if self.defined_pol is not None:
            group["defined_pol"] = self.defined_pol
        if self.eval_pol is not None:
            group["eval_pol"] = self.eval_pol

    def read_chain(self, group: h5py.Group) -> None:
        """Restore this component's sampled state from its chain group, as `write_chain` wrote it.

        Only what a sampler changes is restored; fixed parameters keep their parameter-file value,
        so an edit there takes effect in a run started from another run's chain. The group holds
        the joined component (I and QU together), while `self` may be one polarization view.

        This base version restores the amplitudes' beam, and `beta` when `sample_spectral_index`
        is on (the switch of the spectral-index sampler). Subclasses add their amplitudes, and
        any other sampled state, gated by the switch of the sampler that changes it.
        """
        self.amp_fwhm_rad = np.radians(float(group["amp_fwhm_arcmin"][()])/60.0)
        if hasattr(self, "beta") and "sample_spectral_index" in self.comp_params \
                and self.comp_params.sample_spectral_index:
            beta = np.asarray(group["sed/beta"][()])
            if beta.ndim == 1:  # Joined I and QU views with different values store [I, QU].
                beta = beta[0 if self.eval_pol == "I" else 1]
            self.beta = float(beta)
            logger.debug(f"Component {self.comp_name!r} ({self.eval_pol}): restored sampled "
                         f"beta = {self.beta}.")

    def dot(self, other: "Component") -> float:
        """The dot product of this component's amplitudes with those of `other`.

        `other` must be the same view of the same component. The CG uses this through
        `CompList.dot`. This is the plain dot product; `DiffuseComponent` has its own, for alms.
        """
        return float(arithmetic.dot(self.amps, other.amps))

    def bcast_data_blocking(self, comm:MPI.Comm, root=0):
        """Broadcast the component's data array from the root MPI rank."""
        if not isinstance(self.amps, np.ndarray):
            raise RuntimeError("Component data must be allocated as an array before broadcast.")
        comm.Bcast(self.amps, root=root)

    def accum_data_blocking(self, comm:MPI.Comm, root=0):
        """Sum the component's data array across the communicator onto the root rank."""
        if not isinstance(self.amps, np.ndarray):
            raise RuntimeError("Component data must be allocated as an array before accumulation.")
        myrank=comm.Get_rank()
        send, recv = (MPI.IN_PLACE, self.amps) if myrank == root else (self.amps, None)
        comm.Reduce(send, recv, op=MPI.SUM, root=root)

    def accum_data_non_blocking(self, comm:MPI.Comm, root=0):
        """As `accum_data_blocking`, but returns the MPI request instead of waiting on it."""
        if not isinstance(self.amps, np.ndarray):
            raise RuntimeError("Component data must be allocated as an array before accumulation.")
        myrank=comm.Get_rank()
        send, recv = (MPI.IN_PLACE, self.amps) if myrank == root else (self.amps, None)
        req = comm.Ireduce(send, recv, op=MPI.SUM, root=root)
        return req
