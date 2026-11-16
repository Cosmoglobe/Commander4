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

from commander4.math_utils import arithmetic
from commander4.parameters.schema import read_settings, resolve_param

logger = logging.getLogger(__name__)

class Component:
    """Abstract base class for every sky component.

    Holds the amplitude buffer, the polarization view, and the interface the samplers call.
    Concrete families are defined in `diffuse_components.py` and `point_sources.py`.
    """

    # The settings of the component's `params` block in the parameter file, with their defaults.
    # `self.cfg` holds them under the same names. Each subclass has its own `DEFAULTS`, written as
    # `{**Parent.DEFAULTS, ...}`: a copy of the parent's settings, with the listed ones added or
    # replaced. So the dict of a class lists every setting it accepts.
    # A default of None accepts any value. A default written as a pair [I, QU] marks a setting
    # that is either one value for both polarization views, or such a pair.
    DEFAULTS = {"polarization": None,  # "I", "QU" or "IQU". Must be set.
                "shortname": "comp",   # The name of the component in the chain file.
                "longname": None,      # The name of the component in plot titles.
                "init_from": None}     # A chain file or a sky map to start from.

    # The polarization views ("I", "QU") that this kind of component can have.
    legal_pols: tuple[str, ...] = ("I", "QU")

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

    def __init__(self, params: Bunch, comp_name: str, eval_pol: str):
        """One polarization view of the component `comp_name` of the parameter file.

        Args:
            params: The full parameter file.
            comp_name: The name of the component's block under `components`.
            eval_pol: The view to make, "I" or "QU". A component defined for IQU has both.
        """
        if eval_pol not in self.legal_pols:
            raise ValueError(f"Component {comp_name!r}: a {type(self).__name__} has no "
                             f"{eval_pol!r} view. Its views are {self.legal_pols!r}.")
        self.comp_name = comp_name
        self.eval_pol = eval_pol
        self.cfg = read_settings(params, "params", (f"components.{comp_name}",), self.DEFAULTS)
        self.shortname = self.cfg.shortname
        # `init_from: null` and a left-out `init_from` mean different things (see
        # `load_initial_state`), and `self.cfg` holds None for both.
        self.has_init_from = "init_from" in params.components[comp_name].params
        self.double_prec = resolve_param(params, "double_precision", ("compsep",),
                                         default=False, legal_types=bool)
        # The SED parameters are attributes, because a sampler can change them. They start at the
        # values of the parameter file. SpinningDust and RadioSources have other names for theirs
        # in the parameter file, and set them in their own `__init__`.
        for name in self.sed_param_names:
            if name in self.cfg:
                if self.cfg[name] is None:
                    raise ValueError(f"'components.{comp_name}.params.{name}' must be set.")
                setattr(self, name, self._per_pol(self.cfg[name]))
        # The amplitudes the CompSep solves for, as one array: alms of shape (npol, nalm) for a
        # diffuse component, one value per source for point sources. The subclass allocates it.
        self.amps = None
        # FWHM beam of the component. If the CG solver was used, this will be 0, as it solves for
        # deconvolved components. Only non-zero if the common-resolution per-pix solver was used.
        self.amp_fwhm_rad = 0.0

    def _per_pol(self, value):
        """This view's value of a setting given as one value or as a pair [I, QU]."""
        if isinstance(value, (list, tuple)):
            return value[0] if self.eval_pol == "I" else value[1]
        return value

    def join_split_views(self, other: "Component") -> "Component":
        """Join the I view and the QU view of one IQU component into a new component.

        The result has `eval_pol` "IQU" and all three polarization rows in `amps`.
        """
        if {self.eval_pol, other.eval_pol} != {"I", "QU"}:
            raise ValueError("Joining requires one intensity view and one QU view.")
        intensity_comp, pol_comp = (self, other) if self.eval_pol == "I" else (other, self)
        joined = deepcopy(intensity_comp)
        joined.eval_pol = joined.cfg.polarization
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
        group["defined_pol"] = self.cfg.polarization
        group["eval_pol"] = self.eval_pol

    def read_chain(self, group: h5py.Group, restore_beta: bool) -> None:
        """Restore this component's sampled state from its chain group, as `write_chain` wrote it.

        Only what a sampler changes is restored; fixed parameters keep their parameter-file value,
        so an edit there takes effect in a run started from another run's chain. The group holds
        the joined component (I and QU together), while `self` may be one polarization view.

        This base version restores the amplitudes' beam, and `beta` when `restore_beta` is set
        (the run samples this component's beta in an MCMC sampling group). Subclasses add their
        amplitudes, and any other sampled state.
        """
        self.amp_fwhm_rad = np.radians(float(group["amp_fwhm_arcmin"][()])/60.0)
        if restore_beta:
            beta = np.asarray(group["sed/beta"][()])
            if beta.ndim == 1:  # Joined I and QU views with different values store [I, QU].
                beta = beta[0 if self.eval_pol == "I" else 1]
            self.beta = float(beta)
            logger.debug(f"Component {self.comp_name!r} ({self.eval_pol}): restored sampled "
                         f"beta = {self.beta}.")

    def load_initial_state(self, chain_file: str | None, apply_init_from: bool,
                           restore_beta: bool) -> None:
        """Set this component's starting state from a CompSep chain file or its own `init_from`.

        The source is the `init_from` parameter if `apply_init_from` is set and the component has
        one, and `chain_file` otherwise. So `init_from: null` keeps the component at its
        parameter-file initialization. The file type decides how it is read: a `.h5`/`.hd5` chain
        file goes through `read_chain`, and a `.fits` sky map through `read_amps_from_map`. Without
        a source, the component keeps its parameter-file values and zero amplitudes.

        Args:
            chain_file: The CompSep chain file this chain starts from, or None.
            apply_init_from: False when `chain_file` is this chain's own saved state (on
                `gibbs.chain_status: append`), which must win over the initial guesses.
            restore_beta: True when the run samples this component's beta. A chain file then also
                sets beta; otherwise beta keeps its parameter-file value.
        """
        source = chain_file
        if apply_init_from and self.has_init_from:
            source = self.cfg.init_from
        if not source:
            return
        if str(source).lower().endswith(".fits"):
            self.read_amps_from_map(source)
        elif str(source).lower().endswith((".h5", ".hd5")):
            with h5py.File(source, "r") as file:
                group_path = f"comps/{self.shortname}"
                if group_path not in file:
                    logger.error(f"Component {self.comp_name!r} (shortname {self.shortname!r}) "
                                 f"not found in {source!r}; keeping its parameter-file "
                                 "initialization.")
                    return
                self.read_chain(file[group_path], restore_beta)
        else:
            raise ValueError(f"Unsupported init file {source!r} for component "
                             f"{self.comp_name!r}: expected a .h5/.hd5 chain or a .fits map.")

    def read_amps_from_map(self, map_file: str) -> None:
        """Set the amplitudes from a FITS sky map. Only diffuse components can do this."""
        raise ValueError(f"Component {self.comp_name!r}: a FITS 'init_from' map only works for "
                         "diffuse (alm-based) components.")

    def load_amp_prior_mean(self) -> None:
        """Read the mean of the amplitude prior from the `amp_prior_mean_map` parameter.

        Only diffuse components have such a prior and such a parameter, so this does nothing.
        """

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
