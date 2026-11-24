"""`Component`: the base class of every sky component.

The component families are in `diffuse_components.py` and `point_sources.py`.
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
    """One polarization view ("I" or "QU") of a sky component, with its amplitudes `amps`."""

    # The settings of the `params` block, with their defaults. A subclass writes its dict as
    # `{**Parent.DEFAULTS, ...}`, so each class lists every setting it accepts.
    # A default of None accepts any value. A default pair [I, QU] means one value for both views,
    # or one value per view.
    DEFAULTS = {"polarization": None,  # "I", "QU" or "IQU". Must be set.
                "init_from": None}     # A chain file or a sky map to start from.

    # The polarization views this kind of component can have.
    legal_pols: tuple[str, ...] = ("I", "QU")

    # The attributes that define the SED. The chain stores each under `comps/<comp_name>/sed/`,
    # the fixed ones too, so that it describes the SED without the parameter file.
    sed_param_names: tuple[str, ...] = ()

    # The attributes that define the C(l) prior, stored under `comps/<comp_name>/Cl_prior/`.
    Cl_prior_param_names: tuple[str, ...] = ()

    # None means no prior on the amplitudes: the CG then leaves out every prior term.
    Cl_prior_amplitude = None

    def __init__(self, params: Bunch, comp_name: str, pol: str):
        """One polarization view of the component `comp_name` of the parameter file.

        Args:
            params: The full parameter file.
            comp_name: The component's block under `components`, and its group in the chain file.
            pol: The view, "I" or "QU". A component defined for IQU has both.
        """
        if pol not in self.legal_pols:
            raise ValueError(f"Component {comp_name!r}: a {type(self).__name__} has no "
                             f"{pol!r} view. Its views are {self.legal_pols!r}.")
        self.comp_name = comp_name
        self.pol = pol
        # The component's `params` block of the parameter file, with the `DEFAULTS` filled in.
        self.cfg: Bunch = read_settings(params, "params", (f"components.{comp_name}",),
                                        self.DEFAULTS)
        # `init_from: null` and a left-out `init_from` mean different things (see
        # `load_initial_state`), and `self.cfg` holds None for both.
        self.has_init_from = "init_from" in params.components[comp_name].params
        self.double_precision = resolve_param(params, "double_precision", ("compsep",),
                                              default=False, legal_types=bool)
        # The SED parameters are attributes, because a sampler can change them. SpinningDust sets
        # its peak frequencies, which have other names in the file, in its own `__init__`.
        for name in self.sed_param_names:
            if name in self.cfg:
                if self.cfg[name] is None:
                    raise ValueError(f"'components.{comp_name}.params.{name}' must be set.")
                setattr(self, name, self._per_pol(self.cfg[name]))
        # Alms of shape (npol, nalm) for a diffuse component, one value per source for point
        # sources. Allocated by the subclass.
        self.amps = None
        # The beam the amplitudes carry: 0 after the CG, which solves for the deconvolved sky; the
        # common resolution after the per-pixel solver.
        self.amp_fwhm_rad = 0.0

    def _per_pol(self, value):
        """This view's value of a setting given as one value or as a pair [I, QU]."""
        if isinstance(value, (list, tuple)):
            return value[0] if self.pol == "I" else value[1]
        return value

    def join_split_views(self, other: "Component") -> "Component":
        """Join the I view and the QU view of one IQU component into a new component.

        The result has `pol` "IQU" and all three polarization rows in `amps`.
        """
        if {self.pol, other.pol} != {"I", "QU"}:
            raise ValueError("Joining requires one intensity view and one QU view.")
        intensity_comp, pol_comp = (self, other) if self.pol == "I" else (other, self)
        joined = deepcopy(intensity_comp)
        joined.pol = joined.cfg.polarization
        joined.amps = np.concatenate((intensity_comp.amps, pol_comp.amps), axis=0)

        # The copy of the I view holds only the I value of a per-polarization parameter (such as
        # `nu_ref: [I, QU]`). Restore the pair, so the joined component, which is used for output
        # only, describes both views.
        for param_name in intensity_comp.sed_param_names + intensity_comp.Cl_prior_param_names:
            i_value, qu_value = getattr(intensity_comp, param_name), getattr(pol_comp, param_name)
            if not np.array_equal(i_value, qu_value):
                setattr(joined, param_name, np.array([i_value, qu_value]))
        return joined

    @property
    def npol(self) -> int:
        """The number of polarization rows in `amps`: one per letter of `pol` ("I", "QU", "IQU")."""
        return len(self.pol)

    @property
    def spin(self) -> int:
        """The spin of the harmonic transforms: 0 for "I", 2 for "QU" and "IQU"."""
        return 0 if self.pol == "I" else 2

    @property
    def amp_prior_mean(self):
        """The mean of the amplitude prior, or None for a zero mean (see `DiffuseComponent`)."""
        return None

    def write_chain(self, group: h5py.Group, band_frequencies: dict[str, float]) -> None:
        """Write this component's sample to its chain group, `comps/<comp_name>/`.

        `band_frequencies` maps each band name to its frequency in GHz, for the mixing values.
        """
        group["comp_name"] = self.comp_name
        group["amp_fwhm_arcmin"] = np.degrees(self.amp_fwhm_rad)*60.0
        # A per-polarization `nu_ref` is an array, hence no float(). An unset (None) parameter is
        # not written.
        for param_name in self.sed_param_names:
            value = getattr(self, param_name)
            if value is not None:
                group[f"sed/{param_name}"] = value
        for band_name, nu in band_frequencies.items():
            group[f"mixing/{band_name}"] = self.get_sed(nu)
        group["defined_pol"] = self.cfg.polarization
        group["eval_pol"] = self.pol

    def read_chain(self, group: h5py.Group, restore_beta: bool) -> None:
        """Restore the sampled state from the chain group that `write_chain` wrote.

        Fixed parameters keep their parameter-file value, so an edit there takes effect in a run
        started from another run's chain. The group holds the joined component,
        while `self` may be one polarization view. `beta` is restored only with `restore_beta`
        (when the run samples it).
        """
        self.amp_fwhm_rad = np.radians(float(group["amp_fwhm_arcmin"][()])/60.0)
        if restore_beta:
            beta = np.asarray(group["sed/beta"][()])
            if beta.ndim == 1:  # Joined I and QU views with different values store [I, QU].
                beta = beta[0 if self.pol == "I" else 1]
            self.beta = float(beta)
            logger.debug(f"Component {self.comp_name!r} ({self.pol}): restored sampled "
                         f"beta = {self.beta}.")

    def load_initial_state(self, chain_file: str | None, apply_init_from: bool,
                           restore_beta: bool) -> None:
        """Set the starting state from `init_from`, or else from `chain_file`.

        `init_from` is used if `apply_init_from` is set and the component has one, so
        `init_from: null` keeps the parameter-file values and zero amplitudes. A `.h5`/`.hd5` file
        is read as a chain, a `.fits` file as a sky map.

        Args:
            chain_file: The CompSep chain file this chain starts from, or None.
            apply_init_from: False when `chain_file` is this chain's own saved state
                (`gibbs.chain_status: append`), which must win over `init_from`.
            restore_beta: Also restore `beta` from a chain file (when the run samples it).
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
                group_path = f"comps/{self.comp_name}"
                if group_path not in file:
                    logger.error(f"Component {self.comp_name!r} not found in {source!r}; keeping "
                                 "its parameter-file initialization.")
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
        """Read the mean of the amplitude prior. Only diffuse components have one."""

    def dot(self, other: "Component") -> float:
        """The dot product of the amplitudes with those of `other`, the same view."""
        return float(arithmetic.dot(self.amps, other.amps))

    def bcast_amps(self, comm:MPI.Comm, root=0):
        """Broadcast the component's amplitudes from the root MPI rank."""
        # Blocking on purpose: Intel MPI 2021's non-blocking broadcast fails for arrays above 2 GB
        # (see aux/intel_mpi_ibcast_2gb/).
        comm.Bcast(self.amps, root=root)

    def reduce_amps(self, comm:MPI.Comm, root=0):
        """Sum the component's amplitudes across the communicator onto the root rank."""
        send, recv = (MPI.IN_PLACE, self.amps) if comm.Get_rank() == root else (self.amps, None)
        comm.Reduce(send, recv, op=MPI.SUM, root=root)

    def ireduce_amps(self, comm:MPI.Comm, root=0) -> MPI.Request:
        """Start summing the amplitudes onto the root rank, and return the request.

        The root must `Wait` on the request before it reads `amps`.
        """
        # Unlike the non-blocking broadcast, this works above 2 GB: mpi4py >= 4 calls the MPI-4
        # large-count MPI_Ireduce_c.
        send, recv = (MPI.IN_PLACE, self.amps) if comm.Get_rank() == root else (self.amps, None)
        return comm.Ireduce(send, recv, op=MPI.SUM, root=root)
