from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

from commander4.data_models.detector_map import DetectorMap
from commander4.sky_models.component import CompList, Component
from commander4.solvers.mcmc_sampling import MCMCSamplingGroup


@dataclass(frozen=True)
class SpectralIndexGroup:
    """Metadata for one jointly proposed spectral-index parameter.

    A single logical component defined as ``IQU`` is represented by two execution views (its ``I``
    and ``QU`` views) that share one ``comp_params`` object; both are grouped here so the proposal
    updates them together. ``bounds`` are hard uniform limits (proposals outside are rejected before
    the amplitude re-solve, C3's ``p_uni``); ``prior`` is an optional soft Gaussian ``(mean, rms)``
    added to the log-acceptance ratio (C3's ``p_gauss``). Either is None when unset.
    """

    name: str
    components: tuple[Component, ...]
    proposal_sigma: float
    bounds: tuple[float, float] | None
    prior: tuple[float, float] | None


class SpectralIndexSamplingGroup(MCMCSamplingGroup):
    """MCMC group specialization: a joint Gaussian random walk over component spectral indices.

    All the Metropolis-Hastings machinery (likelihood, MPI lock-step, amplitude buffering/revert,
    accept-reject) lives in :class:`~commander4.solvers.mcmc_sampling.MCMCSamplingGroup`. This
    subclass supplies only the spectral-index-specific state: which components' ``beta`` to propose,
    how to read/write it, the symmetric Gaussian random-walk proposal with optional hard ``bounds``
    (a uniform prior), and an optional soft Gaussian ``prior`` via ``log_prior``. The state passed
    through the base hooks is a ``{group_name: beta_value}`` dict (a fresh dict each step, so it is a
    valid snapshot for the base's revert-on-reject).
    """

    def __init__(self, compsep_comm: MPI.Comm, detector_data: DetectorMap, comp_list: CompList, *,
                 target_pol: str, chisq_active: bool, selected_comps: list[str] | None,
                 root: int = 0):
        """
        Args (in addition to the base class):
            selected_comps: Component names whose spectral indices this group proposes, or None for
                all components that have spectral-index sampling enabled.
        """
        super().__init__(compsep_comm, detector_data, comp_list, target_pol=target_pol,
                         chisq_active=chisq_active, root=root)
        self._groups = _discover_spectral_index_groups(comp_list, selected_comps)

    def has_parameters(self) -> bool:
        return bool(self._groups)

    def capture_state(self) -> dict[str, float]:
        # `beta` is shared across an IQU component's views, so reading the first view suffices.
        return {group.name: float(group.components[0].beta) for group in self._groups}

    def propose(self, current_state: dict[str, float]) -> tuple[dict[str, float], bool]:
        proposed = {}
        in_bounds = True
        for group in self._groups:
            value = float(np.random.normal(current_state[group.name], group.proposal_sigma))
            proposed[group.name] = value
            if group.bounds is not None and not group.bounds[0] <= value <= group.bounds[1]:
                in_bounds = False
        return proposed, in_bounds

    def apply_state(self, state: dict[str, float]) -> None:
        # `get_sed` reads `comp.beta`; the I/QU views of an IQU component each get it set here.
        for group in self._groups:
            for comp in group.components:
                comp.beta = state[group.name]

    def log_prior(self, state: dict[str, float]) -> float:
        # Soft Gaussian prior per group: -0.5 ((beta - mean)/rms)^2. The rms-dependent normalization
        # is state-independent and cancels in the acceptance ratio, so it is dropped. Groups without
        # a prior contribute 0 (flat); hard uniform limits are handled separately via `bounds`.
        total = 0.0
        for group in self._groups:
            if group.prior is not None:
                mean, rms = group.prior
                total += -0.5*((state[group.name] - mean)/rms)**2
        return total


def _discover_spectral_index_groups(comp_list: CompList,
                                    selected_comps: list[str] | None) -> list[SpectralIndexGroup]:
    """Collect the spectral-index parameters to propose jointly in one MCMC group.

    A component contributes a parameter iff it exposes a ``beta`` attribute, has
    ``sample_spectral_index: true`` in its params, and (when ``selected_comps`` is not None) its
    ``comp_name`` is selected. Execution views that share a ``comp_params`` object (the ``I``/``QU``
    views of an IQU component) are grouped into a single parameter.
    """
    selected = None if selected_comps is None else set(selected_comps)
    group_info: dict[int, dict[str, object]] = {}
    seen_name_counts: dict[str, int] = {}

    for comp in comp_list:
        if not hasattr(comp, "beta"):
            continue
        if selected is not None and comp.comp_name not in selected:
            continue
        if not bool(_read_param(comp.comp_params, "sample_spectral_index", False)):
            continue

        group_id = id(comp.comp_params)
        if group_id in group_info:
            group_info[group_id]["components"].append(comp)
            continue

        proposal_sigma = _read_param(comp.comp_params, "spectral_index_proposal_sigma")
        if proposal_sigma is None:
            raise ValueError("Missing required spectral-index sampling parameter "
                             "'spectral_index_proposal_sigma'.")
        proposal_sigma = float(proposal_sigma)
        if proposal_sigma <= 0.0:
            raise ValueError(f"Component '{comp.shortname}' needs a positive "
                             "spectral_index_proposal_sigma.")

        bounds = _read_param(comp.comp_params, "spectral_index_bounds")
        if bounds is not None:
            if len(bounds) != 2:
                raise ValueError(f"Component '{comp.shortname}' spectral_index_bounds must contain "
                                 "exactly two values.")
            bounds = (float(bounds[0]), float(bounds[1]))
            if bounds[0] > bounds[1]:
                raise ValueError(f"Component '{comp.shortname}' spectral_index_bounds must be "
                                 "ordered as [min, max].")

        # Optional soft Gaussian prior (C3's p_gauss). A uniform prior is given by the hard limits in
        # spectral_index_bounds instead, so 'gaussian' is the only supported `spectral_index_prior`.
        prior = _read_param(comp.comp_params, "spectral_index_prior")
        if prior is not None:
            prior_type = _read_param(prior, "type")
            if prior_type != "gaussian":
                raise ValueError(f"Component '{comp.shortname}' spectral_index_prior type "
                                 f"{prior_type!r} is unsupported; use 'gaussian' (a uniform prior is "
                                 "set via spectral_index_bounds).")
            mean, rms = _read_param(prior, "mean"), _read_param(prior, "rms")
            if mean is None or rms is None:
                raise ValueError(f"Component '{comp.shortname}' gaussian spectral_index_prior needs "
                                 "both 'mean' and 'rms'.")
            rms = float(rms)
            if rms <= 0.0:
                raise ValueError(f"Component '{comp.shortname}' spectral_index_prior 'rms' must be "
                                 "positive.")
            prior = (float(mean), rms)

        base_name = str(_read_param(comp.comp_params, "shortname", comp.shortname))
        occurrence = seen_name_counts.get(base_name, 0) + 1
        seen_name_counts[base_name] = occurrence
        group_info[group_id] = {
            "name": base_name if occurrence == 1 else f"{base_name}_{occurrence}",
            "components": [comp],
            "proposal_sigma": proposal_sigma,
            "bounds": bounds,
            "prior": prior,
        }

    groups = []
    for info in group_info.values():
        components = info["components"]
        values = np.array([float(comp.beta) for comp in components], dtype=np.float64)
        if not np.allclose(values, values[0]):
            raise ValueError(f"Spectral-index group '{info['name']}' has inconsistent beta values: "
                             f"{values.tolist()}")
        groups.append(SpectralIndexGroup(name=info["name"], components=tuple(components),
                                         proposal_sigma=info["proposal_sigma"], bounds=info["bounds"],
                                         prior=info["prior"]))
    return groups


def _read_param(container, key: str, default=None):
    return container[key] if key in container else default
