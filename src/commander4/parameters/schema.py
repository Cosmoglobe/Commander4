"""Top-level shape of a Commander4 parameter file, and the quantities derived from it.

A parameter file is seven blocks, each named after the part of the program that reads it:

    gibbs           how long the chain runs
    resources       thread counts (MPI task counts are derived, see `derive_task_counts`)
    output          where everything is written, and what
    components      the sky model (shared: compsep samples it, TOD processing consumes it)
    experiments     the instrument and its data
    tod_processing  what the TOD side does to that data
    compsep         what the component-separation side does with the resulting maps

They replace the old `general` grab-bag, which mixed MPI resources, Gibbs loop control, TOD
sampling steps, compsep solver settings and output paths. `notes/proposed_param_layout.yml` shows
the full layout. This module owns the block list, rejects the old schema loudly, and derives the
MPI task counts and the split of the TOD tasks between bands, which the parameter file no longer
states.
"""

import logging
from dataclasses import dataclass
from typing import Any
from collections.abc import Sequence

from pixell.bunch import Bunch

from commander4.polarization import EXECUTION_POLS, get_execution_band_id

logger = logging.getLogger(__name__)

# The legal top-level entries in the YAML parameter file. Explicitly stating them allows us to
# catch misspelled entries
TOP_LEVEL_BLOCKS = ("gibbs", "resources", "output", "components", "experiments", "tod_processing",
                    "compsep")

# Variable used to identify lack of default. Not using "None" because its a common default.
NO_DEFAULT = object()


@dataclass(frozen=True)
class TODBandConfig:
    """One enabled TOD band, and its ``band_weight`` (None for "auto")."""

    index: int
    experiment_name: str
    band_name: str
    weight: float | None


@dataclass(frozen=True)
class CompSepViewConfig:
    """One enabled CompSep execution view, handled by exactly one rank."""

    rank: int
    band_name: str
    polarization: str
    identifier: str


def enabled_tod_bands(params: Bunch) -> tuple[TODBandConfig, ...]:
    """Return the single ordered inventory used for all TOD rank allocation.

    Band names are routing identifiers between the TOD and CompSep sides, so enabled TOD band
    names must be unique across experiments. This explicit rule is clearer than silently merging
    equal per-experiment band indices into one MPI communicator.

    Each band's ``band_weight`` is a positive number or "auto" (the default), and either every band
    sets a number or none does: an "auto" weight counts detector-samples, which is not on the scale
    of numbers set by hand.
    """
    bands = []
    seen_names: dict[str, str] = {}
    if "experiments" not in params:
        return ()

    for experiment_name in params.experiments:
        experiment = params.experiments[experiment_name]
        if not experiment.enabled:
            continue
        for band_name in experiment.bands:
            band = experiment.bands[band_name]
            if not band.enabled:
                continue
            if band_name in seen_names:
                first_experiment = seen_names[band_name]
                raise ValueError(
                    f"Enabled TOD band name {band_name!r} occurs in both {first_experiment!r} and "
                    f"{experiment_name!r}. Band names route MPI messages and must be globally "
                    "unique across enabled experiments."
                )
            weight = band.band_weight if "band_weight" in band else "auto"
            if weight != "auto" and not (isinstance(weight, (int, float)) and weight > 0):
                raise ValueError(f"experiments.{experiment_name}.bands.{band_name}.band_weight is "
                                 f"{weight!r}; expected a positive number or 'auto'.")
            seen_names[band_name] = experiment_name
            bands.append(TODBandConfig(
                index=len(bands),
                experiment_name=experiment_name,
                band_name=band_name,
                weight=None if weight == "auto" else float(weight),
            ))
    if len({band.weight is None for band in bands}) > 1:
        raise ValueError("Some enabled TOD bands set a number as band_weight, and others use "
                         "'auto'. Set a number on every band, or on none.")
    return tuple(bands)


def split_tod_ranks(weights: Sequence[float], ntask_tod: int) -> list[int]:
    """The number of TOD ranks of each band, in proportion to the bands' weights.

    Every band gets one rank, and each further rank goes to the band that has the most weight per
    rank at that point. This gives the smallest possible largest weight per rank, and whole-number
    weights that sum to ``ntask_tod`` give exactly those numbers.

    Args:
        weights: Each band's weight, e.g. its number of detector-samples.
        ntask_tod: The number of TOD ranks, at least one per band.
    """
    counts = [1]*len(weights)
    for _ in range(ntask_tod - len(weights)):
        busiest = max(range(len(weights)), key=lambda iband: weights[iband]/counts[iband])
        counts[busiest] += 1
    return counts


def enabled_compsep_views(params: Bunch) -> tuple[CompSepViewConfig, ...]:
    """Return CompSep views in rank order: all I views, followed by all QU views."""
    if not compsep_enabled(params) or "bands" not in params.compsep:
        return ()

    views_by_pol: dict[str, list[tuple[str, str]]] = {"I": [], "QU": []}
    for band_name in params.compsep.bands:
        band = params.compsep.bands[band_name]
        if not band.enabled:
            continue
        if band.polarization not in EXECUTION_POLS:
            raise ValueError(
                f"compsep.bands.{band_name}.polarization is {band.polarization!r}; expected "
                f"one of {sorted(EXECUTION_POLS)}."
            )
        for polarization in EXECUTION_POLS[band.polarization]:
            views_by_pol[polarization].append(
                (band_name, get_execution_band_id(band_name, polarization))
            )

    views = []
    for polarization in ("I", "QU"):
        for band_name, identifier in views_by_pol[polarization]:
            views.append(CompSepViewConfig(
                rank=len(views),
                band_name=band_name,
                polarization=polarization,
                identifier=identifier,
            ))
    return tuple(views)


def resolve_param(params: Bunch, key: str, scopes: Sequence[str], default: Any = NO_DEFAULT,
                  legal_values: Sequence[Any] | None = None,
                  legal_types: type | tuple[type, ...] | None = None,
                  raise_on_missing_scope: bool = True):
    """The value of `key`, taken from the first of `scopes` that defines it.

    Several settings may be given at more than one level, the narrower overriding the wider (a
    mapmaker on the band, the experiment, or globally, for instance). `scopes` are dotted paths
    into `params` listed most specific first, e.g.::

        resolve_param(params, "mapmaker", (f"experiments.{exp}.bands.{band}",
                                           f"experiments.{exp}", "tod_processing"))

    Scopes can include "", which just means to look for the key at the root node of `params`.
                                           
    Presence decides, not truthiness: a scope setting ``0`` or ``false`` answers the lookup rather
    than deferring to a wider one. Which scope supplied the value is logged at debug level, since an
    overridden setting is otherwise invisible, and "why did this band use the CG mapmaker" is worth
    being able to answer from a chain's log file.

    Args:
        default: Returned when no scope defines `key`. Omit it to make that an error instead.
        legal_values: If given, the resolved value (or default) must be one of these, so a typo in
            the *value* is caught here rather than as strange behaviour later.
        legal_types: If given, the resolved value (or default) must have one of these types.
        raise_on_missing_scope: What to do about a scope that does not exist in `params` at all.
            There is usually no good reason to list one, so this is an error by default. Pass
            False for a scope that is legitimately optional (a per-band override block that most
            bands do not carry, say), and its absence is then logged at debug level instead.

    Raises:
        ValueError: if no scope defines `key` and no `default` was given, naming every scope that
            was searched; if the resolved value is not in `legal_values`; or if a scope does not
            exist and `raise_on_missing_scope` is set.
    """
    def checked(value, source):
        if legal_types is not None and not isinstance(value, legal_types):
            allowed = legal_types if isinstance(legal_types, tuple) else (legal_types,)
            type_names = [allowed_type.__name__ for allowed_type in allowed]
            raise ValueError(f"'{key}' is {value!r} ({source}), but must have type "
                             f"{' or '.join(type_names)}.")
        if legal_values is not None and value not in legal_values:
            raise ValueError(f"'{key}' is {value!r} ({source}), which is not one of "
                             f"{list(legal_values)}.")
        return value

    for scope in scopes:
        block = params
        # An empty scope addresses `params` itself, for callers that were handed one block rather
        # than the whole parameter file.
        for part in (scope.split(".") if scope else ()):
            if part not in block:
                message = (f"Parameter scope '{scope}' does not exist (no '{part}'), so '{key}' "
                           f"cannot be looked up there; check the parameter file.")
                if raise_on_missing_scope:
                    raise ValueError(message)
                # These prints ended up being very verbose even for 'verbose'. Commenting out.
                # logger.debug(f"Optional parameter scope '{scope}' is absent; skipping it.")
                break
            block = block[part]
        else:
            if key in block:
                source = f"'{scope}'" if scope else "the given parameter block"
                # logger.debug(f"Resolved '{key}' = {block[key]!r} from {source}.")
                return checked(block[key], f"from {source}")
    searched = [scope if scope else "the given parameter block" for scope in scopes]
    if default is NO_DEFAULT:
        raise ValueError(f"'{key}' is not set in any of {searched}, and has no default. Set "
                         f"it in one of them; the first that defines it wins.")
    # logger.debug(f"'{key}' is not set in any of {searched}; using default {default!r}.")
    return checked(default, "the default")


def read_settings(params: Bunch, name: str, scopes: Sequence[str],
                  defaults: dict[str, Any]) -> Bunch:
    """The settings block `name`, with `defaults` for every setting it leaves out.

    The sibling of `resolve_param` for a whole block of settings. `scopes` are dotted paths into
    `params`, most specific first, and each setting comes from the most specific scope whose `name`
    block sets it, so a band can override single settings of a global block::

        read_settings(params, "abs_gain", (f"experiments.{exp}.bands.{band}", "tod_processing"),
                      {"enabled": False, "from_iter": 1, ...})

    A scope that does not exist sets nothing. Every setting must have a default, so a misspelled
    setting fails at startup instead of being ignored. A setting whose default is a bool, int,
    float or str must have that type (an int is accepted for a float); a default of None accepts
    any value.

    Raises:
        ValueError: for a setting without a default, or with the wrong type.
    """
    settings = Bunch(**defaults)
    # The widest scope first, so that each narrower one overrides it.
    for scope in reversed(scopes):
        block = params
        for part in (scope.split(".") if scope else ()):
            block = block[part] if part in block else Bunch()
        if name not in block:
            continue
        for key, value in block[name].items():
            location = f"{scope}.{name}.{key}" if scope else f"{name}.{key}"
            if key not in defaults:
                raise ValueError(f"Unknown setting '{location}'. The known settings of '{name}' "
                                 f"are {list(defaults)}.")
            default = defaults[key]
            if isinstance(default, (bool, int, float, str)):
                legal_types = (int, float) if isinstance(default, float) else type(default)
                # bool is a subclass of int in Python, so it has to be told apart explicitly.
                if (not isinstance(value, legal_types)
                        or isinstance(value, bool) != isinstance(default, bool)):
                    raise ValueError(f"'{location}' is {value!r}, but must be a "
                                     f"{type(default).__name__}.")
            settings[key] = value
    return settings


def resolve_band_lmax(params: Bunch, band_name: str, experiment: str | None, nside: int) -> int:
    """The harmonic bandlimit of one band: its `lmax` parameter, or `3*nside - 1` if unset.

    This is C4's equivalent of C3's `BAND_LMAX`, which is likewise stated per band. It is the lmax
    of the band's `DetectorMap`, and thereby the highest multipole at which any component can be
    compared against this band's data: `Component.project_comp_to_band` truncates the component
    alms to it. `3*nside - 1` is the full bandlimit of a HEALPix map at this resolution, so the
    default lets a band constrain everything its own pixelization can carry.

    Args:
        band_name: Key of the band under `compsep.bands` (and under its experiment, if it has one).
        experiment: Name of the experiment providing the band, or None for a `get_from: file` band,
            whose settings live in its `compsep.bands` entry rather than in an experiment block.
        nside: The band's map resolution, i.e. its `eval_nside`.
    """
    if experiment is None:
        scopes = (f"compsep.bands.{band_name}",)
    else:
        scopes = (f"experiments.{experiment}.bands.{band_name}", f"experiments.{experiment}")
    lmax = resolve_param(params, "lmax", scopes, default=None)
    return 3*nside - 1 if lmax is None else int(lmax)


def validate_param_schema(params_dict: dict) -> None:
    """Reject any top-level key outside `TOP_LEVEL_BLOCKS`."""
    unknown = sorted(set(params_dict) - set(TOP_LEVEL_BLOCKS))
    if unknown:
        raise ValueError(f"Unknown top-level parameter block(s) {unknown}. The valid blocks are "
                         f"{list(TOP_LEVEL_BLOCKS)}.")


# The three method-specific sections a sampling group can live in. Compsep exists to run these, so
# with none of them enabled there is nothing for a compsep rank to do.
SAMPLING_GROUP_SECTIONS = ("cg_sampling_groups", "per_pixel_sampling_groups", "mcmc_sampling_groups")


def has_enabled_sampling_group(params: Bunch) -> bool:
    """Whether any sampling group in any of the three method sections is enabled."""
    if "compsep" not in params:
        return False
    for section in SAMPLING_GROUP_SECTIONS:
        if section not in params.compsep:
            continue
        groups = params.compsep[section]
        for group_name in groups:
            group = groups[group_name]
            if "enabled" not in group or group.enabled:
                return True
    return False


def compsep_enabled(params: Bunch) -> bool:
    """Whether component separation runs at all (default true).

    False means TOD-only mode: no compsep ranks are allocated and `compsep.bands` is ignored, so the
    TOD side runs against the fixed initial sky model it builds itself from `components`. There are
    two ways to get there, and they mean the same thing operationally:

    * `compsep.enabled: false`, stated outright.
    * No enabled sampling group. Compsep ranks exist only to run those groups, so allocating ranks
      that would sample nothing just reserves nodes and memory to forward maps and evaluate a
      chi-squared. TOD-only reaches the same fixed sky far more cheaply.
    """
    if "compsep" not in params:
        return False
    if "enabled" in params.compsep and not params.compsep.enabled:
        return False
    return has_enabled_sampling_group(params)


def derive_task_counts(params: Bunch) -> Bunch:
    """MPI task counts, derived from the band configuration rather than stated in the file.

    Compsep needs exactly one task per enabled band view, one for I and one for QU. Every other
    task is a TOD task, and each enabled TOD band needs at least one (see `split_tod_ranks`).

    Returns:
        Bunch with `tod_bands` (the number of enabled TOD bands), `compsep_I`, `compsep_QU`, and
        `min_total`, their sum: the smallest number of tasks the run can use.
    """
    ntod_bands = len(enabled_tod_bands(params))
    compsep_views = enabled_compsep_views(params)
    ntask_I = sum(view.polarization == "I" for view in compsep_views)
    ntask_QU = sum(view.polarization == "QU" for view in compsep_views)

    return Bunch(tod_bands=ntod_bands, compsep_I=ntask_I, compsep_QU=ntask_QU,
                 min_total=ntod_bands + ntask_I + ntask_QU)


def task_count_breakdown(counts: Bunch) -> str:
    """The derived task counts as one line, for the 'wrong mpirun -n' error message.

    Without TOD bands the count is exact; with them, any further tasks go to the TOD bands.
    """
    return (f"{'at least ' if counts.tod_bands else ''}{counts.min_total} = {counts.tod_bands} TOD "
            f"+ {counts.compsep_I} CompSep-I + {counts.compsep_QU} CompSep-QU")
