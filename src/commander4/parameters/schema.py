"""Top-level shape of a Commander4 parameter file, and the quantities derived from it.

A parameter file is seven blocks, each named after the part of the program that reads it:

    gibbs           how long the chain runs
    resources       thread counts (the MPI task counts follow from the enabled bands)
    output          where everything is written, and what
    components      the sky model (shared: compsep samples it, TOD processing consumes it)
    experiments     the instrument and its data
    tod_processing  what the TOD side does to that data
    compsep         what the component-separation side does with the resulting maps
"""

import copy
import logging
from dataclasses import dataclass
from typing import Any
from collections.abc import Sequence

from pixell.bunch import Bunch

logger = logging.getLogger(__name__)

# How a `polarization` setting of a band or a component splits into the polarizations that C4
# handles separately. An IQU band gives two polbands (a band in one polarization), and each
# CompSep rank works on one polband.
SPLIT_POLS = {"I": ("I",), "QU": ("QU",), "IQU": ("I", "QU")}

# The legal top-level entries in the YAML parameter file. Explicitly stating them allows us to
# catch misspelled entries
TOP_LEVEL_BLOCKS = ("gibbs", "resources", "output", "components", "experiments", "tod_processing",
                    "compsep")

# Variable used to identify lack of default. Not using "None" because its a common default.
NO_DEFAULT = object()


@dataclass(frozen=True)
class TODBandConfig:
    """Info about one TOD band."""
    experiment_name: str    # Name of the experiment this band is on (e.g. "PlanckLFI").
    band_name: str          # Unique name of this band (e.g. "LFI_30GHz").
    weight: float | None    # How expensive TOD processing of this band is. Used for load balancing.


@dataclass(frozen=True)
class CompSepPolBandConfig:
    """Info about one CompSep polband (the I or the QU part of a band)."""
    band_name: str          # Unique name of this band (e.g. "LFI_30GHz"). Used to match with TOD.
    pol: str                # "I" or "QU".


def enabled_tod_bands(params: Bunch) -> tuple[TODBandConfig, ...]:
    """Returns the enabled TOD bands of all experiments, as a tuple of `TODBandConfig`s.
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
                experiment_name=experiment_name,
                band_name=band_name,
                weight=None if weight == "auto" else float(weight),
            ))
    if len({band.weight is None for band in bands}) > 1:
        raise ValueError("Some enabled TOD bands set a number as band_weight, and others use "
                         "'auto'. Set a number on every band, or on none.")
    return tuple(bands)


def enabled_compsep_polbands(params: Bunch) -> tuple[CompSepPolBandConfig, ...]:
    """Returns a tuple of CompSep "polbands" (bands in either I or QU), all I polbands first."""
    if not compsep_enabled(params) or "bands" not in params.compsep:
        return ()

    polbands_by_pol: dict[str, list[CompSepPolBandConfig]] = {"I": [], "QU": []}
    for band_name in params.compsep.bands:
        band = params.compsep.bands[band_name]
        if not band.enabled:
            continue
        if band.polarization not in SPLIT_POLS:
            raise ValueError(
                f"compsep.bands.{band_name}.polarization is {band.polarization!r}; expected "
                f"one of {sorted(SPLIT_POLS)}."
            )
        for pol in SPLIT_POLS[band.polarization]:
            polbands_by_pol[pol].append(CompSepPolBandConfig(band_name=band_name, pol=pol))
    return tuple(polbands_by_pol["I"] + polbands_by_pol["QU"])


def resolve_param(params: Bunch, key: str, scopes: Sequence[str], default: Any = NO_DEFAULT,
                  legal_values: Sequence[Any] | None = None,
                  legal_types: type | tuple[type, ...] | None = None,
                  raise_on_missing_scope: bool = True):
    """The value of `key`, taken from the first of `scopes` that defines it.

    `scopes` are dotted paths into `params`, most specific first, for example
    `(f"experiments.{exp}.bands.{band}", f"experiments.{exp}", "tod_processing")`. The scope ""
    is `params` itself. A scope that sets `0` or `false` still answers.

    Args:
        default: Returned when no scope defines `key`. Without it, that raises `ValueError`.
        legal_values: The value (or default) must be one of these.
        legal_types: The value (or default) must have one of these types.
        raise_on_missing_scope: Raise on a scope that does not exist in `params`. Pass False for an
            optional override block, which is then skipped.
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
    # A copy, so that no two settings blocks share a mutable default such as a list.
    settings = Bunch(**copy.deepcopy(defaults))
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


def names_or_all(names: list[str] | str | None, known_names: list[str],
                 setting: str) -> list[str]:
    """The names a setting selects: `names` itself, or all of `known_names` for "all" or None."""
    if names is None or names == "all":
        return list(known_names)
    if isinstance(names, str):
        raise ValueError(f"'{setting}' must be a list of names or \"all\", not {names!r}.")
    unknown = sorted(set(names) - set(known_names))
    if unknown:
        raise ValueError(f"'{setting}' names unknown {unknown}. The known names are "
                         f"{known_names}.")
    return list(names)


def resolve_band_lmax(params: Bunch, band_name: str, experiment: str | None, nside: int) -> int:
    """The harmonic bandlimit of one band: its `lmax` parameter, or `3*nside - 1` if unset.

    This is C4's equivalent of C3's `BAND_LMAX`, which is likewise stated per band. It is the lmax
    of the band's `BandMap`, and thereby the highest multipole at which any component can be
    compared against this band's data: `Component.comp2band` truncates the component alms to it,
    both in the CG solver and in every realized band sky (`CompList.get_sky_at_nu`). `3*nside - 1`
    is the full bandlimit of a HEALPix map at this resolution, so the default lets a band constrain
    everything its own pixelization can carry.

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
SAMPLING_GROUP_SECTIONS = ("cg_sampling_groups", "per_pixel_sampling_groups",
                           "mcmc_sampling_groups")


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
    """Whether component separation runs at all (default true)."""
    if "compsep" not in params:  # Block missing from parameter file entirely.
        return False
    if "enabled" in params.compsep and not params.compsep.enabled:  # enbaled = False in param-file.
        return False
    # Even if both checks above pass, compsep is implicitly disabled if there are no sampling groups
    return has_enabled_sampling_group(params)