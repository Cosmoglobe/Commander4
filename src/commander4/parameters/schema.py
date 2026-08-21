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
MPI task counts the parameter file no longer states.
"""

import logging
from typing import Any, Sequence

from pixell.bunch import Bunch

logger = logging.getLogger(__name__)

TOP_LEVEL_BLOCKS = ("gibbs", "resources", "output", "components", "experiments", "tod_processing",
                    "compsep")

_LAYOUT_REFERENCE = "See notes/proposed_param_layout.yml for the full layout."


_NO_DEFAULT = object()   # Distinct from None, which is a perfectly good default value.


def resolve_param(params: Bunch, key: str, scopes: Sequence[str], default: Any = _NO_DEFAULT,
                  legal_values: Sequence[Any] | None = None,
                  raise_on_missing_scope: bool = True):
    """The value of `key`, taken from the first of `scopes` that defines it.

    Several settings may be given at more than one level, the narrower overriding the wider (a
    mapmaker on the band, the experiment, or globally, for instance). `scopes` are dotted paths
    into `params` listed most specific first, e.g.::

        resolve_param(params, "mapmaker", (f"experiments.{exp}.bands.{band}",
                                           f"experiments.{exp}", "tod_processing"))

    Presence decides, not truthiness: a scope setting ``0`` or ``false`` answers the lookup rather
    than deferring to a wider one. Which scope supplied the value is logged at debug level, since an
    overridden setting is otherwise invisible, and "why did this band use the CG mapmaker" is worth
    being able to answer from a chain's log file.

    Args:
        default: Returned when no scope defines `key`. Omit it to make that an error instead.
        legal_values: If given, the resolved value (or default) must be one of these, so a typo in
            the *value* is caught here rather than as strange behaviour later.
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
        if legal_values is not None and value not in legal_values:
            raise ValueError(f"'{key}' is {value!r} ({source}), which is not one of "
                             f"{list(legal_values)}.")
        return value

    for scope in scopes:
        block = params
        for part in scope.split("."):
            if part not in block:
                message = (f"Parameter scope '{scope}' does not exist (no '{part}'), so '{key}' "
                           f"cannot be looked up there; check the parameter file.")
                if raise_on_missing_scope:
                    raise ValueError(message)
                logger.debug(f"Optional parameter scope '{scope}' is absent; skipping it.")
                break
            block = block[part]
        else:
            if key in block:
                logger.debug(f"Resolved '{key}' = {block[key]!r} from '{scope}'.")
                return checked(block[key], f"from '{scope}'")
    if default is _NO_DEFAULT:
        raise ValueError(f"'{key}' is not set in any of {list(scopes)}, and has no default. Set "
                         f"it in one of them; the first that defines it wins.")
    logger.debug(f"'{key}' is not set in any of {list(scopes)}; using default {default!r}.")
    return checked(default, "the default")


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
    """Reject the pre-restructure schema, and any top-level key outside `TOP_LEVEL_BLOCKS`.

    Takes the raw parameter dictionary rather than the parsed Bunch, since `parse_params` injects
    `parameter_file_as_string` onto the latter. Called on every rank before anything else, so a
    stale parameter file fails immediately and everywhere rather than at the first stale read.
    """
    if "general" in params_dict:
        raise ValueError(
            "The 'general' block was replaced by blocks named after the part of the program that "
            f"reads them: {', '.join(TOP_LEVEL_BLOCKS)}. Gibbs loop control moved to 'gibbs', "
            "thread counts to 'resources', output paths and chain writing to 'output', TOD "
            "sampling steps to 'tod_processing', and nside / float precision / CG settings to "
            f"'compsep'. {_LAYOUT_REFERENCE}")
    unknown = sorted(set(params_dict) - set(TOP_LEVEL_BLOCKS))
    if unknown:
        raise ValueError(f"Unknown top-level parameter block(s) {unknown}. The valid blocks are "
                         f"{list(TOP_LEVEL_BLOCKS)}. {_LAYOUT_REFERENCE}")


def compsep_enabled(params: Bunch) -> bool:
    """Whether component separation runs at all (default true).

    `compsep.enabled: false` is TOD-only mode: no compsep ranks are allocated and `compsep.bands`
    is ignored, so the TOD side runs against the fixed initial sky model built from `components`.
    """
    if "compsep" not in params:
        return False
    return bool(params.compsep.enabled) if "enabled" in params.compsep else True


def derive_task_counts(params: Bunch) -> Bunch:
    """MPI task counts, derived from the band configuration rather than stated in the file.

    The TOD total is the sum of `num_tasks` over the enabled bands of enabled experiments; compsep
    needs exactly one task per enabled band view, one for I and one for QU. Stating these
    separately is three more numbers to keep in sync, and the most common way to fail a run before
    it starts.

    Returns:
        Bunch with `tod`, `compsep_I`, `compsep_QU` and their sum `total`.
    """
    ntask_tod = 0
    for exp_name in params.experiments if "experiments" in params else ():
        experiment = params.experiments[exp_name]
        if not experiment.enabled:
            continue
        for band_name in experiment.bands:
            band = experiment.bands[band_name]
            if band.enabled:
                ntask_tod += int(band.num_tasks)

    ntask_I = ntask_QU = 0
    if compsep_enabled(params) and "bands" in params.compsep:
        for band_name in params.compsep.bands:
            band = params.compsep.bands[band_name]
            if not band.enabled:
                continue
            ntask_I += "I" in band.polarization
            ntask_QU += "QU" in band.polarization

    return Bunch(tod=ntask_tod, compsep_I=ntask_I, compsep_QU=ntask_QU,
                 total=ntask_tod + ntask_I + ntask_QU)


def task_count_breakdown(counts: Bunch) -> str:
    """The derived task counts as one line, for the 'wrong mpirun -n' error message."""
    return (f"{counts.total} = {counts.tod} TOD + {counts.compsep_I} CompSep-I + "
            f"{counts.compsep_QU} CompSep-QU")
