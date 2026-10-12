"""Which saved Gibbs sample a run starts from, following Commander3's CHAIN_STATUS and INIT_CHAIN.

A run starts in one of three ways, set in the `gibbs` parameter block:

* `chain_status: append` continues the chains in `output.dir` after their newest complete saved
  iteration, and numbers the new samples onward from there. If `output.dir` holds no complete
  iteration, the run starts new chains instead. So the same parameter file can be resubmitted until
  the chains reach `num_iterations`.
* `init_chain: "<run dir>"` or `"<run dir>:<iteration>"` starts new chains (at iteration 1) from the
  newest complete, or the given, iteration of another run.
* Otherwise the chains start from the parameter file.

A saved sample is the ordinary chain files: per chain, one file per band in `chains_bands/` and one
sky file in `chains_compsep/`. An iteration is complete when every file this run reads exists for
every chain the source run wrote. A chain the source run never wrote (normally chain 2, see
`output.chains.write`) starts from the state of a written one; its own history was never saved, so
nothing is lost. A part the source run never wrote at all, such as a band added since, or the sky
of a TOD-only run, starts from the parameter file.
"""
import logging
import os
import re
from dataclasses import dataclass, field

from pixell.bunch import Bunch

from commander4.diagnostics import log  # noqa: F401  (adds the `logger.summary` level)
from commander4.file_io import paths
from commander4.parameters.schema import enabled_tod_bands

logger = logging.getLogger(__name__)

# A chain file name: `<experiment>_<band>_chainNN_iterNNNN.h5` in `chains_bands/`, and
# `chainNN_iterNNNN.h5` in `chains_compsep/`. The `.h5.tmp` files of unfinished writes never match.
_CHAIN_FILE_NAME = re.compile(r"^(?:(.*)_)?chain(\d+)_iter(\d+)\.h5$")


@dataclass
class ChainStart:
    """Where the two chains of a run start. Resolved once, on the world master, and broadcast."""

    first_iteration: int                  # The first iteration this run samples.
    source_dir: str | None = None         # Run directory of the saved sample; None = parameter file.
    source_iteration: int = 0             # Iteration of the saved sample in `source_dir`.
    source_chain: dict[int, int] = field(default_factory=dict)  # Chain here -> saved chain.


def _saved_files(run_dir: str) -> set[tuple[str, int, int]]:
    """Every chain file of the run in `run_dir`, as (part, chain, iteration).

    The part is `<experiment>_<band>` for a band file and `compsep` for a sky file.
    """
    saved = set()
    for subdir in (paths.CHAINS_BANDS, paths.CHAINS_COMPSEP):
        directory = os.path.join(run_dir, subdir)
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            match = _CHAIN_FILE_NAME.match(name)
            if match is not None:
                part = match[1] if subdir == paths.CHAINS_BANDS else "compsep"
                saved.add((part, int(match[2]), int(match[3])))
    return saved


def _complete_iterations(run_dir: str, needed_parts: set[str]) \
        -> tuple[list[int], list[int], list[str]]:
    """The saved iterations of `run_dir` that a run reading `needed_parts` can start from.

    Returns the complete iterations (newest first), the chains the run wrote, and which of the
    needed parts it wrote. An iteration is complete when every written part exists for every
    written chain.
    """
    saved = {entry for entry in _saved_files(run_dir) if entry[0] in needed_parts}
    saved_parts = sorted({part for part, _, _ in saved})
    written_chains = sorted({chain for _, chain, _ in saved})
    complete = []
    for iteration in sorted({it for _, _, it in saved}, reverse=True):
        if all((part, chain, iteration) in saved
               for part in saved_parts for chain in written_chains):
            complete.append(iteration)
    return complete, written_chains, saved_parts


def resolve_chain_start(params: Bunch, tod_active: bool, compsep_active: bool) -> ChainStart:
    """Decide where this run's two chains start, from `gibbs.chain_status` and `gibbs.init_chain`.

    Args:
        tod_active, compsep_active: Whether this run has TOD and CompSep ranks. They decide which
            files of a saved sample the run reads: the band files, the sky file, or both.

    Raises:
        ValueError: If `init_chain` names a run without a complete iteration, or an iteration
            that is not complete.
    """
    output_dir = paths.resolve_output_dir(params.output)
    chain_status = params.gibbs.chain_status if "chain_status" in params.gibbs else "new"
    if chain_status not in ("new", "append"):
        raise ValueError(f"gibbs.chain_status is {chain_status!r}; expected 'new' or 'append'.")
    init_chain = params.gibbs.init_chain if "init_chain" in params.gibbs else None

    # The parts of a saved sample this run reads.
    needed_parts = set()
    if tod_active:
        for band in enabled_tod_bands(params):
            needed_parts.add(f"{band.experiment_name}_{band.band_name}")
    if compsep_active:
        needed_parts.add("compsep")

    start = ChainStart(first_iteration=1)
    if chain_status == "append":
        complete, written_chains, saved_parts = _complete_iterations(output_dir, needed_parts)
        if complete:
            start = ChainStart(complete[0] + 1, output_dir, complete[0])
        else:
            logger.info(f"gibbs.chain_status is 'append', but {output_dir} holds no complete "
                        "saved iteration; starting new chains.")
    if start.source_dir is None and init_chain:
        # C3's `<file>:<iteration>` form; a bare directory means its newest complete iteration.
        source_dir, _, iteration_text = str(init_chain).rpartition(":")
        if not (source_dir and iteration_text.isdigit()):
            source_dir, iteration_text = str(init_chain), ""
        complete, written_chains, saved_parts = _complete_iterations(source_dir, needed_parts)
        if not complete:
            raise ValueError(f"gibbs.init_chain: {source_dir} holds no complete saved iteration "
                             f"of the parts this run reads ({sorted(needed_parts)}).")
        if iteration_text and int(iteration_text) not in complete:
            raise ValueError(f"gibbs.init_chain: iteration {iteration_text} of {source_dir} is "
                             f"not complete for this run. Complete iterations: {complete}.")
        start = ChainStart(1, source_dir, int(iteration_text) if iteration_text else complete[0])

    if start.source_dir is not None:
        # A chain the source run never wrote starts from the first chain it did write.
        for chain in (1, 2):
            start.source_chain[chain] = chain if chain in written_chains else written_chains[0]
        unsaved_parts = sorted(needed_parts - set(saved_parts))
        if unsaved_parts:
            logger.warning(f"{start.source_dir} holds no saved state for {unsaved_parts}; these "
                           "start from the parameter file.")
        logger.summary(f"Starting at iteration {start.first_iteration} from iteration "
                       f"{start.source_iteration} of {start.source_dir} (chain 1 from saved chain "
                       f"{start.source_chain[1]}, chain 2 from saved chain "
                       f"{start.source_chain[2]}).")

    # Files the run will overwrite as it gets to them. Any it does not get to stay as they are.
    later = [it for _, _, it in _saved_files(output_dir) if it >= start.first_iteration]
    if later:
        logger.warning(f"{output_dir} already holds {len(later)} chain files of iterations "
                       f"{min(later)} to {max(later)}. They are overwritten as this run reaches "
                       "them.")
    return start
