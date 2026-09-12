"""How a run starts: fresh parameters, saved initial values, or continuation in place."""
from __future__ import annotations

from dataclasses import dataclass

from pixell.bunch import Bunch

from commander4.file_io import paths

LOAD_FIELDS = ("tod", "amplitudes", "spectral_parameters")


@dataclass(frozen=True)
class RunStart:
    """One source selection shared by both sides, resolved once before loading data.

    `chain="matching"` maps each destination chain to its corresponding source chain. An integer
    seeds both destination chains from that source. `sky_iteration` is resolved internally: zero
    selects a fixed-sky snapshot, and -1 means a legacy TOD-only run uses its configured fixed sky.
    """

    mode: str = "new"
    source: str | None = None
    iteration: int | str = "latest"
    chain: int | str = "matching"
    load: tuple[str, ...] = LOAD_FIELDS
    sky_iteration: int | None = None

    @property
    def start_iteration(self) -> int:
        """New runs start at one; resolved resumes continue after the selected sample."""
        return int(self.iteration) + 1 if self.mode == "resume" else 1

    def source_chain(self, chain: int) -> int:
        """Source chain corresponding to one destination chain."""
        return chain if self.chain == "matching" else int(self.chain)

    def tod_file(self, experiment: str, band: str, chain: int) -> str | None:
        """Resolved source file, or None when TOD state comes from configured defaults."""
        if self.source is None or "tod" not in self.load:
            return None
        return paths.band_chain_file(self.source, experiment, band, self.source_chain(chain),
                                     int(self.iteration))

    def sky_file(self, chain: int) -> str | None:
        """Resolved source file, or None when the sky comes from configured defaults."""
        if self.source is None or self.sky_iteration == -1:
            return None
        if "amplitudes" not in self.load and "spectral_parameters" not in self.load:
            return None
        iteration = self.iteration if self.sky_iteration is None else self.sky_iteration
        if iteration == 0:
            return paths.initial_sky_file(self.source, self.source_chain(chain))
        return paths.compsep_chain_file(self.source, self.source_chain(chain), int(iteration))

    @classmethod
    def from_gibbs(cls, gibbs: Bunch | dict) -> RunStart:
        """Validate `gibbs.start`; file discovery belongs to `resolve_run_start`."""
        gibbs = dict(gibbs)
        removed = set(gibbs) & {"init_from_chain", "initialize", "start_iteration"}
        if removed:
            raise ValueError(f"Replace Gibbs setting(s) {sorted(removed)} with 'gibbs.start': "
                             "mode new/resume, source, iteration, chain and load.")
        block = gibbs.get("start", {})
        if not isinstance(block, (dict, Bunch)):
            raise ValueError("'gibbs.start' must be a mapping.")
        values = dict(block)
        unknown = set(values) - {"mode", "source", "iteration", "chain", "load"}
        if unknown:
            raise ValueError(f"Unknown key(s) {sorted(unknown)} in 'gibbs.start'.")
        mode = values.get("mode", "new")
        if mode not in ("new", "resume"):
            raise ValueError("'gibbs.start.mode' must be new or resume.")
        source = values.get("source")
        if source is not None and (not isinstance(source, str) or not source):
            raise ValueError("'gibbs.start.source' must be a directory path.")
        iteration = values.get("iteration", "latest")
        if iteration != "latest" and (type(iteration) is not int or iteration < 1):
            raise ValueError("'gibbs.start.iteration' must be latest or a positive integer.")
        chain = values.get("chain", "matching")
        if chain != "matching" and (type(chain) is not int or chain not in (1, 2)):
            raise ValueError("'gibbs.start.chain' must be matching, 1 or 2.")
        load = values.get("load", list(LOAD_FIELDS))
        if not isinstance(load, (list, tuple)) or not load:
            raise ValueError("'gibbs.start.load' must be a nonempty list.")
        for name in load:
            if name not in LOAD_FIELDS:
                raise ValueError(f"Unknown load field {name!r}; choose from {LOAD_FIELDS}.")
        if mode == "resume" and (source is not None or chain != "matching"
                                 or set(load) != set(LOAD_FIELDS)):
            raise ValueError("Resume uses output.dir, matching chains and all saved state. "
                             "Use mode new for selective loading or a different source.")
        return cls(mode=mode, source=source, iteration=iteration, chain=chain, load=tuple(load))
