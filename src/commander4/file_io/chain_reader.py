"""Resolve a run's initial state once, from a common saved iteration and source-chain mapping."""
from dataclasses import replace
from pathlib import Path
import logging
import re

import h5py
from pixell.bunch import Bunch

from commander4.file_io import paths
from commander4.parameters.initialization import RunStart
from commander4.parameters.schema import compsep_enabled, enabled_tod_bands

logger = logging.getLogger(__name__)
_SAMPLE_NAME = re.compile(r"(?:^|_)chain(\d+)_iter(\d+)\.h5$")


def _sample_files(run_dir: str) -> dict[str, tuple[int, int]]:
    """Inventory final chain filenames without opening their large datasets."""
    files = {}
    for directory in (paths.CHAINS_BANDS, paths.CHAINS_COMPSEP, paths.INITIAL_STATE):
        dirname = Path(run_dir) / directory
        if not dirname.is_dir():
            continue
        for filename in dirname.iterdir():
            match = _SAMPLE_NAME.search(filename.name)
            if match is not None:
                files[str(filename)] = (int(match[1]), int(match[2]))
    return files


def _is_complete(filename: str, kind: str, components: list[Bunch], start: RunStart) -> bool:
    """Check completion and state datasets; older files have no explicit completion marker."""
    try:
        with h5py.File(filename, "r") as handle:
            if "metadata/complete" in handle and not bool(handle["metadata/complete"][()]):
                return False
            if kind == "tod":
                required = ["scan_ids", "det_names", "abs_gain", "detrel_gain", "temporal_gain",
                            "noise_params", "accept", "present", "metadata/band_unit",
                            "jump_counts", "jump_locations", "jump_offsets"]
            else:
                from commander4 import sky

                required = []
                for component in components:
                    component_class = getattr(sky, component.component_class)
                    shortname = getattr(component.params, "shortname",
                                        component_class.default_shortname)
                    group = f"comps/{shortname}"
                    required.append(group)
                    if "amplitudes" in start.load:
                        name = "source_amps" if component.component_class == "RadioSources" \
                            else "alms"
                        required.extend([f"{group}/{name}", f"{group}/amp_fwhm_arcmin"])
                    if "spectral_parameters" in start.load:
                        required.append(f"{group}/sed")
            for name in required:
                if name not in handle:
                    return False
            if "metadata/chain" in handle:
                match = _SAMPLE_NAME.search(Path(filename).name)
                if int(handle["metadata/chain"][()]) != int(match[1]):
                    return False
                if int(handle["metadata/iteration"][()]) != int(match[2]):
                    return False
    except (OSError, KeyError, ValueError):
        return False
    return True


def resolve_run_start(params: Bunch) -> RunStart:
    """Resolve new/resume semantics, a common complete sample, and both source chains.

    Called once on the world master; broadcast the result before constructing TOD/sky state.
    Resume requires both source chains. It reports later output without moving or deleting it.
    """
    start = RunStart.from_gibbs(params.gibbs)
    output_dir = str(Path(params.output.dir).resolve())
    output_files = _sample_files(output_dir)
    if start.mode == "new":
        if output_files:
            raise ValueError("A new run requires an output directory without chain files. "
                             "Use 'gibbs.start.mode: resume' to continue this directory.")
        if start.source is None:
            return start
        source_dir = str(Path(start.source).resolve())
        if source_dir == output_dir:
            raise ValueError("A new run's source and output directories must differ; use resume.")
    else:
        source_dir = output_dir
    start = replace(start, source=source_dir)
    files = _sample_files(source_dir)
    bands = enabled_tod_bands(params)
    components = []
    for name in params.components:
        component = params.components[name]
        if component.enabled:
            if start.mode == "new" and "init_from" in component.params:
                continue
            components.append(component)

    # A TOD-only run's fixed sky is saved once, at iteration zero. It need not be duplicated at
    # every sampled iteration. Legacy TOD-only runs instead rebuild their configured fixed sky.
    sky_samples = []
    for filename, (_, iteration) in files.items():
        if Path(filename).parent.name == paths.CHAINS_COMPSEP and iteration > 0:
            sky_samples.append(iteration)
    if not sky_samples:
        fixed_files = []
        for chain in (1, 2):
            fixed_files.append(paths.initial_sky_file(source_dir, start.source_chain(chain)))
        if all(filename in files for filename in fixed_files):
            start = replace(start, sky_iteration=0)
        elif start.mode == "resume" and not compsep_enabled(params):
            start = replace(start, sky_iteration=-1)
            logger.warning("This TOD-only run has no fixed-sky snapshot; rebuilding its sky from "
                           "the configured components. Keep their original initialization.")

    iterations = set()
    for _, iteration in files.values():
        if iteration > 0:
            iterations.add(iteration)
    candidates = sorted(iterations, reverse=True)
    if start.iteration != "latest":
        candidates = [start.iteration]
    selected = None
    missing = []
    for iteration in candidates:
        candidate = replace(start, iteration=iteration)
        required: dict[str, str] = {}
        for chain in (1, 2):
            for band in bands:
                filename = candidate.tod_file(band.experiment_name, band.band_name, chain)
                if filename is not None:
                    required[filename] = "tod"
            filename = candidate.sky_file(chain)
            if components and filename is not None:
                required[filename] = "sky"
        if not required:
            raise ValueError("The start selection loads no saved state for this configuration.")
        missing = []
        for filename, kind in required.items():
            if filename not in files or not _is_complete(filename, kind, components, candidate):
                missing.append(filename)
        if not missing:
            selected = candidate
            break
    if selected is None:
        details = "\n".join(missing)
        raise ValueError("No complete saved iteration matches the requested state and source "
                         f"chains ({start.chain}). Resume requires both chains. "
                         f"Missing or incomplete files:\n{details or source_dir}")
    if start.mode == "resume":
        later = []
        for filename, (_, iteration) in output_files.items():
            if iteration > selected.iteration:
                later.append(filename)
        if later:
            raise ValueError(f"The last complete selected iteration is {selected.iteration}, but "
                             "later output exists. Archive or remove these files before resuming; "
                             "no files were changed:\n" + "\n".join(sorted(later)))
    logger.info(f"Starting {selected.mode} run at iteration {selected.start_iteration}; loading "
                f"{selected.load} from {source_dir}, iteration {selected.iteration}, "
                f"source chains {selected.source_chain(1)} and {selected.source_chain(2)}.")
    return selected
