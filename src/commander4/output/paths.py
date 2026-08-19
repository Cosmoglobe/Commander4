"""Layout of a Commander4 run's output directory.

A run writes everything below the single ``output.dir``:

    <output_dir>/logs/              log file and cProfile dumps
    <output_dir>/chains_tod/        per-band TOD sample chains
    <output_dir>/chains_compsep/    component amplitude chains
    <output_dir>/chains_datamaps/   per-band output maps
    <output_dir>/plots/             figures

The subdirectory names live here so that the run, the plotting tool and the standalone tools all
agree on them, and so that reading a run's output needs only the one path the user configured.
"""
import os

LOGS = "logs"
CHAINS_TOD = "chains_tod"
CHAINS_COMPSEP = "chains_compsep"
CHAINS_DATAMAPS = "chains_datamaps"
PLOTS = "plots"

SUBDIRS = (LOGS, CHAINS_TOD, CHAINS_COMPSEP, CHAINS_DATAMAPS, PLOTS)


def resolve_output_dir(output_params) -> str:
    """The run's output directory, given the ``output`` parameter block."""
    if "dir" not in output_params:
        raise ValueError("'output.dir' is required: the single directory a run writes all of its "
                         "output to.")
    return str(output_params.dir)


def subdir(params, name: str) -> str:
    """One of the run's output subdirectories, e.g. ``subdir(params, paths.CHAINS_TOD)``."""
    return os.path.join(resolve_output_dir(params.output), name)


def log_file_path(output_params) -> str:
    """Full path of the run's log file: ``logging.file.filename`` inside the logs subdirectory.

    A file name carrying a directory is refused rather than quietly ignored: a run that silently
    writes somewhere other than where the parameter file says would be discovered far too late.
    """
    filename = output_params.logging.file.filename
    if os.path.basename(filename) != filename:
        raise ValueError(f"'output.logging.file.filename' must be a bare file name, not a path "
                         f"(got {filename!r}); it is always placed in <output_dir>/{LOGS}.")
    return os.path.join(resolve_output_dir(output_params), LOGS, filename)


def create_output_dirs(output_params) -> str:
    """Create the output directory and every subdirectory of it; returns the output directory.

    Called before the loggers are configured, since the file handler opens its file immediately.
    """
    output_dir = resolve_output_dir(output_params)
    for name in SUBDIRS:
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)
    return output_dir
