"""Layout of a Commander4 run's output directory.

A run writes everything below the single ``output.dir``:

    <output_dir>/logs/              log file and cProfile dumps
    <output_dir>/chains_bands/      per-band TOD samples and output maps
    <output_dir>/chains_compsep/    component amplitude chains
    <output_dir>/plots/             figures

The subdirectory names live here so that the run, the plotting tool and the standalone tools all
agree on them, and so that reading a run's output needs only the one path the user configured.

The band chain holds both halves of what the TOD side knows about one band at one Gibbs sample:
the per-scan samples at the top level and the output maps under ``maps/``.
"""
import os

LOGS = "logs"
CHAINS_BANDS = "chains_bands"
CHAINS_COMPSEP = "chains_compsep"
PLOTS = "plots"
INITIAL_STATE = "initial_state"

SUBDIRS = (LOGS, CHAINS_BANDS, CHAINS_COMPSEP, PLOTS)


def band_chain_file(run_dir: str, experiment: str, band: str, chain: int, iteration: int) -> str:
    """Path to one band's saved TOD sample."""
    return os.path.join(run_dir, CHAINS_BANDS,
                        f"{experiment}_{band}_chain{chain:02d}_iter{iteration:04d}.h5")


def compsep_chain_file(run_dir: str, chain: int, iteration: int) -> str:
    """Path to one saved sky sample."""
    return os.path.join(run_dir, CHAINS_COMPSEP, f"chain{chain:02d}_iter{iteration:04d}.h5")


def initial_sky_file(run_dir: str, chain: int) -> str:
    """Fixed sky of a TOD-only run, kept outside the posterior sample directories."""
    return os.path.join(run_dir, INITIAL_STATE, f"chain{chain:02d}_iter0000.h5")


def resolve_output_dir(output_params) -> str:
    """The run's output directory, given the ``output`` parameter block."""
    if "dir" not in output_params:
        raise ValueError("'output.dir' is required: the single directory a run writes all of its "
                         "output to.")
    return str(output_params.dir)


def subdir(params, name: str) -> str:
    """One of the run's output subdirectories, e.g. ``subdir(params, paths.CHAINS_TOD)``."""
    return os.path.join(resolve_output_dir(params.output), name)


def log_file_path(output_params, run_id: str) -> str:
    """Full path of the run's standard ``run-<run-id>.log`` file."""
    return os.path.join(resolve_output_dir(output_params), LOGS, f"run-{run_id}.log")


def create_output_dirs(output_params) -> str:
    """Create the output directory and every subdirectory of it; returns the output directory.

    Called before the loggers are configured, since the file handler opens its file immediately.
    """
    output_dir = resolve_output_dir(output_params)
    for name in SUBDIRS:
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)
    return output_dir
