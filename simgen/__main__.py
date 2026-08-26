"""Command-line entry point.

Run after installing Commander4:

    mpirun -n N c4-simgen -p path/to/param.yml
"""
import sys
import logging
from argparse import ArgumentParser
from traceback import print_exc
from mpi4py import MPI


def configure_logging(rank: int, verbose: bool) -> None:
    """Configure rank-aware output for simgen without enabling third-party INFO logging."""
    logger = logging.getLogger("simgen")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        f"[rank {rank}] %(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.handlers[:] = [handler]


def main() -> None:
    parser = ArgumentParser(description="Modular TOD simulator for Commander4's standard format.")
    parser.add_argument("-p", "--parameter-file", "--parameter_file", required=True,
                        help="Path to a YAML simgen parameter file (see params/param_default.yml).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    rank = MPI.COMM_WORLD.Get_rank()
    configure_logging(rank, args.verbose)

    # Import here (not at module load) so --help works without the heavy sky stack.
    from simgen.pipeline import run
    try:
        sys.exit(run(args.parameter_file))
    except Exception:
        print_exc()
        print(f">>>>>>>> simgen error on rank {rank}; calling MPI abort.")
        MPI.COMM_WORLD.Abort(1)


if __name__ == "__main__":
    main()
