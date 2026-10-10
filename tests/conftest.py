"""Shared pytest setup, loaded by pytest before any test module is imported."""
import os

# Numba starts one thread per CPU on the machine (hundreds on a cluster node) unless told otherwise.
# Waking that many threads costs milliseconds per parallel kernel call, which dominates the small
# arrays used in tests, and multiplies when running tests in parallel with pytest-xdist. A real run
# sets this in mpi/setup.py; tests never call that, so set it here, before numba is imported.
# Four threads still exercise the parallel (prange) code paths.
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
