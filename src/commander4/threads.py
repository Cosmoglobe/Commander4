"""The threads of a TOD rank, and the worker pool that runs `TODView.map`.

A TOD rank has `resources.tod.num_threads` cores. Outside `TODView.map` all of them go to the
library calls (ducc0 FFTs, numba loops). Inside it, `resources.tod.scan_threads` worker threads
each process one detector-scan at a time, and a library call in a worker gets an equal share of the
cores. With `scan_threads: 1` (the default) there are no workers, and everything runs as before.
"""
import itertools
import os
import threading
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor

from commander4.diagnostics.performance import perf_path, set_perf_base

SCAN_THREADS = 1                 # Worker threads per TOD rank, set once by `mpi/setup.py`.
_worker = threading.local()      # `_worker.active` is True inside a worker thread.
_pool: ThreadPoolExecutor | None = None


def library_nthreads() -> int:
    """Threads one library call may use: all of the rank's cores, or a worker's share of them."""
    nthreads = int(os.environ.get("OMP_NUM_THREADS", 1))
    if getattr(_worker, "active", False):
        return max(1, nthreads // SCAN_THREADS)
    return nthreads


def _start_worker() -> None:
    import numba   # Imported here: numba must load after `mpi/setup.py` sets NUMBA_NUM_THREADS.
    _worker.active = True
    numba.set_num_threads(library_nthreads())   # numba's thread count is set per calling thread.


def _serialize_numba_compilation() -> None:
    """Let only one thread at a time compile a numba function.

    numba (0.68) times each compilation with listeners kept in one global registry, which it
    changes outside its own compiler lock. Two threads compiling at once, as the workers do on the
    first call of a kernel, then mix up each other's listeners and crash (AttributeError in
    `numba.core.event`). A function compiles once for each set of argument types, so the lock costs
    nothing afterwards.
    """
    from numba.core.dispatcher import Dispatcher
    if getattr(Dispatcher.compile, "one_at_a_time", False):
        return
    numba_compile = Dispatcher.compile
    lock = threading.RLock()   # Reentrant: compiling one kernel can compile those it calls.

    def compile_alone(self, sig):
        with lock:
            return numba_compile(self, sig)

    compile_alone.one_at_a_time = True
    Dispatcher.compile = compile_alone


def run_in_order(func: Callable, tasks: list[tuple]) -> Iterator:
    """Run `func(*task)` for every task on the worker threads, and yield the results in task order.

    At most four tasks per worker are in flight. Results that finish early wait for the earlier
    ones, so this bounds how many of them are held at once when one task is slow. With one scan
    thread the tasks run here, one after the other.
    """
    if SCAN_THREADS == 1:
        for task in tasks:
            yield func(*task)
        return
    global _pool
    if _pool is None:
        _serialize_numba_compilation()
        _pool = ThreadPoolExecutor(SCAN_THREADS, initializer=_start_worker)
    caller_path = perf_path()

    def run(task: tuple):
        set_perf_base(caller_path)   # Benchmark entries go under the caller's open tags.
        return func(*task)

    remaining = iter(tasks)
    in_flight: deque[Future] = deque(_pool.submit(run, task)
                                     for task in itertools.islice(remaining, 4 * SCAN_THREADS))
    while in_flight:
        result = in_flight.popleft().result()   # Re-raises a worker's exception here.
        in_flight.extend(_pool.submit(run, task) for task in itertools.islice(remaining, 1))
        yield result
