import math
import time
import logging
from collections import defaultdict
from contextlib import ContextDecorator

logger = logging.getLogger(__name__)

class PerfLogger:
    """
    A lightweight, MPI-aware performance profiling tool for HPC applications.

    This logger is designed to track wall-clock time of various code blocks across
    multiple MPI ranks with minimal overhead. It supports hierarchical reporting,
    automatic unit scaling (ns/us/ms/s), and load imbalance detection.

    Nesting is tracked automatically via call paths: each tag is keyed by the full
    tuple of active benchmark tags at the moment it starts, e.g. ("outer", "inner").
    The same tag name called from different parent contexts produces separate data
    entries and appears under the correct parent in the report. Nesting is held on a
    single stack of active frames; stop_bench closes the matching frame and unwinds any
    inner frame left open above it, so an unbalanced or non-LIFO start/stop pair cannot
    desync the stack and corrupt the call paths of subsequent measurements.

    Modes of Operation:
    -------------------
    1. Context Manager (Recommended for blocks):
       >>> with benchmark("io_write"):
       >>>     write_data()

    2. Decorator (Recommended for functions):
       >>> @benchmark("compute_step")
       >>> def compute(): ...

    3. Manual Start/Stop (Recommended for long, linear scripts):
       >>> start_bench("initialization")
       >>> ... setup code ...
       >>> stop_bench("initialization")

    4. Inner-loop timing without inflating the call counter:
       >>> for i in range(N):
       >>>     with benchmark("scan/io",      increment_count=False): ...
       >>>     with benchmark("scan/compute", increment_count=False): ...
       >>> increment_count("scan/io")       # once per outer iteration
       >>> increment_count("scan/compute")
       Note: increment_count resolves the call path from the current active-benchmark
       context, so it must be called in the same benchmark scope as the timed inner calls.

    MPI Behavior:
    -------------
    - Recording is LOCAL: 'benchmark', 'start', and 'stop' only record data
      to the local process memory. No MPI communication happens during profiling.
    - Reporting is COLLECTIVE: 'bench_summary(comm)' must be called by all ranks
      in the provided communicator. It gathers local stats to Rank 0 of that
      communicator for printing.

    Import Warning:
    ---------------
    This module relies on a global singleton instance ('_bench').
    To avoid "split brain" issues where data is recorded to one instance but
    reported from another, ALWAYS import this module using the same absolute path
    throughout your application.

    BAD:  from .utils import benchmark
    GOOD: from my_package.utils.profiling import benchmark
    """

    def __init__(self):
        # Timing data keyed by call-path tuple, e.g. ("outer", "inner").
        # Structure: {path: {'count': int, 'total_ns': int}}
        self.data = defaultdict(lambda: {"count": 0, "total_ns": 0})

        # Memory snapshots keyed by call-path tuple.
        # Structure: {path: {'count': int, 'total_bytes': int, 'min_bytes': float, 'max_bytes': int}}
        self.mem_data = defaultdict(
            lambda: {"count": 0, "total_bytes": 0, "min_bytes": float('inf'), "max_bytes": 0}
        )

        # Tracks insertion order of path tuples across both timing and memory entries.
        self._path_order = []
        self._path_order_set = set()

        # Stack of currently-active frames, each [tag, t_start_ns, path]. Both the manual
        # start/stop API and the benchmark() context manager push/pop on this single stack, so
        # nesting (and the resulting call paths) is tracked uniformly. stop_bench unwinds any
        # leaked inner frames, so a single unbalanced/non-LIFO call cannot permanently desync the
        # stack and poison the call paths (hence the report tree) of everything recorded after it.
        self._stack = []

    def reset(self):
        """
        Clears all accumulated timing and memory data, resetting to a clean slate.
        Any timers that are currently running (via start_bench) are also cancelled.
        """
        self.data.clear()
        self.mem_data.clear()
        self._path_order.clear()
        self._path_order_set.clear()
        self._stack.clear()

    def _register_path(self, path: tuple):
        """Records path in insertion-order list the first time it is seen."""
        if path not in self._path_order_set:
            self._path_order_set.add(path)
            self._path_order.append(path)

    def _start_frame(self, tag: str) -> list:
        """Pushes a new active frame for tag and returns it.

        A frame is [tag, t_start_ns, path], where path is the tuple of all ancestor tags plus
        tag. The same tag opened under different parents therefore produces distinct, correctly
        nested entries. The returned frame is held by the caller so it can be closed by identity.
        """
        path = tuple(f[0] for f in self._stack) + (tag,)
        frame = [tag, time.perf_counter_ns(), path]
        self._stack.append(frame)
        return frame

    def _close_frame(self, idx: int, increment_count: bool, end_ns: int):
        """Records the frame at stack position idx and pops it (along with anything above it).

        Frames left open *above* idx are leaks (an inner start_bench with no matching stop_bench).
        They are discarded with a warning rather than absorbed, so an unbalanced inner call cannot
        leave a stray tag prefixing every subsequent call path.
        """
        stack = self._stack
        if idx != len(stack) - 1:
            stranded = [f[0] for f in stack[idx + 1:]]
            logger.warning(f"Performance Warning: closing '{stack[idx][0]}' discarded un-stopped "
                           f"inner tags {stranded}")
        _, t_start, path = stack[idx]
        del stack[idx:]
        entry = self.data[path]
        if increment_count:
            entry['count'] += 1
        entry['total_ns'] += end_ns - t_start
        self._register_path(path)

    def _current_path(self, tag: str) -> tuple:
        """Call path for a non-scoping record (increment_count / log_memory).

        Normally the active-stack tags plus tag. The exception: when tag already names the
        innermost active block (e.g. log_memory("x") inside `with benchmark("x")`), the record
        attaches to that block's own path instead of spawning a spurious (..., "x", "x") child.
        """
        if self._stack and self._stack[-1][0] == tag:
            return self._stack[-1][2]
        return tuple(f[0] for f in self._stack) + (tag,)

    def benchmark(self, tag: str, increment_count: bool = True):
        """
        Returns a context manager that can also be used as a decorator.

        Args:
            tag: The label for this timing block.
            increment_count: Whether to bump the call counter on exit.
                Set to False inside a tight loop and call increment_count(tag)
                once per outer iteration instead.
        """
        return _BenchmarkContext(self, tag, increment_count)

    def start_bench(self, tag: str):
        """
        Manually starts a timer for 'tag' by pushing an active frame.

        Re-entrant starts of the same tag are supported: each start/stop is matched LIFO, so a
        tag may legitimately nest inside another instance of itself.
        """
        self._start_frame(tag)

    def stop_bench(self, tag: str, increment_count: bool = True):
        """
        Manually stops the timer for 'tag' and records the duration.

        Closes the innermost active frame carrying this tag, unwinding (and warning about) any
        inner frames left open above it. Logs a warning if no matching start() is on the stack.

        Args:
            tag: The label passed to start_bench.
            increment_count: Whether to bump the call counter.
                Set to False inside a tight loop and call increment_count(tag)
                once per outer iteration instead.
        """
        end = time.perf_counter_ns()
        idx = next((i for i in range(len(self._stack) - 1, -1, -1) if self._stack[i][0] == tag),
                   None)
        if idx is None:
            logger.warning(f"Performance Warning: stop('{tag}') called without start()")
            return
        self._close_frame(idx, increment_count, end)

    def increment_count(self, tag: str, n: int = 1):
        """
        Manually bumps the call counter for 'tag' by n, resolving the call path from the
        current active-benchmark context. Must be called in the same benchmark scope as
        the corresponding timed calls.

        Args:
            tag: The label whose counter to increment.
            n:   Amount to add (default 1).
        """
        path = self._current_path(tag)
        self.data[path]['count'] += n
        self._register_path(path)

    def log_memory(self, tag: str):
        """
        Snapshots the current RSS memory and associates it with 'tag' under the current
        active-benchmark context. Can be called multiple times; min/max/avg are tracked.

        If 'tag' names the innermost active block, the snapshot annotates that block rather than
        creating a child entry (so log_memory("x") inside `with benchmark("x")` records on x).

        Args:
            tag: Label under which memory is stored.
        """
        path = self._current_path(tag)
        mem_bytes = _get_current_memory_bytes()
        entry = self.mem_data[path]
        entry['count'] += 1
        entry['total_bytes'] += mem_bytes
        entry['min_bytes'] = min(entry['min_bytes'], mem_bytes)
        entry['max_bytes'] = max(entry['max_bytes'], mem_bytes)
        self._register_path(path)

    def bench_summary(self, comm, label: str = "Stats"):
        """
        Aggregates statistics for the given MPI communicator and prints a report.
        Memory data (from log_memory) is included when present.

        Args:
            comm: The MPI communicator to aggregate over.
            label: A header label for the printed report.
        """
        rank = comm.Get_rank()
        size = comm.Get_size()

        package = (
            {k: dict(v) for k, v in self.data.items()},
            {k: dict(v) for k, v in self.mem_data.items()},
            list(self._path_order),
        )
        all_packages = comm.gather(package, root=0)

        if rank == 0:
            self._print_report(
                [p[0] for p in all_packages],
                [p[1] for p in all_packages],
                [p[2] for p in all_packages],
                size, label,
            )

    def _print_report(self, all_ranks_data, all_mem_data, all_path_orders, size, label):
        # --- Merge timing data ---
        merged_t = {}
        for rank_data in all_ranks_data:
            if not rank_data: continue
            for path, stats in rank_data.items():
                if path not in merged_t:
                    merged_t[path] = {'counts': [], 'times': []}
                merged_t[path]['counts'].append(stats['count'])
                merged_t[path]['times'].append(stats['total_ns'])

        # --- Merge memory data (GB) ---
        GB = 1024 ** 3
        merged_m = {}
        for rank_mem in all_mem_data:
            if not rank_mem: continue
            for path, stats in rank_mem.items():
                if path not in merged_m:
                    merged_m[path] = {'min_gb': [], 'max_gb': [], 'avg_gb': []}
                avg_b = stats['total_bytes'] / stats['count'] if stats['count'] else 0
                merged_m[path]['min_gb'].append(stats['min_bytes'] / GB)
                merged_m[path]['max_gb'].append(stats['max_bytes'] / GB)
                merged_m[path]['avg_gb'].append(avg_b / GB)

        # --- Unified path order: merge per-rank lists, then append stragglers ---
        seen = set()
        path_order = []
        for order in all_path_orders:
            for path in order:
                if path not in seen:
                    seen.add(path)
                    path_order.append(path)
        for path in list(merged_t) + list(merged_m):
            if path not in seen:
                seen.add(path)
                path_order.append(path)

        # --- Build tree: parent of path P is P[:-1]; roots have parent () ---
        children = defaultdict(list)
        for path in path_order:
            children[path[:-1]].append(path)

        ordered_with_depth = []
        def _dfs(path, depth):
            ordered_with_depth.append((path, depth))
            for child in children.get(path, []):
                _dfs(child, depth + 1)
        for root in children[()]:
            _dfs(root, 0)
        # Paths whose parent was never recorded (e.g. only seen on some ranks) fall back to depth 0
        reached = {p for p, _ in ordered_with_depth}
        for path in path_order:
            if path not in reached:
                ordered_with_depth.append((path, 0))

        # --- Auto-scale timing unit ---
        global_max_ns = max((max(d['times']) for d in merged_t.values()), default=0)
        unit, scalar = self._get_auto_unit(global_max_ns)

        has_mem = bool(merged_m)

        # --- Column widths ---
        W_TAG = max((len("  " * d + p[-1]) for p, d in ordered_with_depth), default=18)
        W_N   = 4
        W_VAL = 7
        W_IMB = 5

        def _v(x):   return f"{_fmt3sig(x):>{W_VAL}}"
        def _pct(x): return f"{_fmt3sig(x):>{W_IMB}}"

        t_unit = f"({unit})"
        m_unit = "(GB)"
        h_timing = (f" {'N':>{W_N}} | {'Avg'+t_unit:>{W_VAL}} |"
                    f" {'Min'+t_unit:>{W_VAL}} | {'Max'+t_unit:>{W_VAL}} | {'Imb%':>{W_IMB}}")
        h_mem = (f" || {'Avg'+m_unit:>{W_VAL}} |"
                 f" {'Min'+m_unit:>{W_VAL}} | {'Max'+m_unit:>{W_VAL}} | {'Imb%':>{W_IMB}}") if has_mem else ""
        header = f"{'Tag':<{W_TAG}} |{h_timing}{h_mem}"
        sep = "-" * len(header)

        lines = [f"\n[{label}] Performance Report (Ranks: {size})", header, sep]

        blank_t = f" {'':>{W_N}} | {'':>{W_VAL}} | {'':>{W_VAL}} | {'':>{W_VAL}} | {'':>{W_IMB}}"
        blank_m = f" || {'':>{W_VAL}} | {'':>{W_VAL}} | {'':>{W_VAL}} | {'':>{W_IMB}}" if has_mem else ""

        for path, depth in ordered_with_depth:
            tag_str = f"{'  ' * depth + path[-1]:<{W_TAG}}"

            if path in merged_t:
                d = merged_t[path]
                times = [t / scalar for t in d['times']]
                n_avg = sum(d['counts']) / len(d['counts'])
                t_avg = sum(times) / len(times)
                t_min = min(times)
                t_max = max(times)
                t_imb = ((t_max - t_min) / t_max * 100) if t_max > 0 else 0.0
                t_str = f" {_fmt3sig(n_avg):>{W_N}} | {_v(t_avg)} | {_v(t_min)} | {_v(t_max)} | {_pct(t_imb)}"
            else:
                t_str = blank_t

            if has_mem:
                if path in merged_m:
                    d = merged_m[path]
                    m_avg = sum(d['avg_gb']) / len(d['avg_gb'])
                    m_min = min(d['min_gb'])
                    m_max = max(d['max_gb'])
                    m_imb = ((m_max - m_min) / m_max * 100) if m_max > 0 else 0.0
                    m_str = f" || {_v(m_avg)} | {_v(m_min)} | {_v(m_max)} | {_pct(m_imb)}"
                else:
                    m_str = blank_m
            else:
                m_str = ""

            lines.append(f"{tag_str} |{t_str}{m_str}")

        lines.append(sep)
        logger.info("\n".join(lines))

    def _get_auto_unit(self, max_ns):
        if max_ns >= 1e9:   return "s",  1e9
        elif max_ns >= 1e6: return "ms", 1e6
        elif max_ns >= 1e3: return "us", 1e3
        else:               return "ns", 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt3sig(value):
    """
    Format a float to 3 significant digits without scientific notation.
    Values whose magnitude is below 0.01 are shown as '0.00' to avoid
    runaway decimal places.  Examples:
        1234 -> '1234', 111 -> '111', 11.1 -> '11.1', 1.11 -> '1.11',
        0.111 -> '0.111', 0.00123 -> '0.00', 0 -> '0'.
    """
    if value == 0:
        return "0"
    mag = math.floor(math.log10(abs(value)))
    if mag < -1:
        return "0.00"
    decimals = max(0, 2 - mag)
    return f"{value:.{decimals}f}"


# ---------------------------------------------------------------------------
# Helper: current RSS memory
# ---------------------------------------------------------------------------

def _get_current_memory_bytes():
    """
    Returns the current RSS (Resident Set Size) of this process in bytes.
    Reads /proc/self/status on Linux for an accurate live snapshot;
    falls back to resource.getrusage on other platforms.
    """
    try:
        with open('/proc/self/status') as fh:
            for line in fh:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) * 1024   # kB → bytes
    except OSError:
        pass
    try:
        import resource
        # ru_maxrss is in KB on Linux, bytes on macOS — Linux assumed here
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except Exception:
        return 0


class _BenchmarkContext(ContextDecorator):
    """
    Helper class to handle the context manager protocol.
    Inherits from ContextDecorator to support usage as @decorator.
    """
    __slots__ = ('parent', 'tag', '_increment_count', '_frame')

    def __init__(self, parent, tag, increment_count=True):
        self.parent = parent
        self.tag = tag
        self._increment_count = increment_count
        self._frame = None

    def __enter__(self):
        self._frame = self.parent._start_frame(self.tag)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.perf_counter_ns()
        # Locate our own frame by identity (not by tag), so a same-named inner benchmark is never
        # mistaken for this one. A missing frame means it was already unwound by an outer
        # stop_bench (and warned about); _close_frame discards any frames leaked inside the block.
        stack = self.parent._stack
        idx = next((i for i in range(len(stack) - 1, -1, -1) if stack[i] is self._frame), None)
        if idx is not None:
            self.parent._close_frame(idx, self._increment_count, end)
        return False


# ---------------------------------------------------------------------------
# Global Instance & Public API
# ---------------------------------------------------------------------------
_bench = PerfLogger()

benchmark       = _bench.benchmark       # Context Manager / Decorator
start_bench     = _bench.start_bench     # Manual Start
stop_bench      = _bench.stop_bench      # Manual Stop
increment_count = _bench.increment_count # Manually bump a tag's call counter
log_memory      = _bench.log_memory      # Memory snapshot
bench_reset     = _bench.reset           # Clear all accumulated data
bench_summary   = _bench.bench_summary   # Report Generation (timing + memory)
