import math
import time
import logging
from collections import defaultdict
from contextlib import ContextDecorator

class PerfLogger:
    """
    A lightweight, MPI-aware performance profiling tool for HPC applications.

    This logger is designed to track wall-clock time of various code blocks across
    multiple MPI ranks with minimal overhead. It supports hierarchical reporting,
    automatic unit scaling (ns/us/ms/s), and load imbalance detection.

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

    MPI Behavior:
    -------------
    - Recording is LOCAL: 'benchmark', 'start', and 'stop' only record data
      to the local process memory. No MPI communication happens during profiling.
    - Reporting is COLLECTIVE: 'get_bench_summary(comm)' must be called by all ranks
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
        # Stores the accumulated stats
        # Structure: {'tag': {'count': int, 'total_ns': int, 'last_ns': int}}
        self.data = defaultdict(lambda: {"count": 0, "total_ns": 0, "last_ns": 0})
        
        # Stores the start times for the manual "start/stop" mode
        self.active_timers = {}

        # Stores memory snapshots per tag
        # Structure: {'tag': {'count': int, 'total_bytes': int,
        #                     'min_bytes': float, 'max_bytes': int, 'last_bytes': int}}
        self.mem_data = defaultdict(
            lambda: {"count": 0, "total_bytes": 0,
                     "min_bytes": float('inf'), "max_bytes": 0, "last_bytes": 0}
        )

        # Tracks global insertion order across both timing and memory tags
        self._tag_order = []
        self._tag_order_set = set()

        # Configure logging to ensure output is visible
        self.logger = logging.getLogger(__name__)

    def reset(self):
        """
        Clears all accumulated timing and memory data, resetting to a clean slate.
        Any timers that are currently running (via start_bench) are also cancelled.
        """
        self.data.clear()
        self.active_timers.clear()
        self.mem_data.clear()
        self._tag_order.clear()
        self._tag_order_set.clear()

    def _register_tag(self, tag):
        """Records tag in insertion-order list the first time it is seen."""
        if tag not in self._tag_order_set:
            self._tag_order_set.add(tag)
            self._tag_order.append(tag)

    def benchmark(self, tag, increment_count=True):
        """
        Returns a context manager that can also be used as a decorator.
        
        Args:
            tag (str): The unique label for this timing block.
            increment_count (bool): Whether to bump the call counter on exit.
                Set to False inside a tight loop and call increment_count(tag)
                once per outer iteration instead.
        """
        return _BenchmarkContext(self, tag, increment_count)

    def start_bench(self, tag):
        """
        Manually starts a timer for 'tag'. 
        Overwrites the start time if the tag is already running.
        """
        self.active_timers[tag] = time.perf_counter_ns()

    def stop_bench(self, tag, increment_count=True):
        """
        Manually stops the timer for 'tag' and records the duration.
        Logs a warning if called without a corresponding start().

        Args:
            tag (str): The label passed to start_bench.
            increment_count (bool): Whether to bump the call counter.
                Set to False inside a tight loop and call increment_count(tag)
                once per outer iteration instead.
        """
        if tag not in self.active_timers:
            self.logger.warning(f"Performance Warning: stop('{tag}') called without start()")
            return

        t_start = self.active_timers.pop(tag)
        elapsed = time.perf_counter_ns() - t_start
        
        entry = self.data[tag]
        if increment_count:
            entry['count'] += 1
        entry['total_ns'] += elapsed
        entry['last_ns'] = elapsed
        self._register_tag(tag)

    def increment_count(self, tag, n=1):
        """
        Manually bumps the call counter for 'tag' by n.

        Useful when benchmarking with increment_count=False inside a tight
        loop: call this once per outer iteration (or per logical unit) to
        keep the call counter meaningful without coupling it to the inner
        iteration count.

        Args:
            tag (str): The label whose counter to increment.
            n (int):   Amount to add (default 1).
        """
        self.data[tag]['count'] += n
        self._register_tag(tag)

    def log_memory(self, tag):
        """
        Snapshots the current RSS memory of this process and associates it with
        'tag'.  Can be called multiple times; min/max/avg are tracked.

        Args:
            tag (str): Label under which memory is stored.
        """
        mem_bytes = _get_current_memory_bytes()
        entry = self.mem_data[tag]
        entry['count'] += 1
        entry['total_bytes'] += mem_bytes
        entry['min_bytes'] = min(entry['min_bytes'], mem_bytes)
        entry['max_bytes'] = max(entry['max_bytes'], mem_bytes)
        entry['last_bytes'] = mem_bytes
        self._register_tag(tag)

    def get_last(self, tag):
        """Returns the duration of the last call to 'tag' in seconds."""
        return self.data[tag]['last_ns'] / 1e9

    def get_last_memory(self, tag):
        """Returns the last recorded memory for 'tag' in bytes, or 0 if never logged."""
        return self.mem_data[tag]['last_bytes'] if tag in self.mem_data else 0

    def get_bench_summary(self, comm, label="Stats") -> str:
        """
        Aggregates statistics for the given MPI communicator and returns a string to be printed.
        Memory data (from log_memory) is included when present.
        
        Args:
            comm (MPI.Comm): The communicator to aggregate over.
            label (str): A header label for the printed report.
        """
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # Bundle timing + memory + tag order into a single gather call
        local_dump = dict(self.data)
        local_mem_dump = {k: dict(v) for k, v in self.mem_data.items()}
        package = (local_dump, local_mem_dump, list(self._tag_order))
        all_packages = comm.gather(package, root=0)

        if rank == 0:
            all_timing     = [p[0] for p in all_packages]
            all_mem        = [p[1] for p in all_packages]
            all_tag_orders = [p[2] for p in all_packages]
            return self._print_report(all_timing, all_mem, all_tag_orders, size, label)

    def _print_report(self, all_ranks_data, all_mem_data, all_tag_orders, size, label):
        # --- Merge timing data ---
        merged_t = {}
        for rank_data in all_ranks_data:
            if not rank_data: continue
            for tag, stats in rank_data.items():
                if tag not in merged_t:
                    merged_t[tag] = {'counts': [], 'times': []}
                merged_t[tag]['counts'].append(stats['count'])
                merged_t[tag]['times'].append(stats['total_ns'])

        # --- Merge memory data (GB) ---
        GB = 1024 ** 3
        merged_m = {}
        for rank_mem in all_mem_data:
            if not rank_mem: continue
            for tag, stats in rank_mem.items():
                if tag not in merged_m:
                    merged_m[tag] = {'min_gb': [], 'max_gb': [], 'avg_gb': []}
                avg_b = stats['total_bytes'] / stats['count'] if stats['count'] else 0
                merged_m[tag]['min_gb'].append(stats['min_bytes'] / GB)
                merged_m[tag]['max_gb'].append(stats['max_bytes'] / GB)
                merged_m[tag]['avg_gb'].append(avg_b / GB)

        # --- Unified tag order: merge per-rank order lists, then append any stragglers ---
        seen = set()
        tag_order = []
        for order in all_tag_orders:
            for tag in order:
                if tag not in seen:
                    seen.add(tag)
                    tag_order.append(tag)
        for tag in list(merged_t) + list(merged_m):
            if tag not in seen:
                seen.add(tag)
                tag_order.append(tag)

        # --- Auto-scale timing unit ---
        global_max_ns = 0
        if merged_t:
            global_max_ns = max(max(d['times']) for d in merged_t.values())
        unit, scalar = self._get_auto_unit(global_max_ns)

        has_mem = bool(merged_m)

        # --- Column widths ---
        W_TAG = 18
        W_N   = 4    # call count
        W_VAL = 7    # timing and memory value columns
        W_IMB = 5    # imbalance %

        def _v(x):
            """Format value to 3 significant figures, right-aligned in W_VAL."""
            return f"{_fmt3sig(x):>{W_VAL}}"

        def _pct(x):
            return f"{_fmt3sig(x):>{W_IMB}}"

        # --- Build header ---
        t_unit = f"({unit})"
        m_unit = "(GB)"
        h_timing = (f" {'N':>{W_N}} | {'Avg':>{W_VAL-4}}{t_unit} |"
                    f" {'Min':>{W_VAL-4}}{t_unit} | {'Max':>{W_VAL-4}}{t_unit} | {'Imb%':>{W_IMB}}")
        h_mem = (f" || {'Avg':>{W_VAL-4}}{m_unit} |"
                 f" {'Min':>{W_VAL-4}}{m_unit} | {'Max':>{W_VAL-4}}{m_unit} | {'Imb%':>{W_IMB}}") if has_mem else ""
        header = f"{'Tag':<{W_TAG}} |{h_timing}{h_mem}"
        sep = "-" * len(header)

        lines = []
        lines.append(f"\n[{label}] Performance Report (Ranks: {size})")
        lines.append(header)
        lines.append(sep)

        blank_t = f" {'':>{W_N}} | {'':>{W_VAL}} | {'':>{W_VAL}} | {'':>{W_VAL}} | {'':>{W_IMB}}"
        blank_m = f" || {'':>{W_VAL}} | {'':>{W_VAL}} | {'':>{W_VAL}} | {'':>{W_IMB}}" if has_mem else ""

        for tag in tag_order:
            tag_str = f"{tag:<{W_TAG}}"

            # Timing columns
            if tag in merged_t:
                d = merged_t[tag]
                times  = [t / scalar for t in d['times']]
                n_avg  = sum(d['counts']) / len(d['counts'])
                t_avg  = sum(times) / len(times)
                t_min  = min(times)
                t_max  = max(times)
                t_imb  = ((t_max - t_min) / t_max * 100) if t_max > 0 else 0.0
                t_str = f" {_fmt3sig(n_avg):>{W_N}} | {_v(t_avg)} | {_v(t_min)} | {_v(t_max)} | {_pct(t_imb)}"
            else:
                t_str = blank_t

            # Memory columns
            if has_mem:
                if tag in merged_m:
                    d = merged_m[tag]
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
        # self.logger.info("\n".join(lines))
        return "\n".join(lines)

    def _get_auto_unit(self, max_ns):
        if max_ns >= 1e9:    return "s", 1e9
        elif max_ns >= 1e6:  return "ms", 1e6
        elif max_ns >= 1e3:  return "us", 1e3
        else:                return "ns", 1.0


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
    if mag < -2:
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
    __slots__ = ('parent', 'tag', '_increment_count', 'start')

    def __init__(self, parent, tag, increment_count=True):
        self.parent = parent
        self.tag = tag
        self._increment_count = increment_count
        self.start = 0

    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter_ns() - self.start
        entry = self.parent.data[self.tag]
        if self._increment_count:
            entry['count'] += 1
        entry['total_ns'] += elapsed
        entry['last_ns'] = elapsed
        self.parent._register_tag(self.tag)
        return False


# ---------------------------------------------------------------------------
# Global Instance & Public API
# ---------------------------------------------------------------------------
_bench = PerfLogger()

benchmark         = _bench.benchmark        # Context Manager / Decorator
start_bench       = _bench.start_bench      # Manual Start
stop_bench        = _bench.stop_bench       # Manual Stop
increment_count   = _bench.increment_count  # Manually bump a tag's call counter
log_memory        = _bench.log_memory       # Memory snapshot
bench_reset       = _bench.reset            # Clear all accumulated data
get_bench_summary = _bench.get_bench_summary        # Report Generation (timing + memory)
get_last          = _bench.get_last         # Last timing in seconds
get_last_memory   = _bench.get_last_memory  # Last memory snapshot in bytes