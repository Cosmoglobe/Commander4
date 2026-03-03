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

    MPI Behavior:
    -------------
    - Recording is LOCAL: 'benchmark', 'start', and 'stop' only record data
      to the local process memory. No MPI communication happens during profiling.
    - Reporting is COLLECTIVE: 'summarize(comm)' must be called by all ranks
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
        
        # Configure logging to ensure output is visible
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger(__name__)

    def benchmark(self, tag):
        """
        Returns a context manager that can also be used as a decorator.
        
        Args:
            tag (str): The unique label for this timing block.
        """
        return _BenchmarkContext(self, tag)

    def start_bench(self, tag):
        """
        Manually starts a timer for 'tag'. 
        Overwrites the start time if the tag is already running.
        """
        self.active_timers[tag] = time.perf_counter_ns()

    def stop_bench(self, tag):
        """
        Manually stops the timer for 'tag' and records the duration.
        Logs a warning if called without a corresponding start().
        """
        if tag not in self.active_timers:
            self.logger.warning(f"Performance Warning: stop('{tag}') called without start()")
            return

        t_start = self.active_timers.pop(tag)
        elapsed = time.perf_counter_ns() - t_start
        
        entry = self.data[tag]
        entry['count'] += 1
        entry['total_ns'] += elapsed
        entry['last_ns'] = elapsed

    def get_last(self, tag):
        """Returns the duration of the last call to 'tag' in seconds."""
        return self.data[tag]['last_ns'] / 1e9

    def summarize(self, comm, label="Stats"):
        """
        Aggregates and prints statistics for the given MPI communicator.
        
        Args:
            comm (MPI.Comm): The communicator to aggregate over.
            label (str): A header label for the printed report.
        """
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # Gather data (convert to dict for safe pickling)
        local_dump = dict(self.data)
        all_ranks_data = comm.gather(local_dump, root=0)

        if rank == 0:
            self._print_report(all_ranks_data, size, label)

    def _print_report(self, all_ranks_data, size, label):
        merged = defaultdict(lambda: {'counts': [], 'times': []})

        for rank_data in all_ranks_data:
            if not rank_data: continue
            for tag, stats in rank_data.items():
                merged[tag]['counts'].append(stats['count'])
                merged[tag]['times'].append(stats['total_ns'])

        # Auto-scaling unit logic
        global_max_ns = 0
        if merged:
            # Native python max() 
            global_max_ns = max(max(m['times']) for m in merged.values())
        
        unit, scalar = self._get_auto_unit(global_max_ns)

        lines = []
        lines.append(f"\n[{label}] Hierarchy Report (Ranks: {size})")
        lines.append(f"{'Tag':<25} | {'Calls':<6} | {'Avg ('+unit+')':<8} | {'Min ('+unit+')':<8} | {'Max ('+unit+')':<8} | {'Imbal %':<8}")
        lines.append("-" * 80)

        for tag, data in merged.items():
            raw_times = data['times']
            raw_counts = data['counts']
            
            # List comprehension for division (replaces vector division)
            times = [t / scalar for t in raw_times]
            
            # Native Python math
            avg_calls = sum(raw_counts) / len(raw_counts)
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            
            imbalance = 0.0
            if max_time > 0:
                imbalance = ((max_time - min_time) / max_time) * 100

            lines.append(f"{tag:<25} | {avg_calls:<6.1f} | {avg_time:<8.2f} | {min_time:<8.2f} "\
                         f"| {max_time:<8.2f} | {imbalance:<8.1f}")
        lines.append("-" * 80)
        self.logger.info("\n".join(lines))

    def _get_auto_unit(self, max_ns):
        if max_ns >= 1e9:    return "s", 1e9
        elif max_ns >= 1e6:  return "ms", 1e6
        elif max_ns >= 1e3:  return "us", 1e3
        else:                return "ns", 1.0


class _BenchmarkContext(ContextDecorator):
    """
    Helper class to handle the context manager protocol.
    Inherits from ContextDecorator to support usage as @decorator.
    """
    __slots__ = ('parent', 'tag', 'start') 

    def __init__(self, parent, tag):
        self.parent = parent
        self.tag = tag
        self.start = 0

    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter_ns() - self.start
        entry = self.parent.data[self.tag]
        entry['count'] += 1
        entry['total_ns'] += elapsed
        entry['last_ns'] = elapsed
        return False


# --- Global Instance ---
_bench = PerfLogger()

# Public API
benchmark    = _bench.benchmark   # Context Manager / Decorator
start_bench  = _bench.start_bench # Manual Start
stop_bench   = _bench.stop_bench  # Manual Stop
summarize    = _bench.summarize   # Report Generation
get_last     = _bench.get_last    # Runtime check