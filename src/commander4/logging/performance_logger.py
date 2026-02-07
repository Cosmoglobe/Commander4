import time
import logging
from contextlib import contextmanager
from collections import defaultdict
import numpy as np


class PerfLogger:
    def __init__(self):
        # Using defaultdict to get automatic entry creation on key miss.
        self.data = defaultdict(lambda: {"count": 0, "total_ns": 0, "last_ns": 0})
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def benchmark(self, tag):
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - t0
            entry = self.data[tag]
            entry['count'] += 1
            entry['total_ns'] += elapsed
            entry['last_ns'] = elapsed

    def summarize(self, comm, label="Stats"):
        """
        Aggregates stats ONLY for the ranks in the provided 'comm'.
        """
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Gather data from this specific communicator
        local_dump = dict(self.data)
        all_ranks_data = comm.gather(local_dump, root=0)

        # Only root of this comm prints
        if rank == 0:
            self._print_report(all_ranks_data, size, label)

    def _print_report(self, all_ranks_data, size, label):
        printstring = ""
        merged = defaultdict(lambda: {'counts': [], 'times': []})

        for rank_data in all_ranks_data:
            for tag, stats in rank_data.items():
                merged[tag]['counts'].append(stats['count'])
                merged[tag]['times'].append(stats['total_ns'])

        printstring += f"[{label}] Hierarchy Report (Ranks: {size})\n"
        printstring += f"{'Tag':<25} | {'Calls':<6} | {'Avg (s)':<8} | {'Min (s)':<8} | {'Max (s)':<8} | {'Imbal %':<8}\n"
        printstring += "-" * 78 + "\n"

        for tag, data in sorted(merged.items()):
            times_ms = np.array(data['times']) / 1e9
            avg_time = np.mean(times_ms)
            max_time = np.max(times_ms)
            min_time = np.min(times_ms)
            avg_calls = np.mean(data['counts'])
            
            imbalance = 0.0
            if max_time > 0:
                imbalance = ((max_time - min_time) / max_time) * 100

            printstring += f"{tag:<25} | {avg_calls:<6.1f} | {avg_time:<8.2f} | {min_time:<8.2f} | {max_time:<8.2f} | {imbalance:<8.1f}\n"
        printstring += "-" * 78 + "\n"
        self.logger.info(printstring)

_bench = PerfLogger()
benchmark = _bench.benchmark
summarize = _bench.summarize