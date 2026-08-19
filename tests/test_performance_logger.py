import logging

from commander4.logging.performance_logger import PerfLogger


class _FakeComm:
    """Minimal single-rank stand-in for an MPI communicator (gather returns [local])."""
    def Get_rank(self): return 0
    def Get_size(self): return 1
    def gather(self, x, root=0): return [x]


def test_nested_paths_keyed_by_call_path() -> None:
    b = PerfLogger()
    with b.benchmark("outer"):
        with b.benchmark("inner"):
            pass
    assert ("outer",) in b.data
    assert ("outer", "inner") in b.data
    assert b._stack == []


def test_same_tag_under_different_parents_is_distinct() -> None:
    b = PerfLogger()
    for parent in ("a", "b"):
        with b.benchmark(parent):
            with b.benchmark("leaf"):
                pass
    assert ("a", "leaf") in b.data
    assert ("b", "leaf") in b.data


def test_unbalanced_stop_does_not_corrupt_following_paths() -> None:
    # A stop with no matching start must be a no-op that leaves the stack untouched, so the next
    # benchmark is recorded at the root rather than inheriting a stray prefix.
    b = PerfLogger()
    b.stop_bench("never-started")
    assert b._stack == []
    with b.benchmark("real"):
        pass
    assert ("real",) in b.data


def test_leaked_inner_frame_is_unwound_by_outer_stop() -> None:
    # An inner start_bench that is never stopped is the core failure mode: without unwinding it
    # would prefix every later path. stop_bench("outer") must discard it and restore a clean stack.
    b = PerfLogger()
    b.start_bench("outer")
    b.start_bench("leaked")
    b.stop_bench("outer")
    assert b._stack == []
    assert ("outer",) in b.data
    assert ("outer", "leaked") not in b.data
    with b.benchmark("next"):
        pass
    assert ("next",) in b.data  # no ("leaked", "next") orphan


def test_reentrant_same_tag_start_stop() -> None:
    b = PerfLogger()
    b.start_bench("x")
    b.start_bench("x")
    b.stop_bench("x")  # closes the inner ("x", "x")
    b.stop_bench("x")  # closes the outer ("x",)
    assert b._stack == []
    assert ("x",) in b.data
    assert ("x", "x") in b.data


def test_log_memory_same_name_as_block_attaches_to_block() -> None:
    # Regression: log_memory("x") inside `with benchmark("x")` must annotate x, not split off a
    # separate ("x", "x") memory-only entry (which produced duplicate jump-detect rows).
    b = PerfLogger()
    with b.benchmark("jump-detect"):
        b.log_memory("jump-detect")
    assert ("jump-detect",) in b.data
    assert ("jump-detect",) in b.mem_data
    assert ("jump-detect", "jump-detect") not in b.mem_data


def test_log_memory_new_tag_creates_child_entry() -> None:
    b = PerfLogger()
    with b.benchmark("block"):
        b.log_memory("mem-here")
    assert ("block", "mem-here") in b.mem_data


def test_report_nests_without_duplicate_top_level(caplog) -> None:
    b = PerfLogger()
    with b.benchmark("outer"):
        with b.benchmark("inner"):
            pass
    with caplog.at_level(logging.INFO, logger="commander4.logging.performance_logger"):
        b.bench_summary(_FakeComm(), label="test")
    body = [ln for ln in caplog.text.splitlines() if ln.startswith(("outer", "  inner"))]
    assert sum(ln.startswith("outer") for ln in body) == 1   # single top-level row
    assert any(ln.startswith("  inner") for ln in body)      # inner indented under outer
