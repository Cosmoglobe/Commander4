import pytest

from commander4.diagnostics.log import VERBOSE
from commander4.diagnostics.performance import PerfLogger


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


def test_a_decorated_function_that_calls_itself_records_each_call() -> None:
    b = PerfLogger()

    @b.benchmark("count-down")
    def count_down(n):
        if n > 0:
            count_down(n - 1)

    count_down(2)
    assert b.data[("count-down",)]["count"] == 1
    assert b.data[("count-down", "count-down")]["count"] == 1
    assert b.data[("count-down", "count-down", "count-down")]["count"] == 1
    assert b._stack == []


def test_a_block_left_by_an_exception_is_recorded() -> None:
    b = PerfLogger()
    with pytest.raises(ValueError):
        with b.benchmark("outer"):
            with b.benchmark("inner"):
                raise ValueError
    assert b.data[("outer", "inner")]["count"] == 1
    assert b._stack == []


def test_benchmark_without_a_tag_fails() -> None:
    b = PerfLogger()
    with pytest.raises(TypeError, match="needs a tag"):
        @b.benchmark
        def f():
            pass


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
    with caplog.at_level(VERBOSE, logger="commander4.diagnostics.performance"):
        b.bench_summary(_FakeComm(), label="test")
    body = [ln for ln in caplog.text.splitlines() if ln.startswith(("outer", "  inner"))]
    assert sum(ln.startswith("outer") for ln in body) == 1   # single top-level row
    assert any(ln.startswith("  inner") for ln in body)      # inner indented under outer
