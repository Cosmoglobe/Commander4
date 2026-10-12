"""The scan worker threads (`commander4.threads`) and `TODView.map`.

The threaded runs must give the same numbers as the serial ones to the last bit. These tests check
the pieces that make that so: results come back in task order, each detector-scan's random numbers
depend only on the scan and the step, and benchmark entries of workers land under their caller.
"""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from commander4 import threads
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.pointing import PixelPointing
from commander4.data_models.scan_tod import ScanTOD
from commander4.diagnostics.performance import _bench, bench_reset, benchmark
from commander4.tod.view import TODView


@pytest.fixture
def four_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the test with 4 scan threads, in a pool of its own."""
    monkeypatch.setattr(threads, "SCAN_THREADS", 4)
    monkeypatch.setattr(threads, "_pool", None)


def _band(scan_ids: list[int], ndet: int = 2, ntod: int = 50) -> tuple[DetectorGroupTOD,
                                                                       SimpleNamespace]:
    """A band of tiny detector-scans, and a stand-in for its sampled parameters."""
    detectors = lambda: [DetectorTOD(
        name=f"d{idet}", det_idx_fullband=idet, tod=np.zeros(ntod, dtype=np.float32),
        pointing=PixelPointing(np.zeros(ntod, dtype=np.int64), np.zeros(ntod),
                               np.array([0], dtype=np.int64), None, None, 1, 1, ntod, ntod),
        sampling_rate_hz=10.0, orbital_velocity_m_per_s=None, huffman_tree=None,
        huffman_symbols=None, default_proc_mask=None, specific_proc_masks={})
        for idet in range(ndet)]
    scans = [ScanTOD(detectors(), 0.0, scan_id) for scan_id in scan_ids]
    band = DetectorGroupTOD(scans, "EXP", "BAND", 1, 100.0, 10.0, 10.0, ndet, "I",
                            SimpleNamespace(npar=1))
    samples = SimpleNamespace(seed=1995, chain=1, iteration=3, accept=np.ones((len(scans), ndet),
                              dtype=bool), scan_ids=np.array(scan_ids),
                              scan_runtime=np.zeros(len(scans)))
    return band, samples


def _draws(view: TODView, n: int) -> tuple[int, int, np.ndarray]:
    """A per-scan function: the scan ID, detector and a few random numbers of one detector-scan."""
    time.sleep(0.002 * (view.iscan % 3))   # Uneven run times, so the threads finish out of order.
    return int(view.tod_samples.scan_ids[view.iscan]), view.idet, view.rng.standard_normal(n)


def test_run_in_order_yields_results_in_task_order(four_workers) -> None:
    def slow_square(i: int) -> int:
        time.sleep(0.001 * ((7 * i) % 5))
        return i * i
    assert list(threads.run_in_order(slow_square, [(i,) for i in range(40)])) == \
        [i * i for i in range(40)]


def test_run_in_order_raises_a_worker_error_in_the_caller(four_workers) -> None:
    def fails_on_five(i: int) -> int:
        if i == 5:
            raise ValueError("task five")
        return i
    with pytest.raises(ValueError, match="task five"):
        list(threads.run_in_order(fails_on_five, [(i,) for i in range(20)]))


def test_map_is_identical_for_one_and_four_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    band, samples = _band([11, 12, 13, 14, 15])
    serial = list(TODView(band, samples).map(_draws, 4, step="test"))
    monkeypatch.setattr(threads, "SCAN_THREADS", 4)
    monkeypatch.setattr(threads, "_pool", None)
    threaded = list(TODView(band, samples).map(_draws, 4, step="test"))
    assert len(serial) == 10
    for (scan_a, det_a, draws_a), (scan_b, det_b, draws_b) in zip(serial, threaded):
        assert (scan_a, det_a) == (scan_b, det_b)
        assert np.array_equal(draws_a, draws_b)
    assert np.all(samples.scan_runtime > 0)


def test_random_numbers_follow_the_scan_not_the_rank_layout() -> None:
    """A scan draws the same numbers whatever other scans its rank holds; steps draw different."""
    one_rank = {(scan, det): draws for scan, det, draws
                in TODView(*_band([11, 12, 13, 14])).map(_draws, 3, step="test")}
    other_rank = {(scan, det): draws for scan, det, draws
                  in TODView(*_band([14, 12])).map(_draws, 3, step="test")}
    for key, draws in other_rank.items():
        assert np.array_equal(draws, one_rank[key])
    other_step = {(scan, det): draws for scan, det, draws
                  in TODView(*_band([14, 12])).map(_draws, 3, step="another")}
    assert not np.array_equal(other_step[(14, 0)], other_rank[(14, 0)])


def test_map_skips_rejected_detector_scans() -> None:
    band, samples = _band([11, 12, 13])
    samples.accept[1, 0] = False
    results = list(TODView(band, samples).map(_draws, 1, step="test", accepted_only=True))
    assert [(scan, det) for scan, det, _ in results] == [(11, 0), (11, 1), (12, 1), (13, 0),
                                                         (13, 1)]


def test_worker_benchmarks_are_filed_under_the_caller(four_workers) -> None:
    def timed(view: TODView) -> None:
        with benchmark("inner"):
            time.sleep(0.001)

    bench_reset()
    band, samples = _band([11, 12, 13, 14, 15, 16])
    with benchmark("outer"):
        list(TODView(band, samples).map(timed, step="test"))
    assert _bench.data[("outer", "inner")]["count"] == 12
    assert ("inner",) not in _bench.data
    bench_reset()
