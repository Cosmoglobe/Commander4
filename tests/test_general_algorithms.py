"""Tests for the sorted-array search helpers in commander4.math_utils.search.

`np.searchsorted(..., side='left')` is the reference implementation throughout: all four
routines are documented to return the same leftmost insertion index, so property tests just
compare against it on randomized sorted arrays and query values.
"""

import numpy as np
from numpy.testing import assert_array_equal

from commander4.math_utils.search import (
    bisect_search, bisect_search_many, gallop_search, gallop_search_many,
)


# --- bisect_search -----------------------------------------------------------------------------

def test_bisect_search_matches_searchsorted_with_duplicates() -> None:
    arr = np.array([1, 3, 3, 3, 5, 7, 9], dtype=np.int64)
    for val in [0, 1, 2, 3, 4, 5, 9, 10]:
        assert bisect_search(arr, val) == np.searchsorted(arr, val, side="left")


def test_bisect_search_out_of_range_values() -> None:
    arr = np.array([2, 4, 6, 8], dtype=np.int64)
    assert bisect_search(arr, -100) == 0
    assert bisect_search(arr, 100) == arr.shape[0]


def test_bisect_search_single_element_array() -> None:
    arr = np.array([5], dtype=np.int64)
    assert bisect_search(arr, 4) == 0
    assert bisect_search(arr, 5) == 0
    assert bisect_search(arr, 6) == 1


def test_bisect_search_random_property() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        arr = np.sort(rng.integers(0, 1000, size=rng.integers(1, 200))).astype(np.int64)
        vals = rng.integers(-50, 1050, size=20).astype(np.int64)
        for val in vals:
            assert bisect_search(arr, val) == np.searchsorted(arr, val, side="left")


# --- bisect_search_many -------------------------------------------------------------------------

def test_bisect_search_many_matches_searchsorted() -> None:
    rng = np.random.default_rng(1)
    arr = np.sort(rng.integers(0, 1000, size=200)).astype(np.int64)
    vals = rng.integers(-50, 1050, size=500).astype(np.int64)
    expected = np.searchsorted(arr, vals, side="left")

    out = np.empty(vals.shape[0], dtype=np.int64)
    bisect_search_many(arr, vals, out)

    assert_array_equal(out, expected)


# --- gallop_search -------------------------------------------------------------------------------

def test_gallop_search_matches_bisect_regardless_of_hint() -> None:
    # Correctness must not depend on the hint's quality: `prev` only affects speed, per the
    # docstring's sentinel-boundary argument. Cover in-range, out-of-range, and negative hints.
    arr = np.array([1, 3, 3, 3, 5, 7, 9, 11, 13], dtype=np.int64)
    n = arr.shape[0]
    hints = [-1000, -1, 0, n // 2, n - 1, n, n + 1000]
    for val in [-5, 0, 1, 2, 3, 4, 9, 13, 14, 20]:
        expected = np.searchsorted(arr, val, side="left")
        for prev in hints:
            assert gallop_search(arr, val, prev) == expected


def test_gallop_search_random_property_with_random_hints() -> None:
    rng = np.random.default_rng(2)
    for _ in range(50):
        arr = np.sort(rng.integers(0, 1000, size=rng.integers(1, 200))).astype(np.int64)
        n = arr.shape[0]
        vals = rng.integers(-50, 1050, size=20).astype(np.int64)
        hints = rng.integers(-2 * n - 10, 2 * n + 10, size=20)
        for val, prev in zip(vals, hints):
            expected = np.searchsorted(arr, val, side="left")
            assert gallop_search(arr, int(val), int(prev)) == expected


def test_gallop_search_single_element_array() -> None:
    arr = np.array([5], dtype=np.int64)
    for prev in [-10, 0, 10]:
        assert gallop_search(arr, 4, prev) == 0
        assert gallop_search(arr, 5, prev) == 0
        assert gallop_search(arr, 6, prev) == 1


# --- gallop_search_many -------------------------------------------------------------------------

def test_gallop_search_many_matches_searchsorted() -> None:
    rng = np.random.default_rng(3)
    arr = np.sort(rng.integers(0, 1000, size=200)).astype(np.int64)
    vals = rng.integers(-50, 1050, size=500).astype(np.int64)
    expected = np.searchsorted(arr, vals, side="left")

    out = np.empty(vals.shape[0], dtype=np.int64)
    gallop_search_many(arr, vals, out)

    assert_array_equal(out, expected)


def test_gallop_search_many_preserves_order_for_sorted_queries() -> None:
    # The finger-carrying optimization assumes consecutive queries are close together; a sorted
    # query array is the case it is designed for, and each chunk must stay in query order.
    rng = np.random.default_rng(4)
    arr = np.sort(rng.integers(0, 10_000, size=2000)).astype(np.int64)
    vals = np.sort(rng.integers(-100, 10_100, size=1000)).astype(np.int64)
    expected = np.searchsorted(arr, vals, side="left")

    out = np.empty(vals.shape[0], dtype=np.int64)
    gallop_search_many(arr, vals, out)

    assert_array_equal(out, expected)


def test_gallop_search_many_small_input_fewer_than_threads() -> None:
    # Query count smaller than the thread count degenerates several chunks to zero-length; those
    # must be skipped cleanly rather than indexing out of bounds.
    arr = np.array([1, 3, 5, 7, 9], dtype=np.int64)
    vals = np.array([0, 4, 9], dtype=np.int64)
    expected = np.searchsorted(arr, vals, side="left")

    out = np.empty(vals.shape[0], dtype=np.int64)
    gallop_search_many(arr, vals, out)

    assert_array_equal(out, expected)
