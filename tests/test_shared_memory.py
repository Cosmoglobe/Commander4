"""`SharedArray`: allocation, publication and the explicit free that MPI requires.

These run on one rank, so they cover the allocation contract and the free, but not the sharing
itself. Whether two ranks on a node see the same physical memory needs a real multi-rank run.
"""
import logging

import numpy as np
import pytest
from mpi4py import MPI

from commander4.mpi.shared_memory import SharedArray


def test_owner_writes_are_visible_through_the_array():
    shared = SharedArray(MPI.COMM_SELF, (2, 3), dtype=np.float64)
    try:
        assert shared.is_owner
        assert shared.array.shape == (2, 3)
        assert shared.array.dtype == np.float64
        shared.array[:] = 5.0
        shared.wait_until_filled()
        np.testing.assert_array_equal(shared.array, np.full((2, 3), 5.0))
    finally:
        shared.free()


def test_slices_of_the_buffer_are_contiguous():
    """The far-beam projector hands ducc0 one detector's slice of a stacked buffer."""
    shared = SharedArray(MPI.COMM_SELF, (4, 5, 6))
    try:
        assert shared.array[2].flags["C_CONTIGUOUS"]
    finally:
        shared.free()


def test_repeated_allocate_and_free_does_not_exhaust_windows():
    """One allocation per Gibbs iteration must not accumulate MPI windows."""
    for _ in range(50):
        shared = SharedArray(MPI.COMM_SELF, (64, 64))
        shared.array[:] = 1.0
        shared.free()


def test_free_is_idempotent():
    """A second free must not attempt to release the window twice."""
    shared = SharedArray(MPI.COMM_SELF, (4, 4))
    shared.free()
    shared.free()
    assert shared.array is None


def test_array_is_unusable_after_free():
    """Guards the contract that callers must drop their own views before freeing."""
    shared = SharedArray(MPI.COMM_SELF, (4, 4))
    shared.free()
    with pytest.raises(TypeError):
        shared.array[0, 0]


def test_dropping_an_unfreed_array_is_reported(caplog):
    """A forgotten free() must show up in the log rather than as silent memory growth."""
    shared = SharedArray(MPI.COMM_SELF, (4, 4), name="test cubes")
    with caplog.at_level(logging.ERROR):
        shared.__del__()
    assert "test cubes" in caplog.text
    assert "free()" in caplog.text
    shared.free()


def test_freed_array_reports_nothing(caplog):
    with caplog.at_level(logging.ERROR):
        shared = SharedArray(MPI.COMM_SELF, (4, 4), name="test cubes")
        shared.free()
        shared.__del__()
    assert caplog.text == ""
