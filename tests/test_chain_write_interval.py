"""Thinning of the three chain outputs, one interval each.

`output.chains.interval` used to be a single number that only the datamaps writer honoured, so
the TOD and compsep chains silently wrote every iteration no matter what it said. It is now one
entry per output, and the old scalar form is refused rather than reinterpreted.
"""
import pytest
from pixell.bunch import Bunch

from commander4.file_io.chain_writer import CHAIN_KINDS, should_write_chain


def _params(interval=None):
    chains = Bunch(write=[1])
    if interval is not None:
        chains.interval = interval
    return Bunch(output=Bunch(chains=chains))


def test_each_output_takes_its_own_interval():
    params = _params(Bunch(tod=2, compsep=3, datamaps=6))
    written = {kind: [i for i in range(1, 8) if should_write_chain(params, kind, i)]
               for kind in CHAIN_KINDS}
    assert written["tod"] == [1, 3, 5, 7]
    assert written["compsep"] == [1, 4, 7]
    assert written["datamaps"] == [1, 7]


def test_a_missing_entry_means_every_iteration():
    """Only the output you name is thinned; the other two keep writing every iteration."""
    params = _params(Bunch(datamaps=5))
    assert [i for i in range(1, 12) if should_write_chain(params, "datamaps", i)] == [1, 6, 11]
    assert all(should_write_chain(params, "tod", i) for i in range(1, 12))
    assert all(should_write_chain(params, "compsep", i) for i in range(1, 12))


def test_an_absent_interval_block_means_every_iteration():
    params = _params(None)
    for kind in CHAIN_KINDS:
        assert all(should_write_chain(params, kind, i) for i in range(1, 8))


def test_iterations_are_counted_from_one():
    """Iteration numbers are 1-indexed, so interval N writes iterations 1, N+1, 2N+1, ..."""
    params = _params(Bunch(compsep=3))
    written = [i for i in range(1, 11) if should_write_chain(params, "compsep", i)]
    assert written == [1, 4, 7, 10]


def test_interval_of_one_writes_everything():
    params = _params(Bunch(tod=1))
    assert all(should_write_chain(params, "tod", i) for i in range(1, 8))


def test_the_old_scalar_interval_is_refused():
    """A pre-2026 file said `interval: 2` meaning datamaps only; silently applying that to all
    three outputs would thin the TOD and compsep chains nobody asked to thin."""
    params = _params(2)
    with pytest.raises(ValueError, match="one entry per chain output"):
        should_write_chain(params, "datamaps", 1)


def test_an_unknown_output_name_is_refused():
    params = _params(Bunch(tod=1))
    with pytest.raises(ValueError, match="Unknown chain kind"):
        should_write_chain(params, "plots", 1)
