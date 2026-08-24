"""Thinning of the chain outputs, one interval each.

`output.chains.interval` is a block with one entry per output. Anything else -- a bare number, or a
key no writer reads -- is refused rather than reinterpreted, because either would silently thin the
wrong outputs or nothing at all.

`bands` and `compsep` each gate a whole file. `maps` gates only the `maps/` group inside the band
file, so the maps appear when both `bands` and `maps` fire; the writer applies that, this module
only covers the per-kind schedule.
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
    params = _params(Bunch(bands=2, compsep=3, maps=6))
    written = {kind: [i for i in range(1, 8) if should_write_chain(params, kind, i)]
               for kind in CHAIN_KINDS}
    assert written["bands"] == [1, 3, 5, 7]
    assert written["compsep"] == [1, 4, 7]
    assert written["maps"] == [1, 7]


def test_a_missing_entry_means_every_iteration():
    """Only the output you name is thinned; the other two keep writing every iteration."""
    params = _params(Bunch(maps=5))
    assert [i for i in range(1, 12) if should_write_chain(params, "maps", i)] == [1, 6, 11]
    assert all(should_write_chain(params, "bands", i) for i in range(1, 12))
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
    params = _params(Bunch(bands=1))
    assert all(should_write_chain(params, "bands", i) for i in range(1, 8))


def test_a_scalar_interval_is_refused():
    """A bare `interval: 2` says nothing about *which* output to thin, so it cannot be applied."""
    params = _params(2)
    with pytest.raises(ValueError, match="block of per-output entries"):
        should_write_chain(params, "maps", 1)


def test_unknown_interval_keys_are_refused():
    """A key no writer reads would thin nothing at all, which must not pass silently."""
    params = _params(Bunch(tod=1, compsep=1, datamaps=5))
    with pytest.raises(ValueError, match="unknown entries"):
        should_write_chain(params, "maps", 1)


def test_an_unknown_output_name_is_refused():
    params = _params(Bunch(bands=1))
    with pytest.raises(ValueError, match="Unknown chain kind"):
        should_write_chain(params, "plots", 1)
