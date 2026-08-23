"""The order the two interleaved Gibbs chains are stepped through.

Both halves of the pipeline walk this same sequence, so it is the one place the chain and
iteration numbering is decided. It used to be four separate modular-arithmetic expressions inside
the main loop, one pair per side, which is how chain 2 came to be left one compsep sample short.
"""
from commander4.cli import gibbs_schedule


def test_chains_alternate_within_each_iteration():
    assert gibbs_schedule(3) == [(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3)]


def test_every_chain_gets_every_iteration():
    """The off-by-one this replaces gave chain 2 one sample fewer than chain 1."""
    schedule = gibbs_schedule(7)
    for chain in (1, 2):
        assert [it for c, it in schedule if c == chain] == list(range(1, 8))


def test_length_is_two_chains_times_the_iteration_count():
    for num_iterations in (1, 2, 10):
        assert len(gibbs_schedule(num_iterations)) == 2*num_iterations


def test_a_single_iteration_still_steps_both_chains():
    assert gibbs_schedule(1) == [(1, 1), (2, 1)]
