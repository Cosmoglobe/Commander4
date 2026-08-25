"""The order and independent random streams of the two interleaved Gibbs chains.

Both halves of the pipeline walk this same sequence, so it is the one place the chain and
iteration numbering is decided. It used to be four separate modular-arithmetic expressions inside
the main loop, one pair per side, which is how chain 2 came to be left one compsep sample short.
"""
import numpy as np
from pixell.bunch import Bunch

from commander4.cli import gibbs_schedule, seed_iteration_rng


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


def _mpi_info(side: str = "tod", rank: int = 3) -> Bunch:
    return Bunch(world=Bunch(side=side, rank=rank))


def test_iteration_seed_is_reproducible_and_ignores_previous_random_draws() -> None:
    params = Bunch(gibbs=Bunch(seed=1234))

    seed_iteration_rng(params, _mpi_info(), chain=1, iteration=2)
    first_draw = np.random.normal(size=5)
    np.random.normal(size=100)
    seed_iteration_rng(params, _mpi_info(), chain=1, iteration=2)
    second_draw = np.random.normal(size=5)

    np.testing.assert_array_equal(first_draw, second_draw)


def test_chain_iteration_rank_and_side_select_independent_streams() -> None:
    params = Bunch(gibbs=Bunch(seed=1234))
    seeds = {
        seed_iteration_rng(params, _mpi_info("tod", 3), chain=1, iteration=2),
        seed_iteration_rng(params, _mpi_info("tod", 3), chain=2, iteration=2),
        seed_iteration_rng(params, _mpi_info("tod", 3), chain=1, iteration=3),
        seed_iteration_rng(params, _mpi_info("tod", 4), chain=1, iteration=2),
        seed_iteration_rng(params, _mpi_info("compsep", 3), chain=1, iteration=2),
    }

    assert len(seeds) == 5


def test_default_root_seed_matches_explicit_1995() -> None:
    default = seed_iteration_rng(Bunch(gibbs=Bunch()), _mpi_info(), chain=1, iteration=1)
    explicit = seed_iteration_rng(
        Bunch(gibbs=Bunch(seed=1995)), _mpi_info(), chain=1, iteration=1,
    )

    assert default == explicit
