"""The order and independent random streams of the two interleaved Gibbs chains.

Both halves of the pipeline walk this same sequence, so it is the one place the chain and
iteration numbering is decided. It used to be four separate modular-arithmetic expressions inside
the main loop, one pair per side, which is how chain 2 came to be left one compsep sample short.
"""
import numpy as np
from pixell.bunch import Bunch

from commander4.cli import gibbs_schedule, make_iteration_rng


def test_chains_alternate_within_each_iteration():
    assert gibbs_schedule(1, 3) == [(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3)]


def test_a_continued_chain_samples_only_the_missing_iterations():
    """`num_iterations` is the chains' total length, as Commander3's NUM_GIBBS_ITER."""
    assert gibbs_schedule(3, 4) == [(1, 3), (2, 3), (1, 4), (2, 4)]
    assert gibbs_schedule(5, 4) == []


def _mpi_info(rank: int = 3) -> Bunch:
    return Bunch(world=Bunch(rank=rank))


def _first_draws(params: Bunch, mpi_info: Bunch, chain: int, iteration: int) -> np.ndarray:
    return make_iteration_rng(params, mpi_info, chain=chain, iteration=iteration).normal(size=5)


def test_iteration_rng_is_reproducible_and_ignores_global_numpy_state() -> None:
    params = Bunch(gibbs=Bunch(seed=1234))

    first_draw = _first_draws(params, _mpi_info(), chain=1, iteration=2)
    np.random.seed(7)  # The generator must not read NumPy's legacy global state.
    second_draw = _first_draws(params, _mpi_info(), chain=1, iteration=2)

    np.testing.assert_array_equal(first_draw, second_draw)


def test_chain_iteration_and_rank_select_independent_streams() -> None:
    """The world rank also separates the TOD and CompSep sides, which hold different ranks."""
    params = Bunch(gibbs=Bunch(seed=1234))
    draws = [
        _first_draws(params, _mpi_info(3), chain=1, iteration=2),
        _first_draws(params, _mpi_info(3), chain=2, iteration=2),
        _first_draws(params, _mpi_info(3), chain=1, iteration=3),
        _first_draws(params, _mpi_info(4), chain=1, iteration=2),
    ]

    assert len({tuple(draw) for draw in draws}) == 4


def test_default_root_seed_matches_explicit_1995() -> None:
    default = _first_draws(Bunch(gibbs=Bunch()), _mpi_info(), chain=1, iteration=1)
    explicit = _first_draws(Bunch(gibbs=Bunch(seed=1995)), _mpi_info(), chain=1, iteration=1)

    np.testing.assert_array_equal(default, explicit)
