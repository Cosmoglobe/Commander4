"""Correctness contracts for low-level in-place arithmetic helpers."""

import numpy as np
import pytest

from commander4.math_utils.arithmetic import inplace_arr_add, norm


def test_norm_is_the_euclidean_norm_not_the_squared_norm() -> None:
    assert norm(np.array([3.0, 4.0])) == pytest.approx(5.0)


def test_shape_checks_raise_explicit_exceptions() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        inplace_arr_add(np.ones(3), np.ones(4))
