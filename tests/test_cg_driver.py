import numpy as np
import pytest

from commander4.math_utils.cg_driver import DistributedCG


def test_cg_reports_the_squared_relative_preconditioned_residual() -> None:
    diagonal = np.array([1.0, 2.0])
    solver = DistributedCG(lambda value: diagonal*value, np.ones(2), is_master=True)

    solver.step()

    # After one step r = [1/3, -1/3], so (r.r)/(r0.r0) = (2/9)/2 = 1/9.
    assert solver.err == pytest.approx(1.0/9.0)


def test_with_a_starting_guess_the_error_is_relative_to_the_right_hand_side() -> None:
    diagonal = np.array([1.0, 2.0, 3.0])
    b = np.ones(3)
    solver = DistributedCG(lambda value: diagonal*value, b, is_master=True,
                           x0=np.array([1.0, 0.0, 0.0]))

    solver.step()

    # The first residual is r0 = b - A x0 = [0, 1, 1]. One step gives alpha = (r0.r0)/(r0.A r0)
    # = 2/5 and r = [0, 0.2, -0.2], so (r.r)/(b.b) = 0.08/3 (and not 0.08/2, relative to r0).
    assert solver.err == pytest.approx(0.08/3.0)
    np.testing.assert_array_equal(b, np.ones(3))  # `b` is not changed without `destroy_b`.
    np.testing.assert_allclose(solver.x, [1.0, 0.4, 0.4])  # x0 + alpha*r0
