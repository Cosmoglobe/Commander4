import numpy as np
import pytest

from commander4.compsep.cg_driver import DistributedCGArray


def test_cg_reports_the_squared_relative_preconditioned_residual() -> None:
    diagonal = np.array([1.0, 2.0])
    solver = DistributedCGArray(lambda value: diagonal*value, np.ones(2), is_master=True)

    solver.step()

    # After one step r = [1/3, -1/3], so (r.r)/(r0.r0) = (2/9)/2 = 1/9.
    assert solver.err == pytest.approx(1.0/9.0)
