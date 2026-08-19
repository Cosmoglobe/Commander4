"""Focused tests for the analytic satellite pointing strategy."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "aux"))

from simgen.config import as_bunch_recursive
from simgen.pointing import PlanckScan


def test_planck_scan_anti_sun_period_accelerates_sweep():
    """A half accelerated anti-Sun period reverses the orbital-direction vector."""
    pytest.importorskip("astropy")
    strategy = PlanckScan(as_bunch_recursive({"anti_sun_period_days": 2.0}), fsamp=1.0)

    start = strategy.compute(sample_offset=0, ntod=2).vsun
    half_period = strategy.compute(sample_offset=24 * 3600, ntod=2).vsun
    cosine = np.dot(start, half_period) / (np.linalg.norm(start) * np.linalg.norm(half_period))

    assert cosine < -0.99


def test_planck_scan_default_anti_sun_period_is_physical_year():
    """Omitting the new setting preserves the existing 365.25-day sweep."""
    pytest.importorskip("astropy")
    implicit = PlanckScan(as_bunch_recursive({}), fsamp=1.0)
    explicit = PlanckScan(as_bunch_recursive({"anti_sun_period_days": 365.25}), fsamp=1.0)

    np.testing.assert_allclose(
        implicit.compute(sample_offset=12345, ntod=2).vsun,
        explicit.compute(sample_offset=12345, ntod=2).vsun,
    )