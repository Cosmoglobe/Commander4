"""Tests for detector-scan data selection (the accept-flag machinery in tod/data_selection.py).

Covers the chi-squared z-score judged by the in-loop vetoes and the per-band summary logging (run
single-rank on ``MPI.COMM_SELF`` with a minimal TODSamples stand-in). Parameter gating is tested in
``test_tod_step_schema.py``.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.tod.data_selection import masked_chisq_z, log_dataselect_summary


def test_chisq_z_white_noise_is_standard_normal():
    # Clean white noise with the correct sigma0: z ~ N(0,1), so |z| < 5 is essentially certain.
    rng = np.random.default_rng(0)
    n = 100_000
    residual = rng.normal(0.0, 2.5, n).astype(np.float32)
    mask = np.ones(n, dtype=bool)
    z = masked_chisq_z(residual, mask, 2.5)
    assert abs(z) < 5.0


def test_chisq_z_detects_jump_and_wrong_sigma0():
    rng = np.random.default_rng(1)
    n = 100_000
    residual = rng.normal(0.0, 1.0, n).astype(np.float32)
    mask = np.ones(n, dtype=bool)
    # An uncorrected jump: half the scan offset by 10 sigma -> enormous positive z.
    jumped = residual.copy()
    jumped[n // 2:] += 10.0
    assert masked_chisq_z(jumped, mask, 1.0) > 1_000.0
    # sigma0 overestimated by 2x: chisq far below expectation -> strongly negative z.
    assert masked_chisq_z(residual, mask, 2.0) < -100.0


def test_chisq_z_undefined_cases_are_nan():
    residual = np.ones(100, dtype=np.float32)
    mask = np.ones(100, dtype=bool)
    assert np.isnan(masked_chisq_z(residual, np.zeros(100, dtype=bool), 1.0))  # no good samples
    assert np.isnan(masked_chisq_z(residual, mask, 0.0))     # non-positive sigma0
    assert np.isnan(masked_chisq_z(residual, mask, np.nan))  # non-finite sigma0


# Configuration and iteration gating are tested separately in test_tod_step_schema.py.


def _stub_samples(chisq_z, good_fraction):
    """Minimal TODSamples stand-in with the attributes log_dataselect_summary touches."""
    nscans, ndet = chisq_z.shape
    return SimpleNamespace(nscans=nscans, ndet=ndet, chisq_z=chisq_z,
                           good_fraction=good_fraction,
                           present=np.ones((nscans, ndet), dtype=bool),
                           accept=np.ones((nscans, ndet), dtype=bool), band_name="test-band",
                           chain=1)


def _cfg(**overrides):
    cfg = dict(enabled=True, chisq_abs_threshold=1.0e4, min_good_fraction=0.1)
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def test_log_dataselect_summary_counts_veto_rejections(caplog):
    rng = np.random.default_rng(3)
    nscans, ndet = 200, 2
    z = rng.normal(0.0, 30.0, (nscans, ndet))  # SO-like healthy bulk
    gf = np.full((nscans, ndet), 0.95)
    z[0, 0] = 5e5                      # catastrophic chisq veto
    gf[1, 0], z[1, 0] = 0.02, np.nan   # good-fraction veto (chisq_z never computed)
    z[2, 1] = 130.0                    # marginal noise-model mismatch: not counted as rejected
    z[3, 0] = gf[3, 0] = np.nan        # rejected in an earlier iteration: stale, not re-counted
    stub = _stub_samples(z, gf)
    stub.accept[3, 0] = False

    with caplog.at_level(logging.INFO, logger="commander4.tod.data_selection"):
        log_dataselect_summary(MPI.COMM_SELF, stub, _cfg(), active=True, iteration=2)
    assert "rejected 2 detector-scans" in caplog.text
    assert "low-good-fraction: 1" in caplog.text
    assert "|chisq_z| > 1e+04: 1" in caplog.text
    # Reporting only: the vetoes in the mapmaking loop own accept, the summary never touches it.
    assert stub.accept[0, 0] and stub.accept[1, 0]


def test_log_dataselect_summary_inactive_counts_nothing(caplog):
    # Cuts gated off this iteration (before from_iter_num / past until_iter_num): report-only.
    z = np.full((20, 1), 5e5)
    stub = _stub_samples(z, np.full((20, 1), 0.95))
    with caplog.at_level(logging.INFO, logger="commander4.tod.data_selection"):
        log_dataselect_summary(MPI.COMM_SELF, stub, _cfg(), active=False, iteration=2)
    assert "rejected 0 detector-scans" in caplog.text
