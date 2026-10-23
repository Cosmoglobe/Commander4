"""Tests for detector-scan data selection (the accept-flag machinery in tod/data_selection.py).

Covers the chi-squared z-score judged by the in-loop vetoes and the per-band summary logging (run
single-rank on ``MPI.COMM_SELF`` with a minimal TODSamples stand-in). Parameter gating is tested in
``test_tod_step_schema.py``.
"""

import logging
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI

from commander4.tod.data_selection import DataSelection, masked_chisq_z
from tod_step_fakes import make_step


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
    """Minimal TODSamples stand-in with the attributes `DataSelection.log_summary` touches."""
    nscans, ndet = chisq_z.shape
    selection = make_step(DataSelection, nscans, ndet, settings={"data_selection": {
        "enabled": True, "chisq_abs_threshold": 1.0e4, "min_good_fraction": 0.1}})
    selection.chisq_z, selection.good_fraction = chisq_z, good_fraction
    return SimpleNamespace(band_comm=MPI.COMM_SELF, selection=selection, chain=1,
                           experiment_data=SimpleNamespace(nscans=nscans, band_name="test-band"),
                           present=np.ones((nscans, ndet), dtype=bool))


def test_log_summary_counts_veto_rejections(caplog):
    rng = np.random.default_rng(3)
    nscans, ndet = 200, 2
    z = rng.normal(0.0, 30.0, (nscans, ndet))  # SO-like healthy bulk
    gf = np.full((nscans, ndet), 0.95)
    z[0, 0] = 5e5                      # catastrophic chisq veto
    gf[1, 0], z[1, 0] = 0.02, np.nan   # good-fraction veto (chisq_z never computed)
    z[2, 1] = 130.0                    # marginal noise-model mismatch: not counted as rejected
    z[3, 0] = gf[3, 0] = np.nan        # rejected in an earlier iteration: stale, not re-counted
    stub = _stub_samples(z, gf)
    stub.selection.accept[3, 0] = False

    with caplog.at_level(logging.INFO, logger="commander4.tod.data_selection"):
        stub.selection.log_summary(stub, active=True, iteration=2)
    assert "rejected 2 detector-scans" in caplog.text
    assert "low-good-fraction: 1" in caplog.text
    assert "|chisq_z| > 1e+04: 1" in caplog.text
    # Reporting only: the vetoes in the mapmaking loop own accept, the summary never touches it.
    assert stub.selection.accept[0, 0] and stub.selection.accept[1, 0]


class _VetoView:
    """Stand-in for a TODView focused on detector-scan (1, 0), with what the vetoes read."""
    iscan, idet, sigma0 = 1, 0, 1.0

    def get_mask(self, **kwargs):
        return np.ones(1000, dtype=bool)


def test_vetoes_record_their_diagnostic_and_reject_only_when_active():
    """Both vetoes always record what they judge; they reject only while the cuts are active."""
    settings = {"data_selection": {"min_good_fraction": 0.5, "chisq_abs_threshold": 10.0}}
    noisy = np.random.default_rng(4).normal(0.0, 3.0, 1000)  # sigma0 = 1 is far too small
    for active in (False, True):
        selection = make_step(DataSelection, 2, 1, settings=settings)
        assert not selection.veto_low_good_fraction(_VetoView(), np.ones(10, dtype=bool), active)
        assert selection.good_fraction[1, 0] == 1.0
        rejected = selection.veto_low_good_fraction(_VetoView(), np.arange(10) < 2, active)
        assert rejected == active and selection.accept[1, 0] == (not active)
        assert selection.good_fraction[1, 0] == 0.2

        selection = make_step(DataSelection, 2, 1, settings=settings)
        rejected = selection.veto_high_chisq(_VetoView(), noisy, active)
        assert rejected == active and selection.accept[1, 0] == (not active)
        assert selection.chisq_z[1, 0] == masked_chisq_z(noisy, np.ones(1000, dtype=bool), 1.0)


def test_start_iteration_forgets_the_diagnostics_but_not_the_rejections():
    """Rejection is sticky within a chain; the diagnostics are rebuilt every iteration."""
    selection = make_step(DataSelection, 2, 1)
    selection.chisq_z[:] = 1.0
    selection.good_fraction[:] = 1.0
    selection.accept[0, 0] = False
    selection.start_iteration()
    assert np.isnan(selection.chisq_z).all() and np.isnan(selection.good_fraction).all()
    np.testing.assert_array_equal(selection.accept[:, 0], [False, True])
