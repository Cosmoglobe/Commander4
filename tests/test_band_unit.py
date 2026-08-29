"""Tests for the per-band unit convention (`band_unit`).

`rj_to_band_unit_factor` returns the per-band factor between C4's internal uK_RJ convention and a
band's chosen thermodynamic `band_unit`. The pysm3-dependent conversions are checked against the
analytic CMB<->RJ factor g(x); the identity and error branches need no pysm3.
"""

import numpy as np
import pytest

from commander4.units import rj_to_band_unit_factor


def _analytic_rj_in_cmb(nu_GHz: float) -> float:
    """1 uK_RJ expressed in uK_CMB = 1/g(x), g(x) = x^2 e^x/(e^x-1)^2, x = h nu / (k T_cmb)."""
    h, k, Tcmb = 6.62607015e-34, 1.380649e-23, 2.7255
    x = h * nu_GHz * 1e9 / (k * Tcmb)
    g = x**2 * np.exp(x) / np.expm1(x) ** 2
    return 1.0 / g


def test_uk_rj_and_none_are_identity():
    # The internal convention: no conversion (does not touch pysm3).
    assert rj_to_band_unit_factor(28.4, "uK_RJ") == 1.0
    assert rj_to_band_unit_factor(857.0, None) == 1.0


def test_unsupported_unit_raises():
    with pytest.raises(ValueError):
        rj_to_band_unit_factor(28.4, "bogus_unit")


@pytest.mark.parametrize("nu", [28.4, 44.1, 70.4, 143.0, 353.0])
def test_uk_cmb_matches_analytic_a2t(nu):
    # D = (1 uK_RJ in uK_CMB) should equal the standard RJ->CMB factor 1/g(x).
    assert rj_to_band_unit_factor(nu, "uK_CMB") == pytest.approx(_analytic_rj_in_cmb(nu), rel=1e-3)


def test_metric_prefix_scaling():
    # K_CMB is uK_CMB * 1e-6; mK_CMB is uK_CMB * 1e-3.
    d_uk = rj_to_band_unit_factor(44.1, "uK_CMB")
    assert rj_to_band_unit_factor(44.1, "K_CMB") == pytest.approx(d_uk * 1e-6, rel=1e-6)
    assert rj_to_band_unit_factor(44.1, "mK_CMB") == pytest.approx(d_uk * 1e-3, rel=1e-6)
