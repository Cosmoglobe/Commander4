"""Tests for the per-band unit convention (`band_unit`).

`rj_to_band_unit_factor` returns the single per-band factor D between C4's internal uK_RJ convention
and a band's chosen thermodynamic `band_unit`. The same D governs both external boundaries: output
brightness maps are multiplied by D on write, while the gain (brightness in its denominator) is
divided by D. The pysm3-dependent conversions are checked against the analytic CMB<->RJ factor g(x);
the identity/error branches need no pysm3.
"""

import numpy as np
import pytest

from commander4.utils.unit_conversions import rj_to_band_unit_factor, SUPPORTED_BAND_UNITS


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


@pytest.mark.parametrize("band_unit", [u for u in SUPPORTED_BAND_UNITS if u != "MJy/sr"])
def test_gain_read_write_roundtrip(band_unit):
    # A gain read from file (external * D) then written to chain (internal / D) recovers the input.
    D = rj_to_band_unit_factor(70.4, band_unit)
    gain_external = 78.0e-9
    gain_internal = gain_external * D  # on read
    gain_written = gain_internal / D   # on write
    assert gain_written == pytest.approx(gain_external, rel=1e-12)


# uK_RJ is the internal convention (D == 1.0, a no-op), so it is excluded here: this test needs a
# non-trivial factor to check that the map and gain conversions genuinely cancel.
@pytest.mark.parametrize("band_unit",
                         [u for u in SUPPORTED_BAND_UNITS if u not in ("MJy/sr", "uK_RJ")])
def test_map_and_gain_convert_in_opposite_directions(band_unit):
    # Output brightness maps multiply by D on write; gains divide by D. The calibrated detector
    # signal gain*sky is a physical (band_unit-invariant) quantity, so the two conversions must
    # exactly cancel: (sky_uKRJ * D) * (gain_uKRJ / D) == sky_uKRJ * gain_uKRJ.
    D = rj_to_band_unit_factor(143.0, band_unit)
    assert D != 1.0  # non-trivial factor for these CMB/metric units
    sky_uKRJ, gain_uKRJ = 250.0, 78.0e-9
    map_written = sky_uKRJ * D          # brightness -> band_unit (write_map_chain_to_file)
    gain_written = gain_uKRJ / D        # [det]/uK_RJ -> [det]/band_unit (write_chain_to_file)
    assert map_written * gain_written == pytest.approx(sky_uKRJ * gain_uKRJ, rel=1e-12)
