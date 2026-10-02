"""Tests for the unit conventions: `unit_factor`, and the conversion inside the SED call.

Band quantities stay in each band's `band_unit`; component amplitudes are in the global unit at
their `nu_ref`; `get_sed(nu, unit)` is the one place that converts between the two. The
pysm3-dependent conversions are checked against the analytic CMB<->RJ factor.
"""

import numpy as np
import pytest
from pixell.bunch import Bunch

from commander4.sky.diffuse_components import CMB, ThermalDust
from commander4.units import T_CMB, check_unit_usable, unit_factor


def _analytic_rj_in_cmb(nu_GHz: float) -> float:
    """1 uK_RJ expressed in uK_CMB = 1/g(x), g(x) = x^2 e^x/(e^x-1)^2, x = h nu / (k T_cmb)."""
    h, k = 6.62607015e-34, 1.380649e-23
    x = h * nu_GHz * 1e9 / (k * T_CMB)
    g = x**2 * np.exp(x) / np.expm1(x) ** 2
    return 1.0 / g


def _cmb(nu_ref: float, global_unit: str) -> CMB:
    params = Bunch(polarization="I", lmax=4, spatially_varying_MM=False, Cl_prior_amplitude=None,
                   nu_ref=nu_ref)
    return CMB(params, Bunch(global_unit=global_unit), comp_name="CMB")


def _dust(global_unit: str) -> ThermalDust:
    params = Bunch(polarization="I", lmax=4, spatially_varying_MM=False, Cl_prior_amplitude=None,
                   nu_ref=353.0, beta=1.54, T=20.0)
    return ThermalDust(params, Bunch(global_unit=global_unit), comp_name="ThermalDust")


def test_same_unit_is_identity():
    assert unit_factor(28.4, "uK_RJ", "uK_RJ") == 1.0
    assert unit_factor(857.0, "MJy/sr", "MJy/sr") == 1.0


def test_unsupported_unit_raises():
    with pytest.raises(ValueError, match="Unsupported unit"):
        unit_factor(28.4, "uK_RJ", "bogus_unit")


@pytest.mark.parametrize("nu", [28.4, 44.1, 70.4, 143.0, 353.0, 857.0])
def test_uk_cmb_matches_analytic_a2t_at_the_shared_t_cmb(nu):
    # The orbital dipole uses T_CMB directly, so pysm3's conversions must assume the same value.
    assert unit_factor(nu, "uK_RJ", "uK_CMB") == pytest.approx(_analytic_rj_in_cmb(nu), rel=1e-9)


def test_metric_prefix_scaling():
    d_uk = unit_factor(44.1, "uK_RJ", "uK_CMB")
    assert unit_factor(44.1, "uK_RJ", "K_CMB") == pytest.approx(d_uk * 1e-6, rel=1e-12)
    assert unit_factor(44.1, "uK_RJ", "mK_CMB") == pytest.approx(d_uk * 1e-3, rel=1e-12)
    assert unit_factor(44.1, "K_CMB", "uK_CMB") == pytest.approx(1e6, rel=1e-12)


def test_thermodynamic_units_fail_loudly_in_the_far_infrared():
    # 1 uK_RJ is an infinite number of uK_CMB at 25 THz in double precision.
    with pytest.raises(ValueError, match="far-infrared"):
        unit_factor(25000.0, "uK_RJ", "uK_CMB")
    with pytest.raises(ValueError, match="Band X"):
        check_unit_usable(25000.0, "uK_CMB", "Band X")
    check_unit_usable(4996.0, "MJy/sr", "Band Y")  # MJy/sr works at every frequency.


@pytest.mark.parametrize("nu_ref", [1.0, 100.0, 217.0])
@pytest.mark.parametrize("nu", [30.0, 143.0, 353.0])
def test_cmb_mixing_is_one_between_thermodynamic_units(nu_ref, nu):
    # A CMB amplitude in uK_CMB lands in a uK_CMB band unchanged, whatever nu_ref is.
    assert _cmb(nu_ref, "uK_CMB").get_sed(nu, "uK_CMB") == pytest.approx(1.0, rel=1e-12)


def test_sed_converts_to_the_requested_band_unit():
    dust = _dust("uK_RJ")
    for nu, unit in [(30.0, "uK_CMB"), (143.0, "K_CMB"), (857.0, "MJy/sr")]:
        expected = dust.get_sed(nu, "uK_RJ")*unit_factor(nu, "uK_RJ", unit)
        assert dust.get_sed(nu, unit) == pytest.approx(expected, rel=1e-12)
    # At its own reference frequency and unit, an amplitude is the signal itself.
    assert dust.get_sed(353.0, "uK_RJ") == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("global_unit", ["uK_CMB", "MJy/sr", "K_RJ"])
def test_band_sky_does_not_depend_on_the_global_unit(global_unit):
    # The same physical sky, stored in two global units, realizes to the same band map.
    rng = np.random.default_rng(3)
    for make in (_dust, lambda unit: _cmb(1.0, unit)):
        reference, other = make("uK_RJ"), make(global_unit)
        reference.alms = (rng.normal(size=(1, 15)) + 1j*rng.normal(size=(1, 15)))
        other.alms = reference.alms*unit_factor(reference.nu_ref, "uK_RJ", global_unit)
        for nu, band_unit in [(100.0, "uK_CMB"), (545.0, "MJy/sr")]:
            np.testing.assert_allclose(other.get_sky(nu, band_unit, 4),
                                       reference.get_sky(nu, band_unit, 4), rtol=1e-10)


def test_init_map_is_converted_to_the_global_unit_at_nu_ref():
    dust = _dust("uK_CMB")
    dust.units = "uK_RJ"
    sky_map = np.ones((1, 12))
    np.testing.assert_allclose(dust.init_map_to_amplitude(sky_map),
                               sky_map*unit_factor(353.0, "uK_RJ", "uK_CMB"))
