"""Unit conversions, and which unit each quantity in Commander4 is expressed in.

Commander4 follows Commander3's layout (see notes/unit_handling_transition.md):

* Band quantities stay in the band's own `band_unit`: maps, rms, residuals, gains (detector units
  per band_unit), and the sky model and orbital dipole as the TOD code sees them.
* Diffuse component amplitudes are in the run's global unit (`compsep.global_unit`, default uK_RJ),
  referenced to each component's `nu_ref`.
* `get_sed(nu, unit)` on a component is the one place an amplitude becomes a band signal.

Every conversion goes through `unit_factor`, evaluated at a single frequency (a delta bandpass).
"""

import functools

import numpy as np
import pysm3.units as pysm3_u

# CMB monopole temperature in K. Commander3's default T_CMB, and the value pysm3's CMB
# equivalencies use, so the orbital dipole and every unit conversion agree.
T_CMB = 2.7255

# Units a band, or the component amplitudes, may be expressed in.
SUPPORTED_UNITS = ("uK_RJ", "mK_RJ", "K_RJ", "uK_CMB", "mK_CMB", "K_CMB", "MJy/sr")


@functools.lru_cache(maxsize=None)
def unit_factor(nu_GHz: float, from_unit: str, to_unit: str) -> float:
    """Factor f such that value[to_unit] = f * value[from_unit], for a brightness at `nu_GHz`.

    Uses pysm3's conversions between Rayleigh-Jeans, thermodynamic (CMB) and intensity units at the
    single frequency `nu_GHz`. Cached per (nu_GHz, from_unit, to_unit), so `nu_GHz` must be a
    scalar.

    Raises:
        ValueError: For an unsupported unit, or when the factor is not finite. The latter happens
            for thermodynamic units in the far infrared: 1 uK_RJ is an infinite number of uK_CMB
            in double precision above roughly 20 THz.
    """
    for unit in (from_unit, to_unit):
        if unit not in SUPPORTED_UNITS:
            raise ValueError(f"Unsupported unit {unit!r}; expected one of {SUPPORTED_UNITS}.")
    if from_unit == to_unit:
        return 1.0
    # Far-infrared thermodynamic conversions overflow inside astropy; the check below reports it.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        factor = float((1.0*pysm3_u.Unit(from_unit)).to(
            pysm3_u.Unit(to_unit),
            equivalencies=pysm3_u.cmb_equivalencies(nu_GHz*pysm3_u.GHz)).value)
    if not np.isfinite(factor):
        raise ValueError(f"Converting {from_unit} to {to_unit} at {nu_GHz} GHz gives {factor}. "
                         "Thermodynamic (CMB) units cannot represent far-infrared brightness; "
                         "use an RJ unit or MJy/sr there.")
    return factor


def check_unit_usable(nu_GHz: float, unit: str, what: str) -> None:
    """Stop with a clear error if `unit` cannot express a brightness at `nu_GHz`.

    A unit is usable when converting uK_RJ to it gives a finite, non-zero factor. `what` names the
    band or component in the error message.
    """
    try:
        factor = unit_factor(nu_GHz, "uK_RJ", unit)
    except ValueError as exc:
        raise ValueError(f"{what}: {exc}") from exc
    if factor == 0.0:
        raise ValueError(f"{what}: 1 uK_RJ is 0 {unit} at {nu_GHz} GHz, so {unit} cannot be used "
                         "there.")
