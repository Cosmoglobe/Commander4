"""Conversions between Commander4's internal uK_RJ convention and a band's external `band_unit`.

C4 works internally in Rayleigh-Jeans brightness (uK_RJ); a band may quote its gain and maps in
another thermodynamic unit (e.g. uK_CMB) via `band_unit`. `rj_to_band_unit_factor` gives the single
factor D relating the two, evaluated at band centre with a delta bandpass.
"""

import functools
import pysm3.units as pysm3_u


# Accepted `band_unit` values; uK_RJ is the internal convention (identity), the rest go through
# pysm3's CMB<->RJ equivalency at band centre.
SUPPORTED_BAND_UNITS = ("uK_RJ", "mK_RJ", "K_RJ", "uK_CMB", "mK_CMB", "K_CMB", "MJy/sr")


@functools.lru_cache(maxsize=None)
def rj_to_band_unit_factor(nu_GHz: float, band_unit: str | None) -> float:
    """Factor D such that value[band_unit] = D * value[uK_RJ] at frequency `nu_GHz`.

    Brightness maps multiply by D; a gain (brightness in its denominator) divides by D. `None` or
    "uK_RJ" returns 1.0. Cached per (nu_GHz, band_unit).
    """
    if band_unit is None or band_unit == "uK_RJ":
        return 1.0
    if band_unit not in SUPPORTED_BAND_UNITS:
        raise ValueError(f"Unsupported band_unit {band_unit!r}; expected one of {SUPPORTED_BAND_UNITS}.")
    try:
        return float((1.0 * pysm3_u.uK_RJ).to(
            pysm3_u.Unit(band_unit),
            equivalencies=pysm3_u.cmb_equivalencies(nu_GHz * pysm3_u.GHz)).value)
    except Exception as exc:
        raise ValueError(f"Could not convert uK_RJ to {band_unit!r} at {nu_GHz} GHz: {exc}") from exc
