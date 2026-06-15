"""Helpers for stable execution-view identifiers used in TOD and CompSep routing."""


EXECUTION_POLS = {
    "I": ("I",),
    "QU": ("QU",),
    "IQU": ("I", "QU"),
}


def get_execution_band_id(base_band_id: str, eval_pol: str) -> str:
    """Return the execution-view identifier for one band and one evaluation polarization."""
    if eval_pol not in ("I", "QU"):
        raise ValueError(f"Unsupported execution polarization {eval_pol!r}.")
    return f"{base_band_id}_{eval_pol}"


def get_execution_band_ids(base_band_id: str, defined_pol: str) -> tuple[str, ...]:
    """Return all execution-view identifiers represented by one logical band."""
    if defined_pol not in EXECUTION_POLS:
        raise ValueError(f"Unsupported polarization {defined_pol!r}.")
    return tuple(get_execution_band_id(base_band_id, eval_pol)
                 for eval_pol in EXECUTION_POLS[defined_pol])