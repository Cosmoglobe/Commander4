"""How a band's polarization is named and split into the views the samplers actually run on.

A band defines `I`, `QU` or `IQU`. Commander4 executes one view at a time, so a logical band splits
into execution views with stable identifiers (`030GHz_I`, `030GHz_QU`), which the MPI routing, the
parameter lookup and the component views all key on. Both halves of that vocabulary live here: the
polarization strings themselves, and the identifiers built from them.
"""
import logging

from commander4.diagnostics import log

logger = logging.getLogger(__name__)

POLS_DICT = {"I": 1, "QU": 2, "IQU": 3}  # more allowed in the future.

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


def get_npol(pols:str):
    """
    Return the number of map polarizaiton components given the polarization string `pols`.
    """
    log.logassert(pols in POLS_DICT, "Unrecognised polarization string", logger)
    return POLS_DICT[pols]
    
def is_pol_supported(pols:str):
    """
    Checks if the given polarization string `pols` is matching one of the supported pol configs.
    """
    if pols in POLS_DICT.keys():
        return True
    else:
        return False

def assert_pol_supported(pols:str):
    """
    Asserts if the given polarization string `pols` is matching one of the supported pol configs.
    """
    log.logassert(is_pol_supported(pols), 
                  f"Unsupported polarization string {pols}", 
                  logging.getLogger(__name__))
