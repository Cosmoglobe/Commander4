"""Small simgen parameter helpers shared with Commander4."""

from pixell.bunch import Bunch

from commander4.parameters.bunch import as_bunch_recursive
from commander4.parameters.parse import load_params as load_commander_params


def bget(bunch, key: str, default=None):
    """``dict.get`` for a ``Bunch``, a dictionary, or ``None``."""
    if bunch is None:
        return default
    return bunch[key] if key in bunch else default


def load_params(parameter_file: str) -> tuple[Bunch, dict]:
    """Load simgen parameters with Commander4's normal YAML and ``!import`` handling."""
    params, params_dict, _ = load_commander_params(parameter_file)
    return params, params_dict


__all__ = ["as_bunch_recursive", "bget", "load_params"]
