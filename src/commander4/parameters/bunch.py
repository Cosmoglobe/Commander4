"""Turning a parsed parameter file (nested dicts) into the nested Bunch the code reads.

This lives apart from `parse.py` so tools that already have a resolved dictionary can reuse the
conversion without coupling themselves to YAML file loading.
"""
from pixell.bunch import Bunch


def as_bunch_recursive(dict_of_dicts: dict, name: str | None = None) -> Bunch:
    """Recursively convert nested dicts into nested `Bunch` objects.

    Each nested block also gets a `_name` attribute holding its own key, which is how components
    and bands know what they are called in the parameter file.

    Args:
        dict_of_dicts: The parsed parameter file, or any nested dict.
        name: The key this block was stored under, injected as `_name`.
    """
    res = Bunch()

    # Injected past Bunch's own data dict, so it is an attribute rather than a parameter.
    if name is not None:
        object.__setattr__(res, "_name", name)

    for key, val in dict_of_dicts.items():
        if isinstance(val, dict):
            res[key] = as_bunch_recursive(val, name=key)
        else:
            res[key] = val

    return res
