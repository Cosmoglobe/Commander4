"""Parameter-file loading for simgen.

The simulator reuses Commander4's parameter conventions: YAML with ``!inc`` includes (via
``yaml_include``) and a nested structure converted into ``pixell.bunch.Bunch`` objects whose nested
blocks carry a ``_name`` attribute (the dict key). This mirrors ``commander4.parse_params`` so the
same ``components:`` block and ``experiments -> bands -> detectors`` layout can be shared between a
Commander4 run and a simgen run.
"""
import os
import yaml
import yaml_include
from pixell.bunch import Bunch


def bget(bunch, key: str, default=None):
    """``dict.get`` for a ``pixell.bunch.Bunch`` (which lacks ``.get``), a dict, or ``None``.

    Bunch supports ``key in bunch`` and ``bunch[key]`` but not ``.get``/``.setdefault``; this helper
    gives the missing-key-with-default behaviour the config code relies on.
    """
    if bunch is None:
        return default
    return bunch[key] if key in bunch else default


def as_bunch_recursive(dict_of_dicts: dict, name: str | None = None) -> Bunch:
    """Recursively convert a (possibly nested) dict into a Bunch, tagging each block with ``_name``.

    The ``_name`` attribute matches Commander4's parser and is what the component classes in
    ``commander4.sky_models.component`` read to identify a component (``comp_params._name``).
    """
    res = Bunch()
    if name is not None:
        object.__setattr__(res, "_name", name)
    for key, val in dict_of_dicts.items():
        res[key] = as_bunch_recursive(val, name=key) if isinstance(val, dict) else val
    return res


# Components in the C4 code read `nu_ref`, but some C4 param files (e.g. the LiteBIRD ones) spell the
# reference frequency `nu0`. Accept either by copying `nu0` -> `nu_ref` when only the former is given.
def _normalize_component_params(params_dict: dict) -> None:
    for comp in params_dict.get("components", {}).values():
        cpars = comp.get("params", {}) if isinstance(comp, dict) else {}
        if "nu0" in cpars and "nu_ref" not in cpars:
            cpars["nu_ref"] = cpars["nu0"]


def load_params(parameter_file: str) -> tuple[Bunch, dict]:
    """Load a simgen YAML parameter file, returning ``(params_bunch, params_dict)``.

    ``!inc <relative_path>`` includes are resolved relative to the parameter file's directory, as in
    ``commander4.parse_params``.
    """
    if not os.path.isfile(parameter_file):
        raise FileNotFoundError(f"Could not find parameter file {parameter_file}")
    param_file_dir = os.path.dirname(os.path.abspath(parameter_file))
    # Register the include constructor on the loader used by `yaml.full_load` below.
    yaml.add_constructor("!inc", yaml_include.Constructor(base_dir=param_file_dir))
    with open(parameter_file, "r") as f:
        params_dict = yaml.full_load(f.read())
    _normalize_component_params(params_dict)
    return as_bunch_recursive(params_dict), params_dict
