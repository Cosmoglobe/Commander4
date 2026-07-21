# This program can be run as `c4-diff-params <file1> <file2>`, as long as Commander4 is installed.
#
# It performs a *structural* diff of the Commander4 parameter files of two runs: both parameter
# files are parsed as YAML and their trees compared leaf-by-leaf, so reordering and formatting
# noise is ignored and only genuine parameter changes are reported. Each argument is either a chain
# output .h5 file (parameters read from metadata/parameter_file_as_string) or a raw .yml/.yaml
# parameter file; the two kinds may be mixed freely.

import argparse
import os
import sys

import h5py
import yaml
import yaml_include

PARAM_DATASET = "metadata/parameter_file_as_string"

# ANSI colours keyed by diff marker: '-' only in file1, '+' only in file2, '~' changed value.
COLOURS = {"-": "\033[31m", "+": "\033[32m", "~": "\033[33m", "reset": "\033[0m"}


def _decode(value) -> str:
    """Decode a scalar HDF5 string value (bytes / np.bytes_ / 0-d array) to a Python str."""
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        value = value.item()
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_params(path: str) -> dict:
    """Load a Commander4 parameter dict from a chain .h5 file or a raw YAML file.

    HDF5 inputs are detected by content (not extension) and read from the already fully-resolved
    `metadata/parameter_file_as_string` dataset. Any other file is read as a raw YAML parameter
    file, with `!inc` include directives resolved relative to its directory exactly as Commander4's
    own parser does.
    """
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as f:
            if PARAM_DATASET not in f:
                raise ValueError(f"'{path}' is an HDF5 file but has no '{PARAM_DATASET}' dataset.")
            return yaml.full_load(_decode(f[PARAM_DATASET][()])) or {}
    base_dir = os.path.dirname(os.path.abspath(path))
    yaml.add_constructor("!inc", yaml_include.Constructor(base_dir=base_dir))
    with open(path) as f:
        return yaml.full_load(f.read()) or {}


def _fmt(value) -> str:
    """Render a leaf value in a compact, YAML-flavoured way (true/false/null, {}/[] for empties)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "{}"
    if isinstance(value, list):
        return "[]"
    return str(value)


def _join_path(parts: list) -> str:
    """Join path parts into a dotted path, attaching list-index parts ('[3]') without a dot."""
    out = ""
    for part in parts:
        part = str(part)
        out += part if (not out or part.startswith("[")) else f".{part}"
    return out


def _emit(value, path: list, marker: str, out: list) -> None:
    """Append one (marker, path, text) entry per leaf under `value`, expanding containers."""
    if isinstance(value, dict) and value:
        for key in value:
            _emit(value[key], path + [key], marker, out)
    elif isinstance(value, list) and value:
        for i, item in enumerate(value):
            _emit(item, path + [f"[{i}]"], marker, out)
    else:
        out.append((marker, path, _fmt(value)))


def diff_params(a, b, path: list | None = None, out: list | None = None) -> list:
    """Recursively diff two parsed parameter trees, returning a list of (marker, path, text).

    Dicts and lists are recursed structurally; scalar mismatches become a single '~' change entry
    'old -> new'. A subtree present on only one side (or replaced by a value of another kind) is
    expanded into per-leaf '-' (file1) or '+' (file2) entries.
    """
    path = path if path is not None else []
    out = out if out is not None else []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in a:
            if key in b:
                diff_params(a[key], b[key], path + [key], out)
            else:
                _emit(a[key], path + [key], "-", out)
        for key in b:
            if key not in a:
                _emit(b[key], path + [key], "+", out)
    elif isinstance(a, list) and isinstance(b, list):
        common = min(len(a), len(b))
        for i in range(common):
            diff_params(a[i], b[i], path + [f"[{i}]"], out)
        for i in range(common, len(a)):
            _emit(a[i], path + [f"[{i}]"], "-", out)
        for i in range(common, len(b)):
            _emit(b[i], path + [f"[{i}]"], "+", out)
    elif isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        # Structural mismatch (e.g. a subtree replaced by a scalar): show old vs new leaves.
        _emit(a, path, "-", out)
        _emit(b, path, "+", out)
    elif a != b:
        out.append(("~", path, f"{_fmt(a)} -> {_fmt(b)}"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="c4-diff-params",
        description="Structurally diff the Commander4 parameter files of two runs.",
        epilog="Each FILE is either a chain output .h5 (parameters read from "
        f"{PARAM_DATASET}) or a raw .yml/.yaml parameter file; the two kinds may be mixed. "
        "Exit status: 0 if identical, 1 if differences were found, 2 on error.",
    )
    parser.add_argument("file1", help="First parameter source (.h5 chain file or .yml file).")
    parser.add_argument("file2", help="Second parameter source (.h5 chain file or .yml file).")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colour output.")
    args = parser.parse_args()

    try:
        d1 = load_params(args.file1)
        d2 = load_params(args.file2)
    except (OSError, ValueError, yaml.YAMLError) as err:
        print(f"c4-diff-params: error: {err}", file=sys.stderr)
        return 2

    diffs = diff_params(d1, d2)
    colour = sys.stdout.isatty() and not args.no_color
    c = COLOURS if colour else {k: "" for k in COLOURS}

    print(f"{c['-']}--- {args.file1}{c['reset']}")
    print(f"{c['+']}+++ {args.file2}{c['reset']}")
    if not diffs:
        print("Parameter files are identical.")
        return 0

    for marker, path, text in diffs:
        print(f"{c[marker]}{marker} {_join_path(path)}: {text}{c['reset']}")
    print(f"\n{len(diffs)} difference{'s' if len(diffs) != 1 else ''}.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
