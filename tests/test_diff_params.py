"""Tests for the `c4-diff-params` structural parameter-file diff tool.

The tool parses two Commander4 parameter sources (chain .h5 files or raw .yml files) and reports
their differences leaf-by-leaf. These tests cover the tree-diff semantics and both loader paths;
they only pull in h5py/yaml, so no compiled backend is needed.
"""

import h5py
import yaml

from commander4.standalone_tools.diff_params import diff_params, load_params, _fmt


def _markers(diffs, marker):
    """The set of dotted-ish path tuples carrying a given diff marker."""
    return {tuple(path) for m, path, _ in diffs if m == marker}


def test_changed_added_removed_scalars():
    a = {"general": {"keep": 1, "changed": 10, "only_a": True}}
    b = {"general": {"keep": 1, "changed": 20, "only_b": "x"}}
    diffs = diff_params(a, b)
    assert ("~", ["general", "changed"], "10 -> 20") in diffs
    assert _markers(diffs, "-") == {("general", "only_a")}
    assert _markers(diffs, "+") == {("general", "only_b")}
    # Unchanged leaves are never reported.
    assert all(path != ["general", "keep"] for _, path, _ in diffs)


def test_removed_subtree_expands_to_leaves():
    a = {"blk": {"x": 1, "y": 2}}
    b = {}
    diffs = diff_params(a, b)
    assert _markers(diffs, "-") == {("blk", "x"), ("blk", "y")}
    assert not _markers(diffs, "+")


def test_list_elementwise_and_length_changes():
    a = {"v": [1, 2, 3]}
    b = {"v": [1, 9]}
    diffs = diff_params(a, b)
    assert ("~", ["v", "[1]"], "2 -> 9") in diffs
    assert _markers(diffs, "-") == {("v", "[2]")}  # extra trailing element in a
    assert not _markers(diffs, "+")


def test_scalar_replaced_by_list_is_structural_mismatch():
    # A scalar becoming a list (as with nthreads_compsep) shows the old scalar and the new leaves.
    diffs = diff_params({"n": 128}, {"n": [384, 4]})
    assert ("-", ["n"], "128") in diffs
    assert ("+", ["n", "[0]"], "384") in diffs
    assert ("+", ["n", "[1]"], "4") in diffs


def test_identical_trees_have_no_diffs():
    tree = {"a": {"b": [1, {"c": True}]}, "d": None}
    assert diff_params(tree, tree) == []


def test_fmt_renders_yaml_flavoured_scalars():
    assert _fmt(True) == "true" and _fmt(False) == "false"
    assert _fmt(None) == "null"
    assert _fmt({}) == "{}" and _fmt([]) == "[]"
    assert _fmt(3e-05) == "3e-05"


def test_load_params_from_yaml_file(tmp_path):
    p = tmp_path / "params.yml"
    p.write_text("general:\n  nside: 512\n  flag: true\n")
    assert load_params(str(p)) == {"general": {"nside": 512, "flag": True}}


def test_load_params_resolves_inc_directives(tmp_path):
    (tmp_path / "sub.yml").write_text("det: [a, b]\n")
    main = tmp_path / "params.yml"
    main.write_text("bands: !inc sub.yml\n")
    assert load_params(str(main)) == {"bands": {"det": ["a", "b"]}}


def test_load_params_from_h5(tmp_path):
    src = {"general": {"nside": 256}}
    h5_path = tmp_path / "chain.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("metadata/parameter_file_as_string", data=yaml.dump(src))
    assert load_params(str(h5_path)) == src


def test_load_params_h5_without_dataset_raises(tmp_path):
    h5_path = tmp_path / "bad.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("something_else", data=1)
    try:
        load_params(str(h5_path))
    except ValueError as err:
        assert "parameter_file_as_string" in str(err)
    else:
        raise AssertionError("expected ValueError for missing parameter dataset")
