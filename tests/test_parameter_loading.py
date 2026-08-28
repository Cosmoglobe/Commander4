"""Tests for pure, reusable Commander4 parameter loading."""

import sys
from pathlib import Path

import pytest
import yaml

from commander4.parameters.parse import load_params, params_from_dict

sys.path.insert(0, str(Path(__file__).parent.parent / "sims"))


def test_load_params_resolves_includes_relative_to_their_own_main_file(tmp_path) -> None:
    """Sequential loads must resolve identical include names relative to their own main file."""
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    (first_dir / "included.yml").write_text("name: first\n", encoding="utf-8")
    (second_dir / "included.yml").write_text("name: second\n", encoding="utf-8")
    (first_dir / "params.yml").write_text(
        "included:\n  !import included.yml\n", encoding="utf-8",
    )
    (second_dir / "params.yml").write_text(
        "included:\n  !import included.yml\n", encoding="utf-8",
    )

    first_params, first_dict, first_yaml = load_params(str(first_dir / "params.yml"))
    second_params, second_dict, second_yaml = load_params(str(second_dir / "params.yml"))

    assert first_params.included.name == "first"
    assert second_params.included.name == "second"
    assert yaml.safe_load(first_yaml) == first_dict
    assert yaml.safe_load(second_yaml) == second_dict


def test_import_adds_its_keys_to_the_surrounding_mapping(tmp_path) -> None:
    """An !import among the entries of a mapping must merge into that mapping, not replace it."""
    (tmp_path / "wmap.yml").write_text("WMAPKa:\n  freq: 33.0\n", encoding="utf-8")
    (tmp_path / "lfi.yml").write_text("# LFI\nLFI30:\n  freq: 28.4\n\n", encoding="utf-8")
    (tmp_path / "params.yml").write_text(
        "bands:\n"
        "  Haslam:\n"
        "    freq: 0.408\n"
        "  !import wmap.yml\n"
        "  !import 'lfi.yml'  # quoted, with a trailing comment\n",
        encoding="utf-8",
    )

    _, params_dict, _ = load_params(str(tmp_path / "params.yml"))

    assert params_dict == {"bands": {"Haslam": {"freq": 0.408}, "WMAPKa": {"freq": 33.0},
                                     "LFI30": {"freq": 28.4}}}


def test_nested_imports_resolve_relative_to_their_own_file(tmp_path) -> None:
    """A file pulled in by !import may itself !import, using paths relative to its own directory."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "detectors.yml").write_text("det_1:\n  gain: 1.0\n", encoding="utf-8")
    (tmp_path / "sub" / "band.yml").write_text(
        "freq: 30.0\ndetectors:\n  !import detectors.yml\n", encoding="utf-8",
    )
    (tmp_path / "params.yml").write_text("band:\n  !import sub/band.yml\n", encoding="utf-8")

    _, params_dict, _ = load_params(str(tmp_path / "params.yml"))

    assert params_dict == {"band": {"freq": 30.0, "detectors": {"det_1": {"gain": 1.0}}}}


def test_load_params_rejects_inline_import_and_import_cycles(tmp_path) -> None:
    (tmp_path / "sub.yml").write_text("name: sub\n", encoding="utf-8")
    inline = tmp_path / "inline.yml"
    inline.write_text("included: !import sub.yml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="alone on its own line"):
        load_params(str(inline))

    old_syntax = tmp_path / "old.yml"
    old_syntax.write_text("included: !inc sub.yml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="renamed to '!import'"):
        load_params(str(old_syntax))

    (tmp_path / "a.yml").write_text("a:\n  !import b.yml\n", encoding="utf-8")
    (tmp_path / "b.yml").write_text("b:\n  !import a.yml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Circular !import chain"):
        load_params(str(tmp_path / "a.yml"))


def test_params_from_dict_preserves_block_names_and_metadata() -> None:
    params_dict = {"experiments": {"Example": {"enabled": True}}}

    params = params_from_dict(params_dict)

    assert params.experiments.Example._name == "Example"
    assert yaml.safe_load(params.parameter_file_as_string) == params_dict


def test_load_params_rejects_missing_file_and_non_mapping_root(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Could not find parameter file"):
        load_params(str(tmp_path / "missing.yml"))

    parameter_file = tmp_path / "list.yml"
    parameter_file.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping at its root"):
        load_params(str(parameter_file))


@pytest.mark.parametrize("option", ["--parameter-file", "--parameter_file"])
def test_commander_command_line_accepts_both_parameter_option_spellings(
    monkeypatch, option: str,
) -> None:
    from commander4.cli import parse_command_line

    monkeypatch.setattr(sys, "argv", ["commander4", option, "params.yml"])

    assert parse_command_line() == "params.yml"


def test_simgen_uses_the_shared_loader_without_rewriting_component_fields(tmp_path) -> None:
    from simgen.config import load_params as load_simgen_params

    parameter_file = tmp_path / "sim.yml"
    parameter_file.write_text(
        "components:\n  Dust:\n    params:\n      nu0: 353\n", encoding="utf-8",
    )

    params, params_dict = load_simgen_params(str(parameter_file))

    assert params.components.Dust.params.nu0 == 353
    assert "nu_ref" not in params.components.Dust.params
    assert "nu_ref" not in params_dict["components"]["Dust"]["params"]
