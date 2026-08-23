"""Tests for pure, reusable Commander4 parameter loading."""

import sys
from pathlib import Path

import pytest
import yaml

from commander4.parameters.parse import load_params, params_from_dict

sys.path.insert(0, str(Path(__file__).parent.parent / "sims"))


def test_load_params_resolves_includes_without_global_loader_state(tmp_path) -> None:
    """Sequential loads must resolve identical include names relative to their own main file."""
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    (first_dir / "included.yml").write_text("name: first\n", encoding="utf-8")
    (second_dir / "included.yml").write_text("name: second\n", encoding="utf-8")
    (first_dir / "params.yml").write_text(
        "included: !inc included.yml\n", encoding="utf-8",
    )
    (second_dir / "params.yml").write_text(
        "included: !inc included.yml\n", encoding="utf-8",
    )

    first_params, first_dict, first_yaml = load_params(str(first_dir / "params.yml"))
    second_params, second_dict, second_yaml = load_params(str(second_dir / "params.yml"))

    assert first_params.included.name == "first"
    assert second_params.included.name == "second"
    assert yaml.safe_load(first_yaml) == first_dict
    assert yaml.safe_load(second_yaml) == second_dict


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
