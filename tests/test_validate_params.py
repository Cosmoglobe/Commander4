"""Tests for the MPI-free Commander4 parameter validator."""

import sys

import pytest
import yaml

from commander4.standalone_tools.validate_params import main, validate_parameter_file


def _valid_params() -> dict:
    return {
        "components": {
            "CMB": {
                "enabled": True,
                "component_class": "CMB",
                "params": {"polarization": "I"},
            },
        },
        "experiments": {
            "Example": {
                "enabled": True,
                "experiment_id": "general",
                "bands": {
                    "Band": {
                        "enabled": True,
                        "num_tasks": 2,
                        "detectors": {"detector": {}},
                    },
                },
            },
        },
        "compsep": {"enabled": False},
    }


def _write_params(tmp_path, params: dict, filename: str = "params.yml") -> str:
    parameter_file = tmp_path / filename
    parameter_file.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")
    return str(parameter_file)


def test_valid_file_reports_task_count_and_sampling_groups(tmp_path) -> None:
    params = _valid_params()
    params["compsep"] = {
        "enabled": True,
        "bands": {"Band": {"enabled": True, "polarization": "I"}},
        "cg_sampling_groups": {"amplitudes": {"enabled": True}},
    }

    task_summary, groups = validate_parameter_file(_write_params(tmp_path, params))

    assert task_summary == "3 = 2 TOD + 1 CompSep-I + 0 CompSep-QU"
    assert groups == ["cg_sampling_groups.amplitudes"]


def test_command_prints_a_submission_ready_summary(tmp_path, monkeypatch, capsys) -> None:
    parameter_file = _write_params(tmp_path, _valid_params())
    monkeypatch.setattr(sys, "argv", ["c4-validate-params", parameter_file])

    assert main() == 0

    output = capsys.readouterr().out
    assert f"Valid parameter file: {parameter_file}" in output
    assert "MPI tasks: 2 = 2 TOD + 0 CompSep-I + 0 CompSep-QU" in output
    assert "Enabled sampling groups: none" in output


def test_unknown_component_class_is_rejected(tmp_path) -> None:
    params = _valid_params()
    params["components"]["CMB"]["component_class"] = "NotAComponent"

    with pytest.raises(ValueError, match="NotAComponent"):
        validate_parameter_file(_write_params(tmp_path, params))


def test_unimplemented_component_class_is_rejected(tmp_path) -> None:
    params = _valid_params()
    params["components"]["CMB"]["component_class"] = "TemplateComponent"

    with pytest.raises(ValueError, match="not implemented"):
        validate_parameter_file(_write_params(tmp_path, params))


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("nu0", "nu_ref"),
        ("smoothing_prior_FWHM", "Cl_prior"),
    ],
)
def test_removed_component_fields_have_migration_errors(
    tmp_path, field: str, message: str,
) -> None:
    params = _valid_params()
    params["components"]["CMB"]["params"][field] = 30

    with pytest.raises(ValueError, match=message):
        validate_parameter_file(_write_params(tmp_path, params))


def test_unknown_reader_is_rejected_before_data_access(tmp_path) -> None:
    params = _valid_params()
    params["experiments"]["Example"]["experiment_id"] = "misspelled_reader"

    with pytest.raises(ValueError, match="misspelled_reader"):
        validate_parameter_file(_write_params(tmp_path, params))


def test_sampling_group_unknown_keys_are_rejected(tmp_path) -> None:
    params = _valid_params()
    params["compsep"] = {
        "enabled": True,
        "bands": {"Band": {"enabled": True, "polarization": "I"}},
        "cg_sampling_groups": {"amplitudes": {"not_a_setting": 1}},
    }

    with pytest.raises(ValueError, match="not_a_setting"):
        validate_parameter_file(_write_params(tmp_path, params))


def test_path_checks_are_optional_and_resolve_relative_to_parameter_file(tmp_path) -> None:
    params = _valid_params()
    params["experiments"]["Example"]["bands"]["Band"]["filelist"] = "input.txt"
    parameter_file = _write_params(tmp_path, params)

    validate_parameter_file(parameter_file, check_paths=False)
    with pytest.raises(FileNotFoundError, match="input.txt"):
        validate_parameter_file(parameter_file, check_paths=True)

    (tmp_path / "input.txt").write_text("", encoding="utf-8")
    validate_parameter_file(parameter_file, check_paths=True)


def test_a_file_with_no_enabled_work_is_rejected(tmp_path) -> None:
    params = _valid_params()
    params["experiments"]["Example"]["enabled"] = False

    with pytest.raises(ValueError, match="enables no TOD bands or CompSep views"):
        validate_parameter_file(_write_params(tmp_path, params))
