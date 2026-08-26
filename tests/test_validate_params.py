"""Tests for the MPI-free Commander4 parameter validator."""

import sys

import pytest
import yaml
from pixell.bunch import Bunch

from commander4.parameters.parse import params_from_dict
from commander4.standalone_tools.validate_params import (
    estimate_compsep_sht_work,
    main,
    suggest_compsep_thread_counts,
    validate_parameter_file,
)


def _valid_params() -> dict:
    return {
        "gibbs": {"num_iterations": 1},
        "resources": {
            "tod": {"num_threads": 1},
            "compsep": {"num_threads": 1},
        },
        "output": {
            "dir": "output",
            "logging": {},
            "profiling": False,
            "chains": {"write": [1], "include": {}},
        },
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
        "tod_processing": {},
        "compsep": {"enabled": False},
    }


def _write_params(tmp_path, params: dict, filename: str = "params.yml") -> str:
    parameter_file = tmp_path / filename
    parameter_file.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")
    return str(parameter_file)


def _thread_suggestion_params(num_threads: int | list[int] = 4) -> Bunch:
    params = _valid_params()
    params["resources"]["compsep"]["num_threads"] = num_threads
    params["experiments"]["Example"]["bands"] = {
        "Low": {
            "enabled": True,
            "num_tasks": 1,
            "eval_nside": 128,
            "detectors": {"detector": {}},
        },
        "High": {
            "enabled": True,
            "num_tasks": 1,
            "eval_nside": 1024,
            "detectors": {"detector": {}},
        },
    }
    params["compsep"] = {
        "bands": {
            "Low": {"enabled": True, "polarization": "IQU", "get_from": "Example"},
            "High": {"enabled": True, "polarization": "I", "get_from": "Example"},
        },
        "cg_sampling_groups": {"amplitudes": {"enabled": True}},
    }
    return params_from_dict(params)


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


def test_sht_work_uses_power_law_floor_and_spin_factor() -> None:
    assert estimate_compsep_sht_work(128, "I") == 1.0
    assert estimate_compsep_sht_work(512, "I") == 1.0
    assert estimate_compsep_sht_work(512, "QU") == 2.0
    assert estimate_compsep_sht_work(1024, "I") == pytest.approx(2.0**2.7)


def test_thread_suggestion_preserves_scalar_total_and_compsep_rank_order() -> None:
    suggestion = suggest_compsep_thread_counts(_thread_suggestion_params(num_threads=4))

    # Rank order is Low_I, High_I, Low_QU. The current scalar allocation totals 4 * 3 threads.
    assert suggestion == [1, 8, 3]
    assert sum(suggestion) == 12


def test_thread_suggestion_uses_sum_of_existing_list_as_default() -> None:
    suggestion = suggest_compsep_thread_counts(_thread_suggestion_params(num_threads=[2, 3, 7]))

    assert suggestion == [1, 8, 3]


def test_explicit_node_budget_caps_each_rank_at_one_node() -> None:
    suggestion = suggest_compsep_thread_counts(
        _thread_suggestion_params(), threads_per_node=4, num_nodes=2,
    )

    assert suggestion == [1, 4, 3]
    assert sum(suggestion) == 8
    assert max(suggestion) == 4


def test_thread_suggestion_rejects_budget_smaller_than_rank_count() -> None:
    with pytest.raises(ValueError, match="at least one thread per rank"):
        suggest_compsep_thread_counts(
            _thread_suggestion_params(), threads_per_node=1, num_nodes=1,
        )


def test_command_prints_thread_suggestion(tmp_path, monkeypatch, capsys) -> None:
    params = _valid_params()
    params["resources"]["compsep"]["num_threads"] = 2
    params["components"]["CMB"]["params"]["polarization"] = "IQU"
    params["experiments"]["Example"]["bands"]["Band"]["eval_nside"] = 512
    params["compsep"] = {
        "bands": {
            "Band": {"enabled": True, "polarization": "IQU", "get_from": "Example"},
        },
        "cg_sampling_groups": {"amplitudes": {"enabled": True}},
    }
    parameter_file = _write_params(tmp_path, params)
    monkeypatch.setattr(
        sys, "argv",
        ["c4-validate-params", parameter_file, "--compsep-threads-per-node", "4"],
    )

    assert main() == 0

    output = capsys.readouterr().out
    assert "CompSep SHT scaling:" in output
    assert "Suggested resources.compsep.num_threads: [1, 3]" in output


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


def test_unused_component_parameter_is_rejected(tmp_path) -> None:
    params = _valid_params()
    params["components"]["CMB"]["params"]["unused_setting"] = 1

    with pytest.raises(ValueError, match="unused_setting"):
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


def test_missing_required_top_level_block_is_rejected(tmp_path) -> None:
    params = _valid_params()
    del params["tod_processing"]

    with pytest.raises(ValueError, match="tod_processing"):
        validate_parameter_file(_write_params(tmp_path, params))


def test_empty_top_level_block_must_be_a_mapping(tmp_path) -> None:
    params = _valid_params()
    params["experiments"] = []

    with pytest.raises(ValueError, match="experiments.*mapping"):
        validate_parameter_file(_write_params(tmp_path, params))


def test_chain_selection_is_required(tmp_path) -> None:
    params = _valid_params()
    del params["output"]["chains"]["write"]

    with pytest.raises(ValueError, match="output.chains.write"):
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
