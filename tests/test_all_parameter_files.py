"""Every maintained Commander4 parameter file must satisfy the lightweight validator."""

from pathlib import Path

from commander4.standalone_tools.validate_params import validate_parameter_file


def test_all_commander_parameter_files_validate() -> None:
    parameter_files = []
    for path in Path("params").rglob("*.yml"):
        if any(line.startswith("gibbs:") for line in path.read_text().splitlines()):
            parameter_files.append(path)

    failures = []
    for path in sorted(parameter_files):
        try:
            validate_parameter_file(str(path))
        except Exception as error:
            failures.append(f"{path}: {type(error).__name__}: {error}")

    assert parameter_files
    assert failures == []
