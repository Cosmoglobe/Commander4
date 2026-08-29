"""The unified output-directory layout: `output.dir` and its subdirectories.

Everything a run writes lives under one configured directory. These tests pin the three things that
would otherwise fail silently or far too late: that a run's writers all resolve into that directory,
that a missing `dir` is refused rather than defaulted, and that the log file lands
inside the tree (the file handler opens it before anything else runs, so the directory has to exist
by then).
"""
import os

import pytest
from pixell.bunch import Bunch

from commander4.file_io import paths


def _output(output_dir: str) -> Bunch:
    logging = Bunch(console=Bunch(level="info"), file=Bunch(level="info"))
    return Bunch(dir=output_dir, logging=logging)


def test_every_writer_resolves_into_the_output_directory(tmp_path):
    """The chain writers, the profiler dump and the plots all land under `output.dir`."""
    params = Bunch(output=_output(str(tmp_path / "run")))
    for name in paths.SUBDIRS:
        resolved = paths.subdir(params, name)
        assert os.path.dirname(resolved) == str(tmp_path / "run")
        assert os.path.basename(resolved) == name


def test_subdirectory_names_are_flat():
    """Chain products are siblings, not nested under a shared `chains/` parent.

    The names are part of the on-disk contract that `plot_chain` and the standalone tools read, so
    a rename has to be a deliberate edit here rather than a silent drift.
    """
    assert paths.SUBDIRS == ("logs", "chains_bands", "chains_compsep", "plots")
    assert not any(os.sep in name for name in paths.SUBDIRS)


def test_create_output_dirs_makes_the_whole_tree(tmp_path):
    out = str(tmp_path / "run")
    assert paths.create_output_dirs(_output(out)) == out
    assert sorted(os.listdir(out)) == sorted(paths.SUBDIRS)


def test_create_output_dirs_is_idempotent(tmp_path):
    """Every rank may call it, and a resumed run must not trip over an existing tree."""
    output = _output(str(tmp_path / "run"))
    paths.create_output_dirs(output)
    open(os.path.join(tmp_path, "run", paths.CHAINS_BANDS, "keep.h5"), "w").close()
    paths.create_output_dirs(output)
    assert os.path.exists(os.path.join(tmp_path, "run", paths.CHAINS_BANDS, "keep.h5"))


def test_a_missing_output_dir_is_refused():
    output = Bunch(logging=Bunch(console=Bunch(level="info")))
    with pytest.raises(ValueError, match="output.dir"):
        paths.resolve_output_dir(output)


def test_log_file_goes_in_the_logs_subdirectory(tmp_path):
    output = _output(str(tmp_path / "run"))
    assert "filename" not in output.logging.file
    assert paths.log_file_path(output, "test-run") == str(
        tmp_path / "run" / paths.LOGS / "run-test-run.log")


def test_init_loggers_needs_a_path_for_a_configured_file_logger():
    from commander4.diagnostics import log

    output = _output("unused")
    with pytest.raises(ValueError, match="log_file_path"):
        log.init_loggers(output.logging, None)
