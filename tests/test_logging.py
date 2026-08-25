import io
import logging
import re
import subprocess
import sys

from commander4.diagnostics.log import (
    C4Formatter, SUMMARY, VERBOSE, WorldRankFilter, make_run_id, report_fatal)


def _messages_at_level(level: int) -> str:
    """Emit every routine Commander4 level through a handler at ``level``."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    logger = logging.Logger("commander4.test.levels", level=logging.DEBUG)
    logger.addHandler(handler)

    logger.debug("debug")
    logger.verbose("verbose")
    logger.info("info")
    logger.summary("summary")
    logger.warning("warning")
    handler.flush()
    return stream.getvalue()


def test_custom_level_names() -> None:
    assert logging.getLevelName(VERBOSE) == "VERBOSE"
    assert logging.getLevelName(SUMMARY) == "SUMMARY"
    assert logging.VERBOSE == VERBOSE
    assert logging.SUMMARY == SUMMARY


def test_run_id_starts_with_high_precision_datetime() -> None:
    run_id = make_run_id()
    assert re.fullmatch(r"\d{8}T\d{6}\.\d{6}[+-]\d{4}-[a-z0-9]{4}", run_id)


def test_verbose_handler_includes_detailed_and_high_level_records() -> None:
    output = _messages_at_level(VERBOSE)
    assert "debug" not in output
    assert "verbose" in output
    assert "info" in output
    assert "summary" in output
    assert "warning" in output


def test_summary_handler_suppresses_sampling_detail() -> None:
    output = _messages_at_level(SUMMARY)
    assert "debug" not in output
    assert "verbose" not in output
    assert "info" not in output
    assert "summary" in output
    assert "warning" in output


def test_formatter_includes_aligned_world_rank_without_milliseconds() -> None:
    record = logging.LogRecord("commander4.test", logging.INFO, __file__, 1, "hello", (), None)
    WorldRankFilter(17).filter(record)
    formatter = C4Formatter(
        fmt="{asctime} - rank {world_rank:>3} - {name} - {message}",
        datefmt="%H:%M:%S",
        style="{",
    )
    output = formatter.format(record)
    assert "rank  17 - test - hello" in output
    timestamp = output.split(" - ", maxsplit=1)[0]
    assert len(timestamp) == len("12:34:56")
    assert "." not in timestamp


def test_report_fatal_logs_once_and_writes_rank_traceback(tmp_path, caplog, monkeypatch) -> None:
    monkeypatch.setattr(logging, "shutdown", lambda: None)
    logger = logging.getLogger("commander4.test.fatal")

    try:
        raise ValueError("bad parameter")
    except ValueError as error:
        with caplog.at_level(logging.CRITICAL, logger=logger.name):
            traceback_path = report_fatal(
                error, logger, world_rank=7, error_code=1, traceback_dir=str(tmp_path),
                run_id="test-run")

    assert traceback_path == str(tmp_path / "fatal-test-run.log")
    assert "Fatal error on world rank 7: \nValueError: bad parameter" in caplog.text
    assert "Traceback" in caplog.text
    assert "raise ValueError(\"bad parameter\")" in caplog.text
    traceback_text = (tmp_path / "fatal-test-run.log").read_text()
    assert "Traceback" in traceback_text
    assert "ValueError: bad parameter" in traceback_text


def test_only_first_failing_rank_writes_and_logs(tmp_path, caplog, monkeypatch) -> None:
    monkeypatch.setattr(logging, "shutdown", lambda: None)
    logger = logging.getLogger("commander4.test.fatal")

    with caplog.at_level(logging.CRITICAL, logger=logger.name):
        for rank in (3, 9):
            try:
                raise RuntimeError(f"failure from rank {rank}")
            except RuntimeError as error:
                report_fatal(error, logger, rank, 1, str(tmp_path), run_id="shared-run")

    traceback_text = (tmp_path / "fatal-shared-run.log").read_text()
    assert "failure from rank 3" in traceback_text
    assert "failure from rank 9" not in traceback_text
    assert caplog.text.count("Fatal error on world rank") == 1


def test_validation_remains_active_under_python_optimization() -> None:
    code = (
        "from commander4.polarization import get_npol\n"
        "try:\n"
        "    get_npol('bad')\n"
        "except ValueError:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    result = subprocess.run([sys.executable, "-O", "-c", code], check=False)
    assert result.returncode == 0
