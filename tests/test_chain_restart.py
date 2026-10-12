"""Where a run's two chains start: `gibbs.chain_status` and `gibbs.init_chain` over saved files.

The files here are empty: `resolve_chain_start` decides from file names alone, which is what lets
it run on the world master before any rank reads data.
"""
import os

import pytest
from pixell.bunch import Bunch

from commander4.file_io import paths
from commander4.file_io.chain_restart import resolve_chain_start

ALL_PARTS = ("Exp_B1", "Exp_B2", "compsep")


def _params(output_dir, **gibbs) -> Bunch:
    """A run with two TOD bands, `Exp_B1` and `Exp_B2`, writing to `output_dir`."""
    bands = Bunch(B1=Bunch(enabled=True), B2=Bunch(enabled=True))
    return Bunch(output=Bunch(dir=str(output_dir)), gibbs=Bunch(num_iterations=10, **gibbs),
                 experiments=Bunch(Exp=Bunch(enabled=True, bands=bands)))


def _save(run_dir, chains, iterations, parts=ALL_PARTS) -> None:
    """Create (empty) chain files, as a run writing `chains` would leave them."""
    for subdir in (paths.CHAINS_BANDS, paths.CHAINS_COMPSEP):
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
    for part in parts:
        for chain in chains:
            for iteration in iterations:
                if part == "compsep":
                    path = paths.compsep_chain_file(str(run_dir), chain, iteration)
                else:
                    exp_name, band_name = part.split("_")
                    path = paths.band_chain_file(str(run_dir), exp_name, band_name, chain,
                                                 iteration)
                open(path, "w").close()


def _remove(run_dir, part, chain, iteration) -> None:
    if part == "compsep":
        os.remove(paths.compsep_chain_file(str(run_dir), chain, iteration))
    else:
        exp_name, band_name = part.split("_")
        os.remove(paths.band_chain_file(str(run_dir), exp_name, band_name, chain, iteration))


def test_a_new_run_starts_from_the_parameter_file(tmp_path) -> None:
    start = resolve_chain_start(_params(tmp_path), tod_active=True, compsep_active=True)
    assert start.first_iteration == 1
    assert start.source_dir is None


def test_append_continues_after_the_newest_saved_iteration(tmp_path) -> None:
    """Only chain 1 was written (the usual `output.chains.write: [1]`), so chain 2 restarts from
    chain 1's state."""
    _save(tmp_path, chains=[1], iterations=[1, 2, 3])
    start = resolve_chain_start(_params(tmp_path, chain_status="append"), True, True)
    assert (start.first_iteration, start.source_iteration) == (4, 3)
    assert start.source_dir == str(tmp_path)
    assert start.source_chain == {1: 1, 2: 1}


def test_append_steps_back_to_the_newest_iteration_complete_for_every_written_chain(
        tmp_path) -> None:
    """A run stopped after chain 1's sky of iteration 3 but before chain 2's."""
    _save(tmp_path, chains=[1, 2], iterations=[1, 2, 3])
    _remove(tmp_path, "compsep", 2, 3)
    start = resolve_chain_start(_params(tmp_path, chain_status="append"), True, True)
    assert (start.first_iteration, start.source_iteration) == (3, 2)
    assert start.source_chain == {1: 1, 2: 2}


def test_unfinished_writes_and_unused_bands_do_not_count(tmp_path) -> None:
    """A `.tmp` file is a write that never finished; a band this run does not read is ignored."""
    _save(tmp_path, chains=[1], iterations=[1, 2])
    _save(tmp_path, chains=[1], iterations=[1], parts=["Exp_B3"])
    open(paths.compsep_chain_file(str(tmp_path), 1, 3) + ".tmp", "w").close()
    start = resolve_chain_start(_params(tmp_path, chain_status="append"), True, True)
    assert start.source_iteration == 2


def test_append_without_saved_chains_starts_new_ones(tmp_path) -> None:
    start = resolve_chain_start(_params(tmp_path, chain_status="append"), True, True)
    assert (start.first_iteration, start.source_dir) == (1, None)


def test_append_without_saved_chains_falls_back_to_init_chain(tmp_path) -> None:
    """As in Commander3, so the same parameter file serves the first job and every later one."""
    source = tmp_path / "source"
    _save(source, chains=[1], iterations=[1, 2])
    params = _params(tmp_path / "out", chain_status="append", init_chain=str(source))
    start = resolve_chain_start(params, True, True)
    assert (start.first_iteration, start.source_dir, start.source_iteration) == \
        (1, str(source), 2)


def test_init_chain_takes_the_newest_or_the_named_iteration(tmp_path) -> None:
    source = tmp_path / "source"
    _save(source, chains=[1, 2], iterations=[1, 2, 3])
    start = resolve_chain_start(_params(tmp_path / "out", init_chain=str(source)), True, True)
    assert (start.first_iteration, start.source_iteration) == (1, 3)
    assert start.source_chain == {1: 1, 2: 2}

    start = resolve_chain_start(_params(tmp_path / "out", init_chain=f"{source}:2"), True, True)
    assert (start.first_iteration, start.source_iteration) == (1, 2)


def test_init_chain_refuses_an_incomplete_or_missing_iteration(tmp_path) -> None:
    source = tmp_path / "source"
    _save(source, chains=[1], iterations=[1, 2])
    _remove(source, "Exp_B2", 1, 2)
    with pytest.raises(ValueError, match="iteration 2 .* not complete"):
        resolve_chain_start(_params(tmp_path / "out", init_chain=f"{source}:2"), True, True)
    with pytest.raises(ValueError, match="no complete saved iteration"):
        resolve_chain_start(_params(tmp_path / "out", init_chain=str(tmp_path / "empty")),
                            True, True)


def test_a_part_the_source_never_saved_starts_from_the_parameter_file(tmp_path, caplog) -> None:
    """A TOD-only source has no sky files: a joint run takes its bands and starts its sky anew."""
    source = tmp_path / "source"
    _save(source, chains=[1], iterations=[1, 2], parts=["Exp_B1", "Exp_B2"])
    with caplog.at_level("WARNING"):
        start = resolve_chain_start(_params(tmp_path / "out", init_chain=str(source)), True, True)
    assert start.source_iteration == 2
    assert "compsep" in caplog.text


def test_existing_later_files_are_reported_before_they_are_overwritten(tmp_path, caplog) -> None:
    _save(tmp_path, chains=[1], iterations=[1, 2, 3])
    _remove(tmp_path, "compsep", 1, 3)
    with caplog.at_level("WARNING"):
        start = resolve_chain_start(_params(tmp_path, chain_status="append"), True, True)
    assert start.first_iteration == 3
    assert "iterations 3 to 3" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        resolve_chain_start(_params(tmp_path), True, True)  # A new run over the same directory.
    assert "iterations 1 to 3" in caplog.text


def test_a_tod_only_run_needs_only_its_band_files(tmp_path) -> None:
    _save(tmp_path, chains=[1], iterations=[1, 2], parts=["Exp_B1", "Exp_B2"])
    start = resolve_chain_start(_params(tmp_path, chain_status="append"), True, False)
    assert start.source_iteration == 2
