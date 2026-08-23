"""Pure MPI-layout contracts: enabled inventories, rank assignment, and scan intervals."""

import pytest
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.diagnostics import log as _log  # Registers custom logging methods.
from commander4.mpi.setup import init_mpi, init_mpi_compsep, init_mpi_tod
from commander4.parameters.schema import (
    TODBandConfig,
    derive_task_counts,
    enabled_compsep_views,
    enabled_tod_bands,
    split_integer_range,
)


def _tod_band(num_tasks: int, enabled: bool = True) -> Bunch:
    return Bunch(enabled=enabled, num_tasks=num_tasks, detectors=Bunch(detector={}))


def _params() -> Bunch:
    return Bunch(
        experiments=Bunch(
            First=Bunch(enabled=True, bands=Bunch(
                BandA=_tod_band(2), Disabled=_tod_band(5, enabled=False),
            )),
            Second=Bunch(enabled=True, bands=Bunch(BandB=_tod_band(3))),
        ),
        compsep=Bunch(bands=Bunch(
            BandA=Bunch(enabled=True, polarization="IQU"),
            BandB=Bunch(enabled=True, polarization="QU"),
            FileBand=Bunch(enabled=True, polarization="I"),
        )),
    )


def test_tod_inventory_has_global_band_indices_and_contiguous_rank_intervals() -> None:
    bands = enabled_tod_bands(_params())

    assert bands == (
        TODBandConfig(0, "First", "BandA", 0, 2),
        TODBandConfig(1, "Second", "BandB", 2, 5),
    )


def test_compsep_inventory_places_intensity_before_polarization() -> None:
    views = enabled_compsep_views(_params())

    assert [view.rank for view in views] == [0, 1, 2, 3]
    assert [view.identifier for view in views] == [
        "BandA_I", "FileBand_I", "BandA_QU", "BandB_QU",
    ]
    assert [view.polarization for view in views] == ["I", "I", "QU", "QU"]


def test_task_counts_come_from_the_same_inventories() -> None:
    counts = derive_task_counts(_params())

    assert (counts.tod, counts.compsep_I, counts.compsep_QU, counts.total) == (5, 2, 2, 9)


def test_duplicate_tod_band_names_across_experiments_are_rejected() -> None:
    params = _params()
    params.experiments.Second.bands = Bunch(BandA=_tod_band(3))

    with pytest.raises(ValueError, match="globally unique"):
        enabled_tod_bands(params)


def test_non_positive_tod_task_count_is_rejected() -> None:
    params = _params()
    params.experiments.First.bands.BandA.num_tasks = 0

    with pytest.raises(ValueError, match="num_tasks must be at least 1"):
        enabled_tod_bands(params)


def test_unknown_compsep_polarization_is_rejected() -> None:
    params = _params()
    params.compsep.bands.BandA.polarization = "IQ"

    with pytest.raises(ValueError, match="polarization"):
        enabled_compsep_views(params)


def test_half_open_scan_ranges_cover_every_scan_exactly_once() -> None:
    intervals = []
    assigned_scans = []
    for part in range(3):
        start, stop = split_integer_range(8, 3, part)
        intervals.append((start, stop))
        assigned_scans.extend(range(start, stop))

    assert intervals == [(0, 3), (3, 6), (6, 8)]
    assert assigned_scans == list(range(8))


def test_half_open_scan_ranges_allow_more_ranks_than_scans() -> None:
    intervals = []
    for part in range(4):
        intervals.append(split_integer_range(2, 4, part))

    assert intervals == [(0, 1), (1, 2), (2, 2), (2, 2)]


@pytest.mark.parametrize(
    "length,num_parts,part",
    [(-1, 1, 0), (1, 0, 0), (1, 1, -1), (1, 1, 1)],
)
def test_invalid_scan_range_requests_are_rejected(length: int, num_parts: int, part: int) -> None:
    with pytest.raises(ValueError):
        split_integer_range(length, num_parts, part)


def test_tod_context_has_no_detector_hierarchy() -> None:
    params = Bunch(
        experiments=Bunch(Experiment=Bunch(
            enabled=True, bands=Bunch(Band=_tod_band(1)),
        )),
        compsep=Bunch(enabled=False),
    )
    mpi_info = Bunch(tod=Bunch(
        rank=0, size=1, is_master=False, comm=MPI.COMM_SELF,
    ))

    result = init_mpi_tod(mpi_info, params)

    assert result.experiment.name == "Experiment"
    assert result.band.name == "Band"
    assert result.band.index == 0
    assert result.band.comm.Get_size() == 1
    assert "det" not in result
    assert "det_id" not in result.band


def test_compsep_context_uses_the_inventory_view() -> None:
    params = Bunch(
        experiments=Bunch(),
        compsep=Bunch(bands=Bunch(Band=Bunch(enabled=True, polarization="QU"))),
    )
    mpi_info = Bunch(compsep=Bunch(rank=0, size=1))

    result = init_mpi_compsep(mpi_info, params)

    assert result.band.name == "Band"
    assert result.band.identifier == "Band_QU"
    assert result.band.polarization == "QU"
    assert result.band.comm == MPI.COMM_SELF


@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() != 4, reason="requires exactly four MPI ranks")
def test_four_rank_layout_keeps_experiment_bands_in_separate_communicators() -> None:
    """The first enabled band of each experiment must not share its old local index-zero color."""
    params = Bunch(
        experiments=Bunch(
            First=Bunch(enabled=True, bands=Bunch(BandA=_tod_band(2))),
            Second=Bunch(enabled=True, bands=Bunch(BandB=_tod_band(2))),
        ),
        compsep=Bunch(enabled=False),
        resources=Bunch(
            tod=Bunch(num_threads=1),
            compsep=Bunch(num_threads=1),
        ),
    )

    mpi_info = init_mpi(params)
    world_rank = MPI.COMM_WORLD.Get_rank()

    expected_band = "BandA" if world_rank < 2 else "BandB"
    assert mpi_info.world.side == "tod"
    assert mpi_info.band.name == expected_band
    assert mpi_info.band.size == 2
    assert mpi_info.band.rank == world_rank % 2
    assert "det" not in mpi_info
