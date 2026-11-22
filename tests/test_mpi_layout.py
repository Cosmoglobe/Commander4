"""Pure MPI-layout contracts: enabled inventories, band weights and rank assignment."""
from types import SimpleNamespace

import pytest
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.diagnostics import log as _log  # noqa: F401 (registers logger.summary etc.)
from commander4.mpi.setup import init_mpi, init_mpi_compsep, init_mpi_tod
from commander4.parameters.schema import (
    TODBandConfig,
    derive_task_counts,
    enabled_compsep_views,
    enabled_tod_bands,
    split_tod_ranks,
)


def _tod_band(band_weight: float | str, enabled: bool = True) -> Bunch:
    return Bunch(enabled=enabled, band_weight=band_weight, detectors=Bunch(detector={}))


def _params() -> Bunch:
    return Bunch(
        experiments=Bunch(
            First=Bunch(enabled=True, bands=Bunch(
                BandA=_tod_band(2), Disabled=_tod_band(5, enabled=False),
            )),
            Second=Bunch(enabled=True, bands=Bunch(BandB=_tod_band(3))),
        ),
        # Compsep ranks are only allocated when there is a sampling group to run on them.
        compsep=Bunch(
            bands=Bunch(
                BandA=Bunch(enabled=True, polarization="IQU", get_from="First"),
                BandB=Bunch(enabled=True, polarization="QU", get_from="Second"),
                FileBand=Bunch(enabled=True, polarization="I", get_from="file"),
            ),
            cg_sampling_groups=Bunch(amps=Bunch(enabled=True)),
        ),
    )


def test_tod_inventory_has_global_band_indices_and_weights() -> None:
    bands = enabled_tod_bands(_params())

    assert bands == (
        TODBandConfig(0, "First", "BandA", 2.0),
        TODBandConfig(1, "Second", "BandB", 3.0),
    )


def test_tod_ranks_follow_the_band_weights() -> None:
    # Whole-number weights that sum to the number of ranks give exactly those numbers.
    assert split_tod_ranks([176, 306, 364, 352, 62, 84], 1344) == [176, 306, 364, 352, 62, 84]
    # Otherwise each band gets as many ranks as keeps its weight per rank lowest; every band at
    # least one.
    assert split_tod_ranks([1.0, 1.0, 2.0], 8) == [2, 2, 4]
    assert split_tod_ranks([1000.0, 1.0], 3) == [2, 1]


def test_compsep_inventory_places_intensity_before_polarization() -> None:
    views = enabled_compsep_views(_params())

    assert [view.rank for view in views] == [0, 1, 2, 3]
    assert [view.identifier for view in views] == [
        "BandA_I", "FileBand_I", "BandA_QU", "BandB_QU",
    ]
    assert [view.polarization for view in views] == ["I", "I", "QU", "QU"]


def test_task_counts_come_from_the_same_inventories() -> None:
    counts = derive_task_counts(_params())

    assert (counts.tod_bands, counts.compsep_I, counts.compsep_QU, counts.min_total) == (2, 2, 2, 6)


def test_duplicate_tod_band_names_across_experiments_are_rejected() -> None:
    params = _params()
    params.experiments.Second.bands = Bunch(BandA=_tod_band(3))

    with pytest.raises(ValueError, match="globally unique"):
        enabled_tod_bands(params)


@pytest.mark.parametrize("band_weight", [0, -1.0, "fast"])
def test_band_weight_must_be_positive_or_auto(band_weight: float | str) -> None:
    params = _params()
    params.experiments.First.bands.BandA.band_weight = band_weight

    with pytest.raises(ValueError, match="positive number or 'auto'"):
        enabled_tod_bands(params)


def test_numeric_and_auto_band_weights_are_not_mixed() -> None:
    params = _params()
    params.experiments.First.bands.BandA.band_weight = "auto"

    with pytest.raises(ValueError, match="on every band, or on none"):
        enabled_tod_bands(params)


def test_unknown_compsep_polarization_is_rejected() -> None:
    params = _params()
    params.compsep.bands.BandA.polarization = "IQ"

    with pytest.raises(ValueError, match="polarization"):
        enabled_compsep_views(params)


def test_tod_context_has_no_detector_hierarchy() -> None:
    params = Bunch(
        experiments=Bunch(Experiment=Bunch(
            enabled=True, bands=Bunch(Band=_tod_band(1)),
        )),
        compsep=Bunch(enabled=False),
    )

    tod, band = init_mpi_tod(params, MPI.COMM_SELF, 1, [1], {})

    assert (tod.experiment_name, tod.band_name) == ("Experiment", "Band")
    assert band.comm.Get_size() == 1
    assert not hasattr(tod, "det") and not hasattr(band, "det_id")


def test_band_carries_a_single_node_communicator() -> None:
    """The node communicator is a split of the band, not of the world.

    Two bands sharing a node must not share a node communicator, since anything they put in
    shared memory (the far-sidelobe cubes) is band-specific.
    """
    params = Bunch(
        experiments=Bunch(Experiment=Bunch(
            enabled=True, bands=Bunch(Band=_tod_band(1)),
        )),
        compsep=Bunch(enabled=False),
    )

    _, band = init_mpi_tod(params, MPI.COMM_SELF, 1, [1], {})

    # Serially the band holds one rank, so its node communicator holds exactly that rank.
    assert band.node_comm.Get_size() == 1 and band.node_comm.Get_rank() == 0
    # A node communicator can never be larger than the band it was split from.
    assert band.node_comm.Get_size() <= band.size


# The world ranks of the CompSep views of `_params()` in a run with two TOD ranks, one per band:
# the I views (BandA, FileBand) come first, then the QU views (BandA, BandB).
TOD_MASTERS = {"BandA": 0, "BandB": 1}
COMPSEP_RANKS = {"BandA": {"I": 2, "QU": 4}, "FileBand": {"I": 3}, "BandB": {"QU": 5}}


def test_a_tod_band_gets_its_sky_from_its_i_view_and_sends_each_map_to_its_view() -> None:
    tod, _ = init_mpi_tod(_params(), MPI.COMM_SELF, 1, [1, 1], COMPSEP_RANKS)
    assert (tod.band_name, tod.compsep_sender, tod.compsep_receivers) \
        == ("BandA", 2, {"I": 2, "QU": 4})
    # A band without an I view gets its sky from its QU view. (With no ranks for BandA, the one
    # rank is in BandB.)
    tod, _ = init_mpi_tod(_params(), MPI.COMM_SELF, 1, [0, 1], COMPSEP_RANKS)
    assert (tod.band_name, tod.compsep_sender, tod.compsep_receivers) == ("BandB", 5, {"QU": 5})


def test_a_tod_band_without_a_compsep_entry_is_rejected_at_start_up() -> None:
    with pytest.raises(RuntimeError, match="no enabled entry in 'compsep.bands'"):
        init_mpi_tod(_params(), MPI.COMM_SELF, 1, [1, 1], {"BandB": {"QU": 5}})


@pytest.mark.parametrize("rank, band_name, pol, tod_sender, tod_receiver", [
    (0, "BandA", "I", 0, 0),           # The I view of an IQU band sends the band its sky.
    (1, "FileBand", "I", None, None),  # A band read from file has no TOD side.
    (2, "BandA", "QU", 0, None),       # The QU view of an IQU band gets its map, but sends nothing.
    (3, "BandB", "QU", 1, 1),          # A band without an I view sends its sky from the QU view.
])
def test_a_compsep_rank_knows_its_tod_partners(rank: int, band_name: str, pol: str,
                                               tod_sender: int | None,
                                               tod_receiver: int | None) -> None:
    # A stand-in for CompSep rank `rank` of four; its `Split` result (`subcomm`) is not used here.
    compsep_comm = SimpleNamespace(Get_rank=lambda: rank, Get_size=lambda: 4,
                                   Split=lambda color, key: None)

    compsep = init_mpi_compsep(_params(), compsep_comm, 1, TOD_MASTERS, COMPSEP_RANKS)

    assert (compsep.band_name, compsep.pol, compsep.tod_sender, compsep.tod_receiver) \
        == (band_name, pol, tod_sender, tod_receiver)


def test_a_compsep_band_without_its_tod_band_is_rejected_at_start_up() -> None:
    compsep_comm = SimpleNamespace(Get_rank=lambda: 0, Get_size=lambda: 4,
                                   Split=lambda color, key: None)
    with pytest.raises(RuntimeError, match="has no enabled TOD band 'BandA'"):
        init_mpi_compsep(_params(), compsep_comm, 1, {"BandB": 1}, COMPSEP_RANKS)


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
    assert mpi_info.tod.band_name == expected_band
    assert mpi_info.band.size == 2
    assert mpi_info.band.rank == world_rank % 2
    assert not hasattr(mpi_info, "det")
