import numpy as np
import pytest
from pixell.bunch import Bunch

from commander4.compsep_processing import (
    _filter_sampling_group_components,
    _sampling_group_selection,
    _sampling_group_selects_band,
    _validate_sampling_groups,
    init_compsep_processing,
)
from commander4.communication import _get_compsep_sender_id_for_tod_band, _should_send_compsep_result
from commander4.sky_models.component import CompList
from commander4.sky_models.sky_model import SkyModel
from commander4.utils.execution_ids import get_execution_band_id, get_execution_band_ids


def _make_general(ntask_compsep_qu: int = 1, ntask_compsep_i: int = 1) -> Bunch:
    return Bunch(
        nside=2,
        CG_float_precision="single",
        MPI_config=Bunch(ntask_compsep_I=ntask_compsep_i, ntask_compsep_QU=ntask_compsep_qu),
    )


def _make_component_cfg(polarization: str = "IQU") -> Bunch:
    return Bunch(
        enabled=True,
        component_class="CMB",
        params=Bunch(
            lmax=2,
            polarization=polarization,
            shortname="cmb",
            spatially_varying_MM=False,
            smoothing_prior_FWHM=0.0,
            smoothing_prior_amplitude=1.0,
        ),
    )


def _make_comp_list(polarization: str = "IQU", ntask_compsep_qu: int = 1) -> CompList:
    params = Bunch(general=_make_general(ntask_compsep_qu))
    cmb = _make_component_cfg(polarization)
    object.__setattr__(cmb, "_name", "cmb")
    components = Bunch({"cmb": cmb})
    return CompList.init_from_params(components, params)


def _make_multi_comp_list() -> CompList:
    params = Bunch(general=_make_general())
    cmb = _make_component_cfg("IQU")
    object.__setattr__(cmb, "_name", "CMB")
    dust = Bunch(
        enabled=True,
        component_class="CMB",
        params=Bunch(
            lmax=2,
            polarization="IQU",
            shortname="dust",
            spatially_varying_MM=False,
            smoothing_prior_FWHM=0.0,
            smoothing_prior_amplitude=1.0,
        ),
    )
    object.__setattr__(dust, "_name", "ThermalDust")
    ff = Bunch(
        enabled=True,
        component_class="CMB",
        params=Bunch(
            lmax=2,
            polarization="I",
            shortname="ff",
            spatially_varying_MM=False,
            smoothing_prior_FWHM=0.0,
            smoothing_prior_amplitude=1.0,
        ),
    )
    object.__setattr__(ff, "_name", "FreeFree")
    components = Bunch(
        {
            "CMB": cmb,
            "ThermalDust": dust,
            "FreeFree": ff,
        }
    )
    return CompList.init_from_params(components, params)


def test_execution_band_id_helpers_use_plain_band_names() -> None:
    assert get_execution_band_id("90GHz", "I") == "90GHz_I"
    assert get_execution_band_id("353GHz", "QU") == "353GHz_QU"
    assert get_execution_band_ids("90GHz", "IQU") == ("90GHz_I", "90GHz_QU")


def test_tod_receive_source_prefers_intensity_and_falls_back_to_qu() -> None:
    senders = {"30GHz_I": 3, "30GHz_QU": 4}
    assert _get_compsep_sender_id_for_tod_band("30GHz", senders) == "30GHz_I"

    senders = {"Planck353GHz_QU": 8}
    assert _get_compsep_sender_id_for_tod_band("Planck353GHz", senders) == "Planck353GHz_QU"

    with pytest.raises(KeyError, match="No CompSep sender"):
        _get_compsep_sender_id_for_tod_band("MissingBand", {})


def test_should_send_compsep_result_skips_qu_when_i_sender_exists() -> None:
    destinations = {"LFT_I": 1, "LFT_QU": 2}

    assert _should_send_compsep_result("LFT_I", destinations)
    assert not _should_send_compsep_result("LFT_QU", destinations)
    assert _should_send_compsep_result("Planck353GHz_QU", {"Planck353GHz_QU": 5})
    assert not _should_send_compsep_result("Unused_I", None)


def test_joined_skymodel_realizes_iqu_components() -> None:
    comp_list = _make_comp_list("IQU")
    joined = comp_list.joined()

    sky = SkyModel(joined).get_sky_at_nu(30.0, 2, "IQU", fwhm=0.0)

    assert sky.shape == (3, 48)
    assert np.all(np.isfinite(sky))


def test_sampling_group_component_filter_matches_comp_names_and_preserves_pol_split() -> None:
    comp_list = _make_multi_comp_list()

    intensity = _filter_sampling_group_components(
        comp_list.split_for_eval_pol("I"),
        ["ThermalDust", "FreeFree"],
    )
    polarization = _filter_sampling_group_components(
        comp_list.split_for_eval_pol("QU"),
        ["ThermalDust", "FreeFree"],
    )

    assert [comp.comp_name for comp in intensity] == ["ThermalDust", "FreeFree"]
    assert [comp.eval_pol for comp in intensity] == ["I", "I"]
    assert [comp.comp_name for comp in polarization] == ["ThermalDust"]
    assert [comp.eval_pol for comp in polarization] == ["QU"]


def test_sampling_group_band_filter_accepts_base_and_execution_ids() -> None:
    assert _sampling_group_selects_band(["Planck30GHz"], "Planck30GHz", "Planck30GHz_I")
    assert _sampling_group_selects_band(["Planck30GHz_I"], "Planck30GHz", "Planck30GHz_I")
    assert not _sampling_group_selects_band(["Planck44GHz"], "Planck30GHz", "Planck30GHz_I")


def test_validate_sampling_groups_rejects_unknown_names(caplog) -> None:
    comp_list = _make_multi_comp_list()  # CMB, ThermalDust, FreeFree
    params = Bunch(CompSep_bands=Bunch({"Planck30GHz": Bunch(enabled=True, polarization="IQU")}))

    # Valid references (component name + execution-view band id) pass silently.
    _validate_sampling_groups(
        Bunch(g=Bunch(comps=["CMB"], bands=["Planck30GHz_QU"])), comp_list, params)

    with pytest.raises(AssertionError):
        _validate_sampling_groups(Bunch(g=Bunch(comps=["DoesNotExist"])), comp_list, params)
    assert "unknown component" in caplog.text

    with pytest.raises(AssertionError):
        _validate_sampling_groups(Bunch(g=Bunch(bands=["NoSuchBand"])), comp_list, params)
    assert "unknown band" in caplog.text


def test_sampling_group_selection_resolves_all_and_missing() -> None:
    # Missing entry and the literal "all" both mean "everything" (None); a list is returned as-is.
    assert _sampling_group_selection(Bunch(), "comps") is None
    assert _sampling_group_selection(Bunch(comps="all"), "comps") is None
    assert _sampling_group_selection(Bunch(bands="all"), "bands") is None
    assert _sampling_group_selection(Bunch(comps=["CMB"]), "comps") == ["CMB"]
    assert _sampling_group_selection(Bunch(bands=["Planck30GHz"]), "bands") == ["Planck30GHz"]


def test_validate_sampling_groups_accepts_all_and_missing() -> None:
    comp_list = _make_multi_comp_list()
    params = Bunch(CompSep_bands=Bunch({"Planck30GHz": Bunch(enabled=True, polarization="IQU")}))

    # "all" and omitted entries select everything and must not be checked against names.
    _validate_sampling_groups(Bunch(g=Bunch(comps="all", bands="all")), comp_list, params)
    _validate_sampling_groups(Bunch(g=Bunch()), comp_list, params)


def test_validate_sampling_groups_skips_disabled_group() -> None:
    comp_list = _make_multi_comp_list()
    params = Bunch(CompSep_bands=Bunch({"Planck30GHz": Bunch(enabled=True, polarization="IQU")}))

    # A disabled group is not validated, so its bogus references are tolerated.
    _validate_sampling_groups(
        Bunch(g=Bunch(enabled=False, comps=["DoesNotExist"], bands=["NoSuchBand"])),
        comp_list, params)


def test_comp_name_comes_from_component_bunch_name() -> None:
    params = Bunch(general=_make_general())
    component_cfg = _make_component_cfg("IQU")
    object.__setattr__(component_cfg, "_name", "CMBFromName")
    components = Bunch({"cmb": component_cfg})

    comp_list = CompList.init_from_params(components, params)

    assert [comp.comp_name for comp in comp_list] == ["CMBFromName", "CMBFromName"]


def test_init_compsep_processing_rejects_duplicate_component_names(monkeypatch, caplog) -> None:
    class _FakeCompList:
        def joined(self):
            return [Bunch(comp_name="dup"), Bunch(comp_name="dup")]

    class _FakeComm:
        def allgather(self, data):
            return [data]

    monkeypatch.setattr(CompList, "init_from_params", classmethod(lambda cls, *_: _FakeCompList()))

    mpi_info = Bunch(
        processor_name="test-node",
        world=Bunch(rank=0),
        compsep=Bunch(rank=0, QU_master=1, size=1, comm=_FakeComm()),
    )
    params = Bunch(
        components=Bunch(),
        CompSep_bands=Bunch(
            {
                "BandA": Bunch(enabled=True, polarization="I", get_from="file"),
            }
        ),
    )

    with pytest.raises(AssertionError):
        init_compsep_processing(mpi_info, params)
    assert "Duplicate component names found in CompSep setup" in caplog.text