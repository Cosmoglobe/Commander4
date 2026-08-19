import numpy as np
import pytest
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.compsep_processing import (
    CGSamplingGroupConfig,
    MCMCSamplingGroupConfig,
    PerPixelSamplingGroupConfig,
    _build_conditional_residual,
    _filter_sampling_group_components,
    _read_sampling_groups,
    _resolve_sampling_groups,
    _sampling_group_selects_band,
    _validate_sampling_group_dependencies,
    _validate_sampling_group_references,
    init_compsep_processing,
)
from commander4.communication import _get_compsep_sender_id_for_tod_band, _should_send_compsep_result
from commander4.data_models.detector_map import DetectorMap
from commander4.sky_models.component import CompList
from commander4.sky_models.sky_model import SkyModel
from commander4.solvers.perpix_compsep_solver import solve_compsep_perpix
from commander4.solvers.spectral_index_sampler import (
    _discover_spectral_index_groups, SpectralIndexGroup, SpectralIndexSamplingGroup)
from commander4.utils.execution_ids import get_execution_band_id, get_execution_band_ids


def _make_compsep(ntask_compsep_qu: int = 1, ntask_compsep_i: int = 1) -> Bunch:
    """The `compsep` block components read: its nside and float precision."""
    return Bunch(nside=2, float_precision="single")


def _make_component_cfg(polarization: str = "IQU") -> Bunch:
    return Bunch(
        enabled=True,
        component_class="CMB",
        params=Bunch(
            lmax=2,
            polarization=polarization,
            shortname="cmb",
            spatially_varying_MM=False,
            Cl_prior_amplitude=None,  # identity prior (C_l = 1)
        ),
    )


def _make_comp_list(polarization: str = "IQU", ntask_compsep_qu: int = 1) -> CompList:
    params = Bunch(compsep=_make_compsep(ntask_compsep_qu))
    cmb = _make_component_cfg(polarization)
    object.__setattr__(cmb, "_name", "cmb")
    components = Bunch({"cmb": cmb})
    return CompList.init_from_params(components, params)


def _make_multi_comp_list() -> CompList:
    params = Bunch(compsep=_make_compsep())
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
            Cl_prior_amplitude=None,  # identity prior (C_l = 1)
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
            Cl_prior_amplitude=None,  # identity prior (C_l = 1)
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


def test_group_reference_validation_rejects_unknown_names(caplog) -> None:
    comp_list = _make_multi_comp_list()  # CMB, ThermalDust, FreeFree
    params = Bunch(compsep=Bunch(bands=Bunch(
        {"Planck30GHz": Bunch(enabled=True, polarization="IQU")})))

    # Valid references (component name + execution-view band id) pass silently.
    valid = CGSamplingGroupConfig.from_block(
        "g", Bunch(comps=["CMB"], bands=["Planck30GHz_QU"]))
    _validate_sampling_group_references({"g": valid}, comp_list, params)

    with pytest.raises(AssertionError):
        invalid = CGSamplingGroupConfig.from_block("g", Bunch(comps=["DoesNotExist"]))
        _validate_sampling_group_references({"g": invalid}, comp_list, params)
    assert "unknown component" in caplog.text

    with pytest.raises(AssertionError):
        invalid = CGSamplingGroupConfig.from_block("g", Bunch(bands=["NoSuchBand"]))
        _validate_sampling_group_references({"g": invalid}, comp_list, params)
    assert "unknown band" in caplog.text


def test_group_configs_normalize_all_and_explicit_selections() -> None:
    all_names = CGSamplingGroupConfig.from_block("all", Bunch(comps="all", bands="all"))
    selected = CGSamplingGroupConfig.from_block(
        "selected", Bunch(comps=["CMB"], bands=["Planck30GHz"]))
    assert all_names.comps is None and all_names.bands is None
    assert selected.comps == ("CMB",)
    assert selected.bands == ("Planck30GHz",)


def test_group_reference_validation_accepts_all_and_missing() -> None:
    comp_list = _make_multi_comp_list()
    params = Bunch(compsep=Bunch(bands=Bunch(
        {"Planck30GHz": Bunch(enabled=True, polarization="IQU")})))

    # "all" and omitted entries select everything and must not be checked against names.
    groups = {
        "all": CGSamplingGroupConfig.from_block("all", Bunch(comps="all", bands="all")),
        "missing": CGSamplingGroupConfig.from_block("missing", Bunch()),
    }
    _validate_sampling_group_references(groups, comp_list, params)


def test_read_sampling_groups_filters_disabled_groups() -> None:
    params = Bunch(compsep=Bunch(cg_sampling_groups=Bunch(
        active=Bunch(enabled=True), disabled=Bunch(enabled=False))))
    groups = _read_sampling_groups(params, "cg_sampling_groups", CGSamplingGroupConfig)
    assert list(groups) == ["active"]


def test_comp_name_comes_from_component_bunch_name() -> None:
    params = Bunch(compsep=_make_compsep())
    component_cfg = _make_component_cfg("IQU")
    object.__setattr__(component_cfg, "_name", "CMBFromName")
    components = Bunch({"cmb": component_cfg})

    comp_list = CompList.init_from_params(components, params)

    assert [comp.comp_name for comp in comp_list] == ["CMBFromName", "CMBFromName"]


def _make_spectral_comp_list() -> CompList:
    params = Bunch(compsep=_make_compsep())
    sync = Bunch(
        enabled=True,
        component_class="Synchrotron",
        params=Bunch(
            lmax=2, polarization="IQU", shortname="sync", spatially_varying_MM=False,
            Cl_prior_amplitude=None, beta=-3.1, nu_ref=30.0,
            sample_spectral_index=True, spectral_index_proposal_sigma=0.02,
            spectral_index_bounds=[-4.0, -2.0]),
    )
    object.__setattr__(sync, "_name", "Synchrotron")
    dust = Bunch(
        enabled=True,
        component_class="ThermalDust",
        params=Bunch(
            lmax=2, polarization="IQU", shortname="dust", spatially_varying_MM=False,
            Cl_prior_amplitude=None, beta=1.56, T=20.0,
            nu_ref=545.0, sample_spectral_index=True, spectral_index_proposal_sigma=0.01,
            spectral_index_prior=Bunch(type="gaussian", mean=1.5, rms=0.1)),
    )
    object.__setattr__(dust, "_name", "ThermalDust")
    return CompList.init_from_params(Bunch({"Synchrotron": sync, "ThermalDust": dust}), params)


def test_sampling_group_configs_reject_fields_owned_by_other_methods() -> None:
    with pytest.raises(ValueError, match="sample_class"):
        CGSamplingGroupConfig.from_block("cg", Bunch(sample_class="amplitude_sampler_CG"))
    with pytest.raises(ValueError, match="max_iter"):
        PerPixelSamplingGroupConfig.from_block("pixels", Bunch(max_iter=10))
    with pytest.raises(ValueError, match="parameters"):
        MCMCSamplingGroupConfig.from_block("mcmc", Bunch(parameters="gain"))


def test_cg_and_per_pixel_groups_are_mutually_exclusive() -> None:
    params = Bunch(compsep=Bunch(
        cg_sampling_groups=Bunch(cg=Bunch()),
        per_pixel_sampling_groups=Bunch(pixels=Bunch()),
    ))
    with pytest.raises(ValueError, match="mutually exclusive"):
        _resolve_sampling_groups(params)


def test_per_pixel_solver_accepts_resolved_precision() -> None:
    comp_list = _make_comp_list("I")
    detector_data = DetectorMap(
        map_sky=np.ones((1, 48)), map_rms=np.ones((1, 48)),
        nu=30.0, fwhm=0.0, nside=2,
    )
    result = solve_compsep_perpix(
        MPI.COMM_SELF, detector_data, comp_list, double_precision=False)
    assert result[0].alms.dtype == np.complex64
    assert np.all(np.isfinite(result[0].alms))


def test_validate_mcmc_amplitude_group_dependencies() -> None:
    comp_list = _make_multi_comp_list()  # CMB, ThermalDust, FreeFree
    params = Bunch(compsep=Bunch(bands=Bunch(
        {"Planck30GHz": Bunch(enabled=True, polarization="IQU")})))

    amplitudes = {"amps": CGSamplingGroupConfig.from_block("amps", Bunch(comps=["CMB"]))}
    mcmc = {"beta": MCMCSamplingGroupConfig.from_block(
        "beta", Bunch(comps=["ThermalDust"], update_amplitude_groups=["amps"]))}
    _validate_sampling_group_references(amplitudes, comp_list, params)
    _validate_sampling_group_references(mcmc, comp_list, params)
    _validate_sampling_group_dependencies(amplitudes, mcmc)

    invalid = {"beta": MCMCSamplingGroupConfig.from_block(
        "beta", Bunch(update_amplitude_groups=["does_not_exist"]))}
    with pytest.raises(AssertionError):
        _validate_sampling_group_dependencies(amplitudes, invalid)


def test_build_conditional_residual_subtracts_only_fixed_components() -> None:
    comp_list = _make_multi_comp_list()  # CMB, ThermalDust, FreeFree (all intensity views exist)
    for comp in comp_list.split_for_eval_pol("I"):
        comp.alms = np.full_like(comp.alms, 3.0 + 0.0j)

    nside = 2
    npix = 12 * nside**2
    detector_data = DetectorMap(
        map_sky=np.full((1, npix), 7.0), map_rms=np.ones((1, npix)), nu=30.0, fwhm=60.0, nside=nside)

    active = _filter_sampling_group_components(comp_list.split_for_eval_pol("I"), ["CMB"])
    residual = _build_conditional_residual(detector_data, comp_list, "I", active)

    # The fixed components (everything but CMB) are exactly what should have been subtracted.
    fixed = [c for c in comp_list.split_for_eval_pol("I") if c.comp_name != "CMB"]
    fixed_sky = SkyModel(CompList(fixed)).get_sky_at_nu(30.0, nside, "I", fwhm=detector_data.fwhm_rad)
    assert residual is not detector_data
    assert np.any(fixed_sky != 0.0)  # subtraction is non-trivial
    np.testing.assert_allclose(residual.map_sky, detector_data.map_sky - fixed_sky, rtol=1e-5)
    # The original map is left untouched.
    np.testing.assert_array_equal(detector_data.map_sky, np.full((1, npix), 7.0))


def test_build_conditional_residual_is_noop_when_no_component_is_fixed() -> None:
    comp_list = _make_multi_comp_list()
    active = _filter_sampling_group_components(comp_list.split_for_eval_pol("I"), None)  # all comps
    detector_data = DetectorMap(
        map_sky=np.ones((1, 48)), map_rms=np.ones((1, 48)), nu=30.0, fwhm=60.0, nside=2)
    assert _build_conditional_residual(
        detector_data, comp_list, "I", active) is detector_data


def test_get_sky_removes_amp_fwhm_by_quadrature() -> None:
    # get_sky realizes a component at a *target* band resolution, removing the beam already carried
    # by its amplitudes (amp_fwhm_rad) via applied = sqrt(target^2 - amp_fwhm_rad^2).
    comp = next(c for c in _make_multi_comp_list().split_for_eval_pol("I") if c.comp_name == "CMB")
    comp.alms = np.full_like(comp.alms, 2.0 + 0.0j)  # power at all l so smoothing is non-trivial
    nu, nside, target = 30.0, 2, 0.5  # target beam in radians
    sed = comp.get_sed(nu)
    # Sanity: at this target the beam actually changes the map, so the cases below are distinguishable.
    assert not np.allclose(comp.get_component_map(nside, target), comp.get_component_map(nside, 0.0))

    # Deconvolved amplitudes (amp_fwhm_rad=0) -> apply the full target beam.
    comp.amp_fwhm_rad = 0.0
    np.testing.assert_allclose(comp.get_sky(nu, nside, target),
                               comp.get_component_map(nside, target)*sed)
    # Partially pre-smoothed amplitudes -> apply the quadrature remainder.
    comp.amp_fwhm_rad = 0.3
    np.testing.assert_allclose(comp.get_sky(nu, nside, target),
                               comp.get_component_map(nside, np.sqrt(target**2 - 0.3**2))*sed)
    # Amplitudes already at the band resolution -> no extra smoothing.
    comp.amp_fwhm_rad = target
    np.testing.assert_allclose(comp.get_sky(nu, nside, target),
                               comp.get_component_map(nside, 0.0)*sed)
    # Amplitudes coarser than the target -> clamp to 0 rather than take a sqrt of a negative.
    comp.amp_fwhm_rad = 2*target
    np.testing.assert_allclose(comp.get_sky(nu, nside, target),
                               comp.get_component_map(nside, 0.0)*sed)


def test_smooth_to_resolution_updates_beam_signal_and_is_idempotent() -> None:
    nside = 16
    npix = 12*nside**2
    sky = np.random.default_rng(0).standard_normal((1, npix))
    detector_data = DetectorMap(map_sky=sky.copy(), map_rms=np.ones((1, npix)), nu=30.0,
                                fwhm=30.0, nside=nside)
    detector_data.smooth_to_resolution(60.0)
    assert detector_data.fwhm == 60.0
    # Smoothing to a coarser beam changes the signal and lowers the noise RMS.
    assert not np.allclose(detector_data.map_sky, sky)
    assert detector_data.map_rms.mean() < 1.0
    # Idempotent via fwhm equality: re-requesting the same resolution leaves everything unchanged.
    smoothed, rms = detector_data.map_sky.copy(), detector_data.map_rms.copy()
    detector_data.smooth_to_resolution(60.0)
    np.testing.assert_array_equal(detector_data.map_sky, smoothed)
    np.testing.assert_array_equal(detector_data.map_rms, rms)
    assert detector_data.fwhm == 60.0


def test_smooth_to_resolution_is_noop_when_already_at_target() -> None:
    nside = 16
    npix = 12*nside**2
    sky = np.random.default_rng(1).standard_normal((1, npix))
    detector_data = DetectorMap(map_sky=sky.copy(), map_rms=np.ones((1, npix)), nu=30.0,
                                fwhm=60.0, nside=nside)
    detector_data.smooth_to_resolution(60.0)  # already at target: nothing to do.
    assert detector_data.fwhm == 60.0
    np.testing.assert_array_equal(detector_data.map_sky, sky)


def test_smooth_to_resolution_warns_and_skips_finer_target(caplog) -> None:
    sky = np.random.default_rng(2).standard_normal((1, 48))
    detector_data = DetectorMap(map_sky=sky.copy(), map_rms=np.ones((1, 48)), nu=30.0,
                                fwhm=60.0, nside=2)
    with caplog.at_level("WARNING"):
        detector_data.smooth_to_resolution(30.0)  # finer than native: warn, leave unchanged.
    assert "finer" in caplog.text.lower()
    assert detector_data.fwhm == 60.0
    np.testing.assert_array_equal(detector_data.map_sky, sky)


def test_discover_spectral_index_groups_groups_iqu_views_and_respects_selection() -> None:
    comp_list = _make_spectral_comp_list()

    # All selected: one group per logical component (the I/QU views of each are grouped together).
    all_groups = _discover_spectral_index_groups(comp_list, None)
    assert sorted(g.name for g in all_groups) == ["dust", "sync"]
    sync_group = next(g for g in all_groups if g.name == "sync")
    assert {comp.eval_pol for comp in sync_group.components} == {"I", "QU"}
    assert sync_group.bounds == (-4.0, -2.0)

    # Restricting to one component yields only that component's group.
    only_sync = _discover_spectral_index_groups(comp_list, ["Synchrotron"])
    assert [g.name for g in only_sync] == ["sync"]


def test_discover_spectral_index_groups_reads_gaussian_prior() -> None:
    groups = {g.name: g for g in _discover_spectral_index_groups(_make_spectral_comp_list(), None)}
    assert groups["dust"].prior == (1.5, 0.1)  # gaussian prior parsed as (mean, rms)
    assert groups["sync"].prior is None        # no prior block -> flat prior


def test_spectral_index_gaussian_log_prior_matches_formula() -> None:
    # Construct the sampler without MPI: log_prior only reads self._groups, so bypass __init__.
    sampler = SpectralIndexSamplingGroup.__new__(SpectralIndexSamplingGroup)
    sampler._groups = [
        SpectralIndexGroup("sync", components=(), proposal_sigma=0.1, bounds=None, prior=(-3.0, 0.2)),
        SpectralIndexGroup("dust", components=(), proposal_sigma=0.1, bounds=None, prior=None),
    ]
    # Group at its prior mean and the flat group both contribute 0.
    assert sampler.log_prior({"sync": -3.0, "dust": 1.5}) == 0.0
    # Off-mean group contributes -0.5 ((beta-mean)/rms)^2; flat group still 0.
    np.testing.assert_allclose(sampler.log_prior({"sync": -3.4, "dust": 1.5}),
                               -0.5*((-3.4 + 3.0)/0.2)**2)


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
        compsep=Bunch(bands=Bunch(
            {
                "BandA": Bunch(enabled=True, polarization="I", get_from="file"),
            }
        )),
    )

    with pytest.raises(AssertionError):
        init_compsep_processing(mpi_info, params)
    assert "Duplicate component names found in CompSep setup" in caplog.text
