from copy import deepcopy

import numpy as np
import pytest
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.compsep.processing import (
    CGSamplingGroupConfig,
    MCMCSamplingGroupConfig,
    PerPixelSamplingGroupConfig,
    _build_conditional_residual,
    _evaluate_chi2,
    _filter_sampling_group_components,
    _read_sampling_groups,
    resolve_sampling_groups,
    _sampling_group_selects_band,
    _validate_sampling_group_dependencies,
    _validate_sampling_group_references,
    _validate_component_lmax,
    init_compsep_processing,
    process_compsep,
)
from commander4.diagnostics.log import SUMMARY
from commander4.mpi.transfer import _get_compsep_sender_id_for_tod_band, _should_send_compsep_result
from commander4.data_models.detector_map import DetectorMap
from commander4.sky.comp_list import CompList
from commander4.sky.sky_model import SkyModel
from commander4.compsep.perpix_solver import solve_compsep_perpix
from commander4.compsep.spectral_index import (
    _discover_spectral_index_groups, SpectralIndexGroup, SpectralIndexSamplingGroup)
from commander4.polarization import get_execution_band_id, get_execution_band_ids


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


def test_iteration_chi2_uses_all_observed_map_samples() -> None:
    map_sky = np.arange(1.0, 13.0).reshape(1, -1)
    map_rms = np.full_like(map_sky, 2.0)
    map_rms[0, -1] = np.inf  # An unobserved pixel has zero inverse-noise weight.
    detector_data = DetectorMap(map_sky, map_rms, nu=30.0, fwhm=60.0, nside=1)
    mpi_info = Bunch(compsep=Bunch(comm=MPI.COMM_SELF, rank=0, master=0))

    class ZeroSky:
        def get_sky_at_nu(self, nu: float, nside: int, pols_required: str,
                          fwhm: float | None = None) -> np.ndarray:
            return np.zeros_like(map_sky)

    chi2, ndof = _evaluate_chi2(mpi_info, detector_data, ZeroSky())

    assert chi2 == pytest.approx(np.sum((map_sky[0, :-1] / 2.0)**2))
    assert ndof == 11


def test_compsep_summary_includes_final_chi2_and_z(monkeypatch, caplog) -> None:
    map_sky = np.arange(1.0, 13.0).reshape(1, -1)
    map_rms = np.full_like(map_sky, 2.0)
    detector_data = DetectorMap(map_sky, map_rms, nu=30.0, fwhm=60.0, nside=1)
    mpi_info = Bunch(compsep=Bunch(comm=MPI.COMM_SELF, rank=0, master=0))
    compsep_state = Bunch(amplitude_groups={}, mcmc_groups={})
    monkeypatch.setattr(
        "commander4.compsep.processing.write_compsep_chain_to_file", lambda *args: None)

    with caplog.at_level(SUMMARY, logger="commander4.compsep.processing"):
        result = process_compsep(
            mpi_info, compsep_state, detector_data, iter=4, chain=2,
            params=Bunch(), comp_list=CompList([]))

    chi2 = np.sum((map_sky / 2.0)**2)
    z_chi2 = (chi2 - map_sky.size) / np.sqrt(2.0 * map_sky.size)
    assert isinstance(result, SkyModel)
    assert "Chain 2, iteration 4 complete" in caplog.text
    assert f"chi2={chi2:.6e}" in caplog.text
    assert f"z={z_chi2:.3f}" in caplog.text


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


def test_group_reference_validation_rejects_unknown_names() -> None:
    comp_list = _make_multi_comp_list()  # CMB, ThermalDust, FreeFree
    params = Bunch(compsep=Bunch(bands=Bunch(
        {"Planck30GHz": Bunch(enabled=True, polarization="IQU")})))

    # Valid references (component name + execution-view band id) pass silently.
    valid = CGSamplingGroupConfig.from_block(
        "g", Bunch(comps=["CMB"], bands=["Planck30GHz_QU"]))
    _validate_sampling_group_references({"g": valid}, comp_list, params)

    with pytest.raises(ValueError, match="unknown component"):
        invalid = CGSamplingGroupConfig.from_block("g", Bunch(comps=["DoesNotExist"]))
        _validate_sampling_group_references({"g": invalid}, comp_list, params)

    with pytest.raises(ValueError, match="unknown band"):
        invalid = CGSamplingGroupConfig.from_block("g", Bunch(bands=["NoSuchBand"]))
        _validate_sampling_group_references({"g": invalid}, comp_list, params)


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


def test_gibbs_chains_have_independent_component_state() -> None:
    chain_components = {1: _make_spectral_comp_list()}
    chain_components[2] = deepcopy(chain_components[1])

    chain_1_sync = [comp for comp in chain_components[1] if comp.comp_name == "Synchrotron"]
    chain_2_sync = [comp for comp in chain_components[2] if comp.comp_name == "Synchrotron"]

    # I and QU are two execution views of one component. They keep their shared configuration
    # inside a chain, while the complete component objects are independent between chains.
    assert chain_1_sync[0].comp_params is chain_1_sync[1].comp_params
    assert chain_2_sync[0].comp_params is chain_2_sync[1].comp_params
    assert chain_1_sync[0].comp_params is not chain_2_sync[0].comp_params
    assert all(comp_1 is not comp_2 for comp_1, comp_2 in zip(chain_components[1],
                                                              chain_components[2]))

    sampler = SpectralIndexSamplingGroup.__new__(SpectralIndexSamplingGroup)
    sampler._groups = _discover_spectral_index_groups(chain_components[1], ["Synchrotron"])
    sampler.apply_state({"sync": -2.8})
    chain_1_sync[0].alms[:] = 4.0 + 2.0j
    chain_1_sync[0].amp_fwhm_rad = 0.25

    assert all(comp.beta == -2.8 for comp in chain_1_sync)
    assert all(comp.beta == -3.1 for comp in chain_2_sync)
    assert np.all(chain_2_sync[0].alms == 0.0)
    assert chain_2_sync[0].amp_fwhm_rad == 0.0


def test_sampling_group_configs_reject_fields_owned_by_other_methods() -> None:
    with pytest.raises(ValueError, match="sample_class"):
        CGSamplingGroupConfig.from_block("cg", Bunch(sample_class="amplitude_sampler_CG"))
    with pytest.raises(ValueError, match="max_iter"):
        PerPixelSamplingGroupConfig.from_block("pixels", Bunch(max_iter=10))
    with pytest.raises(ValueError, match="parameters"):
        MCMCSamplingGroupConfig.from_block("mcmc", Bunch(parameters="gain"))


@pytest.mark.parametrize(
    "preconditioner",
    ["BeamOnlyPreconditioner", "NoiseOnlyPreconditioner", "MixingMatrixPreconditioner"],
)
def test_broken_compsep_preconditioners_cannot_be_selected(preconditioner: str) -> None:
    with pytest.raises(ValueError, match="supported values"):
        CGSamplingGroupConfig.from_block("cg", Bunch(preconditioner=preconditioner))


def test_broken_dense_matrix_debug_path_cannot_be_selected() -> None:
    with pytest.raises(ValueError, match="unsupported dense-matrix"):
        CGSamplingGroupConfig.from_block("cg", Bunch(dense_matrix_debug_mode=True))


def test_per_pixel_solver_rejects_non_diffuse_components_before_mpi_work() -> None:
    with pytest.raises(ValueError, match="does not support object"):
        solve_compsep_perpix(None, None, [object()], double_precision=False)


def test_unimplemented_component_classes_are_rejected_during_construction() -> None:
    component = Bunch(
        enabled=True,
        component_class="TemplateComponent",
        params=Bunch(polarization="I"),
    )
    object.__setattr__(component, "_name", "Template")

    with pytest.raises(ValueError, match="not implemented"):
        CompList.init_from_params(Bunch(Template=component), Bunch())


def test_cg_and_per_pixel_groups_are_mutually_exclusive() -> None:
    params = Bunch(compsep=Bunch(
        cg_sampling_groups=Bunch(cg=Bunch()),
        per_pixel_sampling_groups=Bunch(pixels=Bunch()),
    ))
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_sampling_groups(params)


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
    with pytest.raises(ValueError, match="unknown or disabled amplitude group"):
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


def test_spectral_groups_do_not_depend_on_shared_parameter_object_identity() -> None:
    comp_list = _make_spectral_comp_list()
    sync_views = [comp for comp in comp_list if comp.comp_name == "Synchrotron"]
    sync_views[1].comp_params = deepcopy(sync_views[1].comp_params)

    groups = _discover_spectral_index_groups(comp_list, ["Synchrotron"])

    assert len(groups) == 1
    assert {component.eval_pol for component in groups[0].components} == {"I", "QU"}


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


def test_init_compsep_processing_rejects_duplicate_component_names(monkeypatch) -> None:
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

    with pytest.raises(ValueError, match="Duplicate component names found"):
        init_compsep_processing(mpi_info, params)


# --- component lmax vs band lmax ------------------------------------------------------------
# A component multipole above every band's lmax never meets the data, so the C(l) prior decides it
# on its own. C3 leaves this to the user; we report it, either as a warning (the prior is at least
# tapered there by Cl_prior_l_apod) or as an error (nothing suppresses it).

def _lmax_check_params(comp_lmax: int, band_nside: int = 64, l_apod: int | None = None) -> Bunch:
    """One enabled TOD band and one CMB component, with no band `lmax` set (so 3*nside-1)."""
    comp = _make_component_cfg("I")
    comp.params.lmax = comp_lmax
    comp.params.Cl_prior_amplitude = 1.0e+8
    if l_apod is not None:
        comp.params.Cl_prior_l_apod = l_apod
    object.__setattr__(comp, "_name", "CMB")
    params = Bunch(
        compsep=Bunch(nside=band_nside, float_precision="single",
                      bands=Bunch(BandA=Bunch(enabled=True, get_from="EXP", polarization="I"))),
        experiments=Bunch(EXP=Bunch(bands=Bunch(BandA=Bunch(eval_nside=band_nside)))),
    )
    return CompList.init_from_params(Bunch({"CMB": comp}), params), params


def test_component_lmax_within_the_band_lmax_is_silent(caplog) -> None:
    comp_list, params = _lmax_check_params(comp_lmax=191)   # nside 64 -> band lmax 191
    with caplog.at_level("WARNING"):
        _validate_component_lmax(comp_list, params)
    assert caplog.text == ""


def test_component_lmax_above_every_band_with_no_taper_is_an_error(caplog) -> None:
    comp_list, params = _lmax_check_params(comp_lmax=250)
    with caplog.at_level("WARNING"):
        _validate_component_lmax(comp_list, params)
    assert "ERROR" in caplog.text
    assert "l = 192-250 is constrained by the C(l) prior alone" in caplog.text


def test_component_lmax_above_every_band_but_tapered_is_only_a_warning(caplog) -> None:
    comp_list, params = _lmax_check_params(comp_lmax=250, l_apod=191)
    with caplog.at_level("WARNING"):
        _validate_component_lmax(comp_list, params)
    assert "WARNING" in caplog.text and "ERROR" not in caplog.text
    assert "Cl_prior_l_apod = 191" in caplog.text
