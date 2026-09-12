"""Parameter lookup and iteration behavior of the TOD processing steps."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pixell.bunch import Bunch

from commander4.tod.data_selection import data_selection_status
from commander4.tod.config import GainConfig, MapmakingConfig, CGConfig, CorrelatedNoiseConfig
from commander4.tod.config import JumpDetectionConfig, DataSelectionConfig, FarBeamConfig


def _params(**steps) -> Bunch:
    tod_values = {
        "mapmaker": "bin",
        "cg_mapmaker": Bunch(max_iter=10, err_tol=1.0e-10),
        "abs_gain": Bunch(gap_fill_method="wn", downsample_time=1.0),
        "rel_gain": Bunch(gap_fill_method="wn", downsample_time=1.0),
        "temporal_gain": Bunch(gap_fill_method="wn", downsample_time=1.0),
    }
    for name, values in steps.items():
        if name in ("abs_gain", "rel_gain", "temporal_gain") and isinstance(values, dict):
            merged = dict(tod_values[name])
            merged.update(values)
            tod_values[name] = Bunch(**merged)
        else:
            tod_values[name] = Bunch(**values) if isinstance(values, dict) else values
    return Bunch(
        tod_processing=Bunch(**tod_values),
        experiments=Bunch(EXP=Bunch(bands=Bunch(BAND=Bunch()))),
        compsep=Bunch(common_res_fwhm=0.0),
        resources=Bunch(tod=Bunch(num_threads=1)),
        output=Bunch(chains=Bunch(include=Bunch(
            orbital_dipole_maps=True,
            corr_noise_maps=False,
            sky_model_maps=False,
        ))),
    )


EXPERIMENT = SimpleNamespace(
    experiment_name="EXP", band_name="BAND", fsamp=20.0, nu=100.0, nside=64,
)


def _gain(step_name: str = "abs_gain", default: str = "orbital_dipole", **steps) -> GainConfig:
    params = _params(**steps)
    return GainConfig.from_params(params.tod_processing,
                              params.experiments.EXP.bands.BAND, step_name, default)


def test_documented_defaults_are_owned_by_the_config_classes():
    params = _params()
    jump = JumpDetectionConfig.from_params(params.tod_processing, params.experiments.EXP)
    absolute_gain = _gain()
    relative_gain = _gain("rel_gain", "sky")
    temporal_gain = _gain("temporal_gain", "sky")
    correlated_noise = CorrelatedNoiseConfig.from_params(params.tod_processing)
    data_selection = DataSelectionConfig.from_params(params.tod_processing)

    assert jump.window == 10
    assert absolute_gain.calibrate_against == "orbital_dipole"
    assert relative_gain.calibrate_against == "sky"
    assert temporal_gain.calibrate_against == "sky"
    assert absolute_gain.downsample_time == 1.0
    assert absolute_gain.gap_fill_method == "wn"
    assert correlated_noise.sigma0_method == "pairwise"
    assert correlated_noise.sigma0_decimation == 1
    assert correlated_noise.sample_sigma0
    assert not correlated_noise.psd_bin
    assert correlated_noise.cg.max_iter == 0
    assert correlated_noise.cg.err_tol == 1.0e-4
    assert data_selection.chisq_abs_threshold == 1.0e4
    assert data_selection.min_good_fraction == 0.1

    # The far-beam defaults are the Commander3 convolution limits and its LevelS scale factor.
    far_beam = FarBeamConfig.from_params(params.tod_processing)
    assert not far_beam.enabled
    assert far_beam.lmax == 100
    assert far_beam.mmax == 100
    assert far_beam.epsilon == 1.0e-4
    assert far_beam.beam_norm == 2.0


def test_each_config_rejects_unknown_fields_in_its_own_block():
    with pytest.raises(TypeError, match="calibrate_aganist"):
        _gain(abs_gain={"enabled": True, "calibrate_aganist": "sky"})

    params = _params(jump_detection={"windw": 3})
    with pytest.raises(TypeError, match="windw"):
        JumpDetectionConfig.from_params(params.tod_processing, params.experiments.EXP)

    params = _params(data_selection={"minimum_good_fraction": 0.2})
    with pytest.raises(TypeError, match="minimum_good_fraction"):
        DataSelectionConfig.from_params(params.tod_processing)

    # `npoints` was an early far-beam setting that ducc0 makes unnecessary; it must not survive
    # unnoticed in a parameter file.
    params = _params(far_beam_deconvolution={"npoints": [100, 80, 30]})
    with pytest.raises(TypeError, match="npoints"):
        FarBeamConfig.from_params(params.tod_processing)


def test_far_beam_limits_are_validated():
    params = _params(far_beam_deconvolution={"enabled": True, "lmax": 50, "mmax": 80})
    with pytest.raises(ValueError, match="cannot exceed lmax"):
        FarBeamConfig.from_params(params.tod_processing)

    params = _params(far_beam_deconvolution={"lmax": -1})
    with pytest.raises(ValueError, match="non-negative integer"):
        FarBeamConfig.from_params(params.tod_processing)

    # A band-limited beam is legitimate: lmax = mmax = 0 keeps only the monopole.
    params = _params(far_beam_deconvolution={"enabled": True, "lmax": 0, "mmax": 0})
    assert FarBeamConfig.from_params(params.tod_processing).enabled


def test_nested_cg_blocks_validate_their_own_fields():
    params = _params(corr_noise={"cg": Bunch(maxiter=4)})
    with pytest.raises(TypeError, match="maxiter"):
        CorrelatedNoiseConfig.from_params(params.tod_processing)

    params = _params()
    params.tod_processing.cg_mapmaker.maxiter = 4
    with pytest.raises(TypeError, match="maxiter"):
        MapmakingConfig.from_params(params, EXPERIMENT)


def test_mapmaking_config_resolves_resources_and_output_selection():
    params = _params()
    del params.compsep["common_res_fwhm"]
    config = MapmakingConfig.from_params(params, EXPERIMENT)
    assert config.mapmaker == "bin"
    assert config.num_threads == 1
    assert config.sparse_maps == MapmakingConfig.sparse_maps
    assert config.common_res_fwhm == MapmakingConfig.common_res_fwhm
    assert config.include_orbital_dipole_maps
    assert not config.include_corr_noise_maps
    assert not config.include_sky_model_maps
    assert config.band_lmax == 3*EXPERIMENT.nside - 1


def test_sparse_maps_requires_a_boolean():
    params = _params()
    params.experiments.EXP.sparse_maps = "false"
    with pytest.raises(ValueError, match="must have type bool"):
        MapmakingConfig.from_params(params, EXPERIMENT)


def test_mapmaking_config_takes_band_lmax_from_band_then_experiment():
    params = _params()
    assert MapmakingConfig.from_params(params, EXPERIMENT).band_lmax == 3*EXPERIMENT.nside - 1
    params.experiments.EXP.lmax = 100
    assert MapmakingConfig.from_params(params, EXPERIMENT).band_lmax == 100
    params.experiments.EXP.bands.BAND.lmax = 150
    assert MapmakingConfig.from_params(params, EXPERIMENT).band_lmax == 150


def test_each_gain_step_owns_its_gap_fill_and_downsampling():
    abs_gain_block = {"gap_fill_method": "fallback", "downsample_time": 0.25}
    absolute_gain = _gain("abs_gain", "orbital_dipole", abs_gain=abs_gain_block)
    relative_gain = _gain("rel_gain", "sky", abs_gain=abs_gain_block)
    assert absolute_gain.gap_fill_method == "fallback"
    assert absolute_gain.downsample_time == 0.25
    assert relative_gain.gap_fill_method == "wn"
    assert relative_gain.downsample_time == 1.0


def test_until_iter_and_optimize_are_rejected_where_unsupported():
    params = _params(data_selection={"enabled": True, "until_iter": 3})
    DataSelectionConfig.from_params(params.tod_processing)

    with pytest.raises(TypeError, match="until_iter"):
        _gain(abs_gain={"enabled": True, "until_iter": 3})
    with pytest.raises(TypeError, match="optimize"):
        _gain(abs_gain={"enabled": True, "optimize": True})


def test_absent_and_disabled_steps_are_inactive_but_keep_defaults():
    for config in (_gain(), _gain(abs_gain={"enabled": False})):
        assert not config.enabled
        assert config.calibrate_against == "orbital_dipole"




def test_data_selection_until_iter_is_inclusive():
    params = _params(
        corr_noise={"enabled": False},
        data_selection={"enabled": True, "from_iter": 3, "until_iter": 5},
    )
    correlated_noise = CorrelatedNoiseConfig.from_params(params.tod_processing)
    data_selection = DataSelectionConfig.from_params(params.tod_processing)
    active = []
    for iteration in range(1, 8):
        active.append(data_selection_status(iteration, data_selection, correlated_noise)[1])
    assert active == [False, False, True, True, True, False, False]


def test_data_selection_waits_for_configured_correlated_noise():
    params = _params(
        corr_noise={"enabled": True, "from_iter": 5},
        data_selection={"enabled": True, "from_iter": 1},
    )
    correlated_noise = CorrelatedNoiseConfig.from_params(params.tod_processing)
    data_selection = DataSelectionConfig.from_params(params.tod_processing)
    assert data_selection_status(4, data_selection, correlated_noise) == (False, False)
    assert data_selection_status(5, data_selection, correlated_noise) == (True, True)


def test_data_selection_reports_during_its_own_warmup():
    params = _params(
        corr_noise={"enabled": False},
        data_selection={"enabled": True, "from_iter": 3},
    )
    correlated_noise = CorrelatedNoiseConfig.from_params(params.tod_processing)
    data_selection = DataSelectionConfig.from_params(params.tod_processing)
    assert data_selection_status(1, data_selection, correlated_noise) == (True, False)


def test_psd_parameter_sampling_requires_correlated_noise_enabled():
    params = _params(corr_noise={"sample_psd_params": True})
    with pytest.raises(ValueError, match="sample_psd_params"):
        CorrelatedNoiseConfig.from_params(params.tod_processing)


def test_enabled_jump_detection_requires_an_experiment_bitmask():
    params = _params(jump_detection={"enabled": True})
    with pytest.raises(ValueError, match="jump_bitmask"):
        JumpDetectionConfig.from_params(params.tod_processing, params.experiments.EXP)


@pytest.mark.parametrize("config_class, values, error", [
    (GainConfig, {"mask_threshold": 1.0}, "mask_threshold"),
    (CGConfig, {"max_iter": -1}, "max_iter"),
    (CorrelatedNoiseConfig, {"nomono": True, "onlymono": True}, "cannot both be true"),
    (JumpDetectionConfig, {"enabled": True}, "jump_bitmask"),
    (DataSelectionConfig, {"from_iter": 4, "until_iter": 3}, "until_iter"),
    (FarBeamConfig, {"lmax": 50, "mmax": 80}, "cannot exceed lmax"),
    (MapmakingConfig, {"mapmaker": "invalid", "num_threads": 1,
                       "include_orbital_dipole_maps": False, "include_corr_noise_maps": False,
                       "include_sky_model_maps": False}, "mapmaker"),
])
def test_direct_construction_validates_scientific_settings(
    config_class: type, values: dict, error: str,
) -> None:
    """Settings built in Python must satisfy the same checks as parameter-file settings."""
    with pytest.raises(ValueError, match=error):
        config_class(**values)


@pytest.mark.parametrize("config_class", [
    GainConfig, JumpDetectionConfig, CorrelatedNoiseConfig, DataSelectionConfig, FarBeamConfig,
])
def test_direct_construction_rejects_a_non_boolean_enabled(config_class: type) -> None:
    """A quoted "false" is a non-empty string, so without this check the step would run."""
    with pytest.raises(ValueError, match="enabled"):
        config_class(enabled="false")


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> Bunch:
    """Run the real iteration driver with numerical steps and file writes recorded as calls."""
    import commander4.tod.processing as processing

    comm = Mock()
    mpi = Bunch(band=Bunch(comm=comm, node_comm=comm, is_master=True), tod=Bunch(comm=comm))
    experiment = SimpleNamespace(**vars(EXPERIMENT))
    samples = SimpleNamespace(hfi_demodulation=True, chisq_z=np.zeros((1, 1)),
                              good_fraction=np.zeros((1, 1)), residual_tods=None,
                              gather_chain_arrays=Mock(return_value={}),
                              band_unit_factor=1.0, band_unit="uK_RJ")
    calls = Mock()
    for name in ("sample_jump_detection", "sample_hfi_baselines", "sample_absolute_gain",
                 "sample_relative_gain", "sample_temporal_gain_variations"):
        sampler = Mock(return_value=samples)
        calls.attach_mock(sampler, name)
        monkeypatch.setattr(processing, name, sampler)
    for name in ("tod2map_bin", "tod2map_CG"):
        mapmaker = Mock(return_value=({}, {}))
        calls.attach_mock(mapmaker, name)
        monkeypatch.setattr(processing, name, mapmaker)
    for name in ("FarBeamProjector", "write_band_chain_to_file", "log_dataselect_summary"):
        operation = Mock()
        calls.attach_mock(operation, name)
        monkeypatch.setattr(processing, name, operation)
    monkeypatch.setattr(processing, "bench_summary", Mock())
    return Bunch(mpi=mpi, experiment=experiment, samples=samples, calls=calls,
                 run=processing.process_tod)


def test_iteration_gates_keep_the_scientific_order(pipeline: Bunch) -> None:
    """The inclusive start applies to jumps, all three gains, and the far beam."""
    params = _params()
    params.experiments.EXP.jump_bitmask = 1
    for name in ("jump_detection", "abs_gain", "rel_gain", "temporal_gain",
                 "far_beam_deconvolution"):
        params.tod_processing[name] = Bunch(enabled=True, from_iter=3)

    pipeline.run(pipeline.mpi, pipeline.experiment, pipeline.samples, None, params, 1, 2)
    assert [call[0] for call in pipeline.calls.mock_calls] == [
        "sample_hfi_baselines", "tod2map_bin", "write_band_chain_to_file"]

    pipeline.calls.reset_mock()
    pipeline.run(pipeline.mpi, pipeline.experiment, pipeline.samples, None, params, 2, 3)
    assert [call[0] for call in pipeline.calls.mock_calls] == [
        "sample_jump_detection", "sample_hfi_baselines", "sample_absolute_gain",
        "sample_relative_gain", "sample_temporal_gain_variations", "FarBeamProjector",
        "tod2map_bin", "write_band_chain_to_file", "FarBeamProjector().free"]


def test_settings_are_read_again_for_each_iteration(pipeline: Bunch) -> None:
    params = _params(abs_gain={"enabled": True})
    pipeline.run(pipeline.mpi, pipeline.experiment, pipeline.samples, None, params, 1, 1)
    first = pipeline.calls.sample_absolute_gain.call_args.args[4]
    params.experiments.EXP.bands.BAND.abs_gain = Bunch(downsample_time=0.25)
    pipeline.run(pipeline.mpi, pipeline.experiment, pipeline.samples, None, params, 2, 2)
    second = pipeline.calls.sample_absolute_gain.call_args.args[4]
    assert first.downsample_time == 1.0
    assert second.downsample_time == 0.25
    assert params.tod_processing.abs_gain.downsample_time == 1.0


def test_invalid_later_settings_fail_before_sampling(pipeline: Bunch) -> None:
    params = _params(abs_gain={"enabled": True}, data_selection={"min_good_fraction": 2.0})
    with pytest.raises(ValueError, match="min_good_fraction"):
        pipeline.run(pipeline.mpi, pipeline.experiment, pipeline.samples, None, params, 1, 1)
    assert pipeline.calls.mock_calls == []


def test_cg_rejects_far_beam_before_sampling_when_it_becomes_active(pipeline: Bunch) -> None:
    params = _params(mapmaker="CG", far_beam_deconvolution={"enabled": True, "from_iter": 3})
    pipeline.run(pipeline.mpi, pipeline.experiment, pipeline.samples, None, params, 1, 2)
    pipeline.calls.tod2map_CG.assert_called_once()
    pipeline.calls.reset_mock()
    with pytest.raises(ValueError, match="Far-beam"):
        pipeline.run(pipeline.mpi, pipeline.experiment, pipeline.samples, None, params, 1, 3)
    assert pipeline.calls.mock_calls == []


@pytest.mark.parametrize("iteration, report, apply_cuts", [
    (1, False, False), (2, True, False), (3, True, True), (4, True, True), (5, True, False),
])
def test_summary_keeps_noise_wait_and_selection_warmup(
    pipeline: Bunch, iteration: int, report: bool, apply_cuts: bool,
) -> None:
    params = _params(corr_noise={"enabled": True, "from_iter": 2},
                     data_selection={"enabled": True, "from_iter": 3, "until_iter": 4})
    pipeline.run(pipeline.mpi, pipeline.experiment, pipeline.samples, None, params, 1, iteration)
    summary = pipeline.calls.log_dataselect_summary
    assert summary.called == report
    if report:
        assert summary.call_args.kwargs["active"] == apply_cuts


def test_mapmaker_precedence_and_required_cg_controls() -> None:
    params = _params()
    del params.tod_processing["cg_mapmaker"]
    assert MapmakingConfig.from_params(params, EXPERIMENT).mapmaker == "bin"
    params.experiments.EXP.mapmaker = "CG"
    with pytest.raises(ValueError, match="max_iter and err_tol"):
        MapmakingConfig.from_params(params, EXPERIMENT)
    params.tod_processing.cg_mapmaker = Bunch(max_iter=10)
    with pytest.raises(ValueError, match="max_iter and err_tol"):
        MapmakingConfig.from_params(params, EXPERIMENT)
    params.tod_processing.cg_mapmaker.err_tol = 1e-6
    assert MapmakingConfig.from_params(params, EXPERIMENT).mapmaker == "CG"
    params.experiments.EXP.bands.BAND.mapmaker = "bin"
    assert MapmakingConfig.from_params(params, EXPERIMENT).mapmaker == "bin"


def test_parameter_blocks_cannot_override_experiment_metadata() -> None:
    with pytest.raises(TypeError, match="sampling_rate"):
        _gain(abs_gain={"sampling_rate": 10.0})
    params = _params(jump_detection={"jump_bitmask": 4})
    with pytest.raises(TypeError, match="jump_bitmask"):
        JumpDetectionConfig.from_params(params.tod_processing, params.experiments.EXP)
