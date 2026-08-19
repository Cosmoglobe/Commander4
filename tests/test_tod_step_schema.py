"""Validation, resolution, and iteration gates for the TOD-processing config classes."""

import inspect
from types import SimpleNamespace

import pytest
from pixell.bunch import Bunch

from commander4.tod_processing import (
    CorrelatedNoiseConfig,
    DataSelectionConfig,
    GainConfig,
    JumpDetectionConfig,
    MapmakingConfig,
    sample_absolute_gain,
    sample_jump_detection,
    sample_relative_gain,
    sample_temporal_gain_variations,
    tod2map_CG,
    tod2map_bin,
)


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


def _gain(step_name: str = "abs_gain", default: str = "orbital_dipole",
          iteration: int = 1, **steps) -> GainConfig:
    params = _params(**steps)
    return GainConfig.from_params(
        params, EXPERIMENT, step_name, default, iteration, is_master=True,
    )


def test_each_execution_function_accepts_its_config_directly():
    jump_parameters = inspect.signature(sample_jump_detection).parameters
    assert "config" in jump_parameters and "params" not in jump_parameters
    for sampler in (sample_absolute_gain, sample_relative_gain,
                    sample_temporal_gain_variations):
        parameters = inspect.signature(sampler).parameters
        assert "config" in parameters and "params" not in parameters
    for mapmaker in (tod2map_CG, tod2map_bin):
        parameters = inspect.signature(mapmaker).parameters
        assert "mapmaking" in parameters and "params" not in parameters


def test_documented_defaults_are_owned_by_the_config_classes():
    params = _params()
    jump = JumpDetectionConfig.from_params(params, EXPERIMENT)
    absolute_gain = _gain()
    relative_gain = _gain("rel_gain", "sky")
    temporal_gain = _gain("temporal_gain", "sky")
    correlated_noise = CorrelatedNoiseConfig.from_params(params, is_master=True)
    data_selection = DataSelectionConfig.from_params(params)

    assert jump.window == 10
    assert absolute_gain.calibrate_against == "orbital_dipole"
    assert relative_gain.calibrate_against == "sky"
    assert temporal_gain.calibrate_against == "sky"
    assert absolute_gain.downsample_factor == 20
    assert absolute_gain.gap_fill_method == "wn"
    assert correlated_noise.sigma0_method == "pairwise"
    assert correlated_noise.sigma0_decimation == 1
    assert correlated_noise.sample_sigma0
    assert not correlated_noise.psd_bin
    assert correlated_noise.cg.max_iter == 0
    assert correlated_noise.cg.err_tol == 1.0e-4
    assert data_selection.chisq_abs_threshold == 1.0e4
    assert data_selection.min_good_fraction == 0.1


def test_each_config_rejects_unknown_fields_in_its_own_block():
    with pytest.raises(ValueError, match="calibrate_aganist"):
        _gain(abs_gain={"enabled": True, "calibrate_aganist": "sky"})

    params = _params(jump_detection={"windw": 3})
    with pytest.raises(ValueError, match="windw"):
        JumpDetectionConfig.from_params(params, EXPERIMENT)

    params = _params(data_selection={"minimum_good_fraction": 0.2})
    with pytest.raises(ValueError, match="minimum_good_fraction"):
        DataSelectionConfig.from_params(params)


def test_nested_cg_blocks_validate_their_own_fields():
    params = _params(corr_noise={"cg": Bunch(maxiter=4)})
    with pytest.raises(ValueError, match="corr_noise.cg"):
        CorrelatedNoiseConfig.from_params(params, is_master=True)

    params = _params()
    params.tod_processing.cg_mapmaker.maxiter = 4
    with pytest.raises(ValueError, match="cg_mapmaker"):
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


def test_mapmaking_config_takes_band_lmax_from_band_then_experiment():
    params = _params()
    assert MapmakingConfig.from_params(params, EXPERIMENT).band_lmax == 3*EXPERIMENT.nside - 1
    params.experiments.EXP.lmax = 100
    assert MapmakingConfig.from_params(params, EXPERIMENT).band_lmax == 100
    params.experiments.EXP.bands.BAND.lmax = 150
    assert MapmakingConfig.from_params(params, EXPERIMENT).band_lmax == 150


def test_each_gain_step_owns_its_gap_fill_and_downsampling():
    params = _params(
        abs_gain={"gap_fill_method": "fallback", "downsample_time": 0.25},
    )
    absolute_gain = GainConfig.from_params(
        params, EXPERIMENT, "abs_gain", "orbital_dipole", 1, False,
    )
    relative_gain = GainConfig.from_params(
        params, EXPERIMENT, "rel_gain", "sky", 1, False,
    )
    assert absolute_gain.gap_fill_method == "fallback"
    assert absolute_gain.downsample_factor == 5
    assert relative_gain.gap_fill_method == "wn"
    assert relative_gain.downsample_factor == 20


def test_until_iter_and_optimize_are_rejected_where_unsupported():
    params = _params(data_selection={"enabled": True, "until_iter": 3})
    DataSelectionConfig.from_params(params)

    with pytest.raises(ValueError, match="until_iter"):
        _gain(abs_gain={"enabled": True, "until_iter": 3})
    with pytest.raises(ValueError, match="optimize"):
        _gain(abs_gain={"enabled": True, "optimize": True})


def test_absent_and_disabled_steps_are_inactive_but_keep_defaults():
    for config in (_gain(), _gain(abs_gain={"enabled": False})):
        assert not config.enabled
        assert not config.is_active(1)
        assert config.calibrate_against == "orbital_dipole"


def test_from_iter_is_inclusive():
    config = _gain(abs_gain={
        "enabled": True, "from_iter": 3, "calibrate_against": "sky",
    })
    assert not config.is_active(2)
    assert config.is_active(3)
    assert config.calibrate_against == "sky"


def test_data_selection_until_iter_is_inclusive():
    params = _params(
        corr_noise={"enabled": False},
        data_selection={"enabled": True, "from_iter": 3, "until_iter": 5},
    )
    correlated_noise = CorrelatedNoiseConfig.from_params(params, is_master=True)
    data_selection = DataSelectionConfig.from_params(params)
    active = []
    for iteration in range(1, 8):
        active.append(data_selection.cuts_are_active(iteration, correlated_noise))
    assert active == [False, False, True, True, True, False, False]


def test_data_selection_waits_for_configured_correlated_noise():
    params = _params(
        corr_noise={"enabled": True, "from_iter": 5},
        data_selection={"enabled": True, "from_iter": 1},
    )
    correlated_noise = CorrelatedNoiseConfig.from_params(params, is_master=True)
    data_selection = DataSelectionConfig.from_params(params)
    assert not data_selection.is_available(4, correlated_noise)
    assert not data_selection.cuts_are_active(4, correlated_noise)
    assert data_selection.is_available(5, correlated_noise)
    assert data_selection.cuts_are_active(5, correlated_noise)


def test_data_selection_reports_during_its_own_warmup():
    params = _params(
        corr_noise={"enabled": False},
        data_selection={"enabled": True, "from_iter": 3},
    )
    correlated_noise = CorrelatedNoiseConfig.from_params(params, is_master=True)
    data_selection = DataSelectionConfig.from_params(params)
    assert data_selection.is_available(1, correlated_noise)
    assert not data_selection.cuts_are_active(1, correlated_noise)


def test_psd_parameter_sampling_requires_correlated_noise_enabled():
    params = _params(corr_noise={"sample_psd_params": True})
    with pytest.raises(ValueError, match="sample_psd_params"):
        CorrelatedNoiseConfig.from_params(params, is_master=True)


def test_enabled_jump_detection_requires_an_experiment_bitmask():
    params = _params(jump_detection={"enabled": True})
    with pytest.raises(ValueError, match="jump_bitmask"):
        JumpDetectionConfig.from_params(params, EXPERIMENT)
