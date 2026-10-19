"""Parameter lookup and iteration behavior of the TOD processing steps.

Each step reads and checks its own settings when it is built, before the Gibbs loop; `process_tod`
then decides from them which steps run in an iteration.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.data_models.pixel_domain import DistributedMap
from commander4.parameters.schema import read_settings
from commander4.tod.data_selection import DataSelection
from commander4.tod.gain import Gain
from commander4.tod.glitches.sampling import Glitches
from commander4.tod.hfi_demodulation import HFIModulation
from commander4.tod.jumps.sampling import Jumps
from commander4.tod.noise.sampling import Noise
from commander4.tod.sidelobe_deconvolve import FarSidelobes


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


# Band BAND of experiment EXP, on a rank without scans: what the steps read when they are built.
EXPERIMENT = SimpleNamespace(
    experiment_name="EXP", band_name="BAND", fsamp=20.0, nu=100.0, nside=64, nscans=0, ndet=1,
    pols="I", hfi_demodulation=False, instrument_filepath=None, band_unit_factor=1.0,
    noise_model=SimpleNamespace(npar=1), iter_detector_scans=lambda: iter(()),
)

# The steps that have settings, by their name in `TODSamples`.
STEP_CLASSES = {"gain": Gain, "noise": Noise, "selection": DataSelection, "jumps": Jumps,
                "hfi": HFIModulation, "glitches": Glitches, "far_sidelobes": FarSidelobes}


def _steps(params: Bunch, experiment: SimpleNamespace = EXPERIMENT) -> SimpleNamespace:
    """Every step with settings, built from `params` as `TODSamples` builds them."""
    my_band = params.experiments.EXP.bands.BAND
    return SimpleNamespace(**{name: step_class(experiment, params, my_band, MPI.COMM_SELF, 1)
                              for name, step_class in STEP_CLASSES.items()})


# --------------------------------------------------------------------------------------
# read_settings
# --------------------------------------------------------------------------------------
def test_read_settings_fills_defaults_and_lets_narrower_scopes_override() -> None:
    params = Bunch(wide=Bunch(block=Bunch(a=2, b="x")), narrow=Bunch(block=Bunch(a=3)))
    defaults = {"a": 1, "b": "y", "c": 0.5}
    settings = read_settings(params, "block", ("narrow", "wide"), defaults)
    assert (settings.a, settings.b, settings.c) == (3, "x", 0.5)
    # A scope that does not exist, or that has no such block, sets nothing.
    for name, scope in (("block", "missing.scope"), ("other", "wide")):
        settings = read_settings(params, name, (scope,), defaults)
        assert (settings.a, settings.b, settings.c) == (1, "y", 0.5)


@pytest.mark.parametrize("value, ok", [
    (True, False), ("false", False), (3, True), (3.0, False),   # default 1: an int
])
def test_read_settings_checks_the_type_of_an_int_setting(value, ok: bool) -> None:
    params = Bunch(scope=Bunch(block=Bunch(a=value)))
    if ok:
        assert read_settings(params, "block", ("scope",), {"a": 1}).a == value
    else:
        with pytest.raises(ValueError, match="'scope.block.a' is .*must be a int"):
            read_settings(params, "block", ("scope",), {"a": 1})


def test_read_settings_accepts_an_int_for_a_float_and_anything_for_none() -> None:
    params = Bunch(scope=Bunch(block=Bunch(a=2, b=[1, 2])))
    settings = read_settings(params, "block", ("scope",), {"a": 1.0, "b": None})
    assert settings.a == 2 and settings.b == [1, 2]


def test_read_settings_rejects_an_unknown_setting() -> None:
    params = Bunch(scope=Bunch(block=Bunch(windw=3)))
    with pytest.raises(ValueError, match="Unknown setting 'scope.block.windw'"):
        read_settings(params, "block", ("scope",), {"window": 10})


# --------------------------------------------------------------------------------------
# The settings of each step
# --------------------------------------------------------------------------------------
def test_documented_defaults_are_set_by_the_steps():
    steps = _steps(_params())

    assert steps.jumps.cfg.window == 10
    assert steps.gain.cfg.abs_gain.calibrate_against == "orbital_dipole"
    assert steps.gain.cfg.rel_gain.calibrate_against == "sky"
    assert steps.gain.cfg.temporal_gain.calibrate_against == "sky"
    assert steps.gain.cfg.abs_gain.downsample_time == 1.0
    assert steps.gain.cfg.abs_gain.gap_fill_method == "wn"
    noise = steps.noise.cfg
    assert noise.sigma0_method == "pairwise"
    assert noise.sigma0_decimation == 1
    assert noise.sample_sigma0
    assert not noise.psd_bin
    assert noise.cg.max_iter == 0
    assert noise.cg.err_tol == 1.0e-4
    assert steps.selection.cfg.chisq_abs_threshold == 1.0e4
    assert steps.selection.cfg.min_good_fraction == 0.1

    # Glitch sampling is a placeholder, so it stays off unless asked for.
    assert not steps.glitches.cfg.enabled
    assert steps.glitches.cfg.from_iter == 1

    # The far-beam defaults are the Commander3 convolution limits and its LevelS scale factor.
    far_beam = steps.far_sidelobes.cfg
    assert not far_beam.enabled
    assert far_beam.lmax == 100
    assert far_beam.mmax == 100
    assert far_beam.epsilon == 1.0e-4
    assert far_beam.beam_norm == 2.0


@pytest.mark.parametrize("block, settings, error", [
    ("abs_gain", {"enabled": True, "calibrate_aganist": "sky"}, "calibrate_aganist"),
    ("jump_detection", {"windw": 3}, "windw"),
    ("data_selection", {"minimum_good_fraction": 0.2}, "minimum_good_fraction"),
    # `npoints` was an early far-beam setting that ducc0 makes unnecessary; it must not survive
    # unnoticed in a parameter file.
    ("far_beam_deconvolution", {"npoints": [100, 80, 30]}, "npoints"),
    ("glitches", {"threshold": 30}, "threshold"),
    ("corr_noise", {"cg": Bunch(maxiter=4)}, "maxiter"),
    # The gains do not end, and nothing optimizes them.
    ("abs_gain", {"enabled": True, "until_iter": 3}, "until_iter"),
    ("abs_gain", {"enabled": True, "optimize": True}, "optimize"),
    # Experiment metadata is not a setting of a step.
    ("abs_gain", {"sampling_rate": 10.0}, "sampling_rate"),
    ("jump_detection", {"jump_bitmask": 4}, "jump_bitmask"),
])
def test_each_step_rejects_unknown_settings_in_its_own_block(block, settings, error) -> None:
    with pytest.raises(ValueError, match=f"Unknown setting.*{error}"):
        _steps(_params(**{block: settings}))


@pytest.mark.parametrize("block", ["abs_gain", "jump_detection", "corr_noise", "data_selection",
                                   "far_beam_deconvolution", "glitches"])
def test_each_step_rejects_a_non_boolean_enabled(block: str) -> None:
    """A quoted "false" is a non-empty string, so without this check the step would run."""
    with pytest.raises(ValueError, match="enabled"):
        _steps(_params(**{block: {"enabled": "false"}}))


@pytest.mark.parametrize("block, settings, error", [
    ("abs_gain", {"gap_fill_method": "bogus"}, "gap_fill_method"),
    ("corr_noise", {"sigma0_method": "bogus"}, "sigma0_method"),
    ("corr_noise", {"sample_psd_params": True}, "sample_psd_params"),
    ("jump_detection", {"enabled": True}, "jump_bitmask"),
    ("data_selection", {"from_iter": 4, "until_iter": 3}, "until_iter"),
])
def test_settings_that_would_be_ignored_fail_when_the_steps_are_built(block, settings,
                                                                      error) -> None:
    """At startup: each of these would otherwise be ignored without a word, or fail late."""
    with pytest.raises(ValueError, match=error):
        _steps(_params(**{block: settings}))


@pytest.mark.parametrize("name, block", [("jumps", "jump_detection"), ("hfi", None),
                                         ("glitches", "glitches"),
                                         ("far_sidelobes", "far_beam_deconvolution")])
def test_a_switched_off_step_holds_only_its_settings_and_writes_nothing(name, block) -> None:
    steps = _steps(_params(**({block: {"enabled": False}} if block else {})))
    step = getattr(steps, name)
    assert list(vars(step)) == ["cfg"] and not step.cfg.enabled
    arrays = step.chain_arrays()
    assert not (arrays.per_scan or arrays.per_band or arrays.per_detector_scan)


def test_enabled_jump_detection_takes_the_experiments_bitmask() -> None:
    params = _params(jump_detection={"enabled": True})
    params.experiments.EXP.jump_bitmask = 4
    assert _steps(params).jumps.cfg.jump_bitmask == 4


def test_each_gain_term_owns_its_settings_and_a_band_overrides_them_one_by_one():
    params = _params(abs_gain={"gap_fill_method": "fallback", "downsample_time": 0.25})
    params.experiments.EXP.bands.BAND.abs_gain = Bunch(downsample_time=0.5)
    gain = _steps(params).gain.cfg
    assert gain.abs_gain.gap_fill_method == "fallback"
    assert gain.abs_gain.downsample_time == 0.5
    assert gain.rel_gain.gap_fill_method == "wn"
    assert gain.rel_gain.downsample_time == 1.0


def test_absent_and_disabled_steps_are_inactive_but_keep_defaults():
    for params in (_params(), _params(abs_gain={"enabled": False})):
        gain = _steps(params).gain.cfg.abs_gain
        assert not gain.enabled
        assert gain.calibrate_against == "orbital_dipole"


# --------------------------------------------------------------------------------------
# When data selection reports and cuts
# --------------------------------------------------------------------------------------
def _selection_status(iteration: int, **blocks) -> tuple[bool, bool]:
    steps = _steps(_params(**blocks))
    return steps.selection.status(iteration, steps.noise.cfg)


def test_data_selection_until_iter_is_inclusive():
    blocks = dict(corr_noise={"enabled": False},
                  data_selection={"enabled": True, "from_iter": 3, "until_iter": 5})
    active = [_selection_status(iteration, **blocks)[1] for iteration in range(1, 8)]
    assert active == [False, False, True, True, True, False, False]


def test_data_selection_waits_for_configured_correlated_noise():
    blocks = dict(corr_noise={"enabled": True, "from_iter": 5},
                  data_selection={"enabled": True, "from_iter": 1})
    assert _selection_status(4, **blocks) == (False, False)
    assert _selection_status(5, **blocks) == (True, True)


def test_data_selection_reports_during_its_own_warmup():
    blocks = dict(corr_noise={"enabled": False},
                  data_selection={"enabled": True, "from_iter": 3})
    assert _selection_status(1, **blocks) == (True, False)


# --------------------------------------------------------------------------------------
# process_tod
# --------------------------------------------------------------------------------------
def _pipeline(monkeypatch: pytest.MonkeyPatch, params: Bunch) -> Bunch:
    """Run the real iteration driver on steps built from `params`, with their sampling, the
    mapmaker and the file writes recorded as calls instead of run."""
    import commander4.tod.processing as processing

    comm = Mock()
    mpi = Bunch(band=Bunch(comm=comm, node_comm=comm, is_master=True), tod=Bunch(comm=comm))
    samples = _steps(params)
    samples.chain = 1
    # The HFI step is switched on, so that its place in the order is checked too (its sampling
    # is recorded below, so it needs no HFI data).
    samples.hfi.cfg.enabled = True
    # The per-iteration resets (over `steps`) are not recorded.
    samples.steps = []
    samples.gather_chain_arrays = Mock(return_value={})
    calls = Mock()
    for name, method in (("jumps", "sample"), ("hfi", "sample"), ("glitches", "sample"),
                         ("gain", "sample_absolute"), ("gain", "sample_relative"),
                         ("gain", "sample_temporal"), ("far_sidelobes", "build"),
                         ("far_sidelobes", "free"), ("noise", "log_summary"),
                         ("selection", "log_summary")):
        recorder = Mock()
        setattr(getattr(samples, name), method, recorder)
        calls.attach_mock(recorder, f"{name}.{method}")
    for name in ("tod2map", "write_band_chain_to_file"):
        operation = Mock(return_value=({}, {})) if name == "tod2map" else Mock()
        calls.attach_mock(operation, name)
        monkeypatch.setattr(processing, name, operation)
    monkeypatch.setattr(processing, "bench_summary", Mock())
    sky = DistributedMap(local=None, full=None)

    def run(iteration: int) -> None:
        processing.process_tod(mpi, EXPERIMENT, samples, sky, params, iteration)

    return Bunch(run=run, calls=calls, samples=samples)


@pytest.mark.parametrize("mapmaker", ["bin", "CG"])
def test_iteration_gates_keep_the_scientific_order(monkeypatch, mapmaker: str) -> None:
    """The inclusive start applies to jumps, glitches, all three gains, and the far beam, which is
    built before every sampling step, since they all have it removed from their TOD."""
    params = _params(mapmaker=mapmaker)
    params.experiments.EXP.jump_bitmask = 1
    for name in ("jump_detection", "glitches", "abs_gain", "rel_gain", "temporal_gain",
                 "far_beam_deconvolution"):
        params.tod_processing[name] = Bunch(enabled=True, from_iter=3)
    pipeline = _pipeline(monkeypatch, params)

    pipeline.run(2)
    assert [call[0] for call in pipeline.calls.mock_calls] == [
        "hfi.sample", "tod2map", "write_band_chain_to_file"]

    pipeline.calls.reset_mock()
    pipeline.run(3)
    assert [call[0] for call in pipeline.calls.mock_calls] == [
        "far_sidelobes.build", "hfi.sample", "jumps.sample", "glitches.sample",
        "gain.sample_absolute", "gain.sample_relative", "gain.sample_temporal",
        "tod2map", "write_band_chain_to_file", "far_sidelobes.free"]


@pytest.mark.parametrize("iteration, report, apply_cuts", [
    (1, False, False), (2, True, False), (3, True, True), (4, True, True), (5, True, False),
])
def test_summary_keeps_noise_wait_and_selection_warmup(
    monkeypatch, iteration: int, report: bool, apply_cuts: bool,
) -> None:
    params = _params(corr_noise={"enabled": True, "from_iter": 2},
                     data_selection={"enabled": True, "from_iter": 3, "until_iter": 4})
    pipeline = _pipeline(monkeypatch, params)
    pipeline.run(iteration)
    # The noise summary reports every iteration that draws correlated noise.
    assert pipeline.samples.noise.log_summary.called == (iteration >= 2)
    summary = pipeline.samples.selection.log_summary
    assert summary.called == report
    if report:
        assert summary.call_args.kwargs["active"] == apply_cuts
