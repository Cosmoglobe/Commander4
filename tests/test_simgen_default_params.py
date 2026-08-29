"""The exhaustive simgen parameter reference stays complete and constructible."""
from pathlib import Path

from simgen.config import load_params
from simgen.instrument import build_bands
from simgen.modifiers import build_modifiers
from simgen.noise import make_noise_model
from simgen.pointing import POINTING_STRATEGIES, make_pointing
from simgen.sky import _COMPONENT_BUILDERS, build_components
from simgen.transfer import MultiPole, SinglePole


PARAMETER_FILE = Path(__file__).parents[1] / "simgen" / "params" / "param_default.yml"


def test_default_parameter_file_covers_every_registered_option() -> None:
    params, _ = load_params(str(PARAMETER_FILE))

    component_classes = {component.component_class for _, component in params.components.items()}
    assert component_classes == set(_COMPONENT_BUILDERS)

    band_polarizations = {band.polarization for _, band in params.experiments.SimSat.bands.items()}
    assert band_polarizations == {"I", "QU", "IQU"}

    parameter_text = PARAMETER_FILE.read_text()
    for strategy_name in POINTING_STRATEGIES:
        assert f'strategy: "{strategy_name}"' in parameter_text


def test_default_parameter_file_builds_every_active_model() -> None:
    params, _ = load_params(str(PARAMETER_FILE))
    bands = build_bands(params)
    components = build_components(params)

    assert [band.name for band in bands] == ["Band30GHz", "Band100GHz"]
    assert len(components) == 5
    assert type(make_pointing(params.simulation.pointing, bands[0].fsamp)).__name__ == "PlanckScan"
    assert [type(model).__name__ for model in [make_noise_model(band) for band in bands]] == [
        "OofNoise", "WhiteNoise"]
    assert [type(modifier).__name__ for modifier in build_modifiers(params)] == ["CrossTalk"]

    transfer_types = {type(detector.transfer) for band in bands for detector in band.detectors}
    assert {type(None), SinglePole, MultiPole} <= transfer_types
