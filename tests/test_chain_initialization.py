"""Saved TOD and sky state must retain their physical meaning when loaded into a new run."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.cli import gibbs_schedule
from commander4.data_models.jump_corrections import JumpCorrection
from commander4.data_models.tod_samples import TODSamples
from commander4.file_io import paths
from commander4.file_io.chain_reader import resolve_run_start
from commander4.file_io.chain_writer import write_band_chain_to_file, write_compsep_chain_to_file
from commander4.parameters.initialization import RunStart
from commander4.sky.comp_list import CompList
from commander4.sky.sky_model import SkyModel, build_initial_sky_model
from commander4.tod.noise.psd import NoisePSDOof
from commander4.tod.step_config import StepConfig


def _case(run_dir: Path, scan_ids: tuple[int, ...] = (11, 22, 33),
          detectors: tuple[str, ...] = ("a", "b"), hfi: bool = False,
          band_unit: str = "uK_CMB") -> tuple[Bunch, SimpleNamespace, Bunch]:
    """Small detector-scan inputs for the real TOD and component constructors."""
    band = Bunch(enabled=True, num_tasks=1, band_unit=band_unit, detectors=Bunch())
    for name in detectors:
        band.detectors[name] = Bunch(gain=2.0)
    scans = []
    detector_scans = []
    for iscan, scan_id in enumerate(scan_ids):
        scans.append(SimpleNamespace(scan_id=scan_id, start_time=float(scan_id)))
        for idet, name in enumerate(detectors):
            detector_scans.append((iscan, SimpleNamespace(
                name=name, det_idx_fullband=idet, orbital_velocity_m_per_s=np.zeros(3),
                init_scalars=None)))
    noise_model = NoisePSDOof.__new__(NoisePSDOof)
    noise_model.params = np.array([1.0, 0.1, -1.0])
    data = SimpleNamespace(
        experiment_name="EXP", band_name="B", ndet=len(detectors), nscans=len(scans),
        noise_model=noise_model, scan_idx_start=0, scan_idx_stop=len(scans), nu=100.0,
        scans=scans, hfi_demodulation=hfi, iter_detector_scans=lambda: iter(detector_scans))
    dust = Bunch(enabled=True, component_class="ThermalDust", params=Bunch(
        shortname="dust", polarization="IQU", lmax=2, spatially_varying_MM=False,
        Cl_prior_amplitude=None, beta=1.54, T=20.0, nu_ref=[857.0, 353.0]))
    object.__setattr__(dust, "_name", "dust")
    params = Bunch(
        gibbs=Bunch(num_iterations=2),
        experiments=Bunch(EXP=Bunch(enabled=True, bands=Bunch(B=band))),
        components=Bunch(dust=dust), compsep=Bunch(double_precision=True, enabled=False),
        output=Bunch(dir=str(run_dir), chains=Bunch(write=[1, 2], maps_nside="native",
                                                  include=Bunch())),
        parameter_file_as_string="test parameters")
    return params, data, band


def _save(run_dir: Path, hfi: bool = False, chain: int = 1,
          iteration: int = 7) -> tuple[TODSamples, CompList]:
    """Write nontrivial sampled quantities using the production gather and chain writers."""
    params, data, band = _case(run_dir, hfi=hfi)
    paths.create_output_dirs(params.output)
    samples = _load_tod(data, params, band, MPI.COMM_SELF, chain)
    samples.abs_gain = 2.0 + 10.0 * (chain - 1)
    samples.rel_gain[:] = [-0.2, 0.2]
    samples.temporal_gain[:] = np.arange(6).reshape(3, 2) * 0.03
    samples.noise_params[:, :, 0] = np.arange(6).reshape(3, 2) + 1.0
    samples.accept[1, 0] = False
    for iscan in range(3):
        for idet in range(2):
            jump = JumpCorrection([iscan + idet + 2], [-iscan - idet - 0.5])
            samples.jumps.set(iscan, idet, jump)
    if hfi:
        samples.modulation_phase[:] = [[1, -1], [-1, 1], [1, 1]]
        samples.baselines[:] = np.arange(12).reshape(3, 2, 2) * 0.2
    write_band_chain_to_file(params, chain, iteration, "EXP", "B",
                             samples.gather_chain_arrays(iteration), {},
                             band_unit=samples.band_unit,
                             band_unit_factor=samples.band_unit_factor)
    sky = CompList.init_from_params(params.components, params)
    for comp in sky:
        comp.alms[:] = np.arange(comp.alms.size).reshape(comp.alms.shape) + 2.0
        comp.beta = 1.2 + 0.1 * (chain - 1)
        comp.T = 24.0
        comp.amp_fwhm_rad = np.radians(1.0)
    write_compsep_chain_to_file(sky.joined(), params, chain, iteration)
    return samples, sky


def _load_tod(data, params, band, comm, chain):
    start = RunStart.from_gibbs(params.gibbs)
    filename = start.tod_file(data.experiment_name, data.band_name, chain)
    return TODSamples(data, params, band, comm, chain, filename)


def _initialize(params: Bunch, run_dir: Path, *, tod: bool = True, sky: bool = True) -> None:
    load = []
    if tod:
        load.append("tod")
    if sky:
        load.extend(["amplitudes", "spectral_parameters"])
    if load:
        params.gibbs.start = Bunch(source=str(run_dir), chain=1, iteration=7, load=load)
    else:
        params.gibbs.start = Bunch()
    if Path(params.output.dir) == run_dir:
        params.output.dir = str(run_dir / "new_output")


@pytest.mark.parametrize("load_tod, load_sky", [(False, False), (True, False),
                                              (False, True), (True, True)])
def test_tod_and_sky_can_be_initialized_independently(tmp_path, load_tod, load_sky):
    source_dir = tmp_path / "source"
    expected_tod, expected_sky = _save(source_dir)
    params, data, band = _case(tmp_path / "output")
    _initialize(params, source_dir, tod=load_tod, sky=load_sky)
    resolve_run_start(params)
    for chain in (1, 2):
        restored = _load_tod(data, params, band, MPI.COMM_SELF, chain)
        if load_tod:
            np.testing.assert_allclose(restored.gain_all(), expected_tod.gain_all())
            np.testing.assert_array_equal(restored.noise_params, expected_tod.noise_params)
            np.testing.assert_array_equal(restored.accept, expected_tod.accept)
        else:
            np.testing.assert_allclose(restored.gain_all(), 2.0 * restored.band_unit_factor)
    restored_sky = build_initial_sky_model(params, RunStart.from_gibbs(params.gibbs))._components
    for actual, expected in zip(restored_sky, expected_sky):
        if load_sky:
            np.testing.assert_array_equal(actual.alms, expected.alms)
            assert actual.beta == expected.beta
            assert actual.T == expected.T
            assert actual.amp_fwhm_rad == pytest.approx(expected.amp_fwhm_rad)
            assert actual.get_sed(100.0) == expected.get_sed(100.0)
        else:
            assert np.all(actual.alms == 0.0)
            assert actual.beta == 1.54
    if load_sky:
        actual_map = SkyModel(restored_sky).get_sky_at_nu(100.0, 2, "IQU", np.radians(2.0))
        expected_map = SkyModel(expected_sky).get_sky_at_nu(100.0, 2, "IQU", np.radians(2.0))
        np.testing.assert_array_equal(actual_map, expected_map)


@pytest.mark.parametrize("scan_ids", [(33, 11), (), (22,)])
@pytest.mark.parametrize("detectors", [("b", "a"), ("b",)])
def test_tod_reorders_and_subsets_scans_and_detectors_including_hfi(tmp_path, scan_ids, detectors):
    expected, _ = _save(tmp_path, hfi=True)
    params, data, band = _case(tmp_path, scan_ids, detectors, hfi=True, band_unit="uK_RJ")
    _initialize(params, tmp_path, sky=False)
    restored = _load_tod(data, params, band, MPI.COMM_SELF, 2)
    assert restored.modulation_phase_initialized
    assert restored.rel_gain.sum() == pytest.approx(0.0)
    assert restored.noise_params.shape == (len(scan_ids), len(detectors), 3)
    for iscan, scan_id in enumerate(scan_ids):
        source_scan = list(expected.scan_ids).index(scan_id)
        for idet, detector in enumerate(detectors):
            source_detector = expected.det_names.index(detector)
            assert restored.gain(iscan, idet) == pytest.approx(
                expected.gain(source_scan, source_detector))
            for name in ("noise_params", "accept", "modulation_phase", "baselines"):
                np.testing.assert_array_equal(getattr(restored, name)[iscan, idet],
                                              getattr(expected, name)[source_scan, source_detector])
            actual_jump = restored.jumps.get(iscan, idet)
            expected_jump = expected.jumps.get(source_scan, source_detector)
            np.testing.assert_array_equal(actual_jump.locations, expected_jump.locations)
            np.testing.assert_array_equal(actual_jump.offsets, expected_jump.offsets)


def test_tod_resolves_a_different_source_file_for_each_band(tmp_path):
    saved, _ = _save(tmp_path)
    saved.abs_gain = 9.0
    write_band_chain_to_file(saved.params, 1, 7, "EXP", "Other", saved.gather_chain_arrays(7), {},
                             band_unit=saved.band_unit, band_unit_factor=saved.band_unit_factor)
    params, data, band = _case(tmp_path)
    data.band_name = "Other"
    _initialize(params, tmp_path, sky=False)
    restored = _load_tod(data, params, band, MPI.COMM_SELF, 1)
    assert restored.abs_gain == pytest.approx(9.0)


def test_diagnostics_are_recomputed_and_not_required_for_initialization(tmp_path):
    _save(tmp_path)
    filename = paths.band_chain_file(str(tmp_path), "EXP", "B", 1, 7)
    with h5py.File(filename, "r+") as handle:
        del handle["chisq_z"]
        del handle["good_fraction"]
    params, data, band = _case(tmp_path)
    _initialize(params, tmp_path, sky=False)
    restored = _load_tod(data, params, band, MPI.COMM_SELF, 1)
    assert np.isnan(restored.chisq_z).all()
    assert np.isnan(restored.good_fraction).all()


@pytest.mark.parametrize("bad_field", ["noise_params", "det_names", "metadata/band_unit"])
def test_missing_tod_state_is_reported_with_its_source_path(tmp_path, bad_field):
    _save(tmp_path)
    filename = paths.band_chain_file(str(tmp_path), "EXP", "B", 1, 7)
    with h5py.File(filename, "r+") as handle:
        del handle[bad_field]
    params, data, band = _case(tmp_path)
    _initialize(params, tmp_path, sky=False)
    with pytest.raises(ValueError, match=bad_field) as error:
        _load_tod(data, params, band, MPI.COMM_SELF, 1)
    assert filename in str(error.value)


@pytest.mark.parametrize("scan_ids, detectors", [((999,), ("a",)), ((11,), ("unknown",))])
def test_missing_scan_or_detector_is_rejected(tmp_path, scan_ids, detectors):
    _save(tmp_path)
    params, data, band = _case(tmp_path, scan_ids, detectors)
    _initialize(params, tmp_path, sky=False)
    with pytest.raises(ValueError, match="not found"):
        _load_tod(data, params, band, MPI.COMM_SELF, 1)


@pytest.mark.parametrize("field, value, message", [
    ("metadata/noise_model", "OtherNoiseModel", "Noise model differs"),
    ("metadata/nu_ghz", 200.0, "Band frequency differs"),
    ("noise_params", np.ones((3, 2, 4)), "expected"),
    ("present", np.zeros((3, 2), dtype=np.int8), "no saved data"),
])
def test_incompatible_tod_state_is_rejected(tmp_path, field, value, message):
    _save(tmp_path)
    filename = paths.band_chain_file(str(tmp_path), "EXP", "B", 1, 7)
    with h5py.File(filename, "r+") as handle:
        del handle[field]
        handle[field] = value
    params, data, band = _case(tmp_path)
    _initialize(params, tmp_path, sky=False)
    with pytest.raises(ValueError, match=message):
        _load_tod(data, params, band, MPI.COMM_SELF, 1)


def test_sky_amplitudes_and_spectral_parameters_are_independent(tmp_path):
    _, saved = _save(tmp_path)
    for amplitudes, spectral in ((True, False), (False, True)):
        params, _, _ = _case(tmp_path)
        _initialize(params, tmp_path, tod=False)
        params.gibbs.start.load = ["amplitudes"] if amplitudes else ["spectral_parameters"]
        sky = build_initial_sky_model(params, RunStart.from_gibbs(params.gibbs))._components
        for actual, expected in zip(sky, saved):
            np.testing.assert_array_equal(actual.alms, expected.alms if amplitudes else 0.0)
            assert actual.beta == (expected.beta if spectral else 1.54)
            assert actual.T == (expected.T if spectral else 20.0)


def test_component_null_override_keeps_configured_initialization(tmp_path):
    _save(tmp_path)
    params, _, _ = _case(tmp_path)
    _initialize(params, tmp_path, tod=False)
    params.components.dust.params.init_from = None
    sky = build_initial_sky_model(params, RunStart.from_gibbs(params.gibbs))._components
    for comp in sky:
        assert np.all(comp.alms == 0.0)
        assert comp.beta == 1.54


@pytest.mark.parametrize("metadata, value, match", [
    ("sed/nu_ref", 30.0, "reference parameter"),
    ("component_class", "Synchrotron", "component_class"),
    ("amplitude_unit", "uK_CMB", "amplitude_unit"),
])
def test_incompatible_sky_conventions_are_rejected(tmp_path, metadata, value, match):
    _save(tmp_path)
    filename = paths.compsep_chain_file(str(tmp_path), 1, 7)
    with h5py.File(filename, "r+") as handle:
        del handle[f"comps/dust/{metadata}"]
        handle[f"comps/dust/{metadata}"] = value
    params, _, _ = _case(tmp_path)
    _initialize(params, tmp_path, tod=False)
    with pytest.raises(ValueError, match=match):
        build_initial_sky_model(params, RunStart.from_gibbs(params.gibbs))


def test_new_runs_restart_numbering_even_when_loading_a_later_sample(tmp_path):
    _save(tmp_path)
    params, _, _ = _case(tmp_path / "new")
    _initialize(params, tmp_path)
    start = resolve_run_start(params)
    assert gibbs_schedule(2, start.start_iteration) == [(1, 1), (2, 1), (1, 2), (2, 2)]
    assert not StepConfig(enabled=True, from_iter=5).is_active(start.start_iteration)


def test_old_parameter_is_rejected_with_migration_instructions():
    with pytest.raises(ValueError, match="gibbs.start"):
        RunStart.from_gibbs({"init_from_chain": "old.h5"})


def test_loaded_tod_state_can_be_redistributed_across_mpi_ranks(tmp_path):
    """Run under mpirun as well: unequal scan counts, reordered detectors and an empty rank."""
    comm = MPI.COMM_WORLD
    run_dir = Path(comm.bcast(str(tmp_path) if comm.rank == 0 else None, root=0))
    expected = None
    if comm.rank == 0:
        expected, _ = _save(run_dir, hfi=True)
    comm.Barrier()
    if comm.size == 1:
        scan_ids = (33, 11, 22)
    elif comm.rank == 0:
        scan_ids = (33, 11)
    elif comm.rank == 1:
        scan_ids = (22,)
    else:
        scan_ids = ()
    params, data, band = _case(run_dir, scan_ids, ("b", "a"), hfi=True, band_unit="uK_RJ")
    _initialize(params, run_dir)
    restored = _load_tod(data, params, band, comm, 1)
    gathered = restored.gather_chain_arrays(8)
    if comm.rank == 0:
        np.testing.assert_array_equal(gathered["scan_ids"], [33, 11, 22])
        np.testing.assert_allclose(gathered["temporal_gain"],
                                   expected.temporal_gain[[2, 0, 1]][:, [1, 0]])
        np.testing.assert_array_equal(gathered["baselines"],
                                      expected.baselines[[2, 0, 1]][:, [1, 0]])


@pytest.mark.parametrize("key, value", [("chain", True), ("chain", 3), ("iteration", 0),
                                       ("source", False), ("load", ["unknown"])])
def test_invalid_source_parameters_are_rejected(key, value):
    source = {"source": "old", "chain": 1, "iteration": 7, key: value}
    with pytest.raises(ValueError):
        RunStart.from_gibbs({"start": source})


def _resume_params(run_dir: Path) -> Bunch:
    params, _, _ = _case(run_dir)
    params.gibbs.start = Bunch(mode="resume")
    params.compsep.enabled = True
    params.compsep.cg_sampling_groups = Bunch(amps=Bunch(enabled=True))
    return params


def test_resume_restores_each_chain_and_continues_iteration_numbering(tmp_path):
    saved = {}
    for chain in (1, 2):
        saved[chain] = _save(tmp_path, chain=chain)
    params = _resume_params(tmp_path)
    start = resolve_run_start(params)
    assert start.iteration == 7
    assert start.start_iteration == 8
    assert StepConfig(enabled=True, from_iter=5).is_active(start.start_iteration)
    for chain in (1, 2):
        _, data, band = _case(tmp_path)
        actual = TODSamples(data, params, band, MPI.COMM_SELF, chain,
                            start.tod_file("EXP", "B", chain))
        np.testing.assert_allclose(actual.gain_all(), saved[chain][0].gain_all())
        sky = build_initial_sky_model(params, start, chain)._components
        for comp, expected in zip(sky, saved[chain][1]):
            assert comp.beta == expected.beta
    assert saved[1][0].abs_gain != saved[2][0].abs_gain
    assert saved[1][1][0].beta != saved[2][1][0].beta


def test_resume_requires_both_chains_but_a_new_run_can_copy_one(tmp_path):
    _save(tmp_path)
    with pytest.raises(ValueError, match="Resume requires both chains"):
        resolve_run_start(_resume_params(tmp_path))
    params, _, _ = _case(tmp_path / "new")
    _initialize(params, tmp_path)
    start = resolve_run_start(params)
    assert start.source_chain(1) == start.source_chain(2) == 1
    assert start.start_iteration == 1


def test_matching_initialization_selects_the_latest_common_complete_sample(tmp_path):
    _save(tmp_path, chain=1, iteration=6)
    _save(tmp_path, chain=2, iteration=6)
    _save(tmp_path, chain=1, iteration=7)
    params, _, _ = _case(tmp_path / "new")
    params.gibbs.start = Bunch(source=str(tmp_path), iteration="latest", chain="matching")
    start = resolve_run_start(params)
    assert start.iteration == 6
    assert (start.source_chain(1), start.source_chain(2)) == (1, 2)
    assert start.start_iteration == 1


def test_resume_reports_later_output_without_changing_any_files(tmp_path):
    _save(tmp_path, chain=1)
    _save(tmp_path, chain=2)
    _save(tmp_path, chain=1, iteration=8)
    before = sorted(str(path) for path in tmp_path.rglob("*.h5"))
    with pytest.raises(ValueError, match="last complete selected iteration is 7") as error:
        resolve_run_start(_resume_params(tmp_path))
    assert "chain01_iter0008.h5" in str(error.value)
    assert sorted(str(path) for path in tmp_path.rglob("*.h5")) == before


def test_incomplete_file_marker_prevents_selection_of_an_interrupted_sample(tmp_path):
    for chain in (1, 2):
        _save(tmp_path, chain=chain, iteration=6)
        _save(tmp_path, chain=chain, iteration=7)
    filename = paths.compsep_chain_file(str(tmp_path), 2, 7)
    with h5py.File(filename, "r+") as handle:
        handle["metadata/complete"][()] = False
    params, _, _ = _case(tmp_path / "new")
    params.gibbs.start = Bunch(source=str(tmp_path))
    assert resolve_run_start(params).iteration == 6
    with pytest.raises(ValueError, match="later output exists"):
        resolve_run_start(_resume_params(tmp_path))


def test_legacy_complete_files_can_be_resumed(tmp_path):
    for chain in (1, 2):
        _save(tmp_path, chain=chain)
    for filename in tmp_path.rglob("*.h5"):
        with h5py.File(filename, "r+") as handle:
            del handle["metadata/complete"]
    assert resolve_run_start(_resume_params(tmp_path)).start_iteration == 8


def test_resume_ignores_component_initial_guesses(tmp_path):
    for chain in (1, 2):
        _save(tmp_path, chain=chain)
    params = _resume_params(tmp_path)
    params.components.dust.params.init_from = str(tmp_path / "missing_old_initial_guess.fits")
    start = resolve_run_start(params)
    for chain in (1, 2):
        sky = build_initial_sky_model(params, start, chain)._components
        assert sky[0].beta == pytest.approx(1.2 + 0.1 * (chain - 1))


def test_tod_only_resume_uses_distinct_fixed_sky_snapshots(tmp_path):
    for chain in (1, 2):
        _, sky = _save(tmp_path, chain=chain)
        params, _, _ = _case(tmp_path)
        write_compsep_chain_to_file(sky.joined(), params, chain, 0)
        Path(paths.compsep_chain_file(str(tmp_path), chain, 7)).unlink()
    params, _, _ = _case(tmp_path)
    params.gibbs.start = Bunch(mode="resume")
    start = resolve_run_start(params)
    assert start.iteration == 7
    assert start.sky_iteration == 0
    for chain in (1, 2):
        sky = build_initial_sky_model(params, start, chain)._components
        assert sky[0].beta == pytest.approx(1.2 + 0.1 * (chain - 1))


def test_new_run_refuses_a_directory_that_already_contains_chain_files(tmp_path):
    _save(tmp_path)
    params, _, _ = _case(tmp_path)
    with pytest.raises(ValueError, match="new run requires"):
        resolve_run_start(params)


@pytest.mark.parametrize("extra", [{"chain": 1}, {"source": "other"}, {"load": ["tod"]}])
def test_resume_rejects_settings_that_would_start_a_different_run(extra):
    with pytest.raises(ValueError, match="Resume uses"):
        RunStart.from_gibbs({"start": {"mode": "resume", **extra}})
