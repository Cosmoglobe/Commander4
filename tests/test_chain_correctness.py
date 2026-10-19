"""Regression tests for chain selection, the merged band file, and map output conversion."""

import os
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from pixell.bunch import Bunch

from commander4.data_models.tod_samples import TODSamples
from commander4.file_io import paths
from commander4.tod.data_selection import DataSelection
from commander4.tod.gain import Gain
from commander4.tod.glitches.sampling import Glitches
from commander4.tod.hfi_demodulation import HFIModulation
from commander4.tod.jumps.sampling import Jumps
from commander4.tod.noise.sampling import Noise
from commander4.tod.scan_diagnostics import TODDiagnostics
from commander4.tod.sidelobe_deconvolve import FarSidelobes
from commander4.tod.step import SavedBandSample
from commander4.units import rj_to_band_unit_factor
from commander4.file_io.chain_writer import (
    write_band_chain_to_file,
    write_compsep_chain_to_file,
)
from tod_step_fakes import make_step


def _params(output_dir: str | None = None, interval: dict | None = None,
            maps_nside: int = 2) -> Bunch:
    chains = Bunch(write=[1], maps_nside=maps_nside)
    if interval is not None:
        chains.interval = Bunch(**interval)
    output = Bunch(chains=chains)
    if output_dir is not None:
        output.dir = output_dir
    params = Bunch(output=output)
    params.parameter_file_as_string = "parameters"
    return params


def _write_band(params, tod_arrays=None, maps_to_file=None, iter=1, **kwargs) -> str:
    """Write one band file and return its path."""
    write_band_chain_to_file(
        params, chain=1, iter=iter, exp_name="Experiment", band_name="Band",
        tod_arrays=tod_arrays if tod_arrays is not None else {},
        maps_to_file=maps_to_file if maps_to_file is not None else {}, **kwargs)
    return os.path.join(paths.subdir(params, paths.CHAINS_BANDS),
                        f"Experiment_Band_chain01_iter{iter:04d}.h5")


def test_disabled_compsep_chain_returns_before_opening_a_file() -> None:
    write_compsep_chain_to_file([], _params(), chain=2, iter=1)


def _minimal_tod_samples(nscans: int = 3, ndet: int = 2, hfi: bool = False,
                         **include) -> TODSamples:
    """A `TODSamples` carrying only what the chain I/O uses, on a one-rank communicator.

    Gain and noise start at zero instead of from data files, so they are built field by field.
    `include` sets `output.chains.include` for the diagnostics.
    """
    from mpi4py import MPI

    s = TODSamples.__new__(TODSamples)
    s.chain, s.band_comm = 1, MPI.COMM_SELF
    s.experiment_data = SimpleNamespace(nscans=nscans, ndet=ndet, experiment_name="EXP",
                                        band_name="B", nu=100.0, band_unit="uK_RJ",
                                        band_unit_factor=1.0)
    s.scan_ids = np.arange(nscans, dtype=np.int64)
    s.det_names = [f"d{i}" for i in range(ndet)]
    s.present = np.ones((nscans, ndet), dtype=bool)
    s.scan_start_time = np.zeros(nscans)
    s.orbital_velocity = np.zeros((nscans, ndet, 3), dtype=np.float32)
    s.gain = Gain.__new__(Gain)
    s.gain.nu, s.gain.band_unit_factor = 100.0, 1.0
    s.gain.abs_gain, s.gain.rel_gain = 1.0, np.zeros(ndet)
    s.gain.temporal_gain = np.zeros((nscans, ndet))
    s.gain.gain_prior = np.full((ndet, 3), np.nan)
    s.noise = Noise.__new__(Noise)
    s.noise.noise_params = np.zeros((nscans, ndet, 3))
    s.noise.ncorr_cg_residual = np.zeros((nscans, ndet))
    s.noise.ncorr_cg_niter = np.zeros((nscans, ndet), dtype=np.int32)
    s.noise.ncorr_converged = np.zeros((nscans, ndet), dtype=np.int8)
    s.noise.ncorr_high_var = np.zeros((nscans, ndet), dtype=np.int8)
    s.selection = make_step(DataSelection, nscans, ndet)
    s.jumps = make_step(Jumps, nscans, ndet)
    s.hfi = HFIModulation(SimpleNamespace(nscans=nscans, ndet=ndet, hfi_demodulation=hfi),
                          Bunch(), None, MPI.COMM_SELF, 1)
    s.glitches = make_step(Glitches, nscans, ndet, settings={"glitches": {"enabled": True}})
    s.far_sidelobes = make_step(FarSidelobes, nscans, ndet)
    s.diagnostics = make_step(TODDiagnostics, nscans, ndet, **include)
    s.steps = [s.gain, s.noise, s.selection, s.jumps, s.hfi, s.glitches, s.far_sidelobes,
               s.diagnostics]
    return s


def _random_restart_state(samples: TODSamples, rng: np.random.Generator) -> None:
    """Fill the restart state of `samples` with random values, its gains quoted in uK_CMB."""
    nscans, ndet = samples.experiment_data.nscans, samples.experiment_data.ndet
    band_unit_factor = rj_to_band_unit_factor(samples.experiment_data.nu, "uK_CMB")
    samples.experiment_data.band_unit = "uK_CMB"
    samples.experiment_data.band_unit_factor = band_unit_factor
    samples.gain.band_unit_factor = band_unit_factor
    samples.gain.abs_gain = 2.0
    samples.gain.rel_gain = rng.normal(size=ndet)
    samples.gain.temporal_gain = rng.normal(size=(nscans, ndet))
    samples.noise.noise_params = rng.normal(size=(nscans, ndet, 3))
    samples.selection.accept = rng.random((nscans, ndet)) > 0.5
    if samples.hfi.cfg.enabled:
        samples.hfi.modulation_phase = rng.choice([-1, 1], size=(nscans, ndet)).astype(np.int8)
        samples.hfi.baselines = rng.normal(size=(nscans, ndet, 2))


@pytest.mark.parametrize("hfi", [False, True])
def test_a_restart_reads_back_the_sampled_state(tmp_path, hfi: bool) -> None:
    """Gather, write and `read_chain` give back every sampled quantity.

    The reading rank holds its scans and lists its detectors in another order than the writer,
    and the file quotes its gains in uK_CMB: the reader matches scans by ID and detectors by name,
    and converts the gains back to uK_RJ.
    """
    params = _params(str(tmp_path))
    paths.create_output_dirs(params.output)
    saved = _minimal_tod_samples(nscans=4, ndet=3, hfi=hfi)
    _random_restart_state(saved, np.random.default_rng(1))
    path = _write_band(params, saved.gather_chain_arrays(params, 1),
                       band_unit_factor=saved.experiment_data.band_unit_factor)

    restored = _minimal_tod_samples(nscans=4, ndet=3, hfi=hfi)
    scan_order, det_order = [2, 0, 3, 1], [2, 0, 1]
    restored.scan_ids = saved.scan_ids[scan_order]
    restored.det_names = [saved.det_names[i] for i in det_order]
    restored.read_chain(path)

    assert restored.gain.abs_gain == pytest.approx(saved.gain.abs_gain)
    np.testing.assert_allclose(restored.gain.rel_gain, saved.gain.rel_gain[det_order])
    pairs = [(saved.gain, restored.gain, "temporal_gain"),
             (saved.noise, restored.noise, "noise_params"),
             (saved.selection, restored.selection, "accept")]
    if hfi:
        pairs += [(saved.hfi, restored.hfi, "modulation_phase"),
                  (saved.hfi, restored.hfi, "baselines")]
    for saved_step, restored_step, name in pairs:
        expected = getattr(saved_step, name)[scan_order][:, det_order]
        np.testing.assert_allclose(getattr(restored_step, name), expected, err_msg=name)
        assert getattr(restored_step, name).dtype == expected.dtype, name
    assert not restored.hfi.cfg.enabled or restored.hfi.initialized


# The restart state of each sampling step that has one, by its dataset names in the chain file.
_RESTART_DATASETS = {"gain": ["abs_gain", "detrel_gain", "temporal_gain"],
                     "noise": ["noise_params"], "selection": ["accept"],
                     "hfi": ["modulation_phase", "baselines"]}


@pytest.mark.parametrize("step", sorted(_RESTART_DATASETS))
def test_each_sampling_step_reads_back_the_restart_state_it_writes(tmp_path, step: str) -> None:
    """A step's `read_chain` restores, on its own, what its `chain_arrays` wrote: written again,
    the restored step gives the same restart datasets, in this rank's scan and detector order."""
    params = _params(str(tmp_path))
    paths.create_output_dirs(params.output)
    saved = _minimal_tod_samples(nscans=4, ndet=3, hfi=True)
    _random_restart_state(saved, np.random.default_rng(2))
    path = _write_band(params, saved.gather_chain_arrays(params, 1))

    restored = _minimal_tod_samples(nscans=4, ndet=3, hfi=True)
    restored.gain.band_unit_factor = saved.gain.band_unit_factor  # Writes in the same unit.
    scan_order, det_order = [3, 1, 0, 2], [1, 2, 0]
    restored.scan_ids = saved.scan_ids[scan_order]
    restored.det_names = [saved.det_names[i] for i in det_order]
    with h5py.File(path, "r") as f:
        getattr(restored, step).read_chain(SavedBandSample(f, restored.scan_ids,
                                                           restored.det_names, "B"))

    saved_arrays = getattr(saved, step).chain_arrays()
    restored_arrays = getattr(restored, step).chain_arrays()
    for name in _RESTART_DATASETS[step]:
        if name in saved_arrays.per_scan:
            expected = saved_arrays.per_scan[name][scan_order][:, det_order]
            np.testing.assert_allclose(restored_arrays.per_scan[name], expected, err_msg=name)
        elif name == "detrel_gain":
            np.testing.assert_allclose(restored_arrays.per_band[name],
                                       saved_arrays.per_band[name][det_order])
        else:
            np.testing.assert_allclose(restored_arrays.per_band[name], saved_arrays.per_band[name])


def test_per_scan_datasets_are_written_in_time_order() -> None:
    """A rank's scans need not be in time order; the chain rows always are (by scan ID)."""
    samples = _minimal_tod_samples(nscans=4)
    samples.scan_ids = np.array([30, 10, 40, 20], dtype=np.int64)
    ndet = samples.experiment_data.ndet
    samples.gain.temporal_gain = samples.scan_ids[:, None] * np.ones((1, ndet))

    arrays = samples.gather_chain_arrays(_params(), 1)

    np.testing.assert_array_equal(arrays["scan_ids"], [10, 20, 30, 40])
    np.testing.assert_array_equal(arrays["temporal_gain"][:, 0], [10, 20, 30, 40])
    assert arrays["detrel_gain"].shape == (ndet,)


def test_power_spectra_are_written_only_when_kept() -> None:
    """`include.tod_power_spectra: false` leaves the arrays as None; the gather then skips them."""
    assert "tod_ps_raw" in _minimal_tod_samples().gather_chain_arrays(_params(), 1)

    written = _minimal_tod_samples(tod_power_spectra=False).gather_chain_arrays(_params(), 1)
    assert not [name for name in written if name.startswith("tod_ps")]
    assert "noise_params" in written


def test_first_scans_runtime_is_written_as_the_median_of_the_others() -> None:
    """The rank's first timed scan also paid one-time costs (Numba compilation), so the chain gets
    the median of the other timed scans instead. Scan 2 was never processed and stays at zero."""
    samples = _minimal_tod_samples(nscans=4)
    samples.diagnostics.scan_runtime = np.array([5.0, 1.0, 0.0, 3.0])

    arrays = samples.gather_chain_arrays(_params(), 1)

    np.testing.assert_array_equal(arrays["scan_runtime"], [2.0, 1.0, 0.0, 3.0])
    # Kept as measured.
    np.testing.assert_array_equal(samples.diagnostics.scan_runtime, [5.0, 1.0, 0.0, 3.0])


def test_hfi_chain_carries_modulation_phase_and_baselines() -> None:
    """Only HFI bands write the modulation state; other bands write no empty datasets."""
    assert "baselines" not in _minimal_tod_samples().gather_chain_arrays(_params(), 1)
    samples = _minimal_tod_samples(hfi=True)
    nscans, ndet = samples.experiment_data.nscans, samples.experiment_data.ndet
    samples.hfi.baselines = np.arange(nscans * ndet * 2, dtype=np.float64).reshape(nscans, ndet, 2)

    written = samples.gather_chain_arrays(_params(), 1)

    np.testing.assert_array_equal(written["modulation_phase"], samples.hfi.modulation_phase)
    np.testing.assert_array_equal(written["baselines"], samples.hfi.baselines)


def test_glitch_events_are_debug_output_written_per_detector_scan(tmp_path) -> None:
    from commander4.tod.glitches.events import GlitchEvents

    def glitch_names(samples) -> list[str]:
        return sorted(name for name in samples.gather_chain_arrays(_params(), 1)
                      if name.startswith("glitch"))

    samples = _minimal_tod_samples()
    samples.glitches.detected = True
    samples.glitches.events[2, 1] = GlitchEvents([5, 9], [20, 20], [0, 2], [2.0, 0.5], [5, 9],
                                                 [5, 9], [0, 0], [])
    assert glitch_names(samples) == []  # Off unless asked for.

    samples.glitches.write_events = True
    written = samples.gather_chain_arrays(_params(), 1)

    # Only the detector-scan with events gets a group, named by scan ID and detector name.
    event_groups = {name.rsplit("/", 1)[0] for name in glitch_names(samples)
                    if not name.startswith("glitches/template_")}
    assert event_groups == {"glitches/2/d1"}
    np.testing.assert_array_equal(written["glitches/2/d1/start"], [5, 9])
    assert written["glitches/2/d1/shape_params"].size == 0
    assert written["glitches/template_amps"].shape == (2, 3, 8)

    # The chain writer turns the paths into nested HDF5 groups.
    params = _params(str(tmp_path))
    paths.create_output_dirs(params.output)
    with h5py.File(_write_band(params, written)) as handle:
        assert handle["glitches/2/d1/type"].dtype == np.int8

    # Nothing is written before the glitch step has run.
    samples.glitches.detected = False
    assert glitch_names(samples) == []


@pytest.mark.parametrize("save_residual", [False, True])
@pytest.mark.parametrize("save_ncorr", [False, True])
def test_full_tods_round_trip_independently(tmp_path, save_residual: bool,
                                          save_ncorr: bool) -> None:
    """Different-length streams remain associated with their scan IDs and detector names."""
    samples = _minimal_tod_samples()
    params = _params(str(tmp_path))
    paths.create_output_dirs(params.output)
    samples.scan_ids = np.array([11, 3, 29], dtype=np.int64)
    samples.det_names = ["100-1a", "100-1b"]
    streams = [[np.array([1.0, 2.0], dtype=np.float32), None],
               [None, np.array([3.0], dtype=np.float32)],
               [np.array([4.0, 5.0, 6.0], dtype=np.float32), np.array([7.0], dtype=np.float32)]]
    if save_residual:
        samples.diagnostics.residual_tods = streams
    if save_ncorr:
        samples.diagnostics.ncorr_tods = streams

    path = _write_band(params, samples.gather_chain_arrays(params, 1))

    with h5py.File(path) as handle:
        assert ("tods" in handle) == (save_residual or save_ncorr)
        for name, enabled in (("residual", save_residual), ("ncorr", save_ncorr)):
            assert f"{name}_tod_lengths" not in handle
            assert f"{name}_tod_flat" not in handle
            for iscan, scan_id in enumerate(samples.scan_ids):
                for idet, detector in enumerate(samples.det_names):
                    dataset = f"tods/{scan_id}/{detector}/{name}"
                    expected = streams[iscan][idet]
                    assert (dataset in handle) == (enabled and expected is not None)
                    if enabled and expected is not None:
                        np.testing.assert_array_equal(handle[dataset][:], expected)
                        assert handle[dataset].dtype == np.float32


@pytest.mark.parametrize("nscans", [0, 3])
def test_residual_output_without_evaluated_scans(nscans: int) -> None:
    samples = _minimal_tod_samples(nscans=nscans, residual_tods=True)
    arrays = samples.gather_chain_arrays(_params(), 1)
    for key in arrays:
        assert not key.startswith("tods/")


def test_full_tod_gather_preserves_scan_ids_across_ranks() -> None:
    """Also run under mpirun to cover uneven scan counts and a rank with no scans."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    samples = _minimal_tod_samples(nscans=comm.rank)
    samples.band_comm = comm
    samples.scan_ids += 100 * comm.rank
    samples.diagnostics.residual_tods = []
    for scan_id in samples.scan_ids:
        samples.diagnostics.residual_tods.append([np.full(comm.rank + 1, scan_id,
                                                          dtype=np.float32), None])

    arrays = samples.gather_chain_arrays(_params(), 1)

    if comm.rank != 0:
        assert arrays is None
        return
    for rank in range(comm.size):
        for iscan in range(rank):
            scan_id = 100 * rank + iscan
            np.testing.assert_array_equal(arrays[f"tods/{scan_id}/d0/residual"],
                                          np.full(rank + 1, scan_id, dtype=np.float32))
            assert f"tods/{scan_id}/d1/residual" not in arrays


def test_disabled_tod_chain_returns_before_collective_gathers() -> None:
    samples = TODSamples.__new__(TODSamples)
    samples.chain = 2
    samples.band_comm = object()

    assert samples.gather_chain_arrays(_params(), itr=1) is None


def test_band_file_holds_the_tod_samples_and_the_maps_together(tmp_path) -> None:
    """The per-scan samples and the band maps share one file per band per sample."""
    params = _params(str(tmp_path))
    paths.create_output_dirs(params.output)
    tod_arrays = {"scan_ids": np.arange(4, dtype=np.int64),
                  "abs_gain": np.float64(1.5)}
    # 48 pixels = nside 2 = params' `maps_nside`, so the map is written without regrading.
    maps_to_file = {"observed_sky": np.zeros((3, 48), dtype=np.float32),
                    "map_fwhm_arcmin": 30.0}

    path = _write_band(params, tod_arrays, maps_to_file)

    with h5py.File(path) as handle:
        # The TOD samples sit at the top level, which is where TODSamples reads them back from.
        np.testing.assert_array_equal(handle["scan_ids"][:], np.arange(4))
        assert handle["abs_gain"][()] == 1.5
        assert handle["maps/observed_sky"].shape == (3, 48)
        # The beam is metadata about the maps, not a map itself.
        assert handle["metadata/map_fwhm_arcmin"][()] == 30.0
        assert "map_fwhm_arcmin" not in handle["maps"]
    assert not (tmp_path / "chains_datamaps").exists()
    # The file was written under a temporary name and renamed once complete.
    assert os.listdir(paths.subdir(params, paths.CHAINS_BANDS)) == [os.path.basename(path)]


def test_maps_interval_thins_the_group_but_not_the_file(tmp_path) -> None:
    """`bands` gates the file; `maps` gates only the maps group inside it."""
    params = _params(str(tmp_path), interval={"bands": 1, "maps": 3})
    paths.create_output_dirs(params.output)
    tod_arrays = {"scan_ids": np.arange(2, dtype=np.int64)}
    maps_to_file = {"observed_sky": np.zeros((3, 48), dtype=np.float32)}

    for iter in (1, 2, 3, 4):
        path = _write_band(params, tod_arrays, maps_to_file, iter=iter)
        with h5py.File(path) as handle:
            assert "scan_ids" in handle, "the band file itself is written every iteration"
            assert ("maps" in handle) == (iter in (1, 4)), f"maps group at iter {iter}"


def test_rms_downgrade_averages_inverse_variance_and_takes_square_root(tmp_path) -> None:
    params = _params(str(tmp_path))
    paths.create_output_dirs(params.output)

    path = _write_band(params, maps_to_file={"rms": np.full((1, 12), 2.0, dtype=np.float32)})

    with h5py.File(path) as handle:
        np.testing.assert_allclose(handle["maps/rms"][:], 2.0)


def test_hit_counts_are_summed_when_degraded_and_left_unit_free(tmp_path) -> None:
    """A hit map adds over the sub-pixels it merges, and carries no thermodynamic unit."""
    params = _params(str(tmp_path), maps_nside=1)  # 48 input pixels collapse into 12.
    paths.create_output_dirs(params.output)

    path = _write_band(params, maps_to_file={"nhit": np.full(48, 5, dtype=np.int64)},
                       band_unit_factor=3.0)

    with h5py.File(path) as handle:
        written = handle["maps/nhit"][:]
    assert written.shape == (12,)
    np.testing.assert_array_equal(written, 20)  # 4 sub-pixels x 5 hits, no band_unit factor.


def test_the_covariance_map_converts_as_an_inverse_variance(tmp_path) -> None:
    """`cov` is a summed uK_RJ^-2 weight: it sums when degraded and picks up D^-2, not D."""
    params = _params(str(tmp_path), maps_nside=1)  # 48 input pixels collapse into 12.
    paths.create_output_dirs(params.output)
    D = 4.0

    path = _write_band(params, maps_to_file={"cov": np.full((6, 48), 2.0)}, band_unit_factor=D)

    with h5py.File(path) as handle:
        written = handle["maps/cov"][:]
    assert written.shape == (6, 12)
    np.testing.assert_allclose(written, 4*2.0/D**2)
