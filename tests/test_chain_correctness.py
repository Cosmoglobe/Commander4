"""Regression tests for chain selection, the merged band file, and map output conversion."""

import os

import h5py
import numpy as np
from pixell.bunch import Bunch

from commander4.data_models.tod_samples import TODSamples
from commander4.file_io import paths
from commander4.file_io.chain_writer import (
    write_band_chain_to_file,
    write_compsep_chain_to_file,
)


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


def _minimal_tod_samples(nscans: int = 3, ndet: int = 2) -> TODSamples:
    """A `TODSamples` carrying only what `gather_chain_arrays` reads, on a one-rank communicator."""
    from mpi4py import MPI

    from commander4.data_models.jump_corrections import JumpCatalog

    s = TODSamples.__new__(TODSamples)
    s.params, s.chain, s.band_comm = _params(), 1, MPI.COMM_SELF
    s.nscans, s.ndet, s.npar = nscans, ndet, 3
    s.experiment_name, s.band_name = "EXP", "B"
    s.scan_ids = np.arange(nscans, dtype=np.int64)
    s.det_names = [f"d{i}" for i in range(ndet)]
    s.band_unit_factor, s.band_unit = 1.0, "uK_RJ"
    s.abs_gain, s.rel_gain = 1.0, np.zeros(ndet)
    s.temporal_gain = np.zeros((nscans, ndet))
    s.noise_params = np.zeros((nscans, ndet, s.npar))
    s.gain_prior = np.full((ndet, 3), np.nan)
    s.present = np.ones((nscans, ndet), dtype=bool)
    s.accept = np.ones((nscans, ndet), dtype=bool)
    s.chisq_z = np.zeros((nscans, ndet))
    s.good_fraction = np.ones((nscans, ndet))
    s.scan_start_time = np.zeros(nscans)
    s.orbital_velocity = np.zeros((nscans, ndet, 3), dtype=np.float32)
    s.ncorr_cg_residual = np.zeros((nscans, ndet))
    s.ncorr_cg_niter = np.zeros((nscans, ndet), dtype=np.int32)
    s.ncorr_converged = np.zeros((nscans, ndet), dtype=np.int8)
    for name in ("tod_ps_freqs", "tod_ps_ncorr", "tod_ps_raw", "tod_ps_ncorrsub",
                 "tod_ps_residual"):
        setattr(s, name, np.zeros((nscans, ndet, TODSamples.TOD_PS_NBIN), dtype=np.float32))
    s.ncorr_tods = None
    s.jumps = JumpCatalog.empty(nscans, ndet)
    return s


def test_the_band_file_carries_every_dataset_a_restart_reads_back() -> None:
    """`TODSamples.__init__` reads these unconditionally, so the gather has to produce them all.

    A dataset dropped from the gather turns a restart into a `KeyError`, which is the whole reason
    the reader is allowed to be unconditional. This list mirrors the reads in `__init__`.
    """
    read_back = {"scan_ids", "abs_gain", "detrel_gain", "temporal_gain", "noise_params",
                 "accept", "chisq_z", "good_fraction",
                 "jump_counts", "jump_locations", "jump_offsets"}

    written = set(_minimal_tod_samples().gather_chain_arrays(1))

    assert read_back <= written, f"a restart would fail on {sorted(read_back - written)}"


def test_disabled_tod_chain_returns_before_collective_gathers() -> None:
    samples = TODSamples.__new__(TODSamples)
    samples.params = _params()
    samples.chain = 2
    samples.band_comm = object()

    assert samples.gather_chain_arrays(itr=1) is None


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
                       band_unit_factor=3.0, band_unit="uK_CMB")

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
