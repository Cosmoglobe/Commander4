"""The binned mapmaker's residual and hit maps, Commander3's `tod_<freq>_res` and its companion.

`maps/res` is the noise residual `_record_tod_diagnostics` already builds per detector-scan (data
minus sky model, orbital dipole and correlated noise), binned with the same inverse-variance
weights as the signal map. So a TOD that is exactly `gain * sky` must bin to a residual of zero
while still producing the right signal map -- that is the property these tests pin.

`maps/nhit` counts unflagged samples per pixel, which no other output records: `maps/rms` counts
them weighted by `(gain/sigma0)^2`, so it cannot be inverted back into a sample count in general.
"""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.scan_tod import ScanTOD
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.pointing import PixelPointing
from commander4.data_models.tod_samples import TODSamples
import commander4.tod.processing as tod_processing
from commander4.tod.config import MapmakingConfig, CorrelatedNoiseConfig, DataSelectionConfig
import commander4.tod.mapmaking.binned as binned
from commander4.tod.sky_projection import get_s_orb_tod
from commander4.units import unit_factor

_BITMASK = 1
_NSIDE = 1
_NPIX = 12*_NSIDE**2
_GAIN = 1.5   # abs_gain below; rel and temporal gain are zero, so this is the whole gain.


def _build_band(pix, psi, tod, flag=None,
                response_I_P: tuple[float, float] | None = None, unit: str = "uK_RJ",
                velocity: np.ndarray | None = None) -> DetectorGroupTOD:
    """One IQU detector-scan with uncompressed pointing and, by default, no orbital motion.

    The zero orbital velocity is what makes the expected residual exactly the noise: with no
    spacecraft velocity the orbital-dipole TOD the mapmaker subtracts is identically zero, so the
    only model term left is the sky projection.
    """
    ntod = pix.size
    pointing = PixelPointing(pix.astype(np.int64), psi.astype(np.float64), np.array([0], np.int64),
                             None, None, _NSIDE, _NSIDE, ntod, ntod)
    if velocity is None:
        velocity = np.zeros(3)
    det = DetectorTOD(
        name="d0", det_idx_fullband=0, tod=tod.astype(np.float32), pointing=pointing,
        sampling_rate_hz=1.0, orbital_velocity_m_per_s=velocity.astype(np.float32),
        huffman_tree=None, huffman_symbols=None, default_proc_mask=np.ones(_NPIX, bool),
        specific_proc_masks={},
        flag_encoded=(np.zeros(ntod) if flag is None else flag).astype(np.int64),
        bad_data_bitmask=_BITMASK, flag_is_compressed=False,
        response_I_P=response_I_P,
    )
    noise_model = SimpleNamespace(npar=1, params=np.array([np.nan]))
    return DetectorGroupTOD([ScanTOD([det], 0.0, 0)], "EXP", "B", nside=_NSIDE, nu=30.0,
                            unit=unit, fwhm=0.0, fsamp=1.0, ndet=1, pols="IQU",
                            noise_model=noise_model)


def _fake_tod_samples(sigma0: float = 2.0, ndet: int = 1, gain: float = _GAIN,
                      band_unit: str = "uK_RJ") -> SimpleNamespace:
    """Minimal stand-in exposing exactly the fields tod2map_bin / TODView / the diagnostics read."""
    no_jump = SimpleNamespace(is_empty=lambda: True)
    empty_ps = lambda: np.full((1, ndet, 100), np.nan, dtype=np.float32)
    return SimpleNamespace(
        noise_params=np.full((1, ndet, 1), sigma0), abs_gain=gain, rel_gain=np.zeros(ndet),
        temporal_gain=np.zeros((1, ndet)), jumps=SimpleNamespace(get=lambda iscan, idet: no_jump),
        accept=np.ones((1, ndet), dtype=bool), band_unit=band_unit,
        chisq_z=np.full((1, ndet), np.nan), good_fraction=np.full((1, ndet), np.nan),
        TOD_PS_NBIN=100, tod_ps_freqs=empty_ps(), tod_ps_raw=empty_ps(), tod_ps_residual=empty_ps(),
        tod_ps_ncorrsub=empty_ps(), tod_ps_ncorr=empty_ps(), ncorr_tods=None, residual_tods=None)


def _run(band: DetectorGroupTOD, sky_model: np.ndarray, sparse_maps: bool = False,
         gain: float = _GAIN) -> dict[str, np.ndarray]:
    mapmaking = MapmakingConfig(
        mapmaker="bin", num_threads=1,
        include_orbital_dipole_maps=False, include_corr_noise_maps=False,
        include_sky_model_maps=False, include_residual_maps=True, include_hit_maps=True,
        sparse_maps=sparse_maps, common_res_fwhm=0.0,
    )
    tod_samples = _fake_tod_samples(ndet=band.ndet, gain=gain, band_unit=band.unit)
    _, maps = tod_processing.tod2map_bin(
        MPI.COMM_SELF, band, sky_model, tod_samples, 1, mapmaking,
        CorrelatedNoiseConfig(sample_sigma0=False),
        DataSelectionConfig(),
    )
    return maps


def _project(sky: np.ndarray, pix: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """The IQU sky along a pointing: I + Q cos(2 psi) + U sin(2 psi)."""
    return sky[0, pix] + sky[1, pix]*np.cos(2*psi) + sky[2, pix]*np.sin(2*psi)


@pytest.mark.parametrize("enabled", [None, False, True])
def test_residual_tod_parameter_controls_collection(enabled: bool | None) -> None:
    """Omitting the new flag keeps full residual storage disabled."""
    band = _build_band(np.zeros(128, dtype=np.int64), np.zeros(128), np.zeros(128))
    include = Bunch()
    if enabled is not None:
        include.residual_tods = enabled
    params = Bunch(output=Bunch(chains=Bunch(include=include)), gibbs=Bunch())
    my_band = Bunch(band_unit="uK_RJ", detectors=Bunch(d0=Bunch(gain=_GAIN)))

    samples = TODSamples(band, params, my_band, MPI.COMM_SELF, chain=1)

    assert samples.ncorr_tods is None
    if enabled:
        assert samples.residual_tods == [[None]]
    else:
        assert samples.residual_tods is None


def _write_restart_chain(path, band_unit: str, abs_gain: float) -> None:
    """A one-scan, one-detector band chain holding exactly what a restart reads back."""
    with h5py.File(path, "w") as f:
        f["metadata/band_unit"] = band_unit
        f["scan_ids"] = np.array([0], dtype=np.int64)
        f["abs_gain"] = abs_gain
        f["detrel_gain"] = np.zeros(1)
        f["temporal_gain"] = np.zeros((1, 1))
        f["noise_params"] = np.full((1, 1, 1), 2.0)
        f["accept"] = np.ones((1, 1), dtype=np.int8)
        f["chisq_z"] = np.zeros((1, 1))
        f["good_fraction"] = np.ones((1, 1))
        f["jump_counts"] = np.zeros((1, 1), dtype=np.int64)
        f["jump_locations"] = np.zeros(0, dtype=np.int64)
        f["jump_offsets"] = np.zeros(0)


def test_restart_reads_gains_as_stored_and_refuses_another_band_unit(tmp_path):
    """Chain gains are in the chain's band_unit, as in memory, so a restart copies them as they
    are. A parameter file with another band_unit would misread every gain, so it is refused."""
    chain = tmp_path / "band_chain.h5"
    _write_restart_chain(chain, "uK_CMB", abs_gain=2.5)
    params = Bunch(output=Bunch(chains=Bunch(include=Bunch())),
                   gibbs=Bunch(init_from_chain=str(chain)))
    my_band = Bunch(detectors=Bunch(d0=Bunch(gain=_GAIN)))

    band = _build_band(np.zeros(8, dtype=np.int64), np.zeros(8), np.zeros(8), unit="uK_CMB")
    samples = TODSamples(band, params, my_band, MPI.COMM_SELF, chain=1)
    assert samples.abs_gain == 2.5

    band = _build_band(np.zeros(8, dtype=np.int64), np.zeros(8), np.zeros(8), unit="uK_RJ")
    with pytest.raises(ValueError, match="band_unit"):
        TODSamples(band, params, my_band, MPI.COMM_SELF, chain=1)


def test_residual_map_is_zero_for_a_perfect_noiseless_model(monkeypatch):
    """A TOD that is exactly gain*sky leaves nothing behind, so maps/res must vanish."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    rng = np.random.default_rng(4)
    n = 256   # Long enough for the log-binned diagnostic PSD and the per-pixel IQU (3x3) solve.
    pix = rng.integers(0, _NPIX, n).astype(np.int64)
    psi = rng.uniform(0.0, np.pi, n)
    sky = rng.normal(scale=50.0, size=(3, _NPIX))

    maps = _run(_build_band(pix, psi, _GAIN*_project(sky, pix, psi)), sky)

    # float32 TODs at this signal amplitude, so compare at single precision.
    np.testing.assert_allclose(maps["res"], 0.0, atol=1e-3)
    np.testing.assert_allclose(maps["observed_sky"], sky, rtol=0, atol=1e-3)


@pytest.mark.parametrize("band_unit", ["uK_CMB", "K_CMB", "MJy/sr"])
def test_band_unit_changes_only_the_unit_of_the_outputs(monkeypatch, band_unit):
    """The same detector data analysed in another band_unit gives the same maps, converted.

    The TOD is in detector units and does not change. The analysis in `band_unit` gets its gain,
    sky model and orbital dipole in that unit, so a perfect model still leaves a zero residual and
    the binned map equals the uK_RJ map times the uK_RJ -> band_unit factor.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    rng = np.random.default_rng(9)
    n = 256
    pix = rng.integers(0, _NPIX, n).astype(np.int64)
    psi = rng.uniform(0.0, np.pi, n)
    sky_rj = rng.normal(scale=50.0, size=(3, _NPIX))
    velocity = np.array([3.0e4, -1.0e4, 2.0e4])   # A realistic orbital speed, in m/s.
    to_unit = unit_factor(30.0, "uK_RJ", band_unit)

    # The detector data: gain times (sky + orbital dipole), built in uK_RJ.
    reference_band = _build_band(pix, psi, np.zeros(n), velocity=velocity)
    dipole_rj = get_s_orb_tod(reference_band.scans[0].detectors[0], reference_band, pix)
    tod = _GAIN*(_project(sky_rj, pix, psi) + dipole_rj)

    maps_rj = _run(_build_band(pix, psi, tod, velocity=velocity), sky_rj)
    maps_unit = _run(_build_band(pix, psi, tod, unit=band_unit, velocity=velocity),
                     sky_rj*to_unit, gain=_GAIN/to_unit)

    np.testing.assert_allclose(maps_rj["res"], 0.0, atol=1e-3)
    np.testing.assert_allclose(maps_unit["res"], 0.0, atol=1e-3*to_unit)
    np.testing.assert_allclose(maps_unit["observed_sky"], maps_rj["observed_sky"]*to_unit,
                               rtol=1e-5, atol=1e-6*to_unit)
    np.testing.assert_allclose(maps_unit["rms"], maps_rj["rms"]*to_unit, rtol=1e-5)


def test_response_split_detectors_recover_sky_and_zero_residual(monkeypatch):
    """Separate intensity and polarization streams use their own sky-model response."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    rng = np.random.default_rng(17)
    n = 4096
    pix = rng.integers(0, _NPIX, n).astype(np.int64)
    psi = rng.uniform(0.0, np.pi, n)
    sky = rng.normal(size=(3, _NPIX))
    sky[0] *= 500.0
    sky[1:3] *= 5.0

    intensity_tod = _GAIN * sky[0, pix]
    polarization_tod = _GAIN * (
        sky[1, pix] * np.cos(2 * psi) + sky[2, pix] * np.sin(2 * psi)
    )
    intensity = _build_band(
        pix, psi, intensity_tod, response_I_P=(1.0, 0.0),
    ).scans[0].detectors[0]
    polarization = _build_band(
        pix, psi, polarization_tod, response_I_P=(0.0, 1.0),
    ).scans[0].detectors[0]
    intensity.name = "intensity"
    polarization.name = "polarization"
    polarization.det_idx_fullband = 1
    band = DetectorGroupTOD(
        [ScanTOD([intensity, polarization], 0.0, 0)], "EXP", "B", nside=_NSIDE,
        nu=30.0, unit="uK_RJ", fwhm=0.0, fsamp=1.0, ndet=2, pols="IQU",
        noise_model=SimpleNamespace(npar=1, params=np.array([np.nan])),
    )

    maps = _run(band, sky)

    np.testing.assert_allclose(maps["res"], 0.0, atol=1e-3)
    np.testing.assert_allclose(maps["observed_sky"], sky, rtol=0, atol=1e-3)

    # Hits count both streams independently, regardless of their intensity/polarization response.
    expected_hits = np.zeros(_NPIX, dtype=np.int64)
    np.add.at(expected_hits, pix, 2)
    np.testing.assert_array_equal(maps["nhit"], expected_hits)


def test_residual_map_recovers_an_injected_offset(monkeypatch):
    """Adding a constant to the intensity TOD must show up in maps/res, not just in the signal."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    rng = np.random.default_rng(5)
    n = 256
    pix = rng.integers(0, _NPIX, n).astype(np.int64)
    psi = rng.uniform(0.0, np.pi, n)
    sky = rng.normal(scale=50.0, size=(3, _NPIX))
    offset = 7.0   # In sky units, so the TOD gets gain*offset.

    maps = _run(_build_band(pix, psi, _GAIN*(_project(sky, pix, psi) + offset)), sky)

    # An unpolarized offset lands entirely in I; Q and U see it as a cos/sin average over psi.
    np.testing.assert_allclose(maps["res"][0], offset, rtol=1e-3)
    np.testing.assert_allclose(maps["observed_sky"][0], sky[0] + offset, rtol=1e-3)


@pytest.mark.parametrize("sparse_maps", [False, True])
@pytest.mark.parametrize("n_good", [0, 256])
def test_hit_map_counts_unflagged_samples_only(
    monkeypatch: pytest.MonkeyPatch, sparse_maps: bool, n_good: int,
) -> None:
    """Accumulate repeated hits without a dense per-scan temporary, even for empty masks."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    rng = np.random.default_rng(6)
    n, n_flagged = n_good, 40
    pix = rng.choice(np.array([2, 5, 11], dtype=np.int64), n + n_flagged)
    psi = rng.uniform(0.0, np.pi, n + n_flagged)
    flag = np.concatenate([np.zeros(n, np.int64), np.full(n_flagged, _BITMASK, np.int64)])
    sky = np.zeros((3, _NPIX))

    def unexpected_bincount(*args, **kwargs) -> np.ndarray:
        pytest.fail("Hit accumulation must not allocate a dense bincount array for each scan.")

    monkeypatch.setattr(np, "bincount", unexpected_bincount)
    maps = _run(_build_band(pix, psi, np.zeros(n + n_flagged), flag=flag), sky, sparse_maps)

    expected = np.zeros(_NPIX, dtype=np.int64)
    np.add.at(expected, pix[:n], 1)
    np.testing.assert_array_equal(maps["nhit"], expected)
    assert maps["nhit"].dtype == np.int64
    assert maps["nhit"].sum() == n


@pytest.mark.parametrize("sparse_maps", [False, True])
@pytest.mark.parametrize("response_I", [1.0, 0.5])
@pytest.mark.parametrize("all_flagged", [False, True])
def test_intensity_mapmaker_recovers_signal_rms_and_aux_maps(
    monkeypatch: pytest.MonkeyPatch, sparse_maps: bool, response_I: float, all_flagged: bool,
) -> None:
    """I-only maps ignore polarization response and retain the chain's IQU output layout.

    Under mpirun, the last rank has no scans and the others contribute overlapping pixels.
    The noise draw is fixed so signal subtraction and all auxiliary maps have analytic answers.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    comm = MPI.COMM_WORLD
    pix = np.tile(np.array([1, 5, 11], dtype=np.int64), 128)
    flag = np.zeros(pix.size, dtype=np.int64)
    flag[::7] = _BITMASK
    if all_flagged:
        flag[:] = _BITMASK
    sky = (10.0 + np.arange(_NPIX))[None, :]
    offset, corr_amplitude, sidelobe = 0.25, 2.0, 0.125
    band = _build_band(pix, np.zeros(pix.size), np.zeros(pix.size), flag,
                       response_I_P=(response_I, 0.05))
    band.pols = "I"
    det = band.scans[0].detectors[0]
    det.orbital_velocity_m_per_s[:] = [10000.0, 20000.0, 30000.0]
    orbital = get_s_orb_tod(det, band, pix, nthreads=1)
    n_corr = np.full(pix.size, _GAIN * response_I * corr_amplitude, dtype=np.float32)
    det._tod = (_GAIN * (response_I * (sky[0, pix] + offset + sidelobe) + orbital)
                + n_corr).astype(np.float32)
    counts = np.zeros(_NPIX, dtype=np.int64)
    np.add.at(counts, pix[flag == 0], 1)
    if comm.size > 1 and comm.rank == comm.size - 1:
        band.scans = []
        band.nscans = 0
        counts[:] = 0
    counts = comm.allreduce(counts)
    samples = _fake_tod_samples()
    samples.residual_tods = [[None]]
    samples.ncorr_cg_residual = np.zeros((1, 1))
    samples.ncorr_cg_niter = np.zeros((1, 1), dtype=np.int32)
    samples.ncorr_converged = np.zeros((1, 1), dtype=bool)
    samples.chain = 1

    def fixed_noise(*args, **kwargs) -> SimpleNamespace:
        return SimpleNamespace(n_corr=n_corr, noise_params=np.array([2.0]), residual=0.0,
                               niter=0, converged=True, high_var=False)

    monkeypatch.setattr(binned, "sample_correlated_noise", fixed_noise)
    monkeypatch.setattr(binned, "log_corr_noise_stats", lambda *args: None)
    far_beam = SimpleNamespace(get_projection=lambda pix, psi, idet:
                              np.full(pix.size, response_I * sidelobe))
    config = MapmakingConfig(
        mapmaker="bin", num_threads=1, include_orbital_dipole_maps=True,
        include_corr_noise_maps=True, include_sky_model_maps=True, include_residual_maps=True,
        include_sidelobe_maps=True, include_hit_maps=True, include_cov_maps=True,
        sparse_maps=sparse_maps,
    )
    detmaps, maps = binned.tod2map_bin(
        comm, band, sky, samples, 1, config,
        CorrelatedNoiseConfig(enabled=True, sample_sigma0=False),
        DataSelectionConfig(), far_beam,
    )
    if band.nscans:
        np.testing.assert_allclose(samples.residual_tods[0][0], _GAIN * response_I * offset,
                                   atol=2e-4)
    if comm.rank != 0:
        assert detmaps == maps == {}
        return

    observed = counts > 0
    assert set(detmaps) == {"I"}
    for name in ("observed_sky", "res", "corrnoise", "orbdipole", "sidelobe"):
        assert maps[name].shape == (3, _NPIX)
        np.testing.assert_array_equal(maps[name][1:], 0.0)
        np.testing.assert_array_equal(maps[name][0, ~observed], 0.0)
    np.testing.assert_allclose(maps["observed_sky"][0, observed], sky[0, observed] + offset,
                               rtol=1e-5, atol=1e-4)
    # The saved TOD and residual map both remove sky, orbit, n_corr and sidelobes.
    np.testing.assert_allclose(maps["res"][0, observed], offset, atol=1e-4)
    np.testing.assert_allclose(maps["corrnoise"][0, observed], corr_amplitude, atol=1e-6)
    np.testing.assert_allclose(maps["sidelobe"][0, observed], sidelobe, atol=1e-6)
    orbital_map = np.zeros(_NPIX)
    orbital_map[pix] = orbital / response_I
    np.testing.assert_allclose(maps["orbdipole"][0, observed], orbital_map[observed], rtol=1e-6)
    expected_rms = 2.0 / (_GAIN * response_I * np.sqrt(counts[observed]))
    np.testing.assert_allclose(maps["rms"][0, observed], expected_rms, rtol=1e-6)
    assert np.isinf(maps["rms"][0, ~observed]).all()
    assert np.isinf(maps["rms"][1:]).all()
    assert maps["cov"].shape == (6, _NPIX)
    np.testing.assert_allclose(maps["cov"][0], counts * (_GAIN * response_I / 2.0)**2)
    np.testing.assert_array_equal(maps["cov"][1:], 0.0)
    np.testing.assert_array_equal(maps["nhit"], counts)
