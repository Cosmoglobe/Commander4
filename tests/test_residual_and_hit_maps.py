"""The binned mapmaker's residual and hit maps, Commander3's `tod_<freq>_res` and its companion.

`maps/res` is the noise residual `_record_tod_diagnostics` already builds per detector-scan (data
minus sky model, orbital dipole and correlated noise), binned with the same inverse-variance
weights as the signal map. So a TOD that is exactly `gain * sky` must bin to a residual of zero
while still producing the right signal map -- that is the property these tests pin.

`maps/nhit` counts unflagged samples per pixel, which no other output records: `maps/rms` counts
them weighted by `(gain/sigma0)^2`, so it cannot be inverted back into a sample count in general.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from mpi4py import MPI

from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.scan_tod import ScanTOD
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.pointing import PixelPointing
import commander4.tod.processing as tod_processing
import commander4.tod.mapmaking.binned as binned
from commander4.tod.sky_projection import get_s_orb_tod

_BITMASK = 1
_NSIDE = 1
_NPIX = 12*_NSIDE**2
_GAIN = 1.5   # abs_gain below; rel and temporal gain are zero, so this is the whole gain.


def _build_band(pix, psi, tod, flag=None,
                response_I_P: tuple[float, float] | None = None) -> DetectorGroupTOD:
    """One IQU detector-scan with uncompressed pointing and no orbital motion.

    The zero orbital velocity is what makes the expected residual exactly the noise: with no
    spacecraft velocity the orbital-dipole TOD the mapmaker subtracts is identically zero, so the
    only model term left is the sky projection.
    """
    ntod = pix.size
    pointing = PixelPointing(pix.astype(np.int64), psi.astype(np.float64), np.array([0], np.int64),
                             None, None, _NSIDE, _NSIDE, ntod, ntod)
    det = DetectorTOD(
        name="d0", det_idx_fullband=0, tod=tod.astype(np.float32), pointing=pointing,
        sampling_rate_hz=1.0, orbital_velocity_m_per_s=np.zeros(3, dtype=np.float32),
        huffman_tree=None, huffman_symbols=None, default_proc_mask=np.ones(_NPIX, bool),
        specific_proc_masks={},
        flag_encoded=(np.zeros(ntod) if flag is None else flag).astype(np.int64),
        bad_data_bitmask=_BITMASK, flag_is_compressed=False,
        response_I_P=response_I_P,
    )
    noise_model = SimpleNamespace(npar=1, params=np.array([np.nan]))
    return DetectorGroupTOD([ScanTOD([det], 0.0, 0)], "EXP", "B", nside=_NSIDE, nu=30.0, fwhm=0.0,
                            fsamp=1.0, ndet=1, pols="IQU", noise_model=noise_model)


def _fake_tod_samples(sigma0: float = 2.0, ndet: int = 1) -> SimpleNamespace:
    """Minimal stand-in exposing exactly the fields tod2map_bin / TODView / the diagnostics read."""
    no_jump = SimpleNamespace(is_empty=lambda: True)
    empty_ps = lambda: np.full((1, ndet, 100), np.nan, dtype=np.float32)
    return SimpleNamespace(
        noise_params=np.full((1, ndet, 1), sigma0), abs_gain=_GAIN, rel_gain=np.zeros(ndet),
        temporal_gain=np.zeros((1, ndet)), jumps=SimpleNamespace(get=lambda iscan, idet: no_jump),
        accept=np.ones((1, ndet), dtype=bool), band_unit_factor=1.0, band_unit="uK_RJ",
        chisq_z=np.full((1, ndet), np.nan), good_fraction=np.full((1, ndet), np.nan),
        TOD_PS_NBIN=100, tod_ps_freqs=empty_ps(), tod_ps_raw=empty_ps(), tod_ps_residual=empty_ps(),
        tod_ps_ncorrsub=empty_ps(), tod_ps_ncorr=empty_ps(), ncorr_tods=None)


def _run(band: DetectorGroupTOD, sky_model: np.ndarray,
         sparse_maps: bool = False) -> dict[str, np.ndarray]:
    mapmaking = tod_processing.MapmakingConfig(
        mapmaker="bin", num_threads=1,
        include_orbital_dipole_maps=False, include_corr_noise_maps=False,
        include_sky_model_maps=False, include_residual_maps=True, include_hit_maps=True,
        sparse_maps=sparse_maps, common_res_fwhm=0.0,
    )
    _, maps = tod_processing.tod2map_bin(
        MPI.COMM_SELF, band, sky_model, _fake_tod_samples(ndet=band.ndet), 1, mapmaking,
        tod_processing.CorrelatedNoiseConfig(sample_sigma0=False),
        tod_processing.DataSelectionConfig(),
    )
    return maps


def _project(sky: np.ndarray, pix: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """The IQU sky along a pointing: I + Q cos(2 psi) + U sin(2 psi)."""
    return sky[0, pix] + sky[1, pix]*np.cos(2*psi) + sky[2, pix]*np.sin(2*psi)


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
        nu=30.0, fwhm=0.0, fsamp=1.0, ndet=2, pols="IQU",
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
    config = tod_processing.MapmakingConfig(
        mapmaker="bin", num_threads=1, include_orbital_dipole_maps=True,
        include_corr_noise_maps=True, include_sky_model_maps=True, include_residual_maps=True,
        include_sidelobe_maps=True, include_hit_maps=True, include_cov_maps=True,
        sparse_maps=sparse_maps,
    )
    detmaps, maps = binned.tod2map_bin(
        comm, band, sky, samples, 1, config,
        tod_processing.CorrelatedNoiseConfig(enabled=True, sample_sigma0=False),
        tod_processing.DataSelectionConfig(), far_beam,
    )
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
    # Diagnostics subtract sky, orbit and n_corr; the separately removed sidelobe remains in res.
    np.testing.assert_allclose(maps["res"][0, observed], offset + sidelobe, atol=1e-4)
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
