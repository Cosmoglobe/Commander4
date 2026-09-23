"""Check the standalone CMB solver's harmonic range and exclusion masks."""

import sys
from pathlib import Path
from types import SimpleNamespace

import camb
import healpy as hp
import h5py
import numpy as np
import pytest
import yaml
from scipy.special import sph_harm_y

from commander4.standalone_tools import constrained_cmb_realizations as cr


@pytest.mark.parametrize("lmax", [3, 11])
def test_masked_mean_matches_dense_posterior(lmax: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recover modes above 2*nside and also accept a prior shorter than that old cutoff."""
    monkeypatch.setattr(cr, "nthreads", 1)
    nside = 4
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    columns = []
    alm_columns = []
    ells = []
    # Independent real harmonics: m>0 has two real degrees of freedom, each with variance C_l.
    for m in range(lmax + 1):
        for ell in range(m, lmax + 1):
            y = sph_harm_y(ell, m, theta, phi)
            alm = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
            index = hp.Alm.getidx(lmax, ell, m)
            if m == 0:
                columns.append(y.real)
                alm[index] = 1.0
                alm_columns.append(alm)
                ells.append(ell)
            else:
                columns.append(np.sqrt(2) * y.real)
                alm[index] = 1 / np.sqrt(2)
                alm_columns.append(alm.copy())
                ells.append(ell)
                columns.append(-np.sqrt(2) * y.imag)
                alm[index] = 1j / np.sqrt(2)
                alm_columns.append(alm)
                ells.append(ell)
    design = np.array(columns).T
    basis = np.array(alm_columns).T
    ells = np.array(ells)
    cl = 2 / (np.arange(lmax + 1) + 1.0)**2
    beam = np.radians(12.0)
    design *= hp.gauss_beam(beam, lmax=lmax)[ells]
    rng = np.random.default_rng(71)
    truth = rng.normal(size=len(ells)) * np.sqrt(cl[ells])
    ivar = 3 + np.cos(theta)
    mask = (np.abs(np.cos(theta)) > 0.3).astype(float)
    data = design @ truth + rng.normal(size=theta.size) / np.sqrt(ivar)
    weight = ivar * mask
    precision = np.diag(1 / cl[ells]) + design.T @ (weight[:, None] * design)
    expected = np.linalg.solve(precision, design.T @ (weight * data))

    solver = cr.ConstrainedCMB(data[None, :], ivar[None, :], cl, masks=mask[None, :],
                               beam_fwhm=np.array([beam]), maxiter=500)
    assert solver.lmax == lmax
    actual_alm = solver.solve_CG(solver.LHS_func, solver.get_RHS_eqn_mean(), err_tol=1e-22)
    alm_weight = np.full(solver.alm_len, 2.0)
    alm_weight[:lmax + 1] = 1.0
    actual = (basis.conj().T @ (alm_weight * actual_alm)).real
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    fluctuation = solver.get_RHS_eqn_fluct()
    assert fluctuation.shape == actual_alm.shape == (hp.Alm.getsize(lmax),)
    assert np.all(np.isfinite(fluctuation))
    assert np.any(np.abs(actual[ells == lmax]) > 1e-3)


@pytest.mark.parametrize("lmax", [3, 11])
@pytest.mark.parametrize("apodization_deg", [None, 20.0])
def test_cli_uses_component_lmax(
    lmax: int, apodization_deg: float | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Carry the component cutoff and optional mask taper through chain loading and sampling."""
    monkeypatch.setattr(cr, "nthreads", 1)
    params = {
        "components": {"CMB": {"enabled": True, "component_class": "CMB", "params": {
            "polarization": "I", "shortname": "cmb", "lmax": lmax,
            "spatially_varying_MM": False, "Cl_prior_amplitude": None,
        }}},
        "compsep": {"double_precision": True},
        "experiments": {"Sim": {"bands": {"Band100": {"freq": 100.0}}}},
    }
    (tmp_path / "chains_compsep").mkdir()
    (tmp_path / "chains_bands").mkdir()
    with h5py.File(tmp_path / "chains_compsep/chain01_iter0001.h5", "w") as handle:
        handle["metadata/parameter_file_as_string"] = yaml.safe_dump(params)
        handle["comps/cmb/alms"] = np.zeros((1, hp.Alm.getsize(lmax)), dtype=complex)
    npix = hp.nside2npix(4)
    binary_mask = np.ones(npix)
    binary_mask[:20] = 0
    mask_path = tmp_path / "mask.fits"
    hp.write_map(mask_path, binary_mask, dtype=np.float64)
    with h5py.File(tmp_path / "chains_bands/Sim_Band100_chain01_iter0001.h5", "w") as handle:
        handle["maps/observed_sky"] = np.ones((1, npix))
        handle["maps/rms"] = np.ones((1, npix))
        handle["metadata/band_unit"] = "uK_CMB"
        handle["metadata/map_fwhm_arcmin"] = 30.0

    def theory_spectra(*args, **kwargs) -> dict[str, np.ndarray]:
        return {"total": np.ones((200, 4))}

    # Avoid the unrelated CAMB calculation, while retaining real chain loading and CG sampling.
    monkeypatch.setattr(camb, "get_results", lambda pars: SimpleNamespace(
        get_cmb_power_spectra=theory_spectra))
    original_solve = cr.ConstrainedCMB.solve_CG
    solved = []

    def record_solve(self, lhs, rhs, err_tol: float = 1e-6) -> np.ndarray:
        assert self.lmax == lmax
        assert len(self.Cl_prior) == lmax + 1
        np.testing.assert_array_equal(self.masks[0, :20], 0.0)
        if apodization_deg is None:
            np.testing.assert_array_equal(self.masks[0], binary_mask)
        else:
            assert np.any((self.masks[0] > 0) & (self.masks[0] < 1))
        result = original_solve(self, lhs, rhs, err_tol)
        solved.append(result)
        return result

    monkeypatch.setattr(cr.ConstrainedCMB, "solve_CG", record_solve)
    argv = ["c4-cmb-realizations", str(tmp_path), "--iter", "1", "--mask", str(mask_path)]
    if apodization_deg is not None:
        argv.extend(["--mask-fwhm-deg", str(apodization_deg)])
    monkeypatch.setattr(sys, "argv", argv)
    assert cr.main() == 0
    assert len(solved) == 1
    assert solved[0].size == hp.Alm.getsize(lmax)
    output = hp.read_map(tmp_path / "cmb_realizations/chain01_iter0001_cmb_realization.fits")
    np.testing.assert_allclose(output, cr.alm2map(solved[0], 4, lmax), atol=1e-10)


@pytest.mark.parametrize("nside_out", [4, 8, 16])
@pytest.mark.parametrize("nested", [False, True])
def test_mask_preserves_exclusion_at_each_resolution(
    nside_out: int, nested: bool, tmp_path: Path
) -> None:
    """A single excluded child must not reopen when resizing, including NESTED FITS input."""
    # In NESTED ordering, each group of four nside-8 pixels shares an nside-4 parent.
    mask = np.ones(hp.nside2npix(8))
    mask[12] = 0
    mask[40:44] = 0
    stored_mask = mask if nested else hp.reorder(mask, n2r=True)
    path = tmp_path / "mask.fits"
    hp.write_map(path, stored_mask, nest=nested, dtype=np.float64)
    hard = cr._read_mask(str(path), nside_out, 0.0)
    apodized = cr._read_mask(str(path), nside_out, 10.0)
    expected = np.ones(hp.nside2npix(nside_out))
    if nside_out == 4:
        expected[[3, 10]] = 0
    elif nside_out == 8:
        expected[[12, 40, 41, 42, 43]] = 0
    else:
        expected[48:52] = 0
        expected[160:176] = 0
    expected = hp.reorder(expected, n2r=True)
    np.testing.assert_array_equal(hard, expected)
    np.testing.assert_array_equal(apodized == 0, expected == 0)
    assert np.all(np.isfinite(apodized))
    assert np.all((apodized >= 0) & (apodized <= 1))


def test_outward_taper_around_a_single_masked_pixel(tmp_path: Path) -> None:
    """Small mask holes stay excluded and the taper rises monotonically with spherical distance."""
    nside = 16
    npix = hp.nside2npix(nside)
    excluded = npix // 2
    mask = np.ones(npix)
    mask[excluded] = 0
    path = tmp_path / "mask.fits"
    hp.write_map(path, mask, dtype=np.float64)
    vectors = np.array(hp.pix2vec(nside, np.arange(npix)))
    angles = hp.rotator.angdist(vectors[:, excluded], vectors)
    probe = hp.get_all_neighbours(nside, excluded)[0]
    fwhm = 2.0 * np.degrees(angles[probe])
    weights = cr._read_mask(str(path), nside, fwhm)
    assert weights[excluded] == 0
    assert weights[probe] == pytest.approx(0.5, abs=1e-12)
    assert np.all(np.diff(weights[np.argsort(angles)]) >= -1e-12)
    assert np.all(weights[angles > np.radians(5*fwhm)] > 0.999999)


@pytest.mark.parametrize("value", [0.0, 1.0])
@pytest.mark.parametrize("width", [0.0, 3.0])
def test_uniform_masks_are_unchanged(value: float, width: float, tmp_path: Path) -> None:
    path = tmp_path / "mask.fits"
    hp.write_map(path, np.full(hp.nside2npix(4), value), dtype=np.float64)
    np.testing.assert_array_equal(cr._read_mask(str(path), 8, width), value)


def test_apodized_mask_excludes_contamination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing excluded data cannot change the constrained realization with an outward taper."""
    monkeypatch.setattr(cr, "nthreads", 1)
    nside = 8
    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    binary_mask = (np.abs(np.cos(theta)) > 0.3).astype(float)
    path = tmp_path / "mask.fits"
    hp.write_map(path, binary_mask, dtype=np.float64)
    weights = cr._read_mask(str(path), nside, 20.0)
    assert np.any((weights > 0) & (weights < 1))
    rng = np.random.default_rng(7)
    data = rng.normal(size=(1, theta.size))
    cl = 1 / (np.arange(16) + 1.0)**2
    solver = cr.ConstrainedCMB(data, np.ones_like(data), cl, masks=weights[None, :])
    rhs = solver.get_RHS_eqn_mean() + solver.get_RHS_eqn_fluct()
    original = solver.solve_CG(solver.LHS_func, rhs, err_tol=1e-16)
    old_mean_rhs = solver.get_RHS_eqn_mean()
    solver.map_sky[0, binary_mask == 0] += 1e12
    new_mean_rhs = solver.get_RHS_eqn_mean()
    np.testing.assert_array_equal(new_mean_rhs, old_mean_rhs)
    contaminated_rhs = rhs + (new_mean_rhs - old_mean_rhs)
    contaminated = solver.solve_CG(solver.LHS_func, contaminated_rhs, err_tol=1e-16)
    np.testing.assert_array_equal(contaminated, original)


@pytest.mark.parametrize("width", [-1.0, np.nan, np.inf])
def test_invalid_mask_width_is_rejected(width: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        cr._read_mask("unused.fits", 8, width)
