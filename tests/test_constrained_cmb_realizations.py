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
from astropy.io import fits
from scipy.special import sph_harm_y

from commander4.parameters.bunch import as_bunch_recursive
from commander4.standalone_tools import constrained_cmb_realizations as cr
from commander4.units import SUPPORTED_BAND_UNITS, rj_to_band_unit_factor


@pytest.mark.parametrize("nsides", [[4], [2, 4, 8]])
@pytest.mark.parametrize("lmax", [3, 11])
@pytest.mark.parametrize("precond_lmax", [0, 2, 32])
def test_masked_mean_matches_dense_posterior(
    nsides: list[int], lmax: int, precond_lmax: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check native-grid likelihoods and draws against independent real spherical harmonics."""
    monkeypatch.setattr(cr, "nthreads", 1)
    designs = []
    masks = []
    ivars = []
    beams = np.radians(np.linspace(12.0, 30.0, len(nsides)))
    for band, nside in enumerate(nsides):
        theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        columns = []
        alm_columns = []
        ells = []
        # m>0 has two real degrees of freedom, each with variance C_l.
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
        design *= hp.gauss_beam(beams[band], lmax=lmax)[ells]
        designs.append(design)
        ivars.append((band + 1) * (3 + np.cos(theta)))
        masks.append((np.abs(np.cos(theta)) > 0.3).astype(float))
    basis = np.array(alm_columns).T
    ells = np.array(ells)
    cl = 2 / (np.arange(lmax + 1) + 1.0)**2
    rng = np.random.default_rng(71)
    truth = rng.normal(size=len(ells)) * np.sqrt(cl[ells])
    precision = np.diag(1 / cl[ells])
    mean_rhs = np.zeros(len(ells))
    maps = []
    for design, ivar, mask in zip(designs, ivars, masks):
        data = design @ truth + rng.normal(size=ivar.size) / np.sqrt(ivar)
        maps.append(data)
        weight = ivar * mask
        precision += design.T @ (weight[:, None] * design)
        mean_rhs += design.T @ (weight * data)
    expected = np.linalg.solve(precision, mean_rhs)

    solver = cr.ConstrainedCMB(maps, ivars, cl, masks=masks, beam_fwhm=beams,
                               maxiter=500, precond_lmax=precond_lmax)
    assert solver.lmax == lmax
    assert solver.nsides == nsides
    expected_diagonal = np.ones(lmax + 1)
    for band, (ivar, mask) in enumerate(zip(ivars, masks)):
        beam = hp.gauss_beam(beams[band], lmax=lmax)
        expected_diagonal += cl * beam**2 * np.sum(ivar * mask) / (4*np.pi)
    np.testing.assert_allclose(solver._precond_ell, 1 / expected_diagonal, rtol=1e-14)
    if precond_lmax >= lmax and lmax >= max(nsides):
        # With no grid coarsening and all modes in the block, its inverse is exact.
        probe = basis @ truth
        np.testing.assert_allclose(solver.preconditioner(solver.LHS_func(probe)), probe, atol=1e-11)
    actual_alm = solver.solve_CG(solver.LHS_func, solver.get_RHS_eqn_mean(), err_tol=1e-22)
    alm_weight = np.full(solver.alm_len, 2.0)
    alm_weight[:lmax + 1] = 1.0
    actual = (basis.conj().T @ (alm_weight * actual_alm)).real
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    assert np.any(np.abs(actual[ells == lmax]) > 1e-3)

    # Supply known independent standard normals, then compare the complete draw with a dense solve.
    # This checks the prior term and each band's pixel-noise term without using the solver's SHTs.
    prior_noise = rng.normal(size=len(ells))
    pixel_noise = []
    fluct_rhs = prior_noise / np.sqrt(cl[ells])
    for design, ivar, mask in zip(designs, ivars, masks):
        noise = rng.normal(size=ivar.size)
        pixel_noise.append(noise)
        fluct_rhs += design.T @ (np.sqrt(ivar * mask) * noise)
    monkeypatch.setattr(cr.hp, "synalm", lambda *args: basis @ prior_noise)
    noises = iter(pixel_noise)

    def normal(mean: float, sigma: float, size: int) -> np.ndarray:
        noise = next(noises)
        assert size == noise.size
        return noise

    monkeypatch.setattr(cr.np.random, "normal", normal)
    fluctuation = solver.get_RHS_eqn_fluct()
    fluct_coords = (basis.conj().T @ (alm_weight * fluctuation)).real
    np.testing.assert_allclose(fluct_coords, np.sqrt(cl[ells]) * fluct_rhs, atol=1e-11)
    rhs = solver.get_RHS_eqn_mean() + fluctuation
    draw = solver.solve_CG(solver.LHS_func, rhs, err_tol=1e-22)
    draw_coords = (basis.conj().T @ (alm_weight * draw)).real
    expected_draw = np.linalg.solve(precision, mean_rhs + fluct_rhs)
    np.testing.assert_allclose(draw_coords, expected_draw, atol=1e-9)


@pytest.mark.parametrize("lmax", [3, 11])
@pytest.mark.parametrize("apodization_deg", [None, 20.0])
@pytest.mark.parametrize("precond_lmax", [0, 32])
def test_cli_uses_component_lmax(
    lmax: int, apodization_deg: float | None, precond_lmax: int,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        assert (self._lowell_factor is not None) == (precond_lmax > 0)
        np.testing.assert_array_equal(self.masks[0][:20], 0.0)
        if apodization_deg is None:
            np.testing.assert_array_equal(self.masks[0], binary_mask)
        else:
            assert np.any((self.masks[0] > 0) & (self.masks[0] < 1))
        result = original_solve(self, lhs, rhs, err_tol)
        solved.append(result)
        return result

    monkeypatch.setattr(cr.ConstrainedCMB, "solve_CG", record_solve)
    argv = ["c4-cmb-realizations", str(tmp_path), "--iter", "1", "--mask", str(mask_path)]
    if precond_lmax == 0:
        argv.extend(["--precond-lmax", "0"])
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
    solver.map_sky[0][binary_mask == 0] += 1e12
    new_mean_rhs = solver.get_RHS_eqn_mean()
    np.testing.assert_array_equal(new_mean_rhs, old_mean_rhs)
    contaminated_rhs = rhs + (new_mean_rhs - old_mean_rhs)
    contaminated = solver.solve_CG(solver.LHS_func, contaminated_rhs, err_tol=1e-16)
    np.testing.assert_array_equal(contaminated, original)


@pytest.mark.parametrize("width", [-1.0, np.nan, np.inf])
def test_invalid_mask_width_is_rejected(width: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        cr._read_mask("unused.fits", 8, width)


def test_coarse_preconditioner_is_symmetric_positive_and_preserves_likelihood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The coarse-grid block is a CG-safe approximation and changes no likelihood operations."""
    monkeypatch.setattr(cr, "nthreads", 1)
    rng = np.random.default_rng(9)
    nside, lmax = 8, 11
    npix = hp.nside2npix(nside)
    sky = rng.normal(size=(2, npix))
    ivar = rng.uniform(1, 5, size=(2, npix))
    masks = (rng.uniform(size=(2, npix)) > 0.3).astype(float)
    cl = 1 / (np.arange(lmax + 1) + 1.0)**2
    cl[:2] = 0.0
    solver = cr.ConstrainedCMB(sky, ivar, cl, masks=masks, precond_lmax=4)
    diagonal = cr.ConstrainedCMB(sky, ivar, cl, masks=masks, precond_lmax=0)
    x = rng.normal(size=solver.alm_len) + 1j*rng.normal(size=solver.alm_len)
    y = rng.normal(size=solver.alm_len) + 1j*rng.normal(size=solver.alm_len)
    x[:lmax + 1] = x[:lmax + 1].real
    y[:lmax + 1] = y[:lmax + 1].real
    original = x.copy()
    mx = solver.preconditioner(x)
    my = solver.preconditioner(y)
    assert solver.dot_alm(x, my) == pytest.approx(solver.dot_alm(mx, y), rel=1e-12)
    assert solver.dot_alm(x, mx) > 0
    np.testing.assert_array_equal(x, original)
    np.testing.assert_allclose(solver.preconditioner(x + 2*y), mx + 2*my, atol=1e-12)
    np.testing.assert_array_equal(solver.LHS_func(x), diagonal.LHS_func(x))
    np.testing.assert_array_equal(solver.get_RHS_eqn_mean(), diagonal.get_RHS_eqn_mean())
    ell, _ = hp.Alm.getlm(lmax)
    np.testing.assert_array_equal(mx[ell > 4], diagonal.preconditioner(x)[ell > 4])


def test_full_lowell_block_accelerates_the_same_solution(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the block spans the full system, CG should solve in one step with the same answer."""
    monkeypatch.setattr(cr, "nthreads", 1)
    rng = np.random.default_rng(17)
    nside, lmax = 4, 11
    npix = hp.nside2npix(nside)
    sky = rng.normal(size=(1, npix))
    ivar = np.full_like(sky, 1000.0)
    masks = np.ones_like(sky)
    masks[:, :npix//3] = 0
    cl = 1 / (np.arange(lmax + 1) + 1.0)**2
    solutions = []
    counts = []
    for cutoff in [0, lmax]:
        solver = cr.ConstrainedCMB(sky, ivar, cl, masks=masks, maxiter=500, precond_lmax=cutoff)
        count = 0

        def counted_lhs(x: np.ndarray) -> np.ndarray:
            nonlocal count
            count += 1
            return solver.LHS_func(x)

        solutions.append(solver.solve_CG(counted_lhs, solver.get_RHS_eqn_mean(), err_tol=1e-20))
        counts.append(count)
    assert counts[1] == 1
    assert counts[0] > 10
    np.testing.assert_allclose(solutions[0], solutions[1], atol=1e-8)


def test_no_data_preconditioner_and_prior_draw_are_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cr, "nthreads", 1)
    sky = np.zeros((1, hp.nside2npix(4)))
    solver = cr.ConstrainedCMB(sky, np.ones_like(sky), np.ones(9), masks=np.zeros_like(sky))
    assert solver._lowell_factor is None
    np.testing.assert_array_equal(solver._precond_ell, 1.0)
    rhs = solver.get_RHS_eqn_fluct()
    np.testing.assert_allclose(solver.solve_CG(solver.LHS_func, rhs), rhs, atol=1e-14)


def test_full_sky_uses_diagonal_preconditioner() -> None:
    sky = np.zeros((1, hp.nside2npix(4)))
    solver = cr.ConstrainedCMB(sky, np.ones_like(sky), np.ones(9))
    assert solver._lowell_factor is None


def test_invalid_preconditioner_cutoff_is_rejected() -> None:
    sky = np.zeros((1, hp.nside2npix(4)))
    with pytest.raises(ValueError, match="Preconditioner lmax"):
        cr.ConstrainedCMB(sky, np.ones_like(sky), np.ones(9), precond_lmax=-1)


@pytest.mark.parametrize("nsides", [[4, 4], [2, 4]])
def test_updated_data_reuses_lowell_factor_and_solves_current_likelihood(
    nsides: list[int], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing noise and beams refreshes the physics without refactoring the shared block."""
    monkeypatch.setattr(cr, "nthreads", 1)
    rng = np.random.default_rng(51)
    lmax = 8
    sky, ivar, masks, new_sky, new_ivar = [], [], [], [], []
    for nside in nsides:
        npix = hp.nside2npix(nside)
        sky.append(rng.normal(size=npix))
        ivar.append(rng.uniform(1, 2, size=npix))
        mask = np.ones(npix)
        mask[:npix//3] = 0
        masks.append(mask)
        new_sky.append(rng.normal(size=npix))
        new_ivar.append(ivar[-1] * rng.uniform(2, 8, size=npix))
    cl = 2 / (np.arange(lmax + 1) + 1.0)**2
    solver = cr.ConstrainedCMB(sky, ivar, cl, masks=masks, maxiter=500)
    factor = solver._lowell_factor
    first_diagonal = solver._precond_ell.copy()
    new_beams = np.radians([8.0, 15.0])

    def forbid_rebuild(*args, **kwargs) -> None:
        pytest.fail("The low-ell matrix must not be rebuilt between Gibbs samples.")

    monkeypatch.setattr(solver, "_build_lowell_preconditioner", forbid_rebuild)
    solver.update_data(new_sky, new_ivar, new_beams)
    assert solver._lowell_factor is factor
    assert not np.array_equal(solver._precond_ell, first_diagonal)
    reference = cr.ConstrainedCMB(new_sky, new_ivar, cl, masks=masks, beam_fwhm=new_beams,
                                  maxiter=500, precond_lmax=0)
    np.testing.assert_array_equal(solver._precond_ell, reference._precond_ell)
    np.testing.assert_array_equal(solver.get_RHS_eqn_mean(), reference.get_RHS_eqn_mean())
    rhs = reference.get_RHS_eqn_mean() + reference.get_RHS_eqn_fluct()
    np.testing.assert_array_equal(solver.LHS_func(rhs), reference.LHS_func(rhs))
    actual = solver.solve_CG(solver.LHS_func, rhs, err_tol=1e-22)
    expected = reference.solve_CG(reference.LHS_func, rhs, err_tol=1e-22)
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    with pytest.raises(ValueError, match="same band count and map nside"):
        solver.update_data([new_sky[0][:12], new_sky[1]], new_ivar, new_beams)


@pytest.mark.parametrize("nsides", [[2], [2, 4, 4]])
@pytest.mark.parametrize("selection, selected", [
    ([], [1, 3, 5, 8]),
    (["--burn-in", "3"], [5, 8]),
    (["--iter", "3", "8"], [3, 8]),
    (["--burn-in", "8"], []),
])
def test_cli_reuses_setup_across_iterations_and_realizations(
    nsides: list[int], selection: list[str], selected: list[int],
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sparse Gibbs numbering, changing foregrounds/noise, and three independent draws per sample."""
    monkeypatch.setattr(cr, "nthreads", 1)
    lmax = 3
    bands = {}
    for band in range(len(nsides)):
        bands[f"Band{band}"] = {"freq": 100.0}
    params = {
        "components": {
            "CMB": {"enabled": True, "component_class": "CMB", "params": {
                "polarization": "I", "shortname": "cmb", "lmax": lmax,
                "spatially_varying_MM": False, "Cl_prior_amplitude": None,
            }},
            "Dust": {"enabled": True, "component_class": "ThermalDust", "params": {
                "polarization": "I", "shortname": "dust", "lmax": lmax,
                "spatially_varying_MM": False, "Cl_prior_amplitude": None,
                "nu_ref": 100.0, "beta": 1.54, "T": 20.0,
            }},
        },
        "compsep": {"double_precision": True},
        "experiments": {"Sim": {"bands": bands}},
    }
    (tmp_path / "chains_compsep").mkdir()
    (tmp_path / "chains_bands").mkdir()
    for iteration in [1, 3, 5, 8]:
        compsep_path = tmp_path / f"chains_compsep/chain01_iter{iteration:04d}.h5"
        dust = np.zeros((1, hp.Alm.getsize(lmax)), dtype=complex)
        dust[0, 0] = 2*iteration*np.sqrt(4*np.pi)
        with h5py.File(compsep_path, "w") as handle:
            handle["metadata/parameter_file_as_string"] = yaml.safe_dump(params)
            handle["comps/cmb/alms"] = np.zeros_like(dust)
            handle["comps/dust/alms"] = dust
        for band, nside in enumerate(nsides):
            npix = hp.nside2npix(nside)
            band_path = tmp_path / f"chains_bands/Sim_Band{band}_chain01_iter{iteration:04d}.h5"
            with h5py.File(band_path, "w") as handle:
                handle["maps/observed_sky"] = np.full((1, npix), 12.0 + iteration)
                handle["maps/rms"] = np.full((1, npix), 0.5 + 0.1*iteration)
                handle["metadata/band_unit"] = "uK_RJ"
                handle["metadata/map_fwhm_arcmin"] = 10.0*iteration
    mask = np.ones(hp.nside2npix(max(nsides)))
    mask[:12] = 0
    mask_path = tmp_path / "mask.fits"
    hp.write_map(mask_path, mask, dtype=np.float64)

    events = []
    factors = []
    original_load = cr._load_iteration
    original_mask = cr._read_mask
    original_build = cr.ConstrainedCMB._build_lowell_preconditioner
    original_mean = cr.ConstrainedCMB.get_RHS_eqn_mean
    original_solve = cr.ConstrainedCMB.solve_CG

    def load_sample(params, iteration, *args):
        events.append(("load", iteration))
        result = original_load(params, iteration, *args)
        conversion = rj_to_band_unit_factor(100.0, "uK_CMB")
        for band, nside in enumerate(nsides):
            assert result.signal_maps[band].size == hp.nside2npix(nside)
            np.testing.assert_allclose(result.signal_maps[band],
                                       (12.0 - iteration)*conversion, atol=1e-5)
            np.testing.assert_allclose(result.ivar_maps[band],
                                       1/((0.5 + 0.1*iteration)*conversion)**2)
        return result

    def read_mask(*args):
        events.append(("mask", None))
        return original_mask(*args)

    def get_results(pars):
        events.append(("prior", None))
        return SimpleNamespace(get_cmb_power_spectra=lambda *args, **kwargs: {
            "total": np.ones((200, 4))})

    def build(self, weights, cutoff):
        events.append(("build", None))
        return original_build(self, weights, cutoff)

    def mean(self):
        events.append(("mean", None))
        return original_mean(self)

    def solve(self, lhs, rhs, err_tol=1e-6):
        events.append(("solve", None))
        assert self.nsides == nsides
        for band, nside in enumerate(nsides):
            expected_mask = (hp.ud_grade(mask, nside) == 1.0).astype(float)
            np.testing.assert_array_equal(self.masks[band], expected_mask)
        factors.append(self._lowell_factor)
        return original_solve(self, lhs, rhs, err_tol)

    monkeypatch.setattr(cr, "_load_iteration", load_sample)
    monkeypatch.setattr(cr, "_read_mask", read_mask)
    monkeypatch.setattr(camb, "get_results", get_results)
    monkeypatch.setattr(cr.ConstrainedCMB, "_build_lowell_preconditioner", build)
    monkeypatch.setattr(cr.ConstrainedCMB, "get_RHS_eqn_mean", mean)
    monkeypatch.setattr(cr.ConstrainedCMB, "solve_CG", solve)
    monkeypatch.setattr(sys, "argv", ["c4-cmb-realizations", str(tmp_path), "--mask", str(mask_path),
                                     "--n-realizations", "3", *selection])
    initial_figures = cr.plt.get_fignums()
    assert cr.main() == (0 if selected else 1)
    assert cr.plt.get_fignums() == initial_figures
    expected_events = []
    if selected:
        expected_events = [("load", selected[0])]
        expected_events.extend([("mask", None)] * len(set(nsides)))
        expected_events.extend([("prior", None), ("build", None)])
        for index, iteration in enumerate(selected):
            if index > 0:
                expected_events.append(("load", iteration))
            expected_events.append(("mean", None))
            expected_events.extend([("solve", None)]*3)
        for factor in factors:
            assert factor is factors[0]
    assert events == expected_events
    output_dir = tmp_path / "cmb_realizations"
    assert len(list(output_dir.glob("*.fits"))) == 3*len(selected)
    assert len(list(output_dir.glob("*.png"))) == 6*len(selected)
    for iteration in selected:
        maps = []
        for realization in [1, 2, 3]:
            path = output_dir / (
                f"chain01_iter{iteration:04d}_real{realization:04d}_cmb_realization.fits")
            maps.append(hp.read_map(path))
            assert maps[-1].size == hp.nside2npix(max(nsides))
            assert np.all(np.isfinite(maps[-1]))
            with fits.open(path) as handle:
                assert handle[1].header["CHAIN"] == 1
                assert handle[1].header["ITER"] == iteration
                assert handle[1].header["REALIZ"] == realization
                assert handle[1].header["BUNIT"] == "uK_CMB"
        for index in range(1, 3):
            assert not np.array_equal(maps[0], maps[index])


@pytest.mark.parametrize("options", [
    ["--n-realizations", "0"], ["--burn-in", "-1"], ["--burn-in", "3", "--iter", "5"],
])
def test_invalid_batch_options_are_rejected(options: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["c4-cmb-realizations", "unused", *options])
    with pytest.raises(SystemExit) as exc:
        cr.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("stored_unit", [*SUPPORTED_BAND_UNITS, "mixed"])
def test_cli_uses_thermodynamic_units_throughout(
    stored_unit: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equivalent band units give the same residuals, noise weights, and sampled CMB map."""
    monkeypatch.setattr(cr, "nthreads", 1)
    nside, lmax = 4, 5
    npix = hp.nside2npix(nside)
    cmb_ref = 217.0  # A reference far from the RJ limit also exposes spectrum-unit errors.
    params = {
        "components": {
            "CMB": {"enabled": True, "component_class": "CMB", "params": {
                "polarization": "I", "shortname": "cmb", "lmax": lmax, "nu_ref": cmb_ref,
                "spatially_varying_MM": False, "Cl_prior_amplitude": None,
            }},
            "Dust": {"enabled": True, "component_class": "ThermalDust", "params": {
                "polarization": "I", "shortname": "dust", "lmax": lmax,
                "nu_ref": 353.0, "beta": 1.54, "T": 20.0,
                "spatially_varying_MM": False, "Cl_prior_amplitude": None,
            }},
        },
        "compsep": {"double_precision": True},
        "experiments": {"Sim": {"bands": {
            "Band100": {"freq": 100.0}, "Band353": {"freq": 353.0},
        }}},
    }
    (tmp_path / "chains_compsep").mkdir()
    (tmp_path / "chains_bands").mkdir()
    chain_path = tmp_path / "chains_compsep/chain01_iter0001.h5"
    cmb_alm = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
    cmb_alm[hp.Alm.getidx(lmax, 2, 0)] = 3.0
    cmb_alm[hp.Alm.getidx(lmax, 3, 1)] = 2.0 + 1.0j
    dust_alm = np.zeros_like(cmb_alm)
    dust_alm[0] = 50.0
    dust_alm[hp.Alm.getidx(lmax, 4, 2)] = 10.0 + 5.0j
    with h5py.File(chain_path, "w") as handle:
        handle["metadata/parameter_file_as_string"] = yaml.safe_dump(params)
        handle["comps/cmb/alms"] = cmb_alm[None, :] / rj_to_band_unit_factor(cmb_ref, "uK_CMB")
        handle["comps/dust/alms"] = dust_alm[None, :]
    components = cr._build_intensity_components(as_bunch_recursive(params), str(chain_path))
    foregrounds = []
    for component in components:
        if not isinstance(component, cr.CMB):
            foregrounds.append(component)
    foreground_sky = cr.SkyModel(foregrounds)
    signal_maps = []
    ivar_maps = []
    beam_sizes = []
    for band, nu, fwhm_arcmin in [("Band100", 100.0, 20.0), ("Band353", 353.0, 100.0)]:
        beam = np.radians(fwhm_arcmin / 60.0)
        cmb_map = cr.alm2map(hp.almxfl(cmb_alm, hp.gauss_beam(beam, lmax=lmax)), nside, lmax)
        noise = np.linspace(-0.2, 0.2, npix)
        rms = np.linspace(0.5, 1.5, npix)
        foreground_rj = foreground_sky.get_sky_at_nu(nu, nside, "I", fwhm=beam)[0]
        observed_cmb = cmb_map + noise + foreground_rj.astype(float) * rj_to_band_unit_factor(
            nu, "uK_CMB")
        unit = stored_unit
        if stored_unit == "mixed":
            unit = "K_CMB" if nu == 100.0 else "MJy/sr"
        cmb_to_stored = rj_to_band_unit_factor(nu, unit) / rj_to_band_unit_factor(nu, "uK_CMB")
        band_path = tmp_path / f"chains_bands/Sim_{band}_chain01_iter0001.h5"
        with h5py.File(band_path, "w") as handle:
            handle["maps/observed_sky"] = (observed_cmb * cmb_to_stored)[None, :]
            handle["maps/rms"] = (rms * cmb_to_stored)[None, :]
            handle["metadata/band_unit"] = unit
            handle["metadata/map_fwhm_arcmin"] = fwhm_arcmin
        signal_maps.append(cmb_map + noise)
        ivar_maps.append(1 / rms**2)
        beam_sizes.append(beam)

    def theory_spectra(pars, CMB_unit: str, raw_cl: bool) -> dict[str, np.ndarray]:
        assert CMB_unit == "muK" and raw_cl
        return {"total": np.ones((200, 4))}

    monkeypatch.setattr(camb, "get_results", lambda pars: SimpleNamespace(
        get_cmb_power_spectra=theory_spectra))
    prior = np.ones(lmax + 1)
    prior[:2] = 1e6
    reference = cr.ConstrainedCMB(np.array(signal_maps), np.array(ivar_maps), prior,
                                  beam_fwhm=np.array(beam_sizes), maxiter=1000)
    np.random.seed(81)
    rhs = reference.get_RHS_eqn_mean() + reference.get_RHS_eqn_fluct()
    expected_alm = reference.solve_CG(reference.LHS_func, rhs, err_tol=1e-10)
    original_solve = cr.ConstrainedCMB.solve_CG

    def check_units(self, lhs, rhs, err_tol: float = 1e-6) -> np.ndarray:
        np.testing.assert_allclose(self.map_sky, signal_maps, atol=1e-11)
        np.testing.assert_allclose(self.map_ivar, ivar_maps, rtol=1e-12)
        np.testing.assert_array_equal(self.Cl_prior, prior)
        result = original_solve(self, lhs, rhs, err_tol)
        np.testing.assert_allclose(result, expected_alm, atol=1e-8)
        return result

    original_loglog = cr.plt.loglog
    plotted_spectra = []

    def record_spectrum(ell, spectrum, **kwargs):
        plotted_spectra.append(np.array(spectrum))
        return original_loglog(ell, spectrum, **kwargs)

    monkeypatch.setattr(cr.ConstrainedCMB, "solve_CG", check_units)
    monkeypatch.setattr(cr.plt, "loglog", record_spectrum)
    monkeypatch.setattr(sys, "argv", ["c4-cmb-realizations", str(tmp_path), "--iter", "1"])
    np.random.seed(81)
    assert cr.main() == 0
    ell = np.arange(lmax + 1)
    expected_dl = hp.alm2cl(cmb_alm) * ell * (ell + 1) / (2 * np.pi)
    np.testing.assert_allclose(plotted_spectra[0], expected_dl, atol=1e-12)
    output_path = tmp_path / "cmb_realizations/chain01_iter0001_cmb_realization.fits"
    np.testing.assert_allclose(hp.read_map(output_path), cr.alm2map(expected_alm, nside, lmax),
                               atol=1e-8)
    with fits.open(output_path) as handle:
        assert handle[1].header["BUNIT"] == "uK_CMB"
        assert handle[1].header["TUNIT1"] == "uK_CMB"
