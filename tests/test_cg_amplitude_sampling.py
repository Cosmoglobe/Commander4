"""Tests for the constrained-realization (fluctuation) terms of the compsep CG solver.

The statistical tests exploit an exact identity. In the preconditioned variable x = S^{-1/2}a the
solver applies M = 1 + S^{1/2}A^T N^-1 A S^{1/2} and solves Mx = b, with

    b = S^{1/2}A^T N^-1 d  +  S^{1/2}A^T N^{-1/2} eta_1  +  eta_2.

The two fluctuation terms are independent with covariances S^{1/2}A^T N^-1 A S^{1/2} and I, which
sum to exactly M. Hence Cov(x) = M^-1 M M^-1 = M^-1, *independently* of the data, the noise level,
the beam, or the mixing matrix -- nothing about the discretization needs to be derived. Two scalar
consequences are tested:

    E[<x - x_MAP, M (x - x_MAP)>] = tr(M^-1 M) = ndof,
    <xbar - x_MAP, M (xbar - x_MAP)> ~ chi2(ndof)/nsamp.

Both are sensitive to the relative normalization of eta_1 and eta_2: an eta_2 drawn with twice the
correct variance (the pre-2026-08 `gaussian_random_alm` bug) inflates the first by tr(M^-1)/ndof,
about 19% for the configuration used here.

The components use an identity C_l prior (S = I) so that the solver's returned amplitudes
a = S^{1/2}x coincide with x, letting the tests apply M directly to the returned CompList.
"""

import numpy as np
import pytest
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.data_models.detector_map import DetectorMap
from commander4.sky_models.component import CompList
from commander4.solvers.CG_compsep_solver import CompSepSolver
from commander4.utils.math_operations import (complist_dot, gaussian_random_alm,
                                              _dot_complex_alm_1D_arrays)

LMAX = 4
NSIDE = 4
NSAMP = 200


def _make_params(sample_amplitudes: bool = True) -> Bunch:
    return Bunch(
        general=Bunch(nside=NSIDE, CG_float_precision="double", nthreads_compsep=1,
                      CG_max_iter=300, CG_max_iter_pol=300, CG_err_tol=1e-14,
                      MPI_config=Bunch(ntask_compsep_I=1, ntask_compsep_QU=1)),
        compsep=Bunch(preconditioner="NoPreconditioner", dense_matrix_debug_mode=False,
                      sample_amplitudes=sample_amplitudes),
    )


def _make_comp_list(params: Bunch) -> CompList:
    """A single intensity CMB component with an identity C_l prior (Cl_prior_amplitude=None)."""
    cmb = Bunch(enabled=True, component_class="CMB",
                params=Bunch(lmax=LMAX, polarization="I", shortname="cmb",
                             spatially_varying_MM=False, Cl_prior_amplitude=None))
    object.__setattr__(cmb, "_name", "CMB")
    comp_list = CompList.init_from_params(Bunch({"CMB": cmb}), params)
    for comp in comp_list:
        comp.alms[:] = 0.0
    return comp_list


def _make_det_map(seed: int = 0) -> DetectorMap:
    npix = 12*NSIDE**2
    rng = np.random.default_rng(seed)
    return DetectorMap(rng.normal(0.0, 1.0, (1, npix)), np.full((1, npix), 3.0),
                       nu=100.0, fwhm=0.0, nside=NSIDE, double_precision=True, lmax=LMAX)


@pytest.fixture(scope="module")
def realizations():
    """Draw NSAMP constrained realizations plus the corresponding MAP solution."""
    det_map = _make_det_map()
    comm = MPI.COMM_SELF

    map_params = _make_params(sample_amplitudes=False)
    map_solver = CompSepSolver(det_map, map_params, comm)
    x_map = map_solver.solve(_make_comp_list(map_params), seed=0)

    params = _make_params(sample_amplitudes=True)
    solver = CompSepSolver(det_map, params, comm)
    samples = [solver.solve(_make_comp_list(params), seed=1000 + i) for i in range(NSAMP)]
    return solver, x_map, samples


def _ndof(comp_list: CompList) -> int:
    """Real degrees of freedom in the alm vector: (lmax+1)^2 per polarization per component."""
    return sum((comp.lmax + 1)**2 * comp.npol for comp in comp_list)


def test_gaussian_random_alm_is_white_under_the_alm_inner_product():
    """E[<eta,eta>] must equal the number of real degrees of freedom, not twice it."""
    rng_state = np.random.get_state()
    np.random.seed(20260811)
    try:
        for lmax, spin in ((6, 0), (10, 0), (6, 2)):
            # The ncomp axis doubles as the sample axis, so this is one vectorized draw.
            draws = gaussian_random_alm(lmax, lmax, spin, 4000)
            nm0 = lmax + 1
            # <a,a> for the convention of `_dot_complex_alm_1D_arrays`; m=0 entries are real.
            norms = (draws[:, :nm0].real**2).sum(-1) + 2.0*(np.abs(draws[:, nm0:])**2).sum(-1)
            # Spin-s fields have no l < s multipoles; those alms are zeroed, removing s^2 dof.
            expected = (lmax + 1)**2 - spin**2
            assert norms.mean() == pytest.approx(expected, rel=0.03), (lmax, spin)
            # Cross-check the vectorized norm against the numba routine the solver actually uses.
            assert _dot_complex_alm_1D_arrays(draws[0], draws[0], lmax) == pytest.approx(norms[0])
    finally:
        np.random.set_state(rng_state)


def test_constrained_realization_covariance_is_inverse_LHS(realizations):
    """E[<x-x_MAP, M(x-x_MAP)>] == ndof, i.e. Cov(x) = M^-1."""
    solver, x_map, samples = realizations
    ndof = _ndof(x_map)

    quad = []
    for sample in samples:
        delta = sample - x_map
        quad.append(complist_dot(delta, solver.apply_LHS_matrix(delta)))

    mean = np.mean(quad)
    # Each term is a sum of ndof chi2(1) variables, so the standard error is sqrt(2*ndof/NSAMP).
    stderr = np.sqrt(2.0*ndof/len(quad))
    assert abs(mean - ndof) < 4.0*stderr, f"E[<x,Mx>]={mean:.2f} vs ndof={ndof}"


def test_constrained_realization_mean_is_the_MAP_solution(realizations):
    """The sample mean converges on the Wiener/MAP solution: the fluctuations are zero-mean."""
    solver, x_map, samples = realizations
    ndof = _ndof(x_map)

    mean_sample = samples[0] - x_map
    for sample in samples[1:]:
        mean_sample += sample - x_map
    for comp in mean_sample:
        comp.alms[:] /= len(samples)

    # <xbar-x_MAP, M(xbar-x_MAP)> is chi2(ndof)/NSAMP; allow a generous upper bound.
    quad = complist_dot(mean_sample, solver.apply_LHS_matrix(mean_sample))*len(samples)
    assert quad < ndof + 5.0*np.sqrt(2.0*ndof), f"mean deviates from MAP: chi2={quad:.1f}"


def test_sample_amplitudes_toggle_controls_randomness():
    """sample_amplitudes=False is deterministic; True gives a different answer every call."""
    det_map = _make_det_map()
    comm = MPI.COMM_SELF

    map_params = _make_params(sample_amplitudes=False)
    map_solver = CompSepSolver(det_map, map_params, comm)
    first = map_solver.solve(_make_comp_list(map_params), seed=1)
    second = map_solver.solve(_make_comp_list(map_params), seed=2)
    for comp_a, comp_b in zip(first, second):
        np.testing.assert_allclose(comp_a.alms, comp_b.alms, rtol=1e-10, atol=1e-12)

    params = _make_params(sample_amplitudes=True)
    solver = CompSepSolver(det_map, params, comm)
    sampled_a = solver.solve(_make_comp_list(params), seed=1)
    sampled_b = solver.solve(_make_comp_list(params), seed=2)
    assert not np.allclose(sampled_a[0].alms, sampled_b[0].alms)
    # ... and it must not coincide with the MAP solution either.
    assert not np.allclose(sampled_a[0].alms, first[0].alms)


def test_sample_amplitudes_defaults_to_on_when_unset():
    """A parameter file with no compsep.sample_amplitudes entry samples (C3 behaviour)."""
    params = _make_params()
    del params.compsep["sample_amplitudes"]
    solver = CompSepSolver(_make_det_map(), params, MPI.COMM_SELF)
    assert solver.sample_amplitudes is True
