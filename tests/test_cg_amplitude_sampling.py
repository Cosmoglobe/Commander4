"""Tests for the constrained-realization (fluctuation) terms of the compsep CG solver.

The statistical tests exploit an exact identity. In the preconditioned variable x = S^{-1/2}a the
solver applies M = 1 + S^{1/2}A^T N^-1 A S^{1/2} and solves Mx = b, with

    b = S^{1/2}A^T N^-1 d  +  S^{1/2}A^T N^{-1/2} eta_1  +  eta_2.

The two fluctuation terms are independent with covariances S^{1/2}A^T N^-1 A S^{1/2} and I, which
sum to exactly M. Hence Cov(x) = M^-1 M M^-1 = M^-1, *independently* of the data, the noise level,
the beam, or the mixing matrix -- nothing about the discretization needs to be derived. Two scalar
consequences are tested:

    E[<x - x_MAP, M (x - x_MAP)>] = tr(M^-1 M) = ndof,
    <xbar - x_MAP, M (xbar - x_MAP)> ~ chi^2(ndof)/nsamp.

Both are sensitive to the relative normalization of eta_1 and eta_2: an eta_2 drawn with twice the
correct variance (the pre-2026-08 `gaussian_random_alm` bug) inflates the first by tr(M^-1)/ndof,
about 19% for the configuration used here.

The statistical tests run twice: with a C_l prior, and without one (`Cl_prior_amplitude: null`,
C3's `CL_TYPE = none`). Without a prior M = A^T N^-1 A, there is no eta_2, and the identity above
still holds. The solver returns amplitudes a = S^{1/2}x, so the tests scale them back to x before
they apply M. The prior-mean tests below use a prior that is far from flat, so that a wrong
exponent of S or a corrupted mu cache shows.
"""

import healpy as hp
import numpy as np
import pytest
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.compsep.amplitude_sampling import AmplitudeGroup
from commander4.data_models.band_map import BandMap
from commander4.sky.comp_list import CompList
from commander4.compsep.cg_solver import CGAmplitudeSolver
from commander4.math_utils.alm import gaussian_random_alm, _dot_complex_alm_1D_arrays
from commander4.math_utils.sht import alm_to_map

LMAX = 4
NSIDE = 4
NSAMP = 100  # Enough to detect the 19% bug described above at about 7 standard errors.


def _make_params() -> Bunch:
    return Bunch(
        resources=Bunch(compsep=Bunch(num_threads=1)),
        compsep=Bunch(double_precision=True),
    )


def _make_cfg(optimize: bool = False) -> Bunch:
    """The settings of a CG sampling group. `optimize: False` draws a constrained realization."""
    return Bunch(optimize=optimize, max_iter=300, max_iter_pol=300, err_tol=1e-14,
                 preconditioner="NoPreconditioner")


def _make_solver(band_map: BandMap, cfg: Bunch,
                 comm: MPI.Comm = MPI.COMM_SELF) -> CGAmplitudeSolver:
    """A solver for a CG group with settings `cfg`, solved by the ranks of `comm`."""
    group = AmplitudeGroup(name="test", solver="CG", cfg=cfg,
                           solver_ranks={"I": list(range(comm.Get_size())), "QU": []},
                           solver_comm=comm)
    return CGAmplitudeSolver(band_map, group, nthreads=1)


def _make_comp_list(params: Bunch, Cl_prior_amplitude: float | None = None) -> CompList:
    """A single intensity CMB component, by default without a C_l prior."""
    cmb = Bunch(enabled=True, component_class="CMB",
                params=Bunch(lmax=LMAX, polarization="I", 
                             Cl_prior_amplitude=Cl_prior_amplitude))
    comp_list = CompList.init_from_params(Bunch(components=Bunch(CMB=cmb),
                                                compsep=params.compsep))
    for comp in comp_list:
        comp.amps[:] = 0.0
    return comp_list


def _make_band_map(seed: int = 0) -> BandMap:
    npix = 12*NSIDE**2
    rng = np.random.default_rng(seed)
    return BandMap(rng.normal(0.0, 1.0, (1, npix)), np.full((1, npix), 1.0/3.0**2),
                   nu=100.0, fwhm_arcmin=0.0, nside=NSIDE, lmax=LMAX)


# A prior amplitude of 1 gives a prior weight per alm that is comparable to the data weight of
# `_make_band_map`, so both terms of M matter.
@pytest.fixture(scope="module", params=[None, 1.0], ids=["no prior", "with prior"])
def realizations(request):
    """Draw NSAMP constrained realizations plus the corresponding MAP solution, all as x."""
    band_map = _make_band_map()
    comm = MPI.COMM_SELF

    def solve_for_x(solver: CGAmplitudeSolver, seed: int) -> CompList:
        """One solve, scaled back from the amplitudes a = S^{1/2}x to the variable x."""
        solution = _make_comp_list(_make_params(), request.param)
        solver.solve(solution, np.random.default_rng(seed))
        for comp in solution:
            comp.apply_Cl_prior_inv_sqrt(comp.amps)
        return solution

    x_map = solve_for_x(_make_solver(band_map, _make_cfg(optimize=True), comm), 0)
    solver = _make_solver(band_map, _make_cfg(optimize=False), comm)
    samples = [solve_for_x(solver, 1000 + i) for i in range(NSAMP)]
    return solver, x_map, samples


def _ndof(comp_list: CompList) -> int:
    """Real degrees of freedom in the alm vector: (lmax+1)^2 per polarization per component."""
    return sum((comp.lmax + 1)**2 * comp.npol for comp in comp_list)


def test_gaussian_random_alm_is_white_under_the_alm_inner_product():
    """E[<eta,eta>] must equal the number of real degrees of freedom, not twice it."""
    rng = np.random.default_rng(20260811)
    for lmax, spin in ((6, 0), (10, 0), (6, 2)):
        # The npol axis doubles as the sample axis, so this is one vectorized draw.
        draws = gaussian_random_alm(lmax, lmax, spin, 4000, rng)
        nm0 = lmax + 1
        # <a,a> for the convention of `_dot_complex_alm_1D_arrays`; m=0 entries are real.
        norms = (draws[:, :nm0].real**2).sum(-1) + 2.0*(np.abs(draws[:, nm0:])**2).sum(-1)
        # Spin-s fields have no l < s multipoles; those alms are zeroed, removing s^2 dof.
        expected = (lmax + 1)**2 - spin**2
        assert norms.mean() == pytest.approx(expected, rel=0.03), (lmax, spin)
        # Cross-check the vectorized norm against the numba routine the solver actually uses.
        assert _dot_complex_alm_1D_arrays(draws[0], draws[0], lmax) == pytest.approx(norms[0])


def test_constrained_realization_covariance_is_inverse_LHS(realizations):
    """E[<x-x_MAP, M(x-x_MAP)>] == ndof, i.e. Cov(x) = M^-1."""
    solver, x_map, samples = realizations
    ndof = _ndof(x_map)

    quad = []
    for sample in samples:
        delta = sample.copy()
        delta.add_scaled(x_map, -1.0)
        quad.append(delta.dot(solver.apply_LHS_matrix(delta)))

    mean = np.mean(quad)
    # Each term is a sum of ndof chi^2(1) variables, so the standard error is sqrt(2*ndof/NSAMP).
    stderr = np.sqrt(2.0*ndof/len(quad))
    assert abs(mean - ndof) < 4.0*stderr, f"E[<x,Mx>]={mean:.2f} vs ndof={ndof}"


def test_constrained_realization_mean_is_the_MAP_solution(realizations):
    """The sample mean converges on the Wiener/MAP solution: the fluctuations are zero-mean."""
    solver, x_map, samples = realizations
    ndof = _ndof(x_map)

    mean_sample = x_map.zeros_like()
    for sample in samples:
        mean_sample.add_scaled(sample, 1.0/len(samples))
    mean_sample.add_scaled(x_map, -1.0)

    # <xbar-x_MAP, M(xbar-x_MAP)> is chi^2(ndof)/NSAMP; allow a generous upper bound.
    quad = mean_sample.dot(solver.apply_LHS_matrix(mean_sample))*len(samples)
    assert quad < ndof + 5.0*np.sqrt(2.0*ndof), f"mean deviates from MAP: chisq={quad:.1f}"


def test_optimize_controls_randomness():
    """optimize=True is deterministic; optimize=False gives a different answer every call.

    This pins the polarity. Getting it backwards fails no other test, but silently replaces every
    amplitude draw with the Wiener/MAP mean, and the chain is no longer a sample of the posterior.
    """
    band_map = _make_band_map()
    comm = MPI.COMM_SELF

    first, second, sampled_a, sampled_b = [_make_comp_list(_make_params()) for _ in range(4)]
    map_solver = _make_solver(band_map, _make_cfg(optimize=True), comm)
    map_solver.solve(first, np.random.default_rng(1))
    map_solver.solve(second, np.random.default_rng(2))
    for comp_a, comp_b in zip(first, second):
        np.testing.assert_allclose(comp_a.amps, comp_b.amps, rtol=1e-10, atol=1e-12)

    solver = _make_solver(band_map, _make_cfg(optimize=False), comm)
    solver.solve(sampled_a, np.random.default_rng(1))
    solver.solve(sampled_b, np.random.default_rng(2))
    assert not np.allclose(sampled_a[0].amps, sampled_b[0].amps)
    # ... and it must not coincide with the MAP solution either.
    assert not np.allclose(sampled_a[0].amps, first[0].amps)


def test_without_a_prior_the_solution_is_the_data_alone():
    """With `Cl_prior_amplitude: null` (C3's `CL_TYPE = none`) nothing pulls the amplitudes towards
    zero, however little the data weigh. Here the data weight per alm is about 1e-5, so a prior of
    amplitude 1 returns almost nothing of the input."""
    rng = np.random.default_rng(4)
    input_alms = gaussian_random_alm(LMAX, LMAX, 0, 1, rng)
    sky = alm_to_map(input_alms, NSIDE, LMAX, spin=0, nthreads=1)
    band_map = BandMap(sky, np.full(sky.shape, 1.0e-6), nu=100.0, fwhm_arcmin=0.0, nside=NSIDE,
                       lmax=LMAX)
    solver = _make_solver(band_map, _make_cfg(optimize=True))

    no_prior = _make_comp_list(_make_params())[0]
    solver.solve(CompList([no_prior]), rng)
    np.testing.assert_allclose(no_prior.amps*no_prior.get_sed(100.0), input_alms, rtol=1e-5,
                               atol=1e-8)
    tight_prior = _make_comp_list(_make_params(), Cl_prior_amplitude=1.0)[0]
    solver.solve(CompList([tight_prior]), rng)
    assert np.abs(tight_prior.amps).max() < 1e-3*np.abs(no_prior.amps).max()


def test_the_cg_starts_from_the_amplitudes_it_is_given():
    """The CG starts from the amplitudes in the component list (C3's default), not from zero.

    One iteration that starts at the converged solution stays there, and its error, which is
    measured against the right-hand side, is tiny. One iteration from zero gets nowhere near.
    """
    band_map = _make_band_map()
    converged, from_zero, warm = [_make_comp_list(_make_params(), Cl_prior_amplitude=1.0)
                                  for _ in range(3)]
    _make_solver(band_map, _make_cfg(optimize=True)).solve(converged, np.random.default_rng(0))
    one_iteration = Bunch(optimize=True, max_iter=1, max_iter_pol=1, err_tol=0.0,
                          preconditioner="NoPreconditioner")

    residuals = _make_solver(band_map, one_iteration).solve(from_zero, np.random.default_rng(0))
    assert np.abs(from_zero[0].amps - converged[0].amps).max() > \
        1e-2*np.abs(converged[0].amps).max()
    assert residuals[0] > 1e-6

    warm[0].amps[:] = converged[0].amps
    residuals = _make_solver(band_map, one_iteration).solve(warm, np.random.default_rng(0))
    np.testing.assert_allclose(warm[0].amps, converged[0].amps, rtol=1e-5,
                               atol=1e-6*np.abs(converged[0].amps).max())
    assert residuals[0] < 1e-10


def test_solve_leaves_the_solution_in_the_components_it_is_given():
    """The Gibbs loop hands the chain's own components to the solver, and reads the solution from
    them afterwards."""
    comp_list = _make_comp_list(_make_params(), Cl_prior_amplitude=1.0)
    amps = comp_list[0].amps
    solver = _make_solver(_make_band_map(), _make_cfg(optimize=True))

    solver.solve(comp_list, np.random.default_rng(0))

    assert comp_list[0].amps is amps  # The same array, so other views of it see the solution.
    assert np.abs(amps).max() > 0.0   # It was zero on entry.


def test_the_cg_applies_the_full_band_beam_to_amplitudes_from_a_per_pixel_chain():
    """Amplitudes from a per-pixel chain carry a beam (`amp_fwhm_rad`). The CG must still fit the
    unsmoothed sky through the full band beam, so the solution does not depend on that beam."""
    npix = 12*NSIDE**2
    band_map = BandMap(np.random.default_rng(0).normal(0.0, 1.0, (1, npix)),
                       np.full((1, npix), 1.0/3.0**2), nu=100.0, fwhm_arcmin=600.0, nside=NSIDE,
                       lmax=LMAX)
    solver = _make_solver(band_map, _make_cfg(optimize=True))
    sharp, from_per_pixel = _make_comp_list(_make_params()), _make_comp_list(_make_params())
    solver.solve(sharp, np.random.default_rng(0))
    from_per_pixel[0].amp_fwhm_rad = np.deg2rad(8.0)
    solver.solve(from_per_pixel, np.random.default_rng(0))
    np.testing.assert_allclose(from_per_pixel[0].amps, sharp[0].amps, rtol=1e-6)
    assert from_per_pixel[0].amp_fwhm_rad == 0.0


def test_the_lhs_couples_a_diffuse_component_and_point_sources_symmetrically(tmp_path):
    """<x, A y> = <A x, y> for x a CMB alm vector and y point-source amplitudes. The sources enter
    the band in pixel space, so this checks that they are added to Y(alms) before N^-1 and read
    back from the same map after it."""
    from commander4.sky import RadioSources
    with open(tmp_path / "radio.dat", "w") as handle:
        handle.write("# Glon(deg) Glat(deg) I(mJy) alpha_I\n0.0 0.0 1.0 -0.7\n40.0 20.0 1.0 -0.5\n")
    params = _make_params()
    source = Bunch(params=Bunch(polarization="I", nu_ref=30.0,
                                template_path=str(tmp_path/"radio.dat")))
    source_params = Bunch(components=Bunch(radsources=source), compsep=params.compsep)
    rng = np.random.default_rng(0)
    vectors = []
    for random_comp in (0, 1):
        comps = _make_comp_list(params, Cl_prior_amplitude=1.0).components
        comps.append(RadioSources(source_params, "radsources", "I"))
        comps[1].amps[:] = 0.0
        if random_comp == 0:
            comps[0].amps[:] = gaussian_random_alm(LMAX, LMAX, 0, 1, rng)
        else:
            comps[1].amps[:] = rng.normal(0.0, 1e6, comps[1].amps.shape)
        vectors.append(CompList(comps))
    x, y = vectors
    solver = _make_solver(_make_band_map(), _make_cfg())

    x_Ay = x.dot(solver.apply_LHS_matrix(y))
    Ax_y = solver.apply_LHS_matrix(x).dot(y)

    assert x_Ay != 0.0
    assert x_Ay == pytest.approx(Ax_y, rel=1e-5)


def test_a_fixed_number_of_iterations_stops_when_the_residual_vanishes():
    """`err_tol: 0.0` asks for `max_iter` iterations, but the CG stops once the residual is zero
    to machine precision. In single precision this happens long before 300 iterations, and the
    next step would otherwise divide by zero."""
    npix = 12*NSIDE**2
    rng = np.random.default_rng(0)
    band_map = BandMap(rng.normal(0.0, 1.0, (1, npix)), np.full((1, npix), 1.0/3.0**2),
                       nu=100.0, fwhm_arcmin=0.0, nside=NSIDE, lmax=LMAX)
    params = Bunch(resources=Bunch(compsep=Bunch(num_threads=1)),
                   compsep=Bunch(double_precision=False))
    cfg = Bunch(optimize=True, max_iter=300, max_iter_pol=300, err_tol=0.0,
                preconditioner="NoPreconditioner")
    solver = _make_solver(band_map, cfg)

    solution = _make_comp_list(params, Cl_prior_amplitude=1.0)
    residuals = solver.solve(solution, rng)

    assert np.isfinite(solution[0].amps).all()
    assert len(residuals) < 300
    assert residuals[-1] == 0.0


def test_optimize_defaults_to_off_so_the_solver_samples():
    """A group with no `optimize` entry draws a constrained realization (C3 behaviour)."""
    assert AmplitudeGroup.CG_DEFAULTS["optimize"] is False


class TestPriorScalingContract:
    """`apply_Cl_prior_sqrt` and `apply_Cl_prior_inv_sqrt` are an exact pair sharing one contract.

    Both scale an explicitly given alm array by S^{+-1/2} in place and return it, so the call site
    always says what is being scaled -- `comp.apply_Cl_prior_sqrt(comp.amps)` for the component's
    own amplitudes. Uses a non-flat prior throughout, since under S = I both operations are the
    identity and prove nothing.
    """

    @staticmethod
    def _comp():
        return TestPriorMean._comp_list_with_prior(_make_params())[0]

    def test_round_trip_on_a_supplied_array_is_the_identity(self):
        comp = self._comp()
        rng = np.random.default_rng(21)
        alms = (rng.normal(size=comp.amps.shape) + 1j*rng.normal(size=comp.amps.shape))
        original = alms.copy()
        comp.apply_Cl_prior_inv_sqrt(comp.apply_Cl_prior_sqrt(alms))
        np.testing.assert_allclose(alms, original, rtol=1e-10, atol=1e-12)

    def test_supplied_array_is_scaled_and_own_alms_are_untouched(self):
        """Passing an array must not touch self.amps."""
        comp = self._comp()
        comp.amps[:] = 1.0
        own_before = comp.amps.copy()
        supplied = np.ones_like(comp.amps)

        returned = comp.apply_Cl_prior_sqrt(supplied)
        assert returned is supplied                    # in place, and handed back
        assert not np.allclose(supplied, own_before)
        np.testing.assert_array_equal(comp.amps, own_before)

class TestPriorMean:
    """The S^{-1/2} mu term. mu is zero today, so these drive it via a patched `amp_prior_mean`.

    The decisive case is the uninformative-data limit N^-1 -> 0: there the LHS reduces to the
    identity and b -> S^{-1/2}mu, so the returned amplitudes a = S^{1/2}x converge on mu itself,
    for *any* prior S. Running that with a non-identity C_l prior also pins the sign of the
    exponent: applying S^{+1/2} by mistake would return S*mu rather than mu.
    """

    @staticmethod
    def _comp_list_with_prior(params: Bunch) -> CompList:
        """A component with a genuinely non-flat C_l prior, so S != I."""
        cmb = Bunch(enabled=True, component_class="CMB",
                    params=Bunch(lmax=LMAX, polarization="I", 
                                 Cl_prior_amplitude=1.0e3,
                                 Cl_prior_beta=-2.0, Cl_prior_l_pivot=2))
        comp_list = CompList.init_from_params(Bunch(components=Bunch(CMB=cmb),
                                                    compsep=params.compsep))
        for comp in comp_list:
            comp.amps[:] = 0.0
        return comp_list

    @staticmethod
    def _patch_prior_mean(monkeypatch, comp, mu):
        monkeypatch.setattr(type(comp), "amp_prior_mean",
                            property(lambda self: mu.astype(self.amps.dtype, copy=True)))

    def test_uninformative_data_returns_the_prior_mean(self, monkeypatch):
        params = _make_params()
        comp_list = self._comp_list_with_prior(params)
        rng = np.random.default_rng(3)
        mu = rng.normal(0.0, 1.0, comp_list[0].amps.shape) \
            + 1j*rng.normal(0.0, 1.0, comp_list[0].amps.shape)
        mu[:, :LMAX + 1] = mu[:, :LMAX + 1].real     # m = 0 alms are real
        self._patch_prior_mean(monkeypatch, comp_list[0], mu)

        # Enormous noise: A^T N^-1 A is negligible against S^-1, so only the prior informs the fit.
        npix = 12*NSIDE**2
        band_map = BandMap(np.full((1, npix), 5.0), np.full((1, npix), 1e-16),
                           nu=100.0, fwhm_arcmin=0.0, nside=NSIDE, lmax=LMAX)
        _make_solver(band_map, _make_cfg(optimize=True)).solve(comp_list,
                                                                 np.random.default_rng(0))

        np.testing.assert_allclose(comp_list[0].amps, mu, rtol=1e-4, atol=1e-6)

    def test_prior_mean_enters_the_sampled_solve_too(self, monkeypatch):
        """mu shifts the constrained realization as well, not just the MAP solve."""
        params = _make_params()
        band_map = BandMap(np.zeros((1, 12*NSIDE**2)), np.full((1, 12*NSIDE**2), 1e-16),
                           nu=100.0, fwhm_arcmin=0.0, nside=NSIDE, lmax=LMAX)

        # `amp_prior_mean` is a property, so it can only be patched on the class -- which every
        # component instance shares. The unpatched control therefore has to run first.
        unshifted = self._comp_list_with_prior(params)
        _make_solver(band_map, _make_cfg(optimize=False)).solve(unshifted,
                                                                  np.random.default_rng(5))

        shifted = self._comp_list_with_prior(params)
        mu = np.full(shifted[0].amps.shape, 50.0 + 0j)
        self._patch_prior_mean(monkeypatch, shifted[0], mu)
        _make_solver(band_map, _make_cfg(optimize=False)).solve(shifted,
                                                                  np.random.default_rng(5))

        # Same seed, so the eta draws are identical and the difference is exactly the mu term.
        np.testing.assert_allclose(shifted[0].amps - unshifted[0].amps, mu, rtol=1e-4, atol=1e-6)


class TestPriorMeanMap:
    """Loading mu from an `amp_prior_mean_map` FITS sky map (C3's COMP_AMP_PRIOR_MAP)."""

    @staticmethod
    def _comp_list(tmp_path, prior_map=None, flat_prior=True, **extra) -> CompList:
        """A single-component list, optionally with `amp_prior_mean_map` written to `tmp_path`.

        `flat_prior=False` gives a C_l prior that is far from flat. That matters wherever the
        S^{-1/2} of the prior-mean term must actually do something: with the default prior
        S^{-1/2} is close to one, and a test would hardly notice it being applied to the wrong
        array.
        """
        params = _make_params()
        fields = dict(lmax=LMAX, polarization="I", 
                      Cl_prior_amplitude=1.0)
        if not flat_prior:
            fields.update(Cl_prior_amplitude=1.0e3, Cl_prior_beta=-2.0, Cl_prior_l_pivot=2)
        fields.update(extra)
        if prior_map is not None:
            path = tmp_path / "prior_mean.fits"
            hp.write_map(str(path), prior_map, overwrite=True)
            fields["amp_prior_mean_map"] = str(path)
        cmb = Bunch(enabled=True, component_class="CMB", params=Bunch(**fields))
        comp_list = CompList.init_from_params(Bunch(components=Bunch(CMB=cmb),
                                                    compsep=params.compsep))
        for comp in comp_list:
            comp.amps[:] = 0.0
        return comp_list

    def test_defaults_to_none_without_the_parameter(self, tmp_path):
        """A zero-mean prior is signalled by None, so the CG can skip the term entirely."""
        assert self._comp_list(tmp_path)[0].amp_prior_mean is None

    def test_map_round_trips_through_the_transform(self, tmp_path):
        """mu synthesized back to a map must reproduce the input map."""
        nside = 16
        rng = np.random.default_rng(11)
        smooth_map = hp.alm2map(hp.map2alm(rng.normal(0.0, 1.0, hp.nside2npix(nside)),
                                           lmax=LMAX), nside)
        comp_list = self._comp_list(tmp_path, prior_map=smooth_map)
        for comp in comp_list:
            comp.load_amp_prior_mean()

        mu = comp_list[0].amp_prior_mean
        assert not np.allclose(mu, 0.0)
        recovered = alm_to_map(mu.astype(np.complex128), nside, LMAX, spin=0, nthreads=1)
        np.testing.assert_allclose(recovered[0], smooth_map, rtol=1e-3, atol=1e-3)

    def test_cached_mu_survives_repeated_solves(self, tmp_path):
        """The CG applies S^{-1/2} to mu in place, so the cache must be handed out as a copy.

        Uses a non-flat prior: under the default S = I the in-place scaling is the identity and a
        corrupted cache would be indistinguishable from a healthy one.
        """
        nside = 16
        prior_map = np.full(hp.nside2npix(nside), 4.0)
        comp_list = self._comp_list(tmp_path, prior_map=prior_map, flat_prior=False)
        for comp in comp_list:
            comp.load_amp_prior_mean()
        mu_before = comp_list[0].amp_prior_mean.copy()

        # Solve the *same* component objects twice, as the Gibbs loop does across iterations. The
        # solution is left in `comp_list`, so the first one is copied before the second solve.
        band_map, cfg = _make_band_map(), _make_cfg(optimize=True)
        _make_solver(band_map, cfg).solve(comp_list, np.random.default_rng(0))
        first = comp_list[0].amps.copy()
        _make_solver(band_map, cfg).solve(comp_list, np.random.default_rng(0))
        second = comp_list[0].amps

        np.testing.assert_array_equal(comp_list[0].amp_prior_mean, mu_before)
        # The CG stops at err_tol, and threaded SHTs sum in a varying order, so two solves agree
        # only to roughly the CG tolerance. A corrupted cache would change the answer by order one.
        np.testing.assert_allclose(first, second, rtol=1e-4, atol=1e-6)

    def test_uninformative_data_recovers_the_prior_map(self, tmp_path):
        """End-to-end: with no data information the solution is the prior-mean map itself."""
        nside = 16
        rng = np.random.default_rng(4)
        prior_map = hp.alm2map(hp.map2alm(rng.normal(0.0, 1.0, hp.nside2npix(nside)),
                                          lmax=LMAX), nside)
        # Non-flat prior, so the round trip through S^{-1/2} ... S^{1/2} is a real cancellation.
        comp_list = self._comp_list(tmp_path, prior_map=prior_map, flat_prior=False)
        for comp in comp_list:
            comp.load_amp_prior_mean()
        mu = comp_list[0].amp_prior_mean

        npix = 12*NSIDE**2
        band_map = BandMap(np.zeros((1, npix)), np.full((1, npix), 1e-16),
                           nu=100.0, fwhm_arcmin=0.0, nside=NSIDE, lmax=LMAX)
        _make_solver(band_map, _make_cfg(optimize=True)).solve(comp_list,
                                                                 np.random.default_rng(0))

        np.testing.assert_allclose(comp_list[0].amps, mu, rtol=1e-4, atol=1e-6)

    def test_a_tightened_prior_holds_the_solution_to_the_prior_map(self, tmp_path):
        """End-to-end: `Cl_prior_l_tighten` far above lmax pins every multipole to the prior map.

        The prior is loose and the data are precise, so without the tightening the solution
        follows the data. With `Cl_prior_l_tighten` = 4000 the prior variance is scaled by
        sin^8(pi l / 8000), below 1e-22 for l <= 4, and the solution is the prior map.
        """
        nside = 16
        rng = np.random.default_rng(5)
        prior_map = hp.alm2map(hp.map2alm(rng.normal(0.0, 1.0, hp.nside2npix(nside)),
                                          lmax=LMAX), nside)
        npix = 12*NSIDE**2
        data_alms = gaussian_random_alm(LMAX, LMAX, 0, 1, np.random.default_rng(6))
        data_map = alm_to_map(data_alms, NSIDE, LMAX, spin=0, nthreads=1)
        band_map = BandMap(data_map, np.full((1, npix), 1e6),
                           nu=100.0, fwhm_arcmin=0.0, nside=NSIDE, lmax=LMAX)

        solutions = {}
        for l_tighten in (0, 4000):
            comp_list = self._comp_list(tmp_path, prior_map=prior_map, Cl_prior_amplitude=1.0e3,
                                        Cl_prior_l_tighten=l_tighten)
            for comp in comp_list:
                comp.load_amp_prior_mean()
            mu = comp_list[0].amp_prior_mean
            _make_solver(band_map, _make_cfg(optimize=True)).solve(comp_list,
                                                                     np.random.default_rng(0))
            solutions[l_tighten] = comp_list[0].amps
        sed = comp_list[0].get_sed(100.0)

        np.testing.assert_allclose(solutions[0]*sed, data_alms, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(solutions[4000], mu, rtol=1e-6, atol=1e-8)
        assert not np.allclose(data_alms/sed, mu, atol=0.1)  # The two cases are far apart.

    def test_rejects_a_non_fits_prior_mean(self, tmp_path):
        comp_list = self._comp_list(tmp_path, amp_prior_mean_map="/some/chain.h5")
        with pytest.raises(ValueError, match="must be a .fits sky map"):
            for comp in comp_list:
                comp.load_amp_prior_mean()

    def test_setter_rejects_a_wrongly_shaped_mu(self, tmp_path):
        comp = self._comp_list(tmp_path)[0]
        with pytest.raises(ValueError, match="prior mean has shape"):
            comp.amp_prior_mean = np.zeros((comp.npol, hp.Alm.getsize(comp.lmax) + 1),
                                           dtype=comp.amps.dtype)
