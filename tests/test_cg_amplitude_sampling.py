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
a = S^{1/2}x coincide with x, letting the tests apply M directly to the returned CompList. The
prior-mean tests below deliberately do the opposite and use a *non-flat* prior, because S^{-1/2}
is the identity under a flat one and would hide both a wrong exponent and a corrupted mu cache.
"""

import healpy as hp
import numpy as np
import pytest
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.compsep_processing import CGSamplingGroupConfig
from commander4.data_models.detector_map import DetectorMap
from commander4.sky_models.component import CompList
from commander4.solvers.CG_compsep_solver import CompSepSolver
from commander4.utils.math_operations import (alm_to_map, complist_dot, gaussian_random_alm,
                                              _dot_complex_alm_1D_arrays)

LMAX = 4
NSIDE = 4
NSAMP = 200


def _make_params(sample_amplitudes: bool = True) -> Bunch:
    return Bunch(
        resources=Bunch(compsep=Bunch(num_threads=1)),
        compsep=Bunch(nside=NSIDE, float_precision="double"),
    )


def _make_group(sample_amplitudes: bool = True) -> CGSamplingGroupConfig:
    """A CG sampling group. `optimize` is `sample_amplitudes` inverted."""
    return CGSamplingGroupConfig(
        name="test", optimize=not sample_amplitudes,
        max_iter=300, max_iter_pol=300, err_tol=1e-14,
        preconditioner="NoPreconditioner", dense_matrix_debug_mode=False,
    )


def _make_solver(det_map: DetectorMap, group: CGSamplingGroupConfig,
                 comm: MPI.Comm = MPI.COMM_SELF) -> CompSepSolver:
    return CompSepSolver(det_map, comm, group, double_precision=True, nthreads=1)


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

    map_params = _make_params()
    map_solver = _make_solver(det_map, _make_group(False), comm)
    x_map = map_solver.solve(_make_comp_list(map_params), seed=0)

    params = _make_params()
    solver = _make_solver(det_map, _make_group(True), comm)
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
    """optimize=True is deterministic; optimize=False gives a different answer every call."""
    det_map = _make_det_map()
    comm = MPI.COMM_SELF

    map_params = _make_params()
    map_solver = _make_solver(det_map, _make_group(False), comm)
    first = map_solver.solve(_make_comp_list(map_params), seed=1)
    second = map_solver.solve(_make_comp_list(map_params), seed=2)
    for comp_a, comp_b in zip(first, second):
        np.testing.assert_allclose(comp_a.alms, comp_b.alms, rtol=1e-10, atol=1e-12)

    params = _make_params()
    solver = _make_solver(det_map, _make_group(True), comm)
    sampled_a = solver.solve(_make_comp_list(params), seed=1)
    sampled_b = solver.solve(_make_comp_list(params), seed=2)
    assert not np.allclose(sampled_a[0].alms, sampled_b[0].alms)
    # ... and it must not coincide with the MAP solution either.
    assert not np.allclose(sampled_a[0].alms, first[0].alms)


def test_optimize_defaults_to_off_so_the_solver_samples():
    """A group with no `optimize` entry draws a constrained realization (C3 behaviour)."""
    solver = _make_solver(_make_det_map(), CGSamplingGroupConfig(name="default"))
    assert solver.sample_amplitudes is True


def test_optimize_inverts_the_old_sample_amplitudes_flag():
    """Pins the polarity: `optimize: false` samples, `optimize: true` returns the MAP mean.

    Getting this backwards fails no other test but silently replaces every amplitude draw with a
    Wiener/MAP mean, destroying the statistical correctness of the chain.
    """
    det_map = _make_det_map()
    sampling = _make_solver(det_map, _make_group(sample_amplitudes=True))
    assert sampling.sample_amplitudes is True
    assert sampling.config.preconditioner == "NoPreconditioner"
    optimizing = _make_solver(det_map, _make_group(sample_amplitudes=False))
    assert optimizing.sample_amplitudes is False


class TestPriorScalingContract:
    """`apply_Cl_prior_sqrt` and `apply_Cl_prior_inv_sqrt` are an exact pair sharing one contract.

    Both scale an explicitly given alm array by S^{+-1/2} in place and return it, so the call site
    always says what is being scaled -- `comp.apply_Cl_prior_sqrt(comp.alms)` for the component's
    own amplitudes. Uses a non-flat prior throughout, since under S = I both operations are the
    identity and prove nothing.
    """

    @staticmethod
    def _comp():
        return TestPriorMean._comp_list_with_prior(_make_params())[0]

    def test_round_trip_on_a_supplied_array_is_the_identity(self):
        comp = self._comp()
        rng = np.random.default_rng(21)
        alms = (rng.normal(size=comp.alms.shape) + 1j*rng.normal(size=comp.alms.shape))
        original = alms.copy()
        comp.apply_Cl_prior_inv_sqrt(comp.apply_Cl_prior_sqrt(alms))
        np.testing.assert_allclose(alms, original, rtol=1e-10, atol=1e-12)

    def test_round_trip_on_own_alms_is_the_identity(self):
        comp = self._comp()
        rng = np.random.default_rng(22)
        comp.alms[:] = rng.normal(size=comp.alms.shape) + 1j*rng.normal(size=comp.alms.shape)
        original = comp.alms.copy()
        comp.apply_Cl_prior_sqrt(comp.alms)
        assert not np.allclose(comp.alms, original)   # the prior is genuinely non-flat
        comp.apply_Cl_prior_inv_sqrt(comp.alms)
        np.testing.assert_allclose(comp.alms, original, rtol=1e-10, atol=1e-12)

    def test_supplied_array_is_scaled_and_own_alms_are_untouched(self):
        """Passing an array must not touch self._data."""
        comp = self._comp()
        comp.alms[:] = 1.0
        own_before = comp.alms.copy()
        supplied = np.ones_like(comp.alms)

        returned = comp.apply_Cl_prior_sqrt(supplied)
        assert returned is supplied                    # in place, and handed back
        assert not np.allclose(supplied, own_before)
        np.testing.assert_array_equal(comp.alms, own_before)

    def test_the_target_is_required(self):
        """No implicit default: omitting the target is an error, not a silent 'use my own alms'."""
        comp = self._comp()
        with pytest.raises(TypeError):
            comp.apply_Cl_prior_sqrt()
        with pytest.raises(TypeError):
            comp.apply_Cl_prior_inv_sqrt()


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
                    params=Bunch(lmax=LMAX, polarization="I", shortname="cmb",
                                 spatially_varying_MM=False, Cl_prior_amplitude=1.0e3,
                                 Cl_prior_beta=-2.0, Cl_prior_l_pivot=2))
        object.__setattr__(cmb, "_name", "CMB")
        comp_list = CompList.init_from_params(Bunch({"CMB": cmb}), params)
        for comp in comp_list:
            comp.alms[:] = 0.0
        return comp_list

    @staticmethod
    def _patch_prior_mean(monkeypatch, comp, mu):
        monkeypatch.setattr(type(comp), "amp_prior_mean",
                            property(lambda self: mu.astype(self.dtype, copy=True)))

    def test_uninformative_data_returns_the_prior_mean(self, monkeypatch):
        params = _make_params()
        comp_list = self._comp_list_with_prior(params)
        rng = np.random.default_rng(3)
        mu = rng.normal(0.0, 1.0, comp_list[0].alms.shape) \
            + 1j*rng.normal(0.0, 1.0, comp_list[0].alms.shape)
        mu[:, :LMAX + 1] = mu[:, :LMAX + 1].real     # m = 0 alms are real
        self._patch_prior_mean(monkeypatch, comp_list[0], mu)

        # Enormous noise: A^T N^-1 A is negligible against S^-1, so only the prior informs the fit.
        npix = 12*NSIDE**2
        det_map = DetectorMap(np.full((1, npix), 5.0), np.full((1, npix), 1e8),
                              nu=100.0, fwhm=0.0, nside=NSIDE, double_precision=True, lmax=LMAX)
        solution = _make_solver(det_map, _make_group(False)).solve(comp_list)

        np.testing.assert_allclose(solution[0].alms, mu, rtol=1e-4, atol=1e-6)

    def test_zero_prior_mean_leaves_the_solution_unchanged(self):
        """The default mu = 0 must be an exact no-op, not merely a small perturbation."""
        params = _make_params()
        det_map = _make_det_map()
        baseline = _make_solver(det_map, _make_group(False)).solve(_make_comp_list(params))

        comp = _make_comp_list(params)[0]
        assert comp.amp_prior_mean is None
        again = _make_solver(det_map, _make_group(False)).solve(_make_comp_list(params))
        np.testing.assert_array_equal(baseline[0].alms, again[0].alms)

    def test_prior_mean_enters_the_sampled_solve_too(self, monkeypatch):
        """mu shifts the constrained realization as well, not just the MAP solve."""
        params = _make_params()
        det_map = DetectorMap(np.zeros((1, 12*NSIDE**2)), np.full((1, 12*NSIDE**2), 1e8),
                              nu=100.0, fwhm=0.0, nside=NSIDE, double_precision=True, lmax=LMAX)

        # `amp_prior_mean` is a property, so it can only be patched on the class -- which every
        # component instance shares. The unpatched control therefore has to run first.
        unshifted = _make_solver(det_map, _make_group(True)).solve(
            self._comp_list_with_prior(params), seed=5)

        comp_list = self._comp_list_with_prior(params)
        mu = np.full(comp_list[0].alms.shape, 50.0 + 0j)
        self._patch_prior_mean(monkeypatch, comp_list[0], mu)
        shifted = _make_solver(det_map, _make_group(True)).solve(comp_list, seed=5)

        # Same seed, so the eta draws are identical and the difference is exactly the mu term.
        np.testing.assert_allclose(shifted[0].alms - unshifted[0].alms, mu, rtol=1e-4, atol=1e-6)


class TestPriorMeanMap:
    """Loading mu from an `amp_prior_mean_map` FITS sky map (C3's COMP_AMP_PRIOR_MAP)."""

    @staticmethod
    def _comp_list(tmp_path, prior_map=None, flat_prior=True, **extra) -> CompList:
        """A single-component list, optionally with `amp_prior_mean_map` written to `tmp_path`.

        `flat_prior=False` gives a genuinely non-flat C_l prior (S != I). That matters wherever the
        S^{-1/2} of the prior-mean term must actually do something: with the default flat prior
        S^{-1/2} is the identity, and a test would not notice it being applied to the wrong array.
        """
        params = _make_params()
        fields = dict(lmax=LMAX, polarization="I", shortname="cmb", spatially_varying_MM=False,
                      Cl_prior_amplitude=None)
        if not flat_prior:
            fields.update(Cl_prior_amplitude=1.0e3, Cl_prior_beta=-2.0, Cl_prior_l_pivot=2)
        fields.update(extra)
        if prior_map is not None:
            path = tmp_path / "prior_mean.fits"
            hp.write_map(str(path), prior_map, overwrite=True)
            fields["amp_prior_mean_map"] = str(path)
        cmb = Bunch(enabled=True, component_class="CMB", params=Bunch(**fields))
        object.__setattr__(cmb, "_name", "CMB")
        comp_list = CompList.init_from_params(Bunch({"CMB": cmb}), params)
        for comp in comp_list:
            comp.alms[:] = 0.0
        return comp_list

    def test_defaults_to_none_without_the_parameter(self, tmp_path):
        """A zero-mean prior is signalled by None, so the CG can skip the term entirely."""
        assert self._comp_list(tmp_path)[0].amp_prior_mean is None

    def test_absent_prior_mean_skips_the_scaling_entirely(self, tmp_path, monkeypatch):
        """The short-circuit is real: no prior mean means S^{-1/2} is never applied.

        Dropping the None check would now raise (mu is None and the target is required) rather than
        corrupt anything, but this keeps the common zero-mean path from quietly regaining a
        full-size scale-and-add.
        """
        comp_list = self._comp_list(tmp_path)

        def fail(*args, **kwargs):
            raise AssertionError("apply_Cl_prior_inv_sqrt called for a zero-mean prior")
        monkeypatch.setattr(type(comp_list[0]), "apply_Cl_prior_inv_sqrt", fail)

        _make_solver(_make_det_map(), _make_group(False)).solve(comp_list)

    def test_map_round_trips_through_the_transform(self, tmp_path):
        """mu synthesized back to a map must reproduce the input map."""
        nside = 16
        rng = np.random.default_rng(11)
        smooth_map = hp.alm2map(hp.map2alm(rng.normal(0.0, 1.0, hp.nside2npix(nside)),
                                           lmax=LMAX), nside)
        comp_list = self._comp_list(tmp_path, prior_map=smooth_map)
        comp_list.load_amp_prior_means()

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
        comp_list.load_amp_prior_means()
        mu_before = comp_list[0].amp_prior_mean.copy()

        # Solve the *same* component objects twice, as the Gibbs loop does across iterations.
        params = _make_params()
        det_map = _make_det_map()
        first = _make_solver(det_map, _make_group(False)).solve(comp_list)
        second = _make_solver(det_map, _make_group(False)).solve(comp_list)

        np.testing.assert_array_equal(comp_list[0].amp_prior_mean, mu_before)
        np.testing.assert_allclose(first[0].alms, second[0].alms, rtol=1e-10, atol=1e-12)

    def test_uninformative_data_recovers_the_prior_map(self, tmp_path):
        """End-to-end: with no data information the solution is the prior-mean map itself."""
        nside = 16
        rng = np.random.default_rng(4)
        prior_map = hp.alm2map(hp.map2alm(rng.normal(0.0, 1.0, hp.nside2npix(nside)),
                                          lmax=LMAX), nside)
        # Non-flat prior, so the round trip through S^{-1/2} ... S^{1/2} is a real cancellation.
        comp_list = self._comp_list(tmp_path, prior_map=prior_map, flat_prior=False)
        comp_list.load_amp_prior_means()
        mu = comp_list[0].amp_prior_mean

        params = _make_params()
        npix = 12*NSIDE**2
        det_map = DetectorMap(np.zeros((1, npix)), np.full((1, npix), 1e8),
                              nu=100.0, fwhm=0.0, nside=NSIDE, double_precision=True, lmax=LMAX)
        solution = _make_solver(det_map, _make_group(False)).solve(comp_list)

        np.testing.assert_allclose(solution[0].alms, mu, rtol=1e-4, atol=1e-6)

    def test_rejects_a_non_fits_prior_mean(self, tmp_path, caplog):
        comp_list = self._comp_list(tmp_path, amp_prior_mean_map="/some/chain.h5")
        # logassert logs the reason and raises a bare AssertionError, so match on the log.
        with pytest.raises(AssertionError):
            comp_list.load_amp_prior_means()
        assert "must be a .fits sky map" in caplog.text

    def test_setter_rejects_a_wrongly_shaped_mu(self, tmp_path):
        comp = self._comp_list(tmp_path)[0]
        with pytest.raises(ValueError, match="prior mean has shape"):
            comp.amp_prior_mean = np.zeros((comp.npol, comp.alm_len_complex + 1), dtype=comp.dtype)
