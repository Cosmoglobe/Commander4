import numpy as np
import pytest
from numba import njit
from scipy.fft import rfftfreq

from commander4.noise_sampling.sigma0 import calc_sigma0_simple, calc_sigma0_robust
from commander4.noise_sampling.noise_psd import NoisePSDOof
from commander4.noise_sampling.noise_sampling import fill_all_masked
from commander4.noise_sampling.sample_ncorr import (sample_correlated_noise,
                                                    corr_noise_realization_with_gaps)
from commander4.data_models.detector_group_TOD import DetGroupTOD


@njit
def _seed_numba_rng(seed: int) -> None:
    """Seed Numba's internal RNG (independent of NumPy's global RNG, which is used by njit code)."""
    np.random.seed(seed)


def _seed_all_rng(seed: int) -> None:
    """Seed both the NumPy global RNG and Numba's internal RNG for fully reproducible sampling."""
    np.random.seed(seed)
    _seed_numba_rng(seed)


def _synth_corr_noise(seed: int, ntod: int, fsamp: float, sigma0: float, fknee: float,
                      alpha: float) -> np.ndarray:
    """Synthesize a correlated-noise stream whose periodogram follows sigma0^2 (f/fknee)^alpha.

    Uses the rfft/irfft convention (forward unnormalized, inverse divided by ntod) matching the
    project's FFT helpers, so that (1/ntod)|rfft(n_corr)|^2 has expectation P_corr(f).
    """
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(ntod, d=1.0/fsamp)
    P = np.zeros_like(freqs)
    P[1:] = sigma0**2 * (freqs[1:]/fknee)**alpha
    spec = np.sqrt(ntod*P/2.0) * (rng.standard_normal(freqs.size)
                                  + 1j*rng.standard_normal(freqs.size))
    spec[0] = 0.0  # zero-mean
    if ntod % 2 == 0:
        spec[-1] = spec[-1].real  # real Nyquist mode
    return np.fft.irfft(spec, n=ntod)

# ===================================================================
# sigma0 estimation
# ===================================================================

class TestCalcSigma0Simple:
    def test_agreement_with_direct_Numpy_slicing(self):
        """ Validates that the compiled simple estimator exactly matches a vectorized NumPy
            implementation, correctly handling mask gaps.
        """
        rng = np.random.default_rng(seed=105)
        tod = rng.normal(0.0, 2.5, 20000)
        mask = np.ones(20000, dtype=bool)
        # Inject arbitrary mask gaps to simulate dropped packets or glitches
        mask[500:550] = False  # Contiguous gap
        mask[10000] = False    # Single sample gap
        # 1. Compute using the package's compiled implementation
        est_package = calc_sigma0_simple(tod, mask)
        # 2. Compute using native NumPy slicing
        # A pair is only valid if both the current and previous sample are unmasked.
        valid_pairs = mask[1:] & mask[:-1]
        # Extract the differences for valid pairs
        diffs = tod[1:][valid_pairs] - tod[:-1][valid_pairs]
        # Calculate standard deviation. 
        est_native = np.std(diffs, ddof=1) / np.sqrt(2.0)
        # 3. Assert agreement
        assert est_package == pytest.approx(est_native, rel=1e-12)


class TestRobustSigma0:
    def test_basic(self):
        rng = np.random.default_rng(seed=50)
        sigma0_true = 3.0
        tod = rng.normal(0.0, sigma0_true, 10000)
        mask = np.ones(10000, dtype=bool)
        est = calc_sigma0_robust(tod, mask)
        assert est == pytest.approx(sigma0_true, rel=0.1)

    def test_with_outliers(self):
        rng = np.random.default_rng(seed=50)
        sigma0_true = 1.0
        tod = rng.normal(0.0, sigma0_true, 10000)
        # Inject outliers
        tod[500] = 1000.0
        tod[1000] = -500.0
        tod[2000] = 800.0
        mask = np.ones(10000, dtype=bool)
        robust_est = calc_sigma0_robust(tod, mask)
        naive_est = calc_sigma0_simple(tod, mask)
        # Robust should be closer to truth
        assert abs(robust_est - sigma0_true) < abs(naive_est - sigma0_true)

    def test_few_samples(self):
        tod = np.array([1.0, 2.0])
        mask = np.ones(2, dtype=bool)
        # With < 100 valid pairs, should return inf
        est = calc_sigma0_robust(tod, mask)
        assert est == np.inf


# ===================================================================
# NoisePSD model (1/f / "oof")
# ===================================================================

class TestNoisePSDOof:
    def test_eval_matches_formula(self):
        m = NoisePSDOof()
        s0, fk, a = 2.0, 0.5, -2.3
        freqs = np.array([0.0, 0.1, 1.0, 5.0])
        full = m.eval_full(freqs, np.array([s0, fk, a]))
        corr = m.eval_corr(freqs, np.array([s0, fk, a]))
        # At f = 0: full PSD is the white-noise floor, correlated PSD is zero.
        assert full[0] == pytest.approx(s0**2)
        assert corr[0] == 0.0
        pos = freqs > 0
        assert np.allclose(corr[pos], s0**2 * (freqs[pos]/fk)**a)
        assert np.allclose(full[pos], s0**2 * (1.0 + (freqs[pos]/fk)**a))

    def test_inv_corr_spectrum_is_one_over_Pcorr(self):
        """compute_inv_corr_spectrum must equal 1/P_corr (the quantity the noise CG adds to
        1/sigma0^2), matching the previously hardcoded C_1f_inv = 1/(sigma0^2 (f/fknee)^alpha)."""
        m = NoisePSDOof()
        s0, fk, a = 1.3, 0.4, -1.7
        freqs = rfftfreq(2048, d=1.0/10.0)
        inv = m.compute_inv_corr_spectrum(freqs, np.array([s0, fk, a]))
        assert inv[0] == 0.0  # zero-frequency mode excluded
        expected = np.zeros_like(freqs)
        expected[1:] = 1.0 / (s0**2 * (freqs[1:]/fk)**a)
        assert np.allclose(inv, expected)

    def test_sample_params_keeps_sigma0_and_uses_Puni_bounds(self):
        m = NoisePSDOof()
        sigma0 = 1.234
        residual = _synth_corr_noise(1, 2**14, 10.0, sigma0, 0.5, -2.0) \
            + np.random.default_rng(2).normal(0.0, sigma0, 2**14)
        _seed_all_rng(0)
        params = np.array([sigma0, 0.3, -1.5])
        out = m.sample_params(residual, params, 10.0, nu_min=0.0, nu_max=2.0)
        assert out.shape == params.shape
        assert out[0] == params[0]            # sigma0 held fixed
        assert out is not params              # returns a fresh array
        assert params[1] == 0.3 and params[2] == -1.5  # input untouched
        # Grids span the model's uniform priors (single source of truth for the hard bounds).
        assert m.P_uni[1, 0] <= out[1] <= m.P_uni[1, 1]   # fknee in P_uni
        assert m.P_uni[2, 0] <= out[2] <= m.P_uni[2, 1]   # alpha in P_uni ([-4.5, -0.25])

    def test_sample_params_recovers_input(self):
        # The sampler builds its (fknee, alpha) grids from the model's uniform priors, so construct
        # a model whose P_uni spans the injected values rather than relying on the defaults.
        m = NoisePSDOof(P_uni=[[np.nan, np.nan], [0.01, 10.0], [-4.5, -0.25]])
        sigma0, fknee, alpha = 1.0, 0.5, -2.0
        # Fit is performed on the residual (white + correlated noise) with the full PSD model.
        residual = _synth_corr_noise(7, 2**16, 10.0, sigma0, fknee, alpha) \
            + np.random.default_rng(8).normal(0.0, sigma0, 2**16)
        for bin_psd in (False, True):
            _seed_all_rng(123)
            fks, als = [], []
            for _ in range(8):
                out = m.sample_params(residual, np.array([sigma0, 0.2, -1.0]), 10.0,
                                      nu_min=0.0, nu_max=2.0, bin_psd=bin_psd)
                fks.append(out[1])
                als.append(out[2])
            assert np.mean(fks) == pytest.approx(fknee, rel=0.4), f"bin_psd={bin_psd}"
            assert np.mean(als) == pytest.approx(alpha, abs=0.4), f"bin_psd={bin_psd}"


# ===================================================================
# Correlated-noise orchestrator
# ===================================================================

class TestSampleCorrelatedNoise:
    @staticmethod
    def _setup(ntod=2**13, fsamp=10.0, sigma0=1.0, fknee=0.3, alpha=-1.8, seed=3):
        rng = np.random.default_rng(seed)
        tod = _synth_corr_noise(seed, ntod, fsamp, sigma0, fknee, alpha) \
            + rng.normal(0.0, sigma0, ntod)
        mask = np.ones(ntod, dtype=bool)
        mask[1000:1050] = False  # contiguous gap
        mask[5000] = False       # single-sample gap
        return tod, mask, np.array([sigma0, fknee, alpha]), fsamp

    def test_zero_cg_steps_equals_direct_fallback(self):
        """cg_max_iter=0 must skip the masked CG and return the stationary full-mask solution."""
        m = NoisePSDOof()
        tod, mask, params, fsamp = self._setup()
        tod_a, tod_b = tod.copy(), tod.copy()
        _seed_all_rng(42)
        res = sample_correlated_noise(tod_a, mask, params.copy(), m, fsamp, cg_err_tol=1e-6,
                                      cg_max_iter=0, sample_params=False)
        # Reproduce the fallback manually: inpaint, then solve with an all-valid mask.
        _seed_all_rng(42)
        fill_all_masked(tod_b, mask, params[0])
        C = m.compute_inv_corr_spectrum(rfftfreq(2*tod_b.size, d=1.0/fsamp), params)
        ref, _, _, _ = corr_noise_realization_with_gaps(tod_b, np.ones_like(mask), params[0], C)
        assert np.allclose(res.n_corr, ref)
        assert res.niter == 0
        assert res.converged
        assert not res.high_var

    def test_masked_cg_runs_and_converges(self):
        m = NoisePSDOof()
        tod, mask, params, fsamp = self._setup()
        _seed_all_rng(0)
        res = sample_correlated_noise(tod.copy(), mask, params.copy(), m, fsamp, cg_err_tol=1e-6,
                                      cg_max_iter=200, sample_params=False)
        assert res.n_corr.shape == tod.shape
        assert np.all(np.isfinite(res.n_corr))
        assert res.converged          # an easy, well-conditioned system should converge
        assert res.niter > 0          # the masked CG actually ran (unlike cg_max_iter=0)

    def test_param_sampling_toggle(self):
        m = NoisePSDOof()
        tod, mask, params, fsamp = self._setup()
        # With both sampling switches off, noise_params come back untouched.
        _seed_all_rng(0)
        res_off = sample_correlated_noise(tod.copy(), mask, params.copy(), m, fsamp,
                                          cg_err_tol=1e-6, cg_max_iter=0, sample_params=False,
                                          sample_sigma0=False)
        assert np.array_equal(res_off.noise_params, params)
        # Turning on parameter sampling (only) updates fknee/alpha but keeps sigma0.
        _seed_all_rng(0)
        res_on = sample_correlated_noise(tod.copy(), mask, params.copy(), m, fsamp,
                                         cg_err_tol=1e-6, cg_max_iter=0, sample_params=True,
                                         sample_sigma0=False, psd_fit_nu_max=2.0)
        assert res_on.noise_params[0] == params[0]               # sigma0 fixed
        assert not np.array_equal(res_on.noise_params[1:], params[1:])  # fknee/alpha updated

    def test_sigma0_reestimated_from_residual(self):
        """sample_sigma0=True replaces noise_params[0] with a data-driven estimate."""
        m = NoisePSDOof()
        tod, mask, params, fsamp = self._setup(sigma0=1.0)
        params_in = params.copy()
        params_in[0] = 5.0  # deliberately wrong sigma0 going in
        _seed_all_rng(1)
        res = sample_correlated_noise(tod.copy(), mask, params_in, m, fsamp, cg_err_tol=1e-6,
                                      cg_max_iter=50, sample_params=False, sample_sigma0=True)
        assert res.noise_params[0] == pytest.approx(1.0, rel=0.2)  # recovered true sigma0

    def test_monopole_modes(self):
        m = NoisePSDOof()
        ntod, fsamp, sigma0 = 2**13, 10.0, 1.0
        rng = np.random.default_rng(11)
        base = _synth_corr_noise(11, ntod, fsamp, sigma0, 0.3, -1.8) + rng.normal(0, sigma0, ntod)
        offset = 7.0
        mask = np.ones(ntod, dtype=bool)
        mask[2000:2050] = False
        params = np.array([sigma0, 0.3, -1.8])

        _seed_all_rng(5)
        keep = sample_correlated_noise((base + offset).copy(), mask, params.copy(), m, fsamp,
                                       cg_err_tol=1e-6, cg_max_iter=50, sample_params=False)
        _seed_all_rng(5)
        remove = sample_correlated_noise((base + offset).copy(), mask, params.copy(), m, fsamp,
                                         cg_err_tol=1e-6, cg_max_iter=50, sample_params=False,
                                         nomono=True)
        _seed_all_rng(5)
        only = sample_correlated_noise((base + offset).copy(), mask, params.copy(), m, fsamp,
                                       cg_err_tol=1e-6, cg_max_iter=50, sample_params=False,
                                       onlymono=True)
        # Default leaves the offset in n_corr; nomono projects it out; onlymono returns the offset.
        assert np.mean(keep.n_corr[mask]) == pytest.approx(offset, abs=0.3)
        assert np.mean(remove.n_corr[mask]) == pytest.approx(0.0, abs=1e-6)
        assert np.allclose(only.n_corr, np.mean((base + offset)[mask]))
        assert only.niter == 0
        # onlymono takes precedence when both are set (contradictory; the caller logs an error).
        _seed_all_rng(5)
        both = sample_correlated_noise((base + offset).copy(), mask, params.copy(), m, fsamp,
                                       cg_err_tol=1e-6, cg_max_iter=50, sample_params=False,
                                       nomono=True, onlymono=True)
        assert np.allclose(both.n_corr, only.n_corr) and both.niter == 0


# ===================================================================
# Inverse-noise application (apply_N_inv)
# ===================================================================

class TestApplyNInv:
    @staticmethod
    def _detgroup(noise_model, fsamp=10.0):
        return DetGroupTOD(scans=[], experiment_name="x", band_name="b", nside=64, nu=100.0,
                           fwhm=30.0, fsamp=fsamp, ndet=1, pols="IQU", noise_model=noise_model)

    def test_projects_out_dc(self):
        """apply_N_inv must remove the DC (mean) mode, matching Commander multiply_inv_N."""
        dg = self._detgroup(NoisePSDOof())
        tod = np.random.default_rng(0).normal(0.0, 1.0, 4096) + 12.0  # large offset
        out = dg.apply_N_inv(tod, np.array([1.0, 0.5, -2.0]))
        assert abs(np.mean(out)) < 1e-6 * np.max(np.abs(out))

    def test_white_fast_path(self):
        class _White(NoisePSDOof):
            is_white = True
        dg = self._detgroup(_White())
        tod = np.random.default_rng(1).normal(0.0, 2.0, 4096) + 3.0
        sigma0 = 2.0
        out = dg.apply_N_inv(tod, np.array([sigma0, 0.5, -2.0]))
        expected = tod / sigma0**2
        expected -= np.mean(expected)
        assert np.allclose(out, expected)
