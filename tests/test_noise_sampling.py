import numpy as np
import pytest

from commander4.noise_sampling.sigma0 import calc_sigma0_simple, calc_sigma0_robust

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
