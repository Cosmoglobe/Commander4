"""Check the optimized 1/f likelihood against the original vectorized expression."""
import numpy as np
import pytest
from pixell import utils
from scipy.fft import rfftfreq

from commander4.math_utils.fft import forward_rfft
from commander4.tod.noise.psd import (
    NoisePSDOof, _inversion_sampler_1d, _oof_loglike_grid,
)


@pytest.mark.parametrize("weighted", [False, True])
def test_likelihood_grids_match_original(weighted: bool) -> None:
    """Dropping parameter-independent terms must preserve both conditional distributions."""
    rng = np.random.default_rng(182)
    frequencies = np.geomspace(1e-4, 2.0, 1000)
    power = rng.exponential(scale=4.0, size=frequencies.size)
    weight = rng.integers(1, 100, size=frequencies.size).astype(np.float64)
    if not weighted:
        weight[:] = 1.0
    sigma0_sq, fknee, alpha = 4.0, 0.1, -1.7
    for parameter in ("fknee", "alpha"):
        if parameter == "fknee":
            grid = np.geomspace(0.01, 0.5, 150)
            spectrum = sigma0_sq * (1.0 + (frequencies[:, None] / grid)**alpha)
            actual = _oof_loglike_grid(frequencies**alpha, grid**(-alpha),
                                        power / sigma0_sq, weight)
        else:
            grid = np.linspace(-2.5, -0.25, 150)
            spectrum = sigma0_sq * (1.0 + (frequencies[:, None] / fknee)**grid)
            actual = _oof_loglike_grid(np.log(frequencies / fknee), grid,
                                       power / sigma0_sq, weight, exponentiate=True)
        residual = np.log(power)[:, None] - np.log(spectrum)
        reference = np.sum(weight[:, None] * (residual - np.exp(residual)), axis=0)
        np.testing.assert_allclose(actual - actual.max(), reference - reference.max(),
                                   rtol=1e-11, atol=1e-7)


def _reference_draw(model: NoisePSDOof, tod: np.ndarray, start: np.ndarray,
                    fsamp: float, binned: bool) -> np.ndarray:
    """Original likelihood and sampling sequence, retained as a scientific regression reference."""
    freqs = rfftfreq(tod.size, d=1.0 / fsamp)
    power = np.abs(forward_rfft(tod))**2 / tod.size
    keep = (freqs > 0) & (freqs <= 2.0)
    f, p = freqs[keep], power[keep]
    weights = np.ones(f.size)
    if binned:
        bins = utils.expbin(f.size, nbin=100, nmin=1)
        weights = (bins[:, 1] - bins[:, 0]).astype(np.float64)
        f, p = utils.bin_data(bins, f), utils.bin_data(bins, p)
    fknee_grid = np.geomspace(float(model.P_uni[1, 0]), float(model.P_uni[1, 1]), 150)
    alpha_grid = np.linspace(float(model.P_uni[2, 0]), float(model.P_uni[2, 1]), 150)
    result = start.copy()
    for _ in range(5):
        for parameter, grid in ((1, fknee_grid), (2, alpha_grid)):
            if not model.is_sampled(parameter):
                continue
            fknee = grid if parameter == 1 else result[1]
            alpha = grid if parameter == 2 else result[2]
            spectrum = result[0]**2 * (1.0 + (f[:, None] / fknee)**alpha)
            residual = np.log(p)[:, None] - np.log(spectrum)
            loglike = np.sum(weights[:, None] * (residual - np.exp(residual)), axis=0)
            posterior = loglike + model.log_prior(parameter, grid)
            result[parameter] = _inversion_sampler_1d(posterior, grid)
    return result


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("binned", [False, True])
def test_parameter_draws_and_rng_sequence_match_original(dtype: type, binned: bool) -> None:
    """Binning weights, priors, all five sweeps, and random-number consumption are preserved."""
    model = NoisePSDOof(P_uni=[[np.nan, np.nan], [0.01, 0.5], [-2.5, -0.25]])
    start = np.array([2.0, 0.1, -1.5])
    rng = np.random.default_rng(71)
    ntod, fsamp = 32768, 10.0
    freqs = rfftfreq(ntod, d=1.0 / fsamp)
    spectrum = model.eval_full(freqs, start)
    coefficients = rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size)
    coefficients *= np.sqrt(ntod * spectrum / 2.0)
    coefficients[0] = 0.0
    tod = np.fft.irfft(coefficients, ntod).astype(dtype)
    for seed in range(3):
        np.random.seed(seed)
        reference = _reference_draw(model, tod, start, fsamp, binned)
        next_random = np.random.random()
        np.random.seed(seed)
        actual = model.sample_params(tod, start, fsamp, nu_max=2.0, bin_psd=binned)
        tolerance = 2e-6 if dtype == np.float32 else 1e-10
        np.testing.assert_allclose(actual, reference, rtol=tolerance, atol=tolerance * 0.01)
        assert np.random.random() == next_random
