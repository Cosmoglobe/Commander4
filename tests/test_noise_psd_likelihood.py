"""Check the optimized 1/f likelihood against the original vectorized expression."""
import numpy as np
import pytest
from pixell import utils
from scipy.fft import rfftfreq

from commander4.math_utils.fft import forward_rfft
from commander4.tod.noise.psd import (
    NoisePSD2Oof, NoisePSDOof, _inversion_sampler_1d, _oof_loglike_grid,
)


# Hard bounds that bracket the two-component test configuration and keep the red (alpha) and blue
# (alpha2) slopes in disjoint ranges, so the two components cannot swap roles mid-chain.
_2OOF_UNI = [[np.nan, np.nan], [0.01, 0.5], [-2.5, -0.25], [0.5, 5.0], [0.5, 3.0]]
_2OOF_START = np.array([2.0, 0.3, -2.0, 2.0, 1.5])
_2OOF_PARAMETERS = [(1, "fknee"), (2, "alpha"), (3, "fknee2"), (4, "alpha2")]


def _2oof_nu_fit(param_idx: int, window: list[float]) -> list[list[float]]:
    """The five 2oof fit windows, all [0, 2] Hz except *param_idx*, which gets *window*."""
    nu_fit = [[np.nan, np.nan]] + [[0, 2.0] for _ in range(4)]
    nu_fit[param_idx] = window
    return nu_fit


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
    fknee_grid = np.geomspace(float(model.P_uni[1, 0]), float(model.P_uni[1, 1]), 150)
    alpha_grid = np.linspace(float(model.P_uni[2, 0]), float(model.P_uni[2, 1]), 150)
    result = start.copy()
    for _ in range(5):
        for parameter, grid in ((1, fknee_grid), (2, alpha_grid)):
            if not model.is_sampled(parameter):
                continue
            nu_min, nu_max = model.nu_fit[parameter]
            keep = (freqs > 0) & (freqs >= nu_min) & (freqs <= nu_max)
            f, p = freqs[keep], power[keep]
            weights = np.ones(f.size)
            if binned:
                bins = utils.expbin(f.size, nbin=100, nmin=1)
                weights = (bins[:, 1] - bins[:, 0]).astype(np.float64)
                f, p = utils.bin_data(bins, f), utils.bin_data(bins, p)
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
@pytest.mark.parametrize("windows", [([0, 2.0], [0, 2.0]), ([0.25, 1.5], [0.5, 2.0])])
def test_parameter_draws_and_rng_sequence_match_original(
    dtype: type, binned: bool, windows: tuple[list[float], list[float]],
) -> None:
    """Binning weights, priors, all five sweeps, and random-number consumption are preserved."""
    model = NoisePSDOof(P_uni=[[np.nan, np.nan], [0.01, 0.5], [-2.5, -0.25]],
                        nu_fit=[[np.nan, np.nan], *windows])
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
        actual = model.sample_params(tod, start, fsamp, bin_psd=binned)
        tolerance = 2e-6 if dtype == np.float32 else 1e-10
        np.testing.assert_allclose(actual, reference, rtol=tolerance, atol=tolerance * 0.01)
        assert np.random.random() == next_random


@pytest.mark.parametrize("binned", [False, True])
def test_fit_ignores_power_outside_model_window(binned: bool) -> None:
    """Large spectral lines below and above the selected range must not affect the draw."""
    model = NoisePSDOof(nu_fit=[[np.nan, np.nan], [1.0, 2.0], [1.0, 2.0]])
    start = np.array([1.0, 0.1, -1.5])
    tod = np.random.default_rng(4).normal(size=4096)
    time = np.arange(tod.size) / 8.0
    contaminated = tod + 100 * np.cos(2 * np.pi * 0.5 * time)
    contaminated += 100 * np.cos(2 * np.pi * 3.0 * time) + 100
    np.random.seed(19)
    expected = model.sample_params(tod, start, 8.0, bin_psd=binned)
    np.random.seed(19)
    actual = model.sample_params(contaminated, start, 8.0, bin_psd=binned)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("window", [[-1, 2], [2, 1], [1, 1], [np.nan, 2], [0, np.nan]])
def test_invalid_model_window_raises(window: list[float]) -> None:
    model = NoisePSDOof(nu_fit=[[np.nan, np.nan], window, [0, 2]])
    with pytest.raises(ValueError, match="Invalid nu_fit range for fknee"):
        model.sample_params(np.ones(128), np.array([1.0, 0.1, -1.5]), 8.0)


@pytest.mark.parametrize("binned", [False, True])
def test_empty_model_window_raises(binned: bool) -> None:
    model = NoisePSDOof(nu_fit=[[np.nan, np.nan], [5, 6], [0, 2]])
    with pytest.raises(ValueError, match="no positive Fourier modes"):
        model.sample_params(np.ones(128), np.array([1.0, 0.1, -1.5]), 8.0, bin_psd=binned)


def test_model_copies_windows_and_skips_fixed_parameter_window() -> None:
    windows = np.array([[np.nan, np.nan], [np.nan, np.nan], [0, np.inf]])
    model = NoisePSDOof(nu_fit=windows, P_active_rms=[np.nan, 0, np.inf])
    windows[2] = [10, 20]
    np.testing.assert_array_equal(model.nu_fit[2], [0, np.inf])
    start = np.array([1.0, 0.1, -1.5])
    result = model.sample_params(np.ones(128), start, 8.0)
    assert result[1] == start[1]
    assert np.all(np.isfinite(result))


# ===================================================================
# Double 1/f model ("2oof"): the `offset` argument and its four-parameter Gibbs sweep
# ===================================================================

@pytest.mark.parametrize("exponentiate", [False, True])
def test_likelihood_offset_defaults_to_the_single_component_white_floor(exponentiate: bool) -> None:
    """The new `offset` must leave every call the single-1/f model makes bit-for-bit unchanged."""
    rng = np.random.default_rng(913)
    frequencies = np.geomspace(1e-4, 2.0, 500)
    power = rng.exponential(scale=4.0, size=frequencies.size)
    weight = rng.integers(1, 100, size=frequencies.size).astype(np.float64)
    if exponentiate:
        factor, grid = np.log(frequencies / 0.1), np.linspace(-2.5, -0.25, 150)
    else:
        factor, grid = frequencies**-1.7, np.geomspace(0.01, 0.5, 150)**1.7
    baseline = _oof_loglike_grid(factor, grid, power, weight, exponentiate=exponentiate)
    # A scalar 1.0 and an explicit array of ones must both reproduce the hardcoded `+= 1.0` exactly,
    # not merely to within rounding: adding IEEE 1.0 is the same operation either way.
    for offset in (1.0, np.ones(frequencies.size)):
        actual = _oof_loglike_grid(factor, grid, power, weight, exponentiate=exponentiate,
                                   offset=offset)
        np.testing.assert_array_equal(actual, baseline)


@pytest.mark.parametrize("weighted", [False, True])
def test_2oof_likelihood_grids_match_original(weighted: bool) -> None:
    """With the other component folded into `offset`, all four conditionals stay Whittle-exact."""
    rng = np.random.default_rng(182)
    frequencies = np.geomspace(1e-4, 2.0, 1000)
    power = rng.exponential(scale=4.0, size=frequencies.size)
    weight = rng.integers(1, 100, size=frequencies.size).astype(np.float64)
    if not weighted:
        weight[:] = 1.0
    sigma0_sq, fknee, alpha, fknee2, alpha2 = 4.0, 0.1, -1.7, 3.0, 1.6
    red, blue = (frequencies/fknee)**alpha, (frequencies/fknee2)**alpha2
    for parameter in ("fknee", "alpha", "fknee2", "alpha2"):
        # `offset` holds the white floor plus the component *not* being sampled, evaluated on the
        # same frequencies; `sampled` is the (nfreq, ngrid) contribution of the sampled component.
        if parameter == "fknee":
            grid, offset = np.geomspace(0.01, 0.5, 150), 1.0 + blue
            sampled = (frequencies[:, None] / grid)**alpha
            actual = _oof_loglike_grid(frequencies**alpha, grid**(-alpha),
                                       power / sigma0_sq, weight, offset=offset)
        elif parameter == "alpha":
            grid, offset = np.linspace(-2.5, -0.25, 150), 1.0 + blue
            sampled = (frequencies[:, None] / fknee)**grid
            actual = _oof_loglike_grid(np.log(frequencies) - np.log(fknee), grid,
                                       power / sigma0_sq, weight, exponentiate=True, offset=offset)
        elif parameter == "fknee2":
            grid, offset = np.geomspace(1.0, 20.0, 150), 1.0 + red
            sampled = (frequencies[:, None] / grid)**alpha2
            actual = _oof_loglike_grid(frequencies**alpha2, grid**(-alpha2),
                                       power / sigma0_sq, weight, offset=offset)
        else:
            grid, offset = np.linspace(0.5, 3.0, 150), 1.0 + red
            sampled = (frequencies[:, None] / fknee2)**grid
            actual = _oof_loglike_grid(np.log(frequencies) - np.log(fknee2), grid,
                                       power / sigma0_sq, weight, exponentiate=True, offset=offset)
        spectrum = sigma0_sq * (offset[:, None] + sampled)
        residual = np.log(power)[:, None] - np.log(spectrum)
        reference = np.sum(weight[:, None] * (residual - np.exp(residual)), axis=0)
        np.testing.assert_allclose(actual - actual.max(), reference - reference.max(),
                                   rtol=1e-11, atol=1e-7, err_msg=parameter)


def _reference_draw_2oof(model: NoisePSD2Oof, tod: np.ndarray, start: np.ndarray,
                         fsamp: float, binned: bool) -> np.ndarray:
    """Original likelihood and sampling sequence, retained as a scientific regression reference."""
    freqs = rfftfreq(tod.size, d=1.0 / fsamp)
    power = np.abs(forward_rfft(tod))**2 / tod.size
    grids = {1: np.geomspace(float(model.P_uni[1, 0]), float(model.P_uni[1, 1]), 150),
             2: np.linspace(float(model.P_uni[2, 0]), float(model.P_uni[2, 1]), 150),
             3: np.geomspace(float(model.P_uni[3, 0]), float(model.P_uni[3, 1]), 150),
             4: np.linspace(float(model.P_uni[4, 0]), float(model.P_uni[4, 1]), 150)}
    result = start.copy()
    for _ in range(5):
        for parameter in (1, 2, 3, 4):   # the sampler's Gibbs order: fknee, alpha, fknee2, alpha2
            if not model.is_sampled(parameter):
                continue
            grid = grids[parameter]
            nu_min, nu_max = model.nu_fit[parameter]
            keep = (freqs > 0) & (freqs >= nu_min) & (freqs <= nu_max)
            f, p = freqs[keep], power[keep]
            weights = np.ones(f.size)
            if binned:
                bins = utils.expbin(f.size, nbin=100, nmin=1)
                weights = (bins[:, 1] - bins[:, 0]).astype(np.float64)
                f, p = utils.bin_data(bins, f), utils.bin_data(bins, p)
            # The component held fixed in this step, on this parameter's own frequency vector.
            held = (f/result[3])**result[4] if parameter in (1, 2) else (f/result[1])**result[2]
            if parameter == 1:
                sampled = (f[:, None] / grid)**result[2]
            elif parameter == 2:
                sampled = (f[:, None] / result[1])**grid
            elif parameter == 3:
                sampled = (f[:, None] / grid)**result[4]
            else:
                sampled = (f[:, None] / result[3])**grid
            spectrum = result[0]**2 * (1.0 + held[:, None] + sampled)
            residual = np.log(p)[:, None] - np.log(spectrum)
            loglike = np.sum(weights[:, None] * (residual - np.exp(residual)), axis=0)
            posterior = loglike + model.log_prior(parameter, grid)
            result[parameter] = _inversion_sampler_1d(posterior, grid)
    return result


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("binned", [False, True])
@pytest.mark.parametrize("windows", [([0, 2.0], [0, 2.0], [0, 2.0], [0, 2.0]),
                                     ([0, 1.5], [0.02, 2.0], [0.5, 5.0], [1.0, 5.0])])
def test_2oof_parameter_draws_and_rng_sequence_match_original(
    dtype: type, binned: bool, windows: tuple[list[float], ...],
) -> None:
    """Per-parameter offsets and windows, binning weights, priors, all twenty steps, and random-
    number consumption are preserved.

    The start values and windows are tuned so each component rises well above the white floor
    inside its own window (red 9x the floor at 0.1 Hz, blue 3.9x at 5 Hz). That is load-bearing: in
    a window where neither component clears the floor, a conditional goes degenerate and pins at a
    grid edge, and later steps then amplify a 1e-9 likelihood difference chaotically.
    """
    model = NoisePSD2Oof(P_uni=_2OOF_UNI, nu_fit=[[np.nan, np.nan], *windows])
    start = _2OOF_START
    rng = np.random.default_rng(71)
    ntod, fsamp = 8192, 10.0
    freqs = rfftfreq(ntod, d=1.0 / fsamp)
    spectrum = model.eval_full(freqs, start)
    coefficients = rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size)
    coefficients *= np.sqrt(ntod * spectrum / 2.0)
    coefficients[0] = 0.0
    tod = np.fft.irfft(coefficients, ntod).astype(dtype)
    for seed in range(3):
        np.random.seed(seed)
        reference = _reference_draw_2oof(model, tod, start, fsamp, binned)
        next_random = np.random.random()
        np.random.seed(seed)
        actual = model.sample_params(tod, start, fsamp, bin_psd=binned)
        tolerance = 2e-6 if dtype == np.float32 else 1e-10
        np.testing.assert_allclose(actual, reference, rtol=tolerance, atol=tolerance * 0.01)
        assert np.random.random() == next_random


@pytest.mark.parametrize("param_idx, name", _2OOF_PARAMETERS)
@pytest.mark.parametrize("window", [[-1, 2], [2, 1], [1, 1], [np.nan, 2], [0, np.nan]])
def test_invalid_2oof_model_window_raises(param_idx: int, name: str,
                                          window: list[float]) -> None:
    model = NoisePSD2Oof(P_uni=_2OOF_UNI, nu_fit=_2oof_nu_fit(param_idx, window))
    with pytest.raises(ValueError, match=f"Invalid nu_fit range for {name}"):
        model.sample_params(np.ones(128), _2OOF_START, 8.0)


@pytest.mark.parametrize("param_idx, name", _2OOF_PARAMETERS)
@pytest.mark.parametrize("binned", [False, True])
def test_empty_2oof_model_window_raises(param_idx: int, name: str, binned: bool) -> None:
    model = NoisePSD2Oof(P_uni=_2OOF_UNI, nu_fit=_2oof_nu_fit(param_idx, [5, 6]))
    with pytest.raises(ValueError, match="no positive Fourier modes"):
        model.sample_params(np.ones(128), _2OOF_START, 8.0, bin_psd=binned)


@pytest.mark.parametrize("binned", [False, True])
def test_2oof_fit_ignores_power_outside_model_window(binned: bool) -> None:
    """Large spectral lines below and above the selected range must not affect any of the four."""
    model = NoisePSD2Oof(P_uni=_2OOF_UNI,
                         nu_fit=[[np.nan, np.nan]] + [[1.0, 2.0] for _ in range(4)])
    start = np.array([1.0, 0.1, -1.5, 2.0, 1.5])
    tod = np.random.default_rng(4).normal(size=4096)
    time = np.arange(tod.size) / 8.0
    contaminated = tod + 100 * np.cos(2 * np.pi * 0.5 * time)
    contaminated += 100 * np.cos(2 * np.pi * 3.0 * time) + 100
    np.random.seed(19)
    expected = model.sample_params(tod, start, 8.0, bin_psd=binned)
    np.random.seed(19)
    actual = model.sample_params(contaminated, start, 8.0, bin_psd=binned)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)


def test_2oof_model_copies_windows_and_skips_fixed_parameters() -> None:
    """A fixed fknee/fknee2 must still supply the offset the slope steps of both components need."""
    windows = np.array([[np.nan, np.nan], [np.nan, np.nan], [0, np.inf],
                        [np.nan, np.nan], [0, np.inf]])
    model = NoisePSD2Oof(P_uni=_2OOF_UNI, nu_fit=windows,
                         P_active_rms=[np.nan, 0, np.inf, 0, np.inf])
    windows[2] = [10, 20]
    np.testing.assert_array_equal(model.nu_fit[2], [0, np.inf])
    assert [model.is_sampled(i) for i in range(5)] == [False, False, True, False, True]
    start = np.array([1.0, 0.1, -1.5, 2.0, 1.5])
    result = model.sample_params(np.ones(128), start, 8.0)
    assert result[1] == start[1] and result[3] == start[3]
    assert result[2] != start[2] and result[4] != start[4]
    assert np.all(np.isfinite(result))


def test_inversion_sampler_is_unbiased_on_a_peaked_likelihood() -> None:
    """A likelihood peaked on a grid point must sample that point, not the cell below it.

    A plain `cumsum` credits each cell's mass to its right edge, which biased every draw half a grid
    cell low: 3% on the default fknee grid, and cumulative along a flat posterior ridge such as the
    double 1/f model's (fknee2, alpha2).
    """
    grid = np.linspace(0.0, 1.0, 101)
    cell = grid[1] - grid[0]
    lnL = -0.5*((grid - 0.5)/(0.35*cell))**2      # near-delta at grid index 50
    np.random.seed(0)
    draws = np.array([_inversion_sampler_1d(lnL.copy(), grid) for _ in range(20000)])
    assert abs(draws.mean() - 0.5) < 0.05*cell    # was -0.505 cells before the midpoint CDF
    # The draws must straddle the mode rather than sit below it: the right-edge CDF put 98.4% of
    # them below, which is the asymmetry that accumulates along a ridge.
    assert 0.45 < np.mean(draws < 0.5) < 0.55
    assert grid[47] <= draws.min() and draws.max() <= grid[53]
