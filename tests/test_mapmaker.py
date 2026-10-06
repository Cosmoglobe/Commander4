import numpy as np
import pytest
from numpy.typing import NDArray
from mpi4py import MPI

from commander4.tod.mapmaking.binned import Mapmaker, MapmakerIQU, WeightsMapmaker,\
	WeightsMapmakerIQU


def test_scalar_weights_gather() -> None:
    """The scalar weights expose the normal-matrix map through the IQU interface's final_cov_map."""
    weights = WeightsMapmaker(MPI.COMM_SELF, 1)
    with pytest.raises(RuntimeError, match="unfinished weights map"):
        weights.final_map

    pix = np.array([0, 3, 3, 7], dtype=np.int64)
    weights.accumulate_to_map(2.0, pix, response_I_P=(0.5, 0.07))
    weights.gather_map()

    expected_weights = np.zeros(12)
    np.add.at(expected_weights, pix, 2.0 * 0.5**2)
    np.testing.assert_array_equal(weights.final_cov_map, expected_weights)
    assert weights.final_cov_map is weights.final_map


def _build_norm_map_from_A(A: NDArray) -> NDArray:
	"""Pack the symmetric 3x3 A-matrix into the 6-element map layout."""
	npix = A.shape[0]
	norm_map = np.zeros((6, npix), dtype=A.dtype)
	norm_map[0] = A[:, 0, 0]
	norm_map[1] = A[:, 0, 1]
	norm_map[2] = A[:, 0, 2]
	norm_map[3] = A[:, 1, 1]
	norm_map[4] = A[:, 1, 2]
	norm_map[5] = A[:, 2, 2]
	return norm_map


def _solve_expected(norm_map: NDArray, rhs: NDArray) -> NDArray:
	"""Reference IQU solve using explicit per-pixel 3x3 solves."""
	npix = rhs.shape[1]
	expected = np.zeros_like(rhs)
	reg = rhs.dtype.type(1e-12)
	for ipix in range(npix):
		a00 = norm_map[0, ipix]
		a01 = norm_map[1, ipix]
		a02 = norm_map[2, ipix]
		a11 = norm_map[3, ipix]
		a12 = norm_map[4, ipix]
		a22 = norm_map[5, ipix]
		# Skip pixels with no accumulated weights.
		if a00 == 0 and a01 == 0 and a02 == 0 and a11 == 0 and a12 == 0 and a22 == 0:
			continue
		A = np.array(
			[[a00, a01, a02], [a01, a11, a12], [a02, a12, a22]],
			dtype=rhs.dtype,
		)
		A = A + np.eye(3, dtype=rhs.dtype) * reg
		expected[:, ipix] = np.linalg.solve(A, rhs[:, ipix])
	return expected


def _inv_var_expected(norm_map: NDArray) -> NDArray:
	"""Reference inverse variance 1/diag(A^-1) from the per-pixel matrices A.

	Pixels that cannot be inverted (no weights, singular, ill-conditioned) get zero inverse
	variance, as in both mapmaker implementations.
	"""
	npix = norm_map.shape[1]
	expected = np.zeros((3, npix), dtype=norm_map.dtype)
	reg = norm_map.dtype.type(1e-12)
	for ipix in range(npix):
		a00 = norm_map[0, ipix]
		a01 = norm_map[1, ipix]
		a02 = norm_map[2, ipix]
		a11 = norm_map[3, ipix]
		a12 = norm_map[4, ipix]
		a22 = norm_map[5, ipix]
		# Skip pixels with no accumulated weights.
		if a00 == 0 and a01 == 0 and a02 == 0 and a11 == 0 and a12 == 0 and a22 == 0:
			continue
		A = np.array(
			[[a00, a01, a02], [a01, a11, a12], [a02, a12, a22]],
			dtype=norm_map.dtype,
		)
		A = A + np.eye(3, dtype=norm_map.dtype) * reg
		A_inv = np.linalg.inv(A)
		expected[:, ipix] = 1.0/np.diagonal(A_inv)
	return expected


def test_mapmaker_iqu_solve_matches_numpy_float64():
	"""C++IQU solver matches NumPy solve for well-conditioned float64 inputs."""
	rng = np.random.default_rng(123)
	nside = 1
	npix = 12 * nside**2
	M = rng.normal(size=(npix, 3, 3))
	A = np.matmul(np.transpose(M, (0, 2, 1)), M) + np.eye(3)[None, :, :]
	norm_map = _build_norm_map_from_A(A).astype(np.float64)
	rhs = rng.normal(size=(3, npix)).astype(np.float64)

	mapmaker = MapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
	mapmaker._gathered_map = rhs
	mapmaker._has_gathered = True
	mapmaker.normalize_map(norm_map)

	expected = _solve_expected(norm_map, rhs)
	assert np.allclose(mapmaker.final_map, expected, rtol=1e-10, atol=1e-12)


def test_mapmaker_iqu_solve_float32_identity():
	"""C++IQU solver handles float32 outputs with identity normalization."""
	rng = np.random.default_rng(456)
	nside = 1
	npix = 12 * nside**2
	norm_map = np.zeros((6, npix), dtype=np.float32)
	norm_map[0] = 1.0
	norm_map[3] = 1.0
	norm_map[5] = 1.0
	rhs = rng.normal(size=(3, npix)).astype(np.float32)

	mapmaker = MapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float32)
	mapmaker._gathered_map = rhs
	mapmaker._has_gathered = True
	mapmaker.normalize_map(norm_map)

	expected = _solve_expected(norm_map, rhs)
	assert np.allclose(mapmaker.final_map, expected, rtol=1e-5, atol=1e-6)


def test_mapmaker_iqu_singular_pixel_zeroed():
	"""Python reference solver zeros pixels with fully singular normalization."""
	rng = np.random.default_rng(789)
	nside = 1
	npix = 12 * nside**2
	norm_map = np.zeros((6, npix), dtype=np.float64)
	norm_map[0] = 1.0
	norm_map[3] = 1.0
	norm_map[5] = 1.0
	norm_map[:, 0] = 0.0
	rhs = rng.normal(size=(3, npix)).astype(np.float64)

	mapmaker = MapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
	mapmaker._gathered_map = rhs
	mapmaker._has_gathered = True
	mapmaker.normalize_map_Python(norm_map)

	expected = _solve_expected(norm_map, rhs)
	expected[:, 0] = 0.0
	assert np.allclose(mapmaker.final_map, expected, rtol=1e-10, atol=1e-12)


def test_weights_mapmaker_iqu_inv_var_matches_numpy():
	"""C++ inverse-variance computation matches NumPy inverse-diagonal reference."""
	rng = np.random.default_rng(321)
	nside = 1
	npix = 12 * nside**2
	M = rng.normal(size=(npix, 3, 3))
	A = np.matmul(np.transpose(M, (0, 2, 1)), M) + np.eye(3)[None, :, :]
	norm_map = _build_norm_map_from_A(A).astype(np.float64)

	mapmaker = WeightsMapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
	mapmaker._gathered_map = norm_map
	mapmaker._has_gathered = True
	mapmaker.normalize_map()

	expected = _inv_var_expected(norm_map)
	assert np.allclose(mapmaker.final_inv_var_map, expected, rtol=1e-10, atol=1e-12)


def test_weights_mapmaker_iqu_singular_pixel_zero_weight():
	"""Python reference gives zero inverse variance for fully singular pixels."""
	nside = 1
	npix = 12 * nside**2
	norm_map = np.zeros((6, npix), dtype=np.float32)
	norm_map[0] = 2.0
	norm_map[3] = 3.0
	norm_map[5] = 4.0
	norm_map[:, 0] = 0.0

	mapmaker = WeightsMapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float32)
	mapmaker._gathered_map = norm_map
	mapmaker._has_gathered = True
	mapmaker.normalize_map_Python()

	expected = _inv_var_expected(norm_map)
	expected[:, 0] = 0.0
	assert np.allclose(mapmaker.final_inv_var_map, expected, rtol=1e-5, atol=1e-6)


def test_mapmaker_iqu_ill_conditioned_masked_cpp():
	"""C++IQU solver masks ill-conditioned pixels near singularity."""
	nside = 1
	npix = 12 * nside**2
	norm_map = np.zeros((6, npix), dtype=np.float64)
	norm_map[0] = 1.0
	norm_map[3] = 1.0
	norm_map[5] = 1.0

	# Make one pixel ill-conditioned but not strictly singular.
	near_one = 1.0 - 1e-13
	norm_map[1, 0] = near_one
	norm_map[0, 0] = 1.0
	norm_map[3, 0] = 1.0
	norm_map[5, 0] = 1.0

	rhs = np.ones((3, npix), dtype=np.float64)
	mapmaker = MapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
	mapmaker._gathered_map = rhs
	mapmaker._has_gathered = True
	mapmaker.normalize_map(norm_map)

	assert np.allclose(mapmaker.final_map[:, 0], 0.0)
	assert np.allclose(mapmaker.final_map[:, 1], rhs[:, 1], rtol=1e-12, atol=1e-12)


def test_weights_mapmaker_iqu_ill_conditioned_masked_cpp():
	"""C++ inverse-variance computation gives zero weight to ill-conditioned pixels."""
	nside = 1
	npix = 12 * nside**2
	norm_map = np.zeros((6, npix), dtype=np.float64)
	norm_map[0] = 1.0
	norm_map[3] = 1.0
	norm_map[5] = 1.0

	near_one = 1.0 - 1e-13
	norm_map[1, 0] = near_one

	mapmaker = WeightsMapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
	mapmaker._gathered_map = norm_map
	mapmaker._has_gathered = True
	mapmaker.normalize_map()

	assert np.all(mapmaker.final_inv_var_map[:, 0] == 0.0)
	assert np.allclose(mapmaker.final_inv_var_map[:, 1], 1.0, rtol=1e-12, atol=1e-12)


def test_intensity_only_accumulators_do_not_evaluate_polarization_angles():
    """An intensity-only response avoids polarization work in native and reference kernels."""
    nside = 1
    pix = np.array([0, 1, 1, 4], dtype=np.int64)
    psi = np.full(pix.size, np.nan)
    tod = np.array([1.0, 2.0, 3.0, 4.0])
    response_I_P = (1.0, 0.0)
    weight = 2.5

    signal = MapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
    signal_ref = MapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
    signal.accumulate_to_map(tod, weight, pix, psi, response_I_P=response_I_P)
    signal_ref.accumulate_to_map_Python(tod, weight, pix, psi, response_I_P=response_I_P)
    np.testing.assert_allclose(signal._map_signal, signal_ref._map_signal)
    np.testing.assert_array_equal(signal._map_signal[1:3], 0.0)

    weights = WeightsMapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
    weights_ref = WeightsMapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
    weights.accumulate_to_map(weight, pix, psi, response_I_P=response_I_P)
    weights_ref.accumulate_to_map_Python(weight, pix, psi, response_I_P=response_I_P)
    np.testing.assert_allclose(weights._map_signal, weights_ref._map_signal)
    np.testing.assert_array_equal(weights._map_signal[1:6], 0.0)


def test_response_accumulators_match_reference_for_binary_and_general_coefficients():
    """Native fast paths and the general path match the readable Python implementation."""
    rng = np.random.default_rng(654)
    nside = 1
    pix = rng.integers(0, 12 * nside**2, size=100, dtype=np.int64)
    psi = rng.uniform(0.0, np.pi, size=pix.size)
    tod = rng.normal(size=pix.size)
    weight = 2.5
    responses = [
        (1.0, 1.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 0.0),
        (0.35, 0.8),
    ]

    for response_I_P in responses:
        signal = MapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
        signal_ref = MapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
        signal.accumulate_to_map(tod, weight, pix, psi, response_I_P=response_I_P)
        signal_ref.accumulate_to_map_Python(tod, weight, pix, psi, response_I_P=response_I_P)
        np.testing.assert_allclose(signal._map_signal, signal_ref._map_signal, rtol=1e-14)

        weights = WeightsMapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
        weights_ref = WeightsMapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
        weights.accumulate_to_map(weight, pix, psi, response_I_P=response_I_P)
        weights_ref.accumulate_to_map_Python(weight, pix, psi, response_I_P=response_I_P)
        np.testing.assert_allclose(weights._map_signal, weights_ref._map_signal, rtol=1e-14)


def test_intensity_mapmakers_apply_the_intensity_response():
    """The I-only mapmakers weight by the same response the IQU ones put in their II element.

    `pols = "I"` runs the scalar mapmakers while the CG operator still applies `response_I`, so the
    two must agree or the inverse-variance map is inconsistent with A.
    """
    rng = np.random.default_rng(11)
    nside = 1
    pix = rng.integers(0, 12 * nside**2, size=50, dtype=np.int64)
    psi = rng.uniform(0.0, np.pi, size=pix.size)
    tod = rng.normal(size=pix.size)
    weight = 2.5
    response_I_P = (0.4, 0.0)  # QU set to zero so the IQU II element is the I-only answer.

    signal = Mapmaker(MPI.COMM_SELF, nside, dtype=np.float64)
    signal_plain = Mapmaker(MPI.COMM_SELF, nside, dtype=np.float64)
    signal_iqu = MapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
    signal.accumulate_to_map(tod, weight, pix, response_I_P=response_I_P)
    signal_plain.accumulate_to_map(tod, weight, pix)
    signal_iqu.accumulate_to_map(tod, weight, pix, psi, response_I_P=response_I_P)
    np.testing.assert_allclose(signal._map_signal, signal_iqu._map_signal[0], rtol=1e-14)
    np.testing.assert_allclose(signal._map_signal, 0.4 * signal_plain._map_signal, rtol=1e-14)

    weights = WeightsMapmaker(MPI.COMM_SELF, nside)
    weights_plain = WeightsMapmaker(MPI.COMM_SELF, nside)
    weights_iqu = WeightsMapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
    weights.accumulate_to_map(weight, pix, response_I_P=response_I_P)
    weights_plain.accumulate_to_map(weight, pix)
    weights_iqu.accumulate_to_map(weight, pix, psi, response_I_P=response_I_P)
    # P^T N^-1 P, so the response enters squared.
    np.testing.assert_allclose(weights._map_signal, weights_iqu._map_signal[0], rtol=1e-14)
    np.testing.assert_allclose(weights._map_signal, 0.4**2 * weights_plain._map_signal, rtol=1e-14)


def test_the_default_response_leaves_the_intensity_mapmakers_unscaled():
    """The [1, 1] default is a standard detector, so it must accumulate unweighted."""
    nside = 1
    pix = np.array([0, 3, 3, 7], dtype=np.int64)
    tod = np.array([1.0, 2.0, 3.0, 4.0])

    signal = Mapmaker(MPI.COMM_SELF, nside, dtype=np.float64)
    signal.accumulate_to_map(tod, 2.0, pix, response_I_P=(1.0, 1.0))
    expected = np.zeros(12 * nside**2)
    np.add.at(expected, pix, 2.0 * tod)
    np.testing.assert_allclose(signal._map_signal, expected, rtol=1e-14)

    weights = WeightsMapmaker(MPI.COMM_SELF, nside)
    weights.accumulate_to_map(2.0, pix, response_I_P=(1.0, 1.0))
    expected_weights = np.zeros(12 * nside**2)
    np.add.at(expected_weights, pix, 2.0)
    np.testing.assert_allclose(weights._map_signal, expected_weights, rtol=1e-14)
