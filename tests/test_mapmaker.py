import numpy as np
from numpy.typing import NDArray
from mpi4py import MPI

from commander4.tod.mapmaking.binned import MapmakerIQU, WeightsMapmakerIQU


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


def _rms_expected(norm_map: NDArray) -> NDArray:
	"""Reference RMS estimate from per-pixel inverse covariance diagonals.

	Pixels that cannot be inverted (no weights, singular, ill-conditioned) get RMS = inf,
	which is the "zero weight" sentinel used by both mapmaker implementations.
	"""
	npix = norm_map.shape[1]
	expected = np.full((3, npix), np.inf, dtype=norm_map.dtype)
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
		expected[0, ipix] = np.sqrt(A_inv[0, 0]) if A_inv[0, 0] > 0 else np.inf
		expected[1, ipix] = np.sqrt(A_inv[1, 1]) if A_inv[1, 1] > 0 else np.inf
		expected[2, ipix] = np.sqrt(A_inv[2, 2]) if A_inv[2, 2] > 0 else np.inf
	return expected


def test_mapmaker_iqu_solve_matches_numpy_float64():
	"""ctypes IQU solver matches NumPy solve for well-conditioned float64 inputs."""
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
	"""ctypes IQU solver handles float32 outputs with identity normalization."""
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


def test_weights_mapmaker_iqu_invdiag_matches_numpy():
	"""ctypes RMS computation matches NumPy inverse-diagonal reference."""
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

	expected = _rms_expected(norm_map)
	assert np.allclose(mapmaker.final_rms_map, expected, rtol=1e-10, atol=1e-12)


def test_weights_mapmaker_iqu_singular_pixel_infinite():
	"""Python reference RMS computation gives infinite RMS for fully singular pixels."""
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

	expected = _rms_expected(norm_map)
	expected[:, 0] = np.inf
	assert np.allclose(mapmaker.final_rms_map, expected, rtol=1e-5, atol=1e-6)


def test_mapmaker_iqu_ill_conditioned_masked_ctypes():
	"""ctypes IQU solver masks ill-conditioned pixels near singularity."""
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


def test_weights_mapmaker_iqu_ill_conditioned_masked_ctypes():
	"""ctypes RMS computation gives infinite RMS for ill-conditioned pixels."""
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

	assert np.all(np.isinf(mapmaker.final_rms_map[:, 0]))
	assert np.allclose(mapmaker.final_rms_map[:, 1], 1.0, rtol=1e-12, atol=1e-12)


def test_intensity_only_accumulators_do_not_evaluate_polarization_angles():
    """An intensity-only response avoids polarization work in native and reference kernels."""
    nside = 1
    pix = np.array([0, 1, 1, 4], dtype=np.int64)
    psi = np.full(pix.size, np.nan)
    tod = np.array([1.0, 2.0, 3.0, 4.0])
    response = np.array([1.0, 0.0])
    weight = 2.5

    signal = MapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
    signal_ref = MapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
    signal.accumulate_to_map(tod, weight, pix, psi, response=response)
    signal_ref.accumulate_to_map_Python(tod, weight, pix, psi, response=response)
    np.testing.assert_allclose(signal._map_signal, signal_ref._map_signal)
    np.testing.assert_array_equal(signal._map_signal[1:3], 0.0)

    weights = WeightsMapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
    weights_ref = WeightsMapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
    weights.accumulate_to_map(weight, pix, psi, response=response)
    weights_ref.accumulate_to_map_Python(weight, pix, psi, response=response)
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
        None,
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 0.0]),
        np.array([0.35, 0.8]),
    ]

    for response in responses:
        signal = MapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
        signal_ref = MapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
        signal.accumulate_to_map(tod, weight, pix, psi, response=response)
        signal_ref.accumulate_to_map_Python(tod, weight, pix, psi, response=response)
        np.testing.assert_allclose(signal._map_signal, signal_ref._map_signal, rtol=1e-14)

        weights = WeightsMapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
        weights_ref = WeightsMapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
        weights.accumulate_to_map(weight, pix, psi, response=response)
        weights_ref.accumulate_to_map_Python(weight, pix, psi, response=response)
        np.testing.assert_allclose(weights._map_signal, weights_ref._map_signal, rtol=1e-14)
