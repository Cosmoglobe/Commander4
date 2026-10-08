"""Direct tests of the compiled mapmaking kernels in ``commander4.backend.mapmaker``.

The mapmaker classes call these kernels on freshly zeroed buffers, which hides a kernel that skips
or misplaces a write. These tests call the kernels directly, with output buffers pre-filled with
NaN (or, for accumulators, random values that must be added to), and compare with plain NumPy.

Maps are always float64; the accumulators also accept a float32 TOD, which they use as float64.
"""
import numpy as np
import pytest
from numpy.typing import NDArray

from commander4.backend import mapmaker as cpp_mapmaker

NPIX = 12
NTOD = 200
TOL = {"rtol": 1e-12, "atol": 1e-12}
# Response pairs (response_I, response_P) covering every fast path of the response kernels.
RESPONSES = [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.35, 0.8)]
# One unobserved (all-zero) pixel and one seen at a single polarization angle (rank 1).
BAD_PIXELS = [4, 7]
GOOD = np.isin(np.arange(NPIX), BAD_PIXELS, invert=True)


def _scan(tod_dtype: type, seed: int) -> tuple[NDArray, NDArray, NDArray]:
    """Random TOD (in `tod_dtype`), int64 pixel indices with many repeats, and float64 angles."""
    rng = np.random.default_rng(seed)
    return (rng.normal(size=NTOD).astype(tod_dtype), rng.integers(0, NPIX, NTOD, dtype=np.int64),
            rng.uniform(0.0, np.pi, NTOD))


def _normal_matrices(seed: int) -> tuple[NDArray, NDArray]:
    """Per-pixel symmetric 3x3 matrices, as (npix, 3, 3) and packed (II,IQ,IU,QQ,QU,UU, npix)."""
    M = np.random.default_rng(seed).normal(size=(NPIX, 3, 3))
    A = M.transpose(0, 2, 1) @ M + np.eye(3)  # symmetric positive definite
    A[BAD_PIXELS[0]] = 0.0
    A[BAD_PIXELS[1]] = np.outer([1.0, 0.6, -0.3], [1.0, 0.6, -0.3])
    packed = np.ascontiguousarray(A[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]].T)
    return A, packed


def test_hit_accumulator() -> None:
    _, pix, _ = _scan(np.float64, seed=3)
    hits = np.random.default_rng(4).normal(size=NPIX)  # must be added to, not replaced
    expected = hits + np.bincount(pix, minlength=NPIX)
    cpp_mapmaker.hit_accumulator(hits, pix)
    np.testing.assert_allclose(hits, expected, **TOL)


@pytest.mark.parametrize("tod_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ncomp", [3, 1])
@pytest.mark.parametrize("response_I, response_P", RESPONSES)
def test_tod2map(tod_dtype: type, ncomp: int, response_I: float, response_P: float) -> None:
    """map += P^T tod. Without polarization (intensity map, or response_P = 0) psi is not read."""
    tod, pix, psi = _scan(tod_dtype, seed=5)
    m = np.random.default_rng(6).normal(size=(ncomp, NPIX))  # must be added to, not replaced
    # Each sample's pointing row is [r_I, r_P cos(2 psi), r_P sin(2 psi)], or [r_I] for intensity.
    rows = [np.full(NTOD, response_I), response_P*np.cos(2*psi), response_P*np.sin(2*psi)]
    expected = m.copy()
    for k in range(ncomp):
        np.add.at(expected[k], pix, rows[k]*tod.astype(np.float64))
    if ncomp == 1 or response_P == 0.0:
        psi = np.full(NTOD, np.nan)  # would spoil the map if it were read
    cpp_mapmaker.tod2map(m, tod, pix, psi, response_I=response_I, response_P=response_P)
    np.testing.assert_allclose(m, expected, **TOL)


@pytest.mark.parametrize("tod_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("response_I, response_P", RESPONSES)
def test_binned_map_accumulator(tod_dtype: type, response_I: float, response_P: float) -> None:
    """The weights and every TOD map of one pass equal NumPy's separate accumulations."""
    _, pix, psi = _scan(tod_dtype, seed=7)
    rng = np.random.default_rng(8)
    tods = rng.normal(size=(3, NTOD)).astype(tod_dtype)
    weights, maps = rng.normal(size=(6, NPIX)), rng.normal(size=(3, 3, NPIX))
    # The pointing row [r_I, r_P cos(2 psi), r_P sin(2 psi)], and the 6 unique elements (II, IQ,
    # IU, QQ, QU, UU) of its outer product.
    r = [np.full(NTOD, response_I), response_P*np.cos(2*psi), response_P*np.sin(2*psi)]
    outer = [r[0]*r[0], r[0]*r[1], r[0]*r[2], r[1]*r[1], r[1]*r[2], r[2]*r[2]]
    expected_weights, expected_maps = weights.copy(), maps.copy()
    for k in range(6):
        np.add.at(expected_weights[k], pix, 2.5*outer[k])
    for j in range(3):
        for k in range(3):
            np.add.at(expected_maps[j, k], pix, 2.5*r[k]*tods[j].astype(np.float64))
    cpp_mapmaker.binned_map_accumulator(weights, maps, tods, 2.5, pix, psi,
                                        response_I=response_I, response_P=response_P)
    np.testing.assert_allclose(weights, expected_weights, **TOL)
    np.testing.assert_allclose(maps, expected_maps, **TOL)


def test_binned_map_accumulator_intensity_band_and_no_maps() -> None:
    """An I-only band has one weight and one map row per pixel, and never reads psi."""
    tod, pix, _ = _scan(np.float32, seed=22)
    psi = np.full(NTOD, np.nan)  # would turn every map into NaN if it were read
    weights, maps = np.zeros((1, NPIX)), np.zeros((1, 1, NPIX))
    cpp_mapmaker.binned_map_accumulator(weights, maps, tod[None, :], 2.5, pix, psi,
                                        response_I=0.4, response_P=1.0)
    expected_weights, expected_map = np.zeros(NPIX), np.zeros(NPIX)
    np.add.at(expected_weights, pix, 2.5*0.4**2)
    np.add.at(expected_map, pix, 2.5*0.4*tod.astype(np.float64))
    np.testing.assert_allclose(weights[0], expected_weights, **TOL)
    np.testing.assert_allclose(maps[0, 0], expected_map, **TOL)
    # With no TOD maps only the weights are accumulated (the CG mapmaker's case).
    weights6 = np.zeros((6, NPIX))
    cpp_mapmaker.binned_map_accumulator(weights6, np.zeros((0, 3, NPIX)),
                                        np.zeros((0, NTOD), np.float32), 1.0, pix, psi,
                                        response_P=0.0)
    np.testing.assert_allclose(weights6[0], np.bincount(pix, minlength=NPIX), **TOL)


@pytest.mark.parametrize("ncomp", [3, 1])
@pytest.mark.parametrize("response_I, response_P", RESPONSES)
def test_map2tod(ncomp: int, response_I: float, response_P: float) -> None:
    """tod = P map, and tod2map is its exact transpose: <P m, t> = <m, P^T t> for any m and t."""
    other_tod, pix, psi = _scan(np.float64, seed=11)
    m = np.random.default_rng(12).normal(size=(ncomp, NPIX))
    rows = [np.full(NTOD, response_I), response_P*np.cos(2*psi), response_P*np.sin(2*psi)]
    expected = sum(rows[k]*m[k, pix] for k in range(ncomp))
    if ncomp == 1 or response_P == 0.0:
        psi = np.full(NTOD, np.nan)  # would spoil the TOD if it were read
    tod = np.full(NTOD, np.nan)  # NaN shows any sample the kernel fails to write
    cpp_mapmaker.map2tod(m, tod, pix, psi, response_I=response_I, response_P=response_P)
    np.testing.assert_allclose(tod, expected, **TOL)
    transposed = np.zeros((ncomp, NPIX))
    cpp_mapmaker.tod2map(transposed, other_tod, pix, psi, response_I=response_I,
                         response_P=response_P)
    np.testing.assert_allclose(np.dot(tod, other_tod), np.vdot(m, transposed), **TOL)


def test_map_solve_IQU() -> None:
    """Solves A x = b per pixel; pixels whose A cannot be inverted are set to zero."""
    A, norm_map = _normal_matrices(seed=13)
    rhs = np.random.default_rng(14).normal(size=(3, NPIX))
    map_out = np.full((3, NPIX), np.nan)
    cpp_mapmaker.map_solve_IQU(map_out, rhs, norm_map)
    expected = np.zeros((3, NPIX))
    expected[:, GOOD] = np.linalg.solve(A[GOOD], rhs[:, GOOD].T[..., None])[..., 0].T
    np.testing.assert_allclose(map_out, expected, rtol=1e-10, atol=1e-12)


def test_map_inv_var_IQU() -> None:
    """Inverse variance is 1/diag(A^-1) per pixel; pixels whose A cannot be inverted get 0."""
    A, norm_map = _normal_matrices(seed=15)
    inv_var_out = np.full((3, NPIX), np.nan)
    cpp_mapmaker.map_inv_var_IQU(inv_var_out, norm_map)
    expected = np.zeros((3, NPIX))
    expected[:, GOOD] = 1.0/np.diagonal(np.linalg.inv(A[GOOD]), axis1=1, axis2=2).T
    np.testing.assert_allclose(inv_var_out, expected, rtol=1e-10)


def test_apply_invN_to_map_IQU() -> None:
    """Multiplies each pixel's IQU vector by its full (not just diagonal) 3x3 matrix."""
    A, inv_N_map = _normal_matrices(seed=16)
    map_in = np.random.default_rng(17).normal(size=(3, NPIX))
    map_out = np.full((3, NPIX), np.nan)
    cpp_mapmaker.apply_invN_to_map_IQU(map_in, map_out, inv_N_map)
    np.testing.assert_allclose(map_out, np.einsum("pij,jp->ip", A, map_in), **TOL)


def test_solvers_give_up_on_a_nearly_singular_pixel() -> None:
    """A pixel whose 3x3 is only nearly singular (I and Q almost degenerate) is unsolvable too."""
    norm_map = np.zeros((6, NPIX))
    norm_map[[0, 3, 5]] = 1.0
    norm_map[1, 0] = 1.0 - 1e-13  # IQ correlation at rounding distance from exact degeneracy
    rhs = np.ones((3, NPIX))
    solved, inv_var = np.full((3, NPIX), np.nan), np.full((3, NPIX), np.nan)
    cpp_mapmaker.map_solve_IQU(solved, rhs, norm_map)
    cpp_mapmaker.map_inv_var_IQU(inv_var, norm_map)
    np.testing.assert_array_equal(solved[:, 0], 0.0)
    np.testing.assert_array_equal(inv_var[:, 0], 0.0)
    np.testing.assert_allclose(solved[:, 1:], 1.0, **TOL)
    np.testing.assert_allclose(inv_var[:, 1:], 1.0, **TOL)


def test_map_solve_IQU_rejects_wrong_shapes_and_dtypes() -> None:
    _, norm_map = _normal_matrices(seed=18)
    rhs = np.ones((3, NPIX))
    with pytest.raises(RuntimeError):  # (3, npix) instead of (6, npix) would read past the end
        cpp_mapmaker.map_solve_IQU(np.zeros((3, NPIX)), rhs, np.ascontiguousarray(norm_map[:3]))
    with pytest.raises(RuntimeError):  # maps are float64 only
        cpp_mapmaker.map_solve_IQU(np.zeros((3, NPIX), dtype=np.float32), rhs, norm_map)


@pytest.mark.parametrize("bad_pixel", [NPIX, -1])
def test_kernels_reject_out_of_range_pixels(bad_pixel: int) -> None:
    """A pixel index outside [0, npix) must raise instead of touching memory outside the map."""
    tod, pix, psi = _scan(np.float64, seed=20)
    pix[50] = bad_pixel
    m1, m3, m6 = np.zeros((1, NPIX)), np.zeros((3, NPIX)), np.zeros((6, NPIX))
    tod_out = np.empty(NTOD)
    maps3, maps1, tods = np.zeros((1, 3, NPIX)), np.zeros((1, 1, NPIX)), tod[None, :]
    # Every loop of every kernel, including the intensity-only loops.
    calls = [
        lambda: cpp_mapmaker.hit_accumulator(np.zeros(NPIX), pix),
        lambda: cpp_mapmaker.binned_map_accumulator(m6, maps3, tods, 1.0, pix, psi),
        lambda: cpp_mapmaker.binned_map_accumulator(m6, maps3, tods, 1.0, pix, psi,
                                                    response_P=0.0),
        lambda: cpp_mapmaker.binned_map_accumulator(m1, maps1, tods, 1.0, pix, psi),
        lambda: cpp_mapmaker.map2tod(m3, tod_out, pix, psi),
        lambda: cpp_mapmaker.map2tod(m1, tod_out, pix, psi),
        lambda: cpp_mapmaker.tod2map(m3, tod, pix, psi),
        lambda: cpp_mapmaker.tod2map(m1, tod, pix, psi),
    ]
    for call in calls:
        with pytest.raises(RuntimeError, match="pixel index out of range"):
            call()


def test_pixel_hash_round_trip() -> None:
    """global_to_local inverts local_pix, for unsorted pixels anywhere on an nside-16384 sky."""
    rng = np.random.default_rng(21)
    npix_max = 12*16384**2
    # A run of neighbouring pixels, both ends of the valid range, and pixels spread over the sky.
    local_pix = np.unique(np.concatenate([np.arange(1000, 3000), [0, npix_max - 1],
                                          rng.integers(0, npix_max, 5000)]))
    rng.shuffle(local_pix)
    table = cpp_mapmaker.build_pixel_hash(local_pix)
    pix = rng.choice(local_pix, size=20000)
    np.testing.assert_array_equal(local_pix[cpp_mapmaker.global_to_local(table, pix)], pix)
    empty = np.zeros(0, dtype=np.int64)
    assert cpp_mapmaker.global_to_local(cpp_mapmaker.build_pixel_hash(empty), empty).size == 0


def test_pixel_hash_rejects_unknown_duplicate_and_out_of_range_pixels() -> None:
    table = cpp_mapmaker.build_pixel_hash(np.array([5, 9, 2], dtype=np.int64))
    with pytest.raises(RuntimeError, match="not in the local pixel domain"):
        cpp_mapmaker.global_to_local(table, np.array([5, 3], dtype=np.int64))
    for bad in (-1, 2**32 - 1):  # cannot be stored in the 32-bit half of a slot
        with pytest.raises(RuntimeError, match="out of range"):
            cpp_mapmaker.global_to_local(table, np.array([5, bad], dtype=np.int64))
    with pytest.raises(RuntimeError, match="duplicate"):
        cpp_mapmaker.build_pixel_hash(np.array([5, 9, 5], dtype=np.int64))


def test_tod2map_rejects_wrong_dtypes_shapes_and_lengths() -> None:
    tod, pix, psi = _scan(np.float32, seed=19)
    m = np.zeros((3, NPIX))
    with pytest.raises(RuntimeError):  # maps are float64 only
        cpp_mapmaker.tod2map(m.astype(np.float32), tod, pix, psi)
    with pytest.raises(RuntimeError):  # neither (3, npix) nor (1, npix)
        cpp_mapmaker.tod2map(np.zeros((2, NPIX)), tod, pix, psi)
    with pytest.raises(RuntimeError):  # int32 pixel indices
        cpp_mapmaker.tod2map(m, tod, pix.astype(np.int32), psi)
    with pytest.raises(RuntimeError):  # pix shorter than the TOD would read past its end
        cpp_mapmaker.tod2map(m, tod, pix[:10].copy(), psi)
