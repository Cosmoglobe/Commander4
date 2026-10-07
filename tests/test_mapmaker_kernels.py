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


@pytest.mark.parametrize("tod_dtype", [np.float32, np.float64])
def test_map_accumulator(tod_dtype: type) -> None:
    tod, pix, _ = _scan(tod_dtype, seed=1)
    m = np.random.default_rng(2).normal(size=NPIX)  # must be added to, not replaced
    expected = m.copy()
    np.add.at(expected, pix, 2.5*tod.astype(np.float64))
    cpp_mapmaker.map_accumulator(m, tod, 2.5, pix)
    np.testing.assert_allclose(m, expected, **TOL)


def test_map_weight_accumulator() -> None:
    _, pix, _ = _scan(np.float64, seed=3)
    m = np.random.default_rng(4).normal(size=NPIX)
    expected = m.copy()
    np.add.at(expected, pix, 2.5)
    cpp_mapmaker.map_weight_accumulator(m, 2.5, pix)
    np.testing.assert_allclose(m, expected, **TOL)


@pytest.mark.parametrize("tod_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("response_I, response_P", RESPONSES)
def test_map_accumulator_IQU(tod_dtype: type, response_I: float, response_P: float) -> None:
    tod, pix, psi = _scan(tod_dtype, seed=5)
    m = np.random.default_rng(6).normal(size=(3, NPIX))
    # Each sample sees the pointing row [r_I, r_P cos(2 psi), r_P sin(2 psi)].
    rows = [np.full(NTOD, response_I), response_P*np.cos(2*psi), response_P*np.sin(2*psi)]
    expected = m.copy()
    for k in range(3):
        np.add.at(expected[k], pix, 2.5*rows[k]*tod.astype(np.float64))
    cpp_mapmaker.map_accumulator_IQU(m, tod, 2.5, pix, psi,
                                     response_I=response_I, response_P=response_P)
    np.testing.assert_allclose(m, expected, **TOL)


@pytest.mark.parametrize("response_I, response_P", RESPONSES)
def test_map_weight_accumulator_IQU(response_I: float, response_P: float) -> None:
    _, pix, psi = _scan(np.float64, seed=7)
    m = np.random.default_rng(8).normal(size=(6, NPIX))
    # The 6 unique elements (II, IQ, IU, QQ, QU, UU) of the outer product of the pointing row.
    r = [np.full(NTOD, response_I), response_P*np.cos(2*psi), response_P*np.sin(2*psi)]
    outer = [r[0]*r[0], r[0]*r[1], r[0]*r[2], r[1]*r[1], r[1]*r[2], r[2]*r[2]]
    expected = m.copy()
    for k in range(6):
        np.add.at(expected[k], pix, 2.5*outer[k])
    cpp_mapmaker.map_weight_accumulator_IQU(m, 2.5, pix, psi,
                                            response_I=response_I, response_P=response_P)
    np.testing.assert_allclose(m, expected, **TOL)


def test_map2tod() -> None:
    _, pix, _ = _scan(np.float64, seed=9)
    m = np.random.default_rng(10).normal(size=NPIX)
    tod = np.full(NTOD, np.nan)  # NaN shows any sample the kernel fails to write
    cpp_mapmaker.map2tod(m, tod, pix)
    np.testing.assert_allclose(tod, m[pix], **TOL)


def test_map2tod_IQU() -> None:
    _, pix, psi = _scan(np.float64, seed=11)
    m = np.random.default_rng(12).normal(size=(3, NPIX))
    tod = np.full(NTOD, np.nan)
    cpp_mapmaker.map2tod_IQU(m, tod, pix, psi)
    expected = m[0, pix] + m[1, pix]*np.cos(2*psi) + m[2, pix]*np.sin(2*psi)
    np.testing.assert_allclose(tod, expected, **TOL)


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
    m1, m3, m6, tod_out = np.zeros(NPIX), np.zeros((3, NPIX)), np.zeros((6, NPIX)), np.empty(NTOD)
    # Every loop of every kernel, including the intensity-only (response_P = 0) loops.
    calls = [
        lambda: cpp_mapmaker.map_accumulator(m1, tod, 1.0, pix),
        lambda: cpp_mapmaker.map_weight_accumulator(m1, 1.0, pix),
        lambda: cpp_mapmaker.map_accumulator_IQU(m3, tod, 1.0, pix, psi),
        lambda: cpp_mapmaker.map_accumulator_IQU(m3, tod, 1.0, pix, psi, response_P=0.0),
        lambda: cpp_mapmaker.map_weight_accumulator_IQU(m6, 1.0, pix, psi),
        lambda: cpp_mapmaker.map_weight_accumulator_IQU(m6, 1.0, pix, psi, response_P=0.0),
        lambda: cpp_mapmaker.map2tod(m1, tod_out, pix),
        lambda: cpp_mapmaker.map2tod_IQU(m3, tod_out, pix, psi),
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


def test_map_accumulator_IQU_rejects_wrong_dtypes_and_lengths() -> None:
    tod, pix, psi = _scan(np.float32, seed=19)
    m = np.zeros((3, NPIX))
    with pytest.raises(RuntimeError):  # maps are float64 only
        cpp_mapmaker.map_accumulator_IQU(m.astype(np.float32), tod, 1.0, pix, psi)
    with pytest.raises(RuntimeError):  # int32 pixel indices
        cpp_mapmaker.map_accumulator_IQU(m, tod, 1.0, pix.astype(np.int32), psi)
    with pytest.raises(RuntimeError):  # pix shorter than the TOD would read past its end
        cpp_mapmaker.map_accumulator_IQU(m, tod, 1.0, pix[:10].copy(), psi)
