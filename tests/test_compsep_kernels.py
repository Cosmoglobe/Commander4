"""Direct tests of the compiled per-pixel component-separation solver in ``commander4.backend``."""
import numpy as np
import pytest
from numpy.typing import NDArray

from commander4.backend import compsep as cpp_compsep

NBAND, NCOMP, NPIX = 5, 3, 40


def _inputs(seed: int) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Random sky and rms (nband, npix), mixing matrix M (nband, ncomp), eta (npix, nband)."""
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(NBAND, NPIX)), rng.uniform(0.5, 2.0, (NBAND, NPIX)),
            rng.uniform(0.5, 2.0, (NBAND, NCOMP)), rng.normal(size=(NPIX, NBAND)))


@pytest.mark.parametrize("nthreads", [1, 4])
def test_solve_perpix_matches_numpy(nthreads: int) -> None:
    """Each pixel solves (M^T N^-1 M) x = M^T (N^-1 d + N^-1/2 eta), with N^-1 = 1/rms^2."""
    map_sky, map_rms, M, rand = _inputs(seed=1)
    w = 1/map_rms**2                                                   # (nband, npix)
    A = np.einsum("bi,bp,bj->pij", M, w, M)                            # (npix, ncomp, ncomp)
    rhs = np.einsum("bi,bp->pi", M, w*map_sky + np.sqrt(w)*rand.T)     # (npix, ncomp)
    expected = np.linalg.solve(A, rhs[..., None])[..., 0].T            # (ncomp, npix)
    comp_maps = cpp_compsep.solve_perpix(map_sky, map_rms, M, rand, nthreads)
    np.testing.assert_allclose(comp_maps, expected, rtol=1e-10, atol=1e-12)


def test_solve_perpix_zeroes_unsolvable_pixels() -> None:
    """A pixel with infinite RMS in every band carries no information, so its amplitudes are 0."""
    map_sky, map_rms, M, rand = _inputs(seed=2)
    map_rms[:, 7] = np.inf
    comp_maps = cpp_compsep.solve_perpix(map_sky, map_rms, M, rand)
    np.testing.assert_array_equal(comp_maps[:, 7], 0.0)
    assert np.all(comp_maps[:, :7] != 0.0)


def test_solve_perpix_rejects_wrong_shapes() -> None:
    map_sky, map_rms, M, rand = _inputs(seed=3)
    with pytest.raises(RuntimeError):  # rand must be (npix, nband), not (nband, npix)
        cpp_compsep.solve_perpix(map_sky, map_rms, M, np.ascontiguousarray(rand.T))
    with pytest.raises(RuntimeError):  # M must have one row per band
        cpp_compsep.solve_perpix(map_sky, map_rms, M[:4].copy(), rand)
