"""Tests of `BinnedMapmaker`: one pass per detector-scan into the weights and all binned maps, then
a per-pixel solve on the master, compared with the same steps in plain NumPy.

The compiled kernels themselves are tested in test_mapmaker_kernels.py.
"""
import numpy as np
from mpi4py import MPI

from commander4.data_models.pixel_domain import PixelDomain
from commander4.tod.mapmaking.binned import BinnedMapmaker

NSIDE, NPIX = 1, 12
GOOD = np.arange(NPIX) < 10  # pixels the scans below observe at many polarization angles


def _scans(rng: np.random.Generator) -> list[tuple]:
    """Detector-scans over pixels 0-9 at random angles, plus pixel 10 seen at a single angle (so
    its 3x3 cannot be solved); pixel 11 is never observed. Each comes with a weight and two TODs."""
    scans = []
    for _ in range(4):
        pix = np.concatenate([rng.integers(0, 10, 300), [10, 10]])
        psi = np.concatenate([rng.uniform(0.0, np.pi, 300), [0.3, 0.3]])
        tods = {"signal": rng.normal(size=pix.size).astype(np.float32),
                "res": rng.normal(size=pix.size)}
        scans.append((pix, psi, rng.uniform(0.5, 2.0), tods))
    return scans


def test_iqu_maps_match_a_numpy_solve() -> None:
    rng = np.random.default_rng(1)
    response_I, response_P = 0.9, 0.7
    binned = BinnedMapmaker(PixelDomain(MPI.COMM_SELF, NSIDE, "full"), "IQU", ["signal", "res"],
                            count_hits=True)
    A = np.zeros((NPIX, 3, 3))
    b = {name: np.zeros((NPIX, 3)) for name in binned.names}
    hits = np.zeros(NPIX, dtype=np.int64)
    for pix, psi, weight, tods in _scans(rng):
        binned.accumulate(weight, pix, psi, tods, (response_I, response_P))
        binned.count_hits(pix)
        # Each sample's pointing row, and its weighted outer product added to its pixel's A.
        rows = np.stack([np.full(pix.size, response_I), response_P*np.cos(2*psi),
                         response_P*np.sin(2*psi)], axis=1)
        np.add.at(A, pix, weight*rows[:, :, None]*rows[:, None, :])
        for name in b:
            np.add.at(b[name], pix, weight*rows*tods[name][:, None])
        np.add.at(hits, pix, 1)
    binned.finalize()

    np.testing.assert_allclose(binned.map_cov, A[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]].T,
                               rtol=1e-12)
    expected_inv_var = np.zeros((3, NPIX))
    expected_inv_var[:, GOOD] = 1.0/np.diagonal(np.linalg.inv(A[GOOD]), axis1=1, axis2=2).T
    np.testing.assert_allclose(binned.map_inv_var, expected_inv_var, rtol=1e-5)
    for name in b:
        expected = np.zeros((3, NPIX))
        expected[:, GOOD] = np.linalg.solve(A[GOOD], b[name][GOOD][..., None])[..., 0].T
        np.testing.assert_allclose(binned.maps[name], expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(binned.map_nhit, hits)
    assert binned.maps["signal"].dtype == binned.map_inv_var.dtype == np.float32


def test_intensity_band_divides_by_the_weights() -> None:
    """An I-only band's maps are (1, npix), and the intensity response weights each sample."""
    rng = np.random.default_rng(2)
    binned = BinnedMapmaker(PixelDomain(MPI.COMM_SELF, NSIDE, "full"), "I", ["signal"])
    cov, rhs = np.zeros(NPIX), np.zeros(NPIX)
    for pix, psi, weight, tods in _scans(rng):
        binned.accumulate(weight, pix, psi, {"signal": tods["signal"]}, (0.4, 0.0))
        np.add.at(cov, pix, weight*0.4**2)
        np.add.at(rhs, pix, weight*0.4*tods["signal"])
    binned.finalize()

    assert binned.map_cov.shape == binned.map_inv_var.shape == binned.maps["signal"].shape
    assert binned.map_cov.shape == (1, NPIX)
    observed = cov > 0
    np.testing.assert_allclose(binned.map_cov[0], cov, rtol=1e-12)
    np.testing.assert_allclose(binned.map_inv_var[0], cov, rtol=1e-6)
    np.testing.assert_allclose(binned.maps["signal"][0, observed], rhs[observed]/cov[observed],
                               rtol=1e-5)
    np.testing.assert_array_equal(binned.maps["signal"][0, ~observed], 0.0)


def test_intensity_only_detector_never_reads_the_polarization_angle() -> None:
    """A detector without polarization response is binned without psi, so NaN angles cannot spoil
    the maps. Its samples constrain I only, so in an IQU band no pixel can be solved."""
    pix = np.array([0, 1, 1, 4])
    binned = BinnedMapmaker(PixelDomain(MPI.COMM_SELF, NSIDE, "full"), "IQU", ["signal"])
    binned.accumulate(2.5, pix, np.full(pix.size, np.nan),
                      {"signal": np.array([1.0, 2.0, 3.0, 4.0])}, (1.0, 0.0))
    binned.finalize()

    expected_II = np.zeros(NPIX)
    np.add.at(expected_II, pix, 2.5)
    np.testing.assert_array_equal(binned.map_cov[0], expected_II)
    np.testing.assert_array_equal(binned.map_cov[1:], 0.0)
    np.testing.assert_array_equal(binned.maps["signal"], 0.0)
    np.testing.assert_array_equal(binned.map_inv_var, 0.0)


def test_weights_alone_without_any_map() -> None:
    """The CG mapmaker bins the weights alone when no aux map is wanted."""
    pix = np.array([0, 3, 3, 7])
    binned = BinnedMapmaker(PixelDomain(MPI.COMM_SELF, NSIDE, "full"), "I", [])
    binned.accumulate(2.0, pix, np.zeros(pix.size), {}, (0.5, 0.0))
    binned.finalize()

    expected = np.zeros(NPIX)
    np.add.at(expected, pix, 2.0*0.5**2)
    np.testing.assert_array_equal(binned.map_cov[0], expected)
    assert binned.maps == {}
