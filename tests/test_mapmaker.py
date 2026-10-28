"""Tests of `BinnedMapmaker`: one pass per detector-scan into the weights and all binned maps, then
a per-pixel solve on the master. The C++ methods `accumulate` and `finalize` are compared with
their plain-NumPy versions `accumulate_Python` and `finalize_Python`.

The compiled kernels themselves are tested in test_mapmaker_kernels.py.
"""
import numpy as np
import pytest
from mpi4py import MPI

from commander4.data_models.pixel_domain import PixelDomain
from commander4.tod.mapmaking.binned import BinnedMapmaker

NSIDE, NPIX = 1, 12


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


def _bin(pols: str, names: list[str], scans: list[tuple], response_I_P: tuple[float, float],
         python: bool) -> BinnedMapmaker:
    """Bin the scans with the C++ methods, or with their NumPy versions if `python` is set."""
    binned = BinnedMapmaker(PixelDomain(MPI.COMM_SELF, NSIDE, "full"), pols, names, count_hits=True)
    accumulate = binned.accumulate_Python if python else binned.accumulate
    for pix, psi, weight, tods in scans:
        accumulate(weight, pix, psi, tods, response_I_P)
        binned.count_hits(pix)
    if python:
        binned.finalize_Python()
    else:
        binned.finalize()
    return binned


@pytest.mark.parametrize("pols", ["IQU", "I"])
@pytest.mark.parametrize("response_I_P", [(1.0, 1.0), (0.9, 0.7), (1.0, 0.0), (0.0, 1.0),
                                          (0.0, 0.0)])
def test_cpp_methods_match_the_python_reference(pols: str, response_I_P: tuple) -> None:
    scans = _scans(np.random.default_rng(1))
    cpp = _bin(pols, ["signal", "res"], scans, response_I_P, python=False)
    ref = _bin(pols, ["signal", "res"], scans, response_I_P, python=True)

    ncomp = 1 if pols == "I" else 3
    assert cpp.map_cov.shape == ref.map_cov.shape == (1 if pols == "I" else 6, NPIX)
    assert cpp.map_inv_var.shape == cpp.maps["signal"].shape == (ncomp, NPIX)
    assert cpp.maps["signal"].dtype == cpp.map_inv_var.dtype == np.float32
    np.testing.assert_allclose(cpp.map_cov, ref.map_cov, rtol=1e-12)
    np.testing.assert_allclose(cpp.map_inv_var, ref.map_inv_var, rtol=1e-6)
    for name in ("signal", "res"):
        np.testing.assert_allclose(cpp.maps[name], ref.maps[name], rtol=1e-6, atol=1e-9)
    np.testing.assert_array_equal(cpp.map_nhit, ref.map_nhit)


@pytest.mark.parametrize("python", [False, True])
def test_intensity_only_detector_never_reads_the_polarization_angle(python: bool) -> None:
    """A detector without polarization response is binned without psi, so NaN angles cannot spoil
    the maps. Its samples constrain I only, so in an IQU band no pixel can be solved."""
    pix = np.array([0, 1, 1, 4])
    scans = [(pix, np.full(pix.size, np.nan), 2.5, {"signal": np.array([1.0, 2.0, 3.0, 4.0])})]
    binned = _bin("IQU", ["signal"], scans, (1.0, 0.0), python)

    expected_II = np.zeros(NPIX)
    np.add.at(expected_II, pix, 2.5)
    np.testing.assert_array_equal(binned.map_cov[0], expected_II)
    np.testing.assert_array_equal(binned.map_cov[1:], 0.0)
    np.testing.assert_array_equal(binned.maps["signal"], 0.0)
    np.testing.assert_array_equal(binned.map_inv_var, 0.0)


@pytest.mark.parametrize("python", [False, True])
def test_weights_alone_without_any_map(python: bool) -> None:
    """The CG mapmaker bins the weights alone when no aux map is wanted."""
    pix = np.array([0, 3, 3, 7])
    binned = _bin("I", [], [(pix, np.zeros(pix.size), 2.0, {})], (0.5, 0.0), python)

    expected = np.zeros(NPIX)
    np.add.at(expected, pix, 2.0*0.5**2)
    np.testing.assert_array_equal(binned.map_cov[0], expected)
    assert binned.maps == {}
