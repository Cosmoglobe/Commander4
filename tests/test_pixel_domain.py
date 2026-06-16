"""Tests for PixelDomain and the sparse map storage it enables.

The core correctness checks (to_local, reduce/scatter round-trips, and sparse-vs-full mapmaker
equivalence) run under a plain single-rank ``pytest`` because they use COMM_SELF / a size-1
COMM_WORLD. The genuinely distributed checks are written collectively against COMM_WORLD, so they
exercise real multi-rank Gatherv/Scatterv when launched as::

    mpirun -n 4 python -m pytest tests/test_pixel_domain.py
"""

import numpy as np
from mpi4py import MPI
from numpy.testing import assert_allclose
import pytest

from commander4.utils.pixel_domain import PixelDomain
from commander4.utils.mapmaker import MapmakerIQU, WeightsMapmakerIQU, Mapmaker, WeightsMapmaker


def _sparse_domain(comm, nside, local_pix):
    return PixelDomain(comm, nside, "sparse", local_pix=np.asarray(local_pix, dtype=np.int64))


# --- to_local --------------------------------------------------------------------------------

def test_to_local_is_identity_in_full_mode():
    dom = PixelDomain(MPI.COMM_SELF, nside=2, mode="full")
    pix = np.array([5, 0, 47, 12], dtype=np.int64)
    assert dom.to_local(pix) is pix or np.array_equal(dom.to_local(pix), pix)


def test_to_local_maps_global_to_compact_indices():
    local_pix = [3, 7, 8, 20, 41]
    dom = _sparse_domain(MPI.COMM_SELF, nside=2, local_pix=local_pix)
    pix = np.array([20, 3, 41, 8, 7], dtype=np.int64)
    expected = np.array([3, 0, 4, 2, 1], dtype=np.int64)  # index of each pix within local_pix
    assert_allclose(dom.to_local(pix), expected)
    # The mapped indices must round-trip back to the original global pixels.
    assert_allclose(np.asarray(local_pix)[dom.to_local(pix)], pix)


# --- from_view -------------------------------------------------------------------------------

class _FakeView:
    def __init__(self, pix):
        self.pix = pix

class _FakeScanView:
    """Minimal stand-in exposing the only thing PixelDomain.from_view touches: view.pix."""
    def __init__(self, pix_arrays):
        self._pix_arrays = pix_arrays
    def iter_focused(self, *, accepted_only=False):
        for pix in self._pix_arrays:
            yield _FakeView(pix)

def test_from_view_collects_union_of_observed_pixels():
    scan_view = _FakeScanView([np.array([5, 5, 1]), np.array([1, 40, 7])])
    dom = PixelDomain.from_view(scan_view, MPI.COMM_SELF, "sparse", nside=2)
    assert_allclose(dom.local_pix, np.array([1, 5, 7, 40]))
    assert dom.n_local == 4


# --- reduce / scatter round-trips (collective; meaningful under mpirun) ----------------------

def test_reduce_to_full_matches_reference():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    nside, npix = 4, 12 * 16
    rng = np.random.default_rng(100 + rank)
    # Each rank observes an overlapping window of pixels, so several ranks share pixels.
    local_pix = np.unique(rng.integers(0, npix, size=30))
    dom = _sparse_domain(comm, nside, local_pix)

    local = rng.normal(size=(3, dom.n_local))
    full = dom.reduce_to_full(local)

    # Reference: scatter-add each rank's contribution into a full-sky map and sum across ranks.
    ref_local = np.zeros((3, npix))
    np.add.at(ref_local[0], local_pix, local[0])
    np.add.at(ref_local[1], local_pix, local[1])
    np.add.at(ref_local[2], local_pix, local[2])
    ref = np.zeros((3, npix))
    comm.Reduce(ref_local, ref if rank == 0 else None, op=MPI.SUM, root=0)
    if rank == 0:
        assert_allclose(full, ref, rtol=1e-12, atol=1e-12)
    else:
        assert full is None


def test_scatter_from_full_round_trips():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nside, npix = 4, 12 * 16
    rng = np.random.default_rng(200 + rank)
    local_pix = np.unique(rng.integers(0, npix, size=25))
    dom = _sparse_domain(comm, nside, local_pix)

    full = rng.normal(size=(2, npix)) if rank == 0 else None
    local = dom.scatter_from_full(full, ncomp=2)
    # Each rank must receive exactly the master's values at its own pixels.
    full_b = comm.bcast(full, root=0)
    assert_allclose(local, full_b[:, local_pix], rtol=1e-12, atol=1e-12)


# --- sparse-vs-full mapmaker equivalence (single-rank, exercises the local-buffer kernels) ---

def _random_scan(rng, npix_observed, ntod):
    pix = rng.integers(0, npix_observed, size=ntod).astype(np.int64)
    psi = rng.uniform(0, np.pi, size=ntod)
    tod = rng.normal(size=ntod)
    return pix, psi, tod

@pytest.mark.parametrize("response", [None, np.array([1.0, 0.7])])
def test_weights_iqu_sparse_matches_full(response):
    rng = np.random.default_rng(7)
    nside = 4
    pix, psi, _ = _random_scan(rng, npix_observed=40, ntod=500)
    local_pix = np.unique(pix)

    full = WeightsMapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
    sparse = WeightsMapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64,
                               pixel_domain=_sparse_domain(MPI.COMM_SELF, nside, local_pix))
    full.accumulate_to_map(2.5, pix, psi, response=response)
    sparse.accumulate_to_map(2.5, pix, psi, response=response)
    full.gather_map()
    sparse.gather_map()
    assert_allclose(sparse._gathered_map, full._gathered_map, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("response", [None, np.array([1.0, 0.7])])
def test_signal_iqu_sparse_matches_full(response):
    rng = np.random.default_rng(11)
    nside = 4
    pix, psi, tod = _random_scan(rng, npix_observed=40, ntod=500)
    local_pix = np.unique(pix)

    full = MapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64)
    sparse = MapmakerIQU(MPI.COMM_SELF, nside, dtype=np.float64,
                        pixel_domain=_sparse_domain(MPI.COMM_SELF, nside, local_pix))
    full.accumulate_to_map(tod, 2.5, pix, psi, response=response)
    sparse.accumulate_to_map(tod, 2.5, pix, psi, response=response)
    full.gather_map()
    sparse.gather_map()
    assert_allclose(sparse._gathered_map, full._gathered_map, rtol=1e-12, atol=1e-12)


def test_scalar_sparse_matches_full():
    rng = np.random.default_rng(13)
    nside = 4
    pix, _, tod = _random_scan(rng, npix_observed=40, ntod=500)
    local_pix = np.unique(pix)

    full_sig = Mapmaker(MPI.COMM_SELF, nside, dtype=np.float64)
    sparse_sig = Mapmaker(MPI.COMM_SELF, nside, dtype=np.float64,
                         pixel_domain=_sparse_domain(MPI.COMM_SELF, nside, local_pix))
    full_w = WeightsMapmaker(MPI.COMM_SELF, nside, dtype=np.float64)
    sparse_w = WeightsMapmaker(MPI.COMM_SELF, nside, dtype=np.float64,
                              pixel_domain=_sparse_domain(MPI.COMM_SELF, nside, local_pix))
    full_sig.accumulate_to_map(tod, 2.5, pix)
    sparse_sig.accumulate_to_map(tod, 2.5, pix)
    full_w.accumulate_to_map(2.5, pix)
    sparse_w.accumulate_to_map(2.5, pix)
    full_sig.gather_map(); sparse_sig.gather_map()
    full_w.gather_map(); sparse_w.gather_map()
    assert_allclose(sparse_sig._gathered_map, full_sig._gathered_map, rtol=1e-12, atol=1e-12)
    assert_allclose(sparse_w._gathered_map, full_w._gathered_map, rtol=1e-12, atol=1e-12)
