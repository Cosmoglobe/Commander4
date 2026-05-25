"""Hard-coded benchmark for the final ang2pix comparison.

- ``healpy.ang2pix`` as the public upstream reference and ground truth
- ``healpy.pixelfunc.pixlib._ang2pix_ring`` as the direct compiled healpy ring path
- the exact local C++ kernel
- the faster local degree-9 polynomial C++ kernel

The exact C++ speedup does not come from changing the pixelization logic. It comes
from comparing the public ``healpy.ang2pix`` entry point with a dedicated standalone
compiled kernel that runs the same RING formulas under a more optimized implementation.

The benchmark also times healpy's internal compiled ring entry point so the output
separates public-wrapper overhead from the remaining gap to the standalone kernel.

The benchmark deliberately uses the default longitude-normalization path in the local
wrapper so the exact call is closer to ``healpy.ang2pix`` rather than relying on a
special fast-path benchmark contract.

The polynomial kernel is faster again because it replaces ``cos(theta)`` with a fixed
polynomial approximation and keeps the same integer ``loc2pix`` logic afterwards.

Result of running this benchmark on pelican2.uio.no (5x and 10x speedup for exact and non-exact C++ implementations, compared to Healpy):
nside=2048, sizes=[1000, 10000, 100000, 1000000, 5000000], repeats=5, warmups=2, seed=0, cpp_threads=1, poly_boundary_tol=0.0
        size   healpy [ms]    hp-core [ms]    cpp [ms]    cpp-poly [ms]   core/hp  cpp/core  cpoly/core
        1000         0.025           0.017       0.019            0.011      0.70      1.11        0.66
  errors vs healpy: hp-core=0 (0.00000000e+00), cpp=0 (0.00000000e+00), cpp-poly=0 (0.00000000e+00)
       10000         0.224           0.211       0.073            0.027      0.94      0.34        0.13
  errors vs healpy: hp-core=0 (0.00000000e+00), cpp=0 (0.00000000e+00), cpp-poly=0 (0.00000000e+00)
      100000         2.582           2.542       0.578            0.201      0.98      0.23        0.08
  errors vs healpy: hp-core=0 (0.00000000e+00), cpp=0 (0.00000000e+00), cpp-poly=0 (0.00000000e+00)
     1000000        26.132          25.426       5.804            1.952      0.97      0.23        0.08
  errors vs healpy: hp-core=0 (0.00000000e+00), cpp=0 (0.00000000e+00), cpp-poly=15 (1.50000000e-05)
     5000000       135.182         132.016      30.219           14.833      0.98      0.23        0.11
  errors vs healpy: hp-core=0 (0.00000000e+00), cpp=0 (0.00000000e+00), cpp-poly=82 (1.64000000e-05)
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import healpy as hp
import numpy as np
from healpy import pixelfunc
import ducc0

from fast_ang2pix import ang2pix_ring_ctypes, ang2pix_ring_poly_ctypes, load_ang2pix_ctypes_lib

NSIDE = 2048
SIZES = [1_000, 10_000, 100_000, 1_000_000, 5_000_000]
REPEATS = 5
WARMUPS = 2
SEED = 0
CPP_THREADS = 1
POLY_BOUNDARY_TOL = 0.0


def build_coordinates(size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Build benchmark coordinates."""
    theta = rng.uniform(0.0, np.pi, size=size)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=size)
    return theta, phi


def time_operation(operation: Callable[[], object]) -> np.ndarray:
    """Run the configured warmups and collect elapsed times in seconds."""
    for _ in range(WARMUPS):
        operation()

    samples = np.empty(REPEATS, dtype=np.float64)
    for index in range(REPEATS):
        t0 = perf_counter()
        operation()
        samples[index] = perf_counter() - t0
    return samples


def median_seconds(samples: np.ndarray) -> float:
    """Return the median elapsed time in seconds."""
    return float(np.median(samples))


def count_errors(reference: np.ndarray, candidate: np.ndarray) -> tuple[int, float]:
    """Count mismatching pixels and return both count and fraction."""
    mismatch_count = int(np.count_nonzero(reference != candidate))
    return mismatch_count, mismatch_count / reference.size


def main() -> int:
    """Benchmark healpy against the exact and polynomial local C++ kernels."""
    rng = np.random.default_rng(SEED)
    load_ang2pix_ctypes_lib()
    healpy_ring_core = pixelfunc.pixlib._ang2pix_ring

    print("Benchmarking public healpy, healpy core, and the final exact/polynomial C++ kernels.")
    print(
        f"nside={NSIDE}, sizes={SIZES}, repeats={REPEATS}, warmups={WARMUPS}, seed={SEED}, "
        f"cpp_threads={CPP_THREADS}, poly_boundary_tol={POLY_BOUNDARY_TOL}"
    )
    print(
        f"{'size':>12} {'healpy [ms]':>13} {'hp-core [ms]':>15} {'ducc [ms]':>15} {'cpp [ms]':>11} {'cpp-poly [ms]':>16} "
        f"{'core/hp':>9} {'cpp/core':>9} {'cpoly/core':>11}"
    )

    for size in SIZES:
        theta, phi = build_coordinates(size, rng)
        healpy_pixels = hp.ang2pix(NSIDE, theta, phi)
        healpy_core_pixels = healpy_ring_core(NSIDE, theta, phi)
        cpp_pixels = ang2pix_ring_ctypes(theta, phi, NSIDE, nthreads=CPP_THREADS)
        cpp_poly_pixels = ang2pix_ring_poly_ctypes(
            theta,
            phi,
            NSIDE,
            boundary_tol=POLY_BOUNDARY_TOL,
            nthreads=CPP_THREADS,
        )
        base = ducc0.healpix.Healpix_Base(NSIDE,"RING")
        ptg = np.stack([theta,phi],axis=1)
        res = np.empty(len(theta),dtype=np.int64)
        ducc_pixels = base.ang2pix(ptg,nthreads=CPP_THREADS,out=res).copy()

        healpy_time = median_seconds(time_operation(lambda: hp.ang2pix(NSIDE, theta, phi)))
        healpy_core_time = median_seconds(time_operation(lambda: healpy_ring_core(NSIDE, theta, phi)))
        ducc_time = median_seconds(time_operation(lambda: base.ang2pix(ptg,nthreads=CPP_THREADS,out=res)))
        cpp_time = median_seconds(
            time_operation(lambda: ang2pix_ring_ctypes(theta, phi, NSIDE, nthreads=CPP_THREADS))
        )
        cpp_poly_time = median_seconds(
            time_operation(
                lambda: ang2pix_ring_poly_ctypes(
                    theta,
                    phi,
                    NSIDE,
                    boundary_tol=POLY_BOUNDARY_TOL,
                    nthreads=CPP_THREADS,
                )
            )
        )

        print(
            f"{size:12d} {1.0e3 * healpy_time:13.3f} {1.0e3 * healpy_core_time:15.3f} {1.0e3 * ducc_time:15.3f} {1.0e3 * cpp_time:11.3f} "
            f"{1.0e3 * cpp_poly_time:16.3f} {healpy_core_time / healpy_time:9.2f} {cpp_time / healpy_core_time:9.2f} "
            f"{cpp_poly_time / healpy_core_time:11.2f}"
        )

        healpy_core_errors, healpy_core_error_fraction = count_errors(healpy_pixels, healpy_core_pixels)
        ducc_errors, ducc_error_fraction = count_errors(healpy_pixels, ducc_pixels)
        cpp_errors, cpp_error_fraction = count_errors(healpy_pixels, cpp_pixels)
        cpp_poly_errors, cpp_poly_error_fraction = count_errors(healpy_pixels, cpp_poly_pixels)
        print(
            "  errors vs healpy: "
            f"hp-core={healpy_core_errors} ({healpy_core_error_fraction:.8e}), "
            f"ducc={ducc_errors} ({ducc_error_fraction:.8e}), "
            f"cpp={cpp_errors} ({cpp_error_fraction:.8e}), "
            f"cpp-poly={cpp_poly_errors} ({cpp_poly_error_fraction:.8e})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
