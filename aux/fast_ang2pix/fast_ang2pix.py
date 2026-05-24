"""ctypes wrappers for the surviving ang2pix experiments.

This module only exposes the two kernels that still matter in the benchmark:

- ``ang2pix_ring_ctypes``: exact RING ``theta, phi -> pixel`` mapping.
- ``ang2pix_ring_poly_ctypes``: faster degree-9 polynomial approximation.

The exact kernel is faster than the public ``healpy.ang2pix`` API in this benchmark
because it calls a dedicated standalone compiled implementation of the same RING
mapping logic. Measured speedups were dominated by that kernel-level implementation
difference rather than by special benchmark input assumptions.

The polynomial kernel keeps the same integer ``loc2pix`` logic but replaces
``cos(theta)`` with a fixed degree-9 approximation. An optional ``boundary_tol``
cleanup pass recomputes only outputs that are close to a decision boundary.
"""

from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path

import numpy as np

_CTYPES_LIB: ct.CDLL | None = None
_CTYPES_CONFIGURED = False

__all__ = [
	"ang2pix_ring_ctypes",
	"ang2pix_ring_poly_ctypes",
	"load_ang2pix_ctypes_lib",
]


def _configure_ang2pix_ctypes(lib: ct.CDLL) -> None:
	"""Register ctypes signatures for the exact and polynomial C++ entry points."""
	global _CTYPES_CONFIGURED
	if _CTYPES_CONFIGURED:
		return

	ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="contiguous")
	ct_f64_dim1 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="contiguous")
	ct_f32_dim1 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="contiguous")

	lib.ang2pix_ring_theta_f64.argtypes = [
		ct_f64_dim1,
		ct_f64_dim1,
		ct_i64_dim1,
		ct.c_int64,
		ct.c_int64,
		ct.c_int,
		ct.c_int,
		ct.c_int,
	]
	lib.ang2pix_ring_theta_f32.argtypes = [
		ct_f32_dim1,
		ct_f32_dim1,
		ct_i64_dim1,
		ct.c_int64,
		ct.c_int64,
		ct.c_int,
		ct.c_int,
		ct.c_int,
	]
	lib.ang2pix_ring_poly_f64.argtypes = [
		ct_f64_dim1,
		ct_f64_dim1,
		ct_i64_dim1,
		ct.c_int64,
		ct.c_int64,
		ct.c_double,
		ct.c_int,
		ct.c_int,
		ct.c_int,
	]
	lib.ang2pix_ring_poly_f32.argtypes = [
		ct_f32_dim1,
		ct_f32_dim1,
		ct_i64_dim1,
		ct.c_int64,
		ct.c_int64,
		ct.c_double,
		ct.c_int,
		ct.c_int,
		ct.c_int,
	]

	for name in (
		"ang2pix_ring_theta_f64",
		"ang2pix_ring_theta_f32",
		"ang2pix_ring_poly_f64",
		"ang2pix_ring_poly_f32",
	):
		getattr(lib, name).restype = None

	_CTYPES_CONFIGURED = True


def load_ang2pix_ctypes_lib() -> ct.CDLL:
	"""Load the standalone shared library from ``aux/``.

	The library is built with ``-march=native``, so it must be rebuilt on each machine
	where it is benchmarked.
	"""
	global _CTYPES_LIB
	if _CTYPES_LIB is not None:
		return _CTYPES_LIB

	base_dir = Path(__file__).resolve().parent
	candidates: list[Path] = []
	if override := os.environ.get("FAST_ANG2PIX_LIB"):
		candidates.append(Path(override).expanduser())
	candidates.extend(
		(
			base_dir / "fast_ang2pix_ctypes.so",
			base_dir / "libfast_ang2pix.so",
			base_dir / "fast_ang2pix.so",
		)
	)

	lib: ct.CDLL | None = None
	for candidate in candidates:
		if candidate.exists():
			lib = ct.CDLL(str(candidate.resolve()))
			break

	if lib is None:
		raise RuntimeError(
			"Could not load aux/fast_ang2pix_ctypes.so. Rebuild it on the current machine "
			"with `bash aux/build_fast_ang2pix.sh`."
		)

	_configure_ang2pix_ctypes(lib)
	_CTYPES_LIB = lib
	return lib


def _validate_nside(nside: int) -> None:
	"""Validate that ``nside`` is a positive power of two."""
	if nside <= 0 or (nside & (nside - 1)) != 0:
		raise ValueError("nside must be a positive power of two.")


def _prepare_pair(
	first: np.ndarray | float,
	second: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], np.dtype]:
	"""Convert two same-shaped inputs to contiguous 1D arrays with a shared dtype."""
	first_arr = np.asarray(first)
	second_arr = np.asarray(second)
	if first_arr.shape != second_arr.shape:
		raise ValueError("Input arrays must have identical shapes.")

	dtype = np.result_type(first_arr.dtype, second_arr.dtype, np.float32)
	first_flat = np.ascontiguousarray(first_arr.astype(dtype, copy=False).reshape(-1))
	second_flat = np.ascontiguousarray(second_arr.astype(dtype, copy=False).reshape(-1))
	return first_flat, second_flat, first_arr.shape, np.dtype(dtype)


def _reshape_output(out: np.ndarray, shape: tuple[int, ...]) -> np.ndarray | np.int64:
	"""Restore a flat output array to the caller's original shape."""
	out = out.reshape(shape)
	return out[()] if out.ndim == 0 else out


def ang2pix_ring_ctypes(
	theta: np.ndarray | float,
	phi: np.ndarray | float,
	nside: int,
	*,
	refine_poles: bool = True,
	phi_in_range: bool = False,
	nthreads: int = 1,
) -> np.ndarray | np.int64:
	"""Run the exact C++ RING ``theta, phi -> pixel`` kernel.

	Compared with the public ``healpy.ang2pix`` API, this benchmark path is faster
	because it calls the standalone compiled kernel directly. In the measured runs,
	the dominant gain came from that kernel-level implementation difference rather than
	from optional wrapper flags such as ``phi_in_range``.
	"""
	_validate_nside(nside)

	theta_flat, phi_flat, shape, dtype = _prepare_pair(theta, phi)
	out = np.empty(theta_flat.size, dtype=np.int64)
	lib = load_ang2pix_ctypes_lib()
	if dtype == np.dtype(np.float32):
		lib.ang2pix_ring_theta_f32(
			theta_flat,
			phi_flat,
			out,
			out.size,
			nside,
			int(refine_poles),
			int(phi_in_range),
			nthreads,
		)
	else:
		lib.ang2pix_ring_theta_f64(
			theta_flat,
			phi_flat,
			out,
			out.size,
			nside,
			int(refine_poles),
			int(phi_in_range),
			nthreads,
		)
	return _reshape_output(out, shape)


def ang2pix_ring_poly_ctypes(
	theta: np.ndarray | float,
	phi: np.ndarray | float,
	nside: int,
	*,
	boundary_tol: float = 0.0,
	refine_poles: bool = True,
	phi_in_range: bool = False,
	nthreads: int = 1,
) -> np.ndarray | np.int64:
	"""Run the degree-9 polynomial C++ RING ``theta, phi -> pixel`` kernel.

	This path keeps the exact integer ``loc2pix`` logic but replaces ``cos(theta)`` with
	a fixed degree-9 approximation, which is why it is substantially faster and no longer
	exact. If ``boundary_tol`` is positive, the kernel recomputes only outputs that are
	close to a pixel-index decision boundary.
	"""
	_validate_nside(nside)
	if boundary_tol < 0.0:
		raise ValueError("boundary_tol must be non-negative.")

	theta_flat, phi_flat, shape, dtype = _prepare_pair(theta, phi)
	out = np.empty(theta_flat.size, dtype=np.int64)
	lib = load_ang2pix_ctypes_lib()
	if dtype == np.dtype(np.float32):
		lib.ang2pix_ring_poly_f32(
			theta_flat,
			phi_flat,
			out,
			out.size,
			nside,
			boundary_tol,
			int(refine_poles),
			int(phi_in_range),
			nthreads,
		)
	else:
		lib.ang2pix_ring_poly_f64(
			theta_flat,
			phi_flat,
			out,
			out.size,
			nside,
			boundary_tol,
			int(refine_poles),
			int(phi_in_range),
			nthreads,
		)
	return _reshape_output(out, shape)
