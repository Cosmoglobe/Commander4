"""Preconditioners M ~ A^-1 for the CG mapmaker's normal equations.

The mapmaking operator is ``A = P^T T^T N^-1 T P``. Dropping the transfer function ``T`` leaves
``P^T N^-1 P``, which is exactly the per-pixel normal matrix the binned mapmaker accumulates
(`WeightsMapmaker` / `WeightsMapmakerIQU`) -- block diagonal in pixels, and therefore cheap to
invert. Every preconditioner here is built from that matrix; they differ only in how much of it they
keep.

All of them zero the pixels the binned mapmaker cannot solve either, which is what makes the CG
usable on a partial sky: an unobserved or degenerately-sampled pixel is projected out of the solve
rather than left as an unconstrained direction for the CG to amplify.

(The component-separation solver's preconditioners are a separate family, in
`compsep/preconditioners.py`.)
"""
import ctypes as ct
import logging

import numpy as np
from numpy.typing import NDArray

from commander4.backend.ctypes_lib import load_cmdr4_ctypes_lib
from commander4.math_utils.arithmetic import inplace_arr_prod

logger = logging.getLogger(__name__)

# Reciprocal-condition floor below which a per-pixel 3x3 counts as unsolvable. Matches the binned
# mapmaker's C solver (mapmaker.cpp::_invert_SPD_3x3), so both mapmakers drop the same pixels.
_RCOND_FLOOR = 1e-12


def invert_normal_matrix_IQU(normal_matrix: NDArray) -> tuple[NDArray, NDArray]:
    """Invert the per-pixel 3x3 normal matrix, given and returned as its 6 unique elements.

    Args:
        normal_matrix: (6, npix) unique elements (II, IQ, IU, QQ, QU, UU) of the accumulated
            per-pixel inverse-noise matrix, i.e. `WeightsMapmakerIQU.final_cov_map`.

    Returns:
        ``(inverse, solvable)``. ``inverse`` is (6, npix), the unique elements of A_pp^-1, left at
        zero wherever the 3x3 is singular or too ill-conditioned to invert. ``solvable`` is the
        (npix,) boolean mask of the pixels that were inverted.
    """
    if normal_matrix.ndim != 2 or normal_matrix.shape[0] != 6:
        raise ValueError(f"Normal matrix must have shape (6, npix), got {normal_matrix.shape}.")
    a00, a01, a02, a11, a12, a22 = np.asarray(normal_matrix, dtype=np.float64)
    # Cofactors of the symmetric 3x3, then Cramer's rule. A zero diagonal entry means the matrix is
    # singular (or not positive definite); det <= _RCOND_FLOOR*diag_prod means it is too
    # ill-conditioned to invert. Either way the whole block stays at zero.
    c00, c01, c02 = a11*a22 - a12*a12, a02*a12 - a01*a22, a01*a12 - a02*a11
    det = a00*c00 + a01*c01 + a02*c02
    diag_prod = a00*a11*a22
    solvable = (diag_prod > np.finfo(np.float64).tiny) & (det > _RCOND_FLOOR*diag_prod)
    inv_det = np.zeros_like(det)
    np.divide(1.0, det, out=inv_det, where=solvable)
    inverse = np.ascontiguousarray(
        np.stack([c00, c01, c02, a00*a22 - a02*a02, a01*a02 - a00*a12, a00*a11 - a01*a01])*inv_det)
    return inverse, solvable


class BlockInvNPreconditionerIQU:
    """ Block-Jacobi preconditioner for the polarized CG mapmaker: M = A_pp^-1, pixel by pixel.

        A_pp is the per-pixel 3x3 inverse-noise (I,Q,U) normal matrix the weights mapmaker
        accumulates. With an identity transfer function the mapmaking operator P^T N^-1 P is exactly
        block diagonal in pixels, so this M is the exact inverse and the CG converges in one
        iteration; with a non-trivial T it is the same operator with T dropped, still the dominant
        part. This is the default, and the right choice unless you are deliberately studying the
        solver: `InvNPreconditionerIQU` keeps only the diagonal and is strictly worse.
    """

    def __init__(self, normal_matrix:NDArray):
        """ Initialize by inverting the per-pixel normal matrix.

        Args:
            normal_matrix: (6, npix) unique elements (II, IQ, IU, QQ, QU, UU) of the accumulated
                per-pixel inverse-noise matrix.
        """
        self.inv_N_IQU, _ = invert_normal_matrix_IQU(normal_matrix)
        self.npix = self.inv_N_IQU.shape[1]

        self.maplib = load_cmdr4_ctypes_lib()
        ct_f64_dim2 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=2, flags="contiguous")
        self.maplib.apply_invN_to_map_IQU_f64.argtypes = [ct_f64_dim2,  # map_in
                                                          ct_f64_dim2,  # map_out
                                                          ct_f64_dim2,  # inv_N_map
                                                          ct.c_int64]   # num_pix

    def __call__(self, map: NDArray) -> NDArray:
        if map.shape != (3, self.npix):
            raise ValueError(f"Map must have shape (3, {self.npix}), got {map.shape}.")
        map_in = np.ascontiguousarray(map, dtype=np.float64)
        map_out = np.empty_like(map_in)
        self.maplib.apply_invN_to_map_IQU_f64(map_in, map_out, self.inv_N_IQU, self.npix)
        return map_out


class InvNPreconditionerIQU:
    """ Jacobi (diagonal) preconditioner for the polarized CG mapmaker: M = 1/diag(A).

        Keeps only the I, Q and U diagonals of the per-pixel normal matrix, ignoring the
        correlations between them. Kept for solver studies; `BlockInvNPreconditionerIQU` inverts the
        full 3x3 for the same input and converges far faster.

        1/diag(A) must not be taken at face value on a partial sky. A ragged-edge pixel seen at a
        single polarization angle has, say, A_QQ = sum w*cos^2(2 psi) at rounding level (~1e-35, not
        exactly 0) while A_II and A_UU are large: guarding only against an exactly-zero diagonal
        leaves M with an entry of ~1e35, which swamps every dot product in the CG and wrecks the
        whole map. The solvability test is therefore the full 3x3 one, the same criterion the binned
        mapmaker uses to decide a pixel has no solution; those pixels get M = 0, which projects them
        out of the solve.
    """

    def __init__(self, normal_matrix:NDArray):
        """ Initialize from the per-pixel normal matrix, keeping the reciprocal of its diagonal.

        Args:
            normal_matrix: (6, npix) unique elements (II, IQ, IU, QQ, QU, UU) of the accumulated
                per-pixel inverse-noise matrix.
        """
        _, solvable = invert_normal_matrix_IQU(normal_matrix)
        A_diag = np.asarray(normal_matrix, dtype=np.float64)[(0, 3, 5), :]
        self.inv_N_IQU = np.zeros_like(A_diag)
        np.divide(1.0, A_diag, out=self.inv_N_IQU, where=solvable[np.newaxis, :])
        self.npix = self.inv_N_IQU.shape[1]

    def __call__(self, map: NDArray) -> NDArray:
        if map.shape != (3, self.npix):
            raise ValueError(f"Map must have shape (3, {self.npix}), got {map.shape}.")
        map_out = np.copy(map)
        inplace_arr_prod(map_out, self.inv_N_IQU)
        return map_out


class InvNPreconditionerI:
    """ Jacobi (diagonal) preconditioner for the temperature-only CG mapmaker: M = 1/diag(A).

        The intensity counterpart of `BlockInvNPreconditionerIQU`; here A is diagonal per pixel, so
        1/diag(A) is the exact inverse and there is no block version to prefer. An unobserved pixel
        has weight exactly 0 -- no near-degenerate case as in polarization -- and gets M = 0.
    """

    def __init__(self, weight_map:NDArray):
        """ Initialize from the per-pixel inverse-variance weights, shape (npix,) or (1, npix). """
        weights = np.asarray(weight_map, dtype=np.float64)
        self.inv_N_map = weights.reshape((1, -1))
        self.npix = self.inv_N_map.shape[1]
        observed = self.inv_N_map > 0
        inv = np.zeros_like(self.inv_N_map)
        np.divide(1.0, self.inv_N_map, out=inv, where=observed)
        self.inv_N_map = inv

    def __call__(self, map: NDArray) -> NDArray:
        if map.shape[-1] != self.npix:
            raise ValueError(f"Map must have {self.npix} pixels, got shape {map.shape}.")
        map_out = np.copy(map)
        logger.debug(f"## Preconditioner called. map shape: {map.shape}, inv N shape: {self.inv_N_map.shape}")
        # this allows it to be applied to IQU maps as well
        map_out = map_out.reshape((1,-1)) if map_out.ndim == 1 else map_out
        if map_out.shape[0] == 1:
            inplace_arr_prod(map_out, self.inv_N_map)
        else:
            for i in range(map_out.shape[0]):
                inplace_arr_prod(map_out[i,:], self.inv_N_map)
        return map_out
