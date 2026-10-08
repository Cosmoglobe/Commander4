"""Preconditioners M ~ A^-1 for the CG mapmaker's normal equations.

The mapmaking operator is ``A = P^T T^T N^-1 T P``. Dropping the transfer function ``T`` leaves
``P^T N^-1 P``, which is exactly the per-pixel normal matrix the binned mapmaker accumulates
(`BinnedMapmaker.map_cov`) -- block diagonal in pixels, and therefore cheap to
invert. Every preconditioner here is built from that matrix; they differ only in how much of it they
keep.

All of them zero the pixels the binned mapmaker cannot solve either, which is what makes the CG
usable on a partial sky: an unobserved or degenerately-sampled pixel is projected out of the solve
rather than left as an unconstrained direction for the CG to amplify. The per-pixel 3x3 inverse
comes from the binned mapmaker's own C++ solver (`map_invert_IQU`), so both mapmakers decide the
same way which pixels they can solve.

(The component-separation solver's preconditioners are a separate family, in
`compsep/preconditioners.py`.)
"""
import logging

import numpy as np
from numpy.typing import NDArray

from commander4.backend import mapmaker as cpp_mapmaker
from commander4.math_utils.arithmetic import inplace_arr_prod

logger = logging.getLogger(__name__)


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
                per-pixel inverse-noise matrix, i.e. `BinnedMapmaker.map_cov`.
        """
        # The unique elements of each pixel's inverse 3x3; all zero where it cannot be inverted.
        self.inv_N_IQU = np.empty((6, normal_matrix.shape[1]))
        cpp_mapmaker.map_invert_IQU(self.inv_N_IQU, normal_matrix)
        self.npix = self.inv_N_IQU.shape[1]

    def __call__(self, map: NDArray) -> NDArray:
        if map.shape != (3, self.npix):
            raise ValueError(f"Map must have shape (3, {self.npix}), got {map.shape}.")
        map_in = np.ascontiguousarray(map, dtype=np.float64)
        map_out = np.empty_like(map_in)
        cpp_mapmaker.apply_invN_to_map_IQU(map_in, map_out, self.inv_N_IQU)
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
        # The pixels the binned mapmaker can solve are those whose 3x3 inverse is not all zero.
        inverse = np.empty((6, normal_matrix.shape[1]))
        cpp_mapmaker.map_invert_IQU(inverse, normal_matrix)
        solvable = np.any(inverse != 0.0, axis=0)
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
