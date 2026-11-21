"""The conjugate-gradient iteration, adapted from pixell.utils for operators spread over MPI ranks.

The component-separation amplitude solver runs it on `CompList` vectors, and the CG mapmaker on
NumPy arrays.
"""
import numpy as np

from commander4.math_utils.arithmetic import inplace_add_scaled_vec, inplace_scale_add, dot
from commander4.sky.comp_list import CompList


def identity_preconditioner(x):     return x.copy()


class DistributedCG:
    """Preconditioned CG iteration for A x = b, on `CompList` vectors or NumPy arrays.

    Every rank calls `A`, which does the MPI communication, but only the master gets its result
    and does the rest of the iteration. Call `step()` until ``err`` is small enough, and read the
    solution from ``x``. ``err`` is ``(r^T M r) / (b^T M b)``, the squared preconditioned residual
    relative to that of the right-hand side, as in C3. It means the same with and without a
    starting guess.
    """
    def __init__(self, A, b, is_master: bool, x0=None, M=identity_preconditioner,
                 destroy_b: bool = False):
        """Set up the iteration from the starting guess `x0` (zero if None).

        `x0` is not copied: it becomes the solution vector and is changed in place. With
        `destroy_b`, `b` becomes the residual vector and is changed in place too.
        """
        self.is_master = is_master
        self.A   = A
        self.M   = M
        # The vector arithmetic. Both kinds run the same numba kernels (a CompList applies them
        # component by component), with the number of threads set per rank in `mpi/setup.py`.
        # Both kinds also have a `copy()` method.
        if isinstance(b, CompList):
            self.dot        = CompList.dot
            self.add_scaled = CompList.add_scaled     # x += a*y
            self.scale_add  = CompList.scale_and_add  # x = a*x + y
            zeros_like      = CompList.zeros_like
        elif isinstance(b, np.ndarray):
            self.dot        = dot
            self.add_scaled = inplace_add_scaled_vec
            self.scale_add  = inplace_scale_add
            zeros_like      = np.zeros_like
        else:
            raise TypeError(f"The CG works on a CompList or a NumPy array, not on {type(b)}.")

        # CG meta-parameters
        self.err = np.inf
        self.i   = 0
        if x0 is None:
            self.x = zeros_like(b)
            self.r = b.copy() if not destroy_b else b
        else:
            if is_master:  # Before `b` is changed in place below.
                self.rz0 = float(self.dot(b, self.M(b)))
            self.x = x0
            Ax = self.A(self.x)  # Collective: every rank calls A, only the master gets the result.
            self.r = b.copy() if not destroy_b else b
            if is_master:
                self.add_scaled(self.r, Ax, -1.0)  # r = b - A x0
        if is_master:  # Only the master needs these.
            z = self.M(self.r)
            self.rz  = self.dot(self.r, z)
            if x0 is None:
                self.rz0 = float(self.rz)  # Here r = b.
            self.p   = z
        else:
            # The other ranks pass `p` to A only as a work buffer, and never use `r`. So the two
            # share one vector.
            self.p = self.r

    def step(self):
        """Take one CG step, which updates ``x``, ``i`` and ``err``.

        ``err`` becomes exactly 0 when the residual is zero to machine precision. The solution
        cannot improve after that, and further steps change nothing.
        """
        # Full vectors: p, Ap, x, r, z. Ap and z not in memory at the
        # same time. Total memory cost: 4 vectors + 1 temporary = 5 vectors
        self.i += 1
        Ap = self.A(self.p)
        if not self.is_master:  # The rest of the CG iteration is done by the master alone.
            return
        pAp = self.dot(self.p, Ap)
        if pAp == 0.0:
            # The residual is zero to machine precision: p = 0, or A p is too small to represent.
            # Stop here, because alpha below would divide by zero.
            self.err = 0.0
            return
        alpha = self.rz/pAp
        self.add_scaled(self.x, self.p, alpha)  # x += alpha*p
        self.add_scaled(self.r, Ap, -alpha)     # r -= alpha*Ap
        del Ap
        z       = self.M(self.r)
        next_rz = self.dot(self.r, z)
        self.err = next_rz/self.rz0
        beta = next_rz/self.rz
        self.rz = next_rz
        self.scale_add(self.p, z, beta)         # p = beta*p + z
