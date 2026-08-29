"""The conjugate-gradient iteration itself, adapted from pixell for distributed operators.

`DistributedCG` works on `CompList` vectors (component amplitudes spread over MPI ranks) and
`DistributedCGArray` on plain arrays. Both differ from a textbook CG only in taking the dot product
and the vector arithmetic as callables, so the collective reductions happen inside the caller's
operator rather than here.
"""
import numpy as np
import logging
from copy import deepcopy, copy
from commander4.math_utils.arithmetic import inplace_add_scaled_vec, inplace_scale_add, dot
from commander4.sky.comp_list import inplace_complist_add_scaled_array,\
    inplace_complist_scale_and_add, complist_dot

logger = logging.getLogger(__name__)


def identity_preconditioner(x):     return np.copy(x)


class DistributedCG:
    """CG iteration on `CompList` vectors, used by the component-separation amplitude solver.

    Borrowed from pixell.utils and modified for Commander4's distributed operators and for
    CompList's overriding of certain NumPy operations. ``err`` is the squared relative
    preconditioned residual, ``(r^T M r) / (r_0^T M r_0)``, matching Commander3.
    """
    def __init__(self, A, b, is_master, x0=None, M=identity_preconditioner, dot=complist_dot,
                 destroy_b=False):
        """Initialize a solver for the system Ax=b, with a starting guess of x0 (0
        if not provided). Vectors b and x0 must provide addition and multiplication,
        as well as the .copy() method, such as provided by numpy arrays. The
        preconditioner is given by M. A and M must be functors acting on vectors
        and returning vectors. The dot product may be manually specified using the
        dot argument. This is useful for MPI-parallelization, for example."""
        self.is_master = is_master
        self.A   = A
        self.M   = M
        self.dot = dot
        self.b   = b

        # CG meta-parameters
        self.err = np.inf
        self.i   = 0
        if x0 is None:
            self.x = np.zeros_like(b)
            self.r = deepcopy(b) if not destroy_b else b
        else:
            self.x  = deepcopy(x0)
            self.r  = b - self.A(self.x)
        if is_master:  # Only the master needs these.
            # Internal work variables
            z = self.M(self.r)
            self.rz  = self.dot(self.r, z)  # Avoid calling custom dot func on non-master ranks.
            self.rz0 = float(self.rz)
            self.p   = z
        else:
            self.p = np.zeros_like(b)
            
    def step(self):
        """Take a single step in the iteration. Results in .x, .i
        and .err being updated. To solve the system, call step() in
        a loop until you are satisfied with the accuracy. The result
        can then be read off from .x."""
        # Full vectors: p, Ap, x, r, z. Ap and z not in memory at the
        # same time. Total memory cost: 4 vectors + 1 temporary = 5 vectors
        Ap = self.A(self.p)
        if self.is_master:  # The rest of the CG iteration is done by the master alone.

            alpha = self.rz/self.dot(self.p, Ap)

            # Line below equivalent to: self.x = [_x + alpha*_p for _x, _p in zip(self.x, self.p)]
            self.x.inplace_add_scaled(self.p, alpha)

            # Line below equivalent to: self.r = [_r - alpha*_Ap for _r, _Ap in zip(self.r, Ap)]
            self.r.inplace_add_scaled(Ap, -alpha)

            del Ap
            z       = self.M(self.r)
            next_rz = self.dot(self.r, z)
            self.err = next_rz/self.rz0
            beta = next_rz/self.rz
            self.rz = next_rz

            # Line below equivalent to: self.p = [_p*beta + _z for _p, _z in zip(self.p, z)]
            self.p.inplace_scale_and_add(z, beta)

        self.i += 1


class DistributedCGArray:
    """CG iteration on plain NumPy arrays, used by the CG mapmaker.

    Same algorithm and squared-residual convention as `DistributedCG`, but the vector arithmetic
    goes through NumPy directly rather than through CompList's componentwise operations.
    """
    def __init__(self, A, b, is_master, x0=None, M=identity_preconditioner, dot=dot,
                 destroy_b=False):
        """Initialize a solver for the system Ax=b, with a starting guess of x0 (0
        if not provided). Vectors b and x0 must provide addition and multiplication,
        as well as the .copy() method, such as provided by numpy arrays. The
        preconditioner is given by M. A and M must be functors acting on vectors
        and returning vectors. The dot product may be manually specified using the
        dot argument. This is useful for MPI-parallelization, for example."""
        self.is_master = is_master
        self.A   = A
        self.M   = M
        self.dot = dot

        # CG meta-parameters
        self.err = np.inf
        self.i   = 0
        if x0 is None:
            self.x = np.zeros_like(b)
            self.r = copy(b) if not destroy_b else b 
        else:
            self.x  = copy(x0)
            self.r  = b - self.A(self.x)
        if is_master:  # Only the master needs these.
            # Internal work variables
            z = self.M(self.r)
            self.rz  = self.dot(self.r, z)  # Avoid calling custom dot func on non-master ranks.
            self.rz0 = float(self.rz)
            self.p   = z
        else:
            self.p = np.zeros_like(b)
            
    def step(self):
        """Take a single step in the iteration. Results in .x, .i
        and .err being updated. To solve the system, call step() in
        a loop until you are satisfied with the accuracy. The result
        can then be read off from .x."""
        # Full vectors: p, Ap, x, r, z. Ap and z not in memory at the
        # same time. Total memory cost: 4 vectors + 1 temporary = 5 vectors
        Ap = self.A(self.p)
        if self.is_master:  # The rest of the CG iteration is done by the master alone.

            alpha = self.rz/self.dot(self.p, Ap)

            # Line below equivalent to: self.x = [_x + alpha*_p for _x, _p in zip(self.x, self.p)]
            inplace_add_scaled_vec(self.x, self.p, alpha)

            # Line below equivalent to: self.r = [_r - alpha*_Ap for _r, _Ap in zip(self.r, Ap)]
            inplace_add_scaled_vec(self.r, Ap, -alpha)

            del Ap
            z       = self.M(self.r)
            next_rz = self.dot(self.r, z)
            # print("CG step 4: ", next_rz)
            self.err = next_rz/self.rz0
            beta = next_rz/self.rz
            # print("CG step 5: ", beta)
            self.rz = next_rz

            # Line below equivalent to: self.p = [_p*beta + _z for _p, _z in zip(self.p, z)]
            inplace_scale_add(self.p, z, beta)

        self.i += 1
