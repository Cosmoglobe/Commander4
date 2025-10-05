import numpy as np
import logging

def default_M(x):     return np.copy(x)
def default_dot(a,b): return a.dot(np.conj(b))

class distributed_CG:
    """Preconditioner borrowed from pixell.utils, and modified to accomodate both the distributed
    computations of Commander4 component separation, and overriding of certain Numpy operations.
    """
    def __init__(self, A, b, is_master, x0=None, M=default_M, dot=default_dot, destroy_b=False):
        """Initialize a solver for the system Ax=b, with a starting guess of x0 (0
        if not provided). Vectors b and x0 must provide addition and multiplication,
        as well as the .copy() method, such as provided by numpy arrays. The
        preconditioner is given by M. A and M must be functors acting on vectors
        and returning vectors. The dot product may be manually specified using the
        dot argument. This is useful for MPI-parallelization, for example."""
        self.is_master = is_master
        self.logger = logging.getLogger(__name__)
        self.A   = A
        self.M   = M
        self.dot = dot
        self.b   = b

        # CG meta-parameters
        self.err = np.inf
        self.i   = 0
        if x0 is None:
            self.x = b.copy()
            self.r = b.copy() if not destroy_b else b
        else:
            self.x  = x0.copy()
            self.r  = [_b - _Ax for _b,_Ax in zip(b,self.A(self.x))]
        if is_master:  # Only the master needs these.
            # Internal work variables
            z = self.M(self.r)
            self.rz  = self.dot(self.r, z)  # Avoid calling custom dot func on non-master ranks.
            self.rz0 = float(self.rz)
            self.p   = z
        else:
            self.p = []
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
            self.x = [_x + alpha*_p for _x, _p in zip(self.x, self.p)]
            self.r = [_r - alpha*_Ap for _r, _Ap in zip(self.r, Ap)]
            del Ap
            z       = self.M(self.r)
            next_rz = self.dot(self.r, z)
            self.err = next_rz/self.rz0
            beta = next_rz/self.rz
            self.rz = next_rz
            self.p = [_p*beta + _z for _p, _z in zip(self.p, z)]
        self.i += 1