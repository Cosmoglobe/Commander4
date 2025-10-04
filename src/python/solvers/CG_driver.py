import numpy as np

def default_M(x):     return np.copy(x)
def default_dot(a,b): return a.dot(np.conj(b))

class CG:
	"""Preconditioner borrowed from pixell.utils, and modified to accomodate both the distributed
    computations of Commander4 component separation, and overriding of certain Numpy operations.
	"""
	def __init__(self, A, b, x0=None, M=default_M, dot=default_dot, destroy_b=False):
		"""Initialize a solver for the system Ax=b, with a starting guess of x0 (0
		if not provided). Vectors b and x0 must provide addition and multiplication,
		as well as the .copy() method, such as provided by numpy arrays. The
		preconditioner is given by M. A and M must be functors acting on vectors
		and returning vectors. The dot product may be manually specified using the
		dot argument. This is useful for MPI-parallelization, for example."""
		# Init parameters
		self.A   = A
		self.b   = b # not necessary to store this. Delete?
		self.M   = M
		self.dot = dot
		if x0 is None:
			self.x = np.zeros_like(b)
			self.r = b.copy() if not destroy_b else b
		else:
			self.x  = x0.copy()
			self.r  = b-self.A(self.x)
		# Internal work variables
		n = b.size
		z = self.M(self.r)
		self.rz  = self.dot(self.r, z)
		self.rz0 = float(self.rz)
		self.p   = z
		self.i   = 0
		self.err = np.inf
	def step(self):
		"""Take a single step in the iteration. Results in .x, .i
		and .err being updated. To solve the system, call step() in
		a loop until you are satisfied with the accuracy. The result
		can then be read off from .x."""
		# Full vectors: p, Ap, x, r, z. Ap and z not in memory at the
		# same time. Total memory cost: 4 vectors + 1 temporary = 5 vectors
		Ap = self.A(self.p)
		alpha = self.rz/self.dot(self.p, Ap)
		self.x += alpha*self.p
		self.r -= alpha*Ap
		del Ap
		z       = self.M(self.r)
		next_rz = self.dot(self.r, z)
		self.err = next_rz/self.rz0
		beta = next_rz/self.rz
		self.rz = next_rz
		self.p  = z + beta*self.p
		self.i += 1
	def save(self, fname):
		"""Save the volatile internal parameters to hdf file fname. Useful
		for later restoring cg iteration"""
		import h5py
		with h5py.File(fname, "w") as hfile:
			for key in ["i","rz","rz0","x","r","p","err"]:
				hfile[key] = getattr(self, key)
	def load(self, fname):
		"""Load the volatile internal parameters from the hdf file fname.
		Useful for restoring a saved cg state, after first initializing the
		object normally."""
		import h5py
		with h5py.File(fname, "r") as hfile:
			for key in ["i","rz","rz0","x","r","p","err"]:
				setattr(self, key, hfile[key][()])
