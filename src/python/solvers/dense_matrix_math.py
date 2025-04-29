### This file contains dense matrix math for performing testing of CompSep systems for very low nsides (~16) ###
### It is designed to work specifically with the CompSep system, including the specfic MPI setup ###
### Ideally this class would support calculating e.g. the conditioning number without constructing the full matrix, ###
### but that would need to be fine-tailored with the MPI implementation of the Ax application, which I haven't bothered. ###

import numpy as np
from tqdm import trange
from pixell import utils
from mpi4py.MPI import Comm
import scipy
from collections.abc import Callable
from scipy.linalg import svd

class DenseMatrix:
    def __init__(self, CompSep_comm: Comm, A_operator: Callable, size: int):
        """ Class for performing dense matrix math, specifically designed to work with the CompSep class (regarding MPI setup).
        Args:
            CompSep_comm (MPI.Comm): MPI communicator for the component separation processes, which should contain one rank per band.
            A_operator (Callable): Function to apply the matrix A to a vector, in an MPI-distributed fashion (master and helper tasks).
            size (int): Size of the matrix.
        """
        self.CompSep_comm = CompSep_comm
        self.is_master = self.CompSep_comm.Get_rank() == 0
        self.A_operator = A_operator
        self.size = size
        self.construct_dense_matrix()


    def construct_dense_matrix(self):
        """ Function for constructing the dense matrix A, and storing it as "self.A_matrix".
            Also calculates the diagonal of the matrix and stores it as "self.A_diag".
            Both of these are stored in the master task (rank 0).
        """
        if self.is_master:
            self.A_matrix = np.zeros((self.size, self.size))
            for i in trange(self.size):
                u = utils.uvec(self.size, i)
                self.A_matrix[i] = self.A_operator(u)
            self.A_diag = np.diag(self.A_matrix)
        else:
            for i in trange(self.size):
                self.A_operator(None)  # The helper tasks need to call the A-operator, but do not store the results.


    def solve_by_inversion(self, RHS):
        """ Solves the equation Ax=b for x given b (RHS) using direct inversion. The dense LHS matrix is already constructed.
            Assumes that both x and b are in alm space.

            Args:
                RHS: A Numpy array representing b, in alm space.
            Returns:
                x_bestfit: The resulting best-fit solution to x (if rank==0, else None).
        """
        if self.is_master:
            x_bestfit = scipy.linalg.solve(self.A_matrix, RHS)
            return x_bestfit


    def get_sing_vals(self):
        """ Calculates the singular values of the matrix A using the SVD method.
            This is done by calculating the singular values of A and taking the ratio of the largest to smallest.
            Returns:
                cond_num: The singular values of the matrix A, in descending order (if rank==0, else None).
        """
        if self.is_master:
            s = svd(self.A_matrix, compute_uv=False)
            return s


    def test_matrix_symmetry(self):
        """ Tests the symmetry of the matrix A by checking if A == A^T.
            Useful for debugging CG, as it requires a symmetric matrix.
            Returns:
                is_symmetric: True if the matrix is symmetric, False otherwise (if rank==0, else None).
        """
        if self.is_master:
            is_symmetric = np.allclose(self.A_matrix, self.A_matrix.T)
            return is_symmetric