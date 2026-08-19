### This file contains dense matrix math for performing testing of CompSep systems for very low nsides (~16) ###
### It is designed to work specifically with the CompSep system, including the specfic MPI setup ###
### Ideally this class would support calculating e.g. the conditioning number without constructing the full matrix, ###
### but that would need to be fine-tailored with the MPI implementation of the Ax application, which I haven't bothered. ###

import logging
import numpy as np
from tqdm import trange
from pixell import utils
from mpi4py import MPI
from mpi4py.MPI import Comm
from copy import deepcopy
import scipy
from collections.abc import Callable


class DenseMatrix:
    def __init__(self, CompSep, A_operator, matrix_name):
                #  CompSep_comm: Comm, A_operator: Callable, alm_len_percomp: int,
                #  float_dtype, matrix_name: str = ""):
        """ Class for performing dense matrix math, specifically designed to work with the CompSep class (regarding MPI setup).
        Args:
            CompSep_comm (MPI.Comm): MPI communicator for the component separation processes, which should contain one rank per band.
            A_operator (Callable): Function to apply the matrix A to a vector, in an MPI-distributed fashion (master and helper tasks).
            size (int): Size of the matrix.
            alm_len_percomp (int): The size of the alms of each component (each held by one MPI rank). 
            matrix_name (str): Name used for printing.
        """
        self.logger = logging.getLogger(__name__)
        self.matrix_name = matrix_name
        self.CompSep_comm = CompSep.CompSep_comm
        self.A_operator = A_operator
        self.alm_len_percomp = CompSep.alm_len_percomp
        self.is_master = self.CompSep_comm.Get_rank() == 0
        self.my_comp = self.CompSep_comm.Get_rank()
        self.ncomps = self.alm_len_percomp.shape[0]
        self.is_holding_comp = self.my_comp < self.ncomps
        self.full_size = np.sum(self.alm_len_percomp)
        self.float_dtype = CompSep.float_dtype
        self.construct_dense_matrix()


    def construct_dense_matrix(self):
        """ Function for constructing the dense matrix A, and storing it as "self.A_matrix".
            The matrix is stored on all ranks.
        """
        if self.is_master:
            self.logger.info(f"Starting construction of dense matrix {self.matrix_name}")
            self.A_matrix = np.zeros((self.full_size, self.full_size))
            a_in_zeros = [np.zeros((1,nalm), dtype=self.float_dtype) for nalm in self.alm_len_percomp]
            i = 0
            for icomp in trange(self.ncomps):
                nalm = self.alm_len_percomp[icomp]
                for ialm in range(nalm):
                    a_in = deepcopy(a_in_zeros)
                    a_in[icomp][0,ialm] = 1.0
                    a_out = self.A_operator(a_in)
                    a_out = np.concatenate(a_out, axis=-1)
                    self.A_matrix[i,:] = a_out
                    i += 1
        else:
            for icomp in range(self.ncomps):
                nalm = self.alm_len_percomp[icomp]
                for ialm in range(nalm):
                    self.A_operator([])


    def solve_by_inversion(self, RHS):
        """ Solves the equation Ax=b for x given b (RHS) using direct inversion. The dense LHS matrix is already constructed.
            Assumes that both x and b are in alm space.

            Args:
                RHS: A Numpy array representing b, in alm space.
            Returns:
                x_bestfit: The resulting best-fit solution to x for the component owned by this rank.
        """
        if self.CompSep_comm.Get_rank() == 0:
            self.logger.info("Solving LHS matrix by direct inversion.")

        if self.is_master:
            RHS = np.concatenate(RHS, axis=-1)
            x_bestfit = scipy.linalg.solve(self.A_matrix, RHS[0])
            return x_bestfit
        else:
            return []


    def print_sing_vals(self):
        """ Calculates and prints the singular values of the dense matrix A, as well as the condition number.
            Useful for debugging CG preconditioners, as their primary purpose is to improve the condition number.
        """
        if self.is_master:
            sing_vals = scipy.linalg.svd(self.A_matrix, compute_uv=False)
            self.logger.info(f"Condition number of matrix {self.matrix_name}: {sing_vals[0]/sing_vals[-1]:.3e}")
            self.logger.info(f"Singular values of matrix {self.matrix_name}: {sing_vals[0]:.1e} .. {sing_vals[sing_vals.size//4]:.1e} .. {sing_vals[sing_vals.size//2]:.1e} .. {sing_vals[3*sing_vals.size//4]:.1e} .. {sing_vals[-1]:.1e}")


    def test_matrix_hermitian(self):
        """ Cheks that the dense matrix is Hermitian by checking if A^H == A, and the deviation from this.
            Useful for debugging CG, as it requires a symmetric matrix.
        """
        if self.is_master:
            diff = np.mean(np.abs(self.A_matrix - np.conjugate(self.A_matrix.T)))/np.std(self.A_matrix)
            is_hermitian = np.allclose(self.A_matrix, np.conjugate(self.A_matrix.T))
            if is_hermitian:
                self.logger.info(f"Matrix {self.matrix_name} is Hermitian with mean(A^H - A)/std(A) = {diff:.2e}")
            else:
                self.logger.warning(f"Matrix {self.matrix_name} is NOT HERMITIAN with mean(A^H - A)/std(A) = {diff:.2e}")


    def print_matrix_diag(self):
        """ Prints 8 uniformily space diagonal elements of the dense matrix.
            Can be used to see whether the preconditioner was able to accurately capture diagonal of matrix.
        """
        if self.is_master:
            diag = np.diag(self.A_matrix)
            size = diag.shape[0]
            self.logger.info(f"Matrix {self.matrix_name} diag: {diag[0]:.1e} .. {diag[size//8]:.1e} .. {diag[(2*size)//8]:.1e} .. {diag[(3*size)//8]:.1e} .. {diag[(4*size)//8]:.1e} .. {diag[(5*size)//8]:.1e} .. {diag[(6*size)//8]:.1e} .. {diag[(7*size)//8]:.1e} .. {diag[-1]:.1e}")


    def test_matrix_eigenvalues(self):
        """ Calculate and print the eigenvalues of the A-matrix.
            Prints a warning if they do not align with a Hermitian positive definite matrix. 
        """
        if self.is_master:
            eigvals = scipy.linalg.eigvals(self.A_matrix)
            min_eigval = np.min(np.abs(eigvals))
            max_eigval = np.max(np.abs(eigvals))
            imag_max_eigval = np.max(eigvals.imag)
            if imag_max_eigval > 1e-10 or min_eigval < -1e-10:
                self.logger.warning(f"Matrix {self.matrix_name} IS NOT symmetric positive-definite!")
            self.logger.info(f"Eigvals: min={min_eigval:.1e}, max={max_eigval:.1e} highest imag={imag_max_eigval:.1e}")