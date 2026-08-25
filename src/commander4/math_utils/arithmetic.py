"""Simple arithmetic on arrays, done in place and without copies.

These exist because 1. NumPy is not threaded, which matters on the CompSep side where a rank has
many cores, and 2. several NumPy operations allocate a copy, while these write into the array they
are given. `dot`, `norm` and `MPI_dot` are the inner products the CG solvers reduce with.
"""
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
from scipy.linalg import blas as blas_wrapper
from mpi4py import MPI


# Collection of Numba functions and BLAS wrappers for simply array manipulation.
# These exist because 1. Numpy is not threaded, which is a problem on the comp-sep module which has
# a lot of cores available, and 2. Certain Numpy operations create copies, these functions do not.

AXPY_ROUTINES = {
    np.dtype('float32'): blas_wrapper.saxpy,
    np.dtype('float64'): blas_wrapper.daxpy,
    np.dtype('complex64'): blas_wrapper.caxpy,
    np.dtype('complex128'): blas_wrapper.zaxpy,
}
def inplace_axpy(inplace_array, add_array, multiply_value):
    """`inplace_array += add_array*multiply_value`. Performs in-place scaled vector addition using
    BLAS AXPY routines. Support f32, 64, c64, and c128 data types, but all arguments must match.
    NB: Seems to fail for arrays larger than 2**32, which is a bit of an issue...
    """
    if inplace_array.size == 0: return
    if inplace_array.shape != add_array.shape:
        raise ValueError("AXPY input arrays must have matching shapes.")
    if inplace_array.dtype != add_array.dtype:
        raise TypeError("AXPY input arrays must have matching dtypes.")
    # Select the Correct BLAS Routine
    axpy_func = AXPY_ROUTINES[inplace_array.dtype]
    axpy_func(x=add_array, y=inplace_array, n=inplace_array.size, a=multiply_value)


@njit(fastmath=True, parallel=True)
def inplace_scale_add(arr_main, arr_add, float_mult):
    if arr_main.shape != arr_add.shape:
        raise ValueError("Input arrays must have matching shapes.")
    flat1 = arr_main.ravel()
    flat2 = arr_add.ravel()
    for i in prange(arr_main.size):
        flat1[i] = flat1[i]*float_mult + flat2[i]

@njit(fastmath=True)
def inplace_add_scaled_vec_serial(arr_main, arr_add, float_mult):
    if arr_main.shape != arr_add.shape:
        raise ValueError("Input arrays must have matching shapes.")
    flat1 = arr_main.ravel()
    flat2 = arr_add.ravel()
    for i in range(arr_main.size):
        flat1[i] += flat2[i]*float_mult

@njit(fastmath=True, parallel=True)
def inplace_add_scaled_vec(arr_main, arr_add, float_mult):
    if arr_main.shape != arr_add.shape:
        raise ValueError("Input arrays must have matching shapes.")
    flat1 = arr_main.ravel()
    flat2 = arr_add.ravel()
    for i in prange(arr_main.size):
        flat1[i] += flat2[i]*float_mult

@njit(fastmath=True, parallel=True)
def inplace_arr_add(arr_main, arr_add):
    if arr_main.shape != arr_add.shape:
        raise ValueError("Input arrays must have matching shapes.")
    flat1 = arr_main.ravel()
    flat2 = arr_add.ravel()
    for i in prange(arr_main.size):
        flat1[i] += flat2[i]

@njit(fastmath=True, parallel=True)
def inplace_arr_sub(arr_main, arr_add):
    if arr_main.shape != arr_add.shape:
        raise ValueError("Input arrays must have matching shapes.")
    flat1 = arr_main.ravel()
    flat2 = arr_add.ravel()
    for i in prange(arr_main.size):
        flat1[i] -= flat2[i]

@njit(fastmath=True, parallel=True)
def inplace_arr_prod(arr_main, arr_prod):
    len = arr_main.size
    if arr_main.shape != arr_prod.shape:
        raise ValueError("Input arrays must have matching shapes.")
    flat1 = arr_main.ravel()
    flat2 = arr_prod.ravel()
    for i in prange(len):
        flat1[i] *= flat2[i]

@njit(fastmath=True, parallel=True)
def inplace_arr_truediv(arr_main, arr_prod):
    len = arr_main.size
    if arr_main.shape != arr_prod.shape:
        raise ValueError("Input arrays must have matching shapes.")
    flat1 = arr_main.ravel()
    flat2 = arr_prod.ravel()
    for i in prange(len):
        flat1[i] /= flat2[i]

@njit(fastmath=True, parallel=True)
def inplace_scale(arr_main, scalar_prod):
    len = arr_main.size
    flat1 = arr_main.ravel()
    for i in prange(len):
        flat1[i] *= scalar_prod

@njit(fastmath=True, parallel=True)
def dot(arr1, arr2):
    len = arr1.size
    res = 0.0
    flat1 = arr1.ravel()
    flat2 = arr2.ravel()
    for i in prange(len):
        res += flat1[i]*flat2[i]
    return res

def norm(arr):
    """Return the Euclidean norm of an array flattened over all axes."""
    return np.sqrt(dot(arr, arr))

def MPI_dot(arr1, arr2, comm:MPI.Comm, double_prec:bool = False):
    """
    Computes the dot product locally and accumulates it on all the ranks involved in `comm`.
    """
    local_res = np.array(dot(arr1, arr2), dtype=np.float64 if double_prec else np.float32) #single-value array so mpi Allreduce does not complain.
    res = comm.allreduce(local_res, op=MPI.SUM) #comm.Allreduce(MPI.IN_PLACE, local_res, op=MPI.SUM)
    return res #np.float64(local_res) if double_prec else np.float32(local_res) #unpack it
