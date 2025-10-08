import numpy as np
import pytest
import sys
import os

# Ugly trick to be able to import.
module_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(module_root_path)

from src.python.utils.math_operations import calculate_sigma0, inplace_scale_add, inplace_add_scaled_vec,\
    inplace_arr_prod, inplace_scale, dot


### calculate_sigma0(tod, mask) ###
def test_sigma0():
    arr = np.random.normal(2.0, 3.0, 10)
    mask = np.random.randint(0, 2, 10).astype(bool)
    masked_arr = arr[mask]
    numpy_sol = np.std(masked_arr[1:] - masked_arr[:-1])/np.sqrt(2)
    assert calculate_sigma0(arr, mask) == pytest.approx(numpy_sol)

def test_sigma0_masked():
    arr = np.random.normal(2.0, 3.0, 10)
    mask = np.random.randint(0, 2, 10).astype(bool)
    mask = np.zeros(10).astype(bool)
    assert calculate_sigma0(arr, mask) == np.inf
    mask[3] = True
    assert calculate_sigma0(arr, mask) == np.inf


### dot(arr1, arr2) ###
def test_dot():
    arr1 = np.random.normal(2.0, 3.0, 100)
    arr2 = np.random.normal(-3.5, 1.0, 100)
    assert dot(arr1, arr2) == pytest.approx(np.dot(arr1, arr2))

def test_dot_f32_f64():
    arr1 = np.random.normal(2.0, 3.0, 100).astype(np.float32, copy=False)
    arr2 = np.random.normal(-3.5, 1.0, 100)
    assert dot(arr1, arr2) == pytest.approx(np.dot(arr1, arr2))

def test_dot_ndim():
    arr1 = np.random.normal(2.0, 3.0, (10,20))
    arr2 = np.random.normal(-3.5, 1.0, (10,20))
    assert dot(arr1, arr2) == pytest.approx(np.dot(arr1.flatten(), arr2.flatten()))


### inplace_scale_add(arr_main, arr_add, float_mult) ###
def test_inplace_add_scaled_vec():
    arr1 = np.random.normal(2.0, 3.0, 100)
    arr2 = np.random.normal(-3.5, 1.0, 100)
    value = -12.5
    answer = arr1 + value*arr2
    inplace_add_scaled_vec(arr1, arr2, value)
    assert arr1 == pytest.approx(answer)

def test_inplace_add_scaled_vec_integer():
    arr1 = np.random.normal(2.0, 3.0, 100)
    arr2 = np.random.normal(-3.5, 1.0, 100)
    value = 5
    answer = arr1 + value*arr2
    inplace_add_scaled_vec(arr1, arr2, value)
    assert arr1 == pytest.approx(answer)

def test_inplace_add_scaled_vec_f32_f64():
    arr1 = np.random.normal(2.0, 3.0, 100).astype(np.float32, copy=False)
    arr2 = np.random.normal(-3.5, 1.0, 100)
    value = 5
    answer = arr1 + value*arr2
    inplace_add_scaled_vec(arr1, arr2, value)
    assert arr1 == pytest.approx(answer)


### inplace_arr_prod(arr_main, arr_prod) ###
def test_inplace_arr_prod():
    arr1 = np.random.normal(2.0, 3.0, 100)
    arr2 = np.random.normal(-3.5, 1.0, 100)
    answer = arr1*arr2
    inplace_arr_prod(arr1, arr2)
    assert arr1 == pytest.approx(answer)
