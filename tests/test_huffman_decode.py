import numpy as np
import pytest
from numpy.testing import assert_array_equal

from commander4.backend import utils as cpp_utils
from commander4.compression import huffman


def test_huffman_decode_roundtrip_int8() -> None:
    values = np.array([-8, -3, 7, -3, 2, 7, -8, 2], dtype=np.int8)
    tree, symb, sym_codes, sym_lengths = huffman.build_huffman_tree([values])
    encoded = np.frombuffer(huffman.huffman_compress_array(values, sym_codes, sym_lengths), dtype=np.uint8)
    out = np.empty(values.size, dtype=values.dtype)

    decoded = cpp_utils.huffman_decode(encoded, tree, symb, out)

    assert decoded.dtype == values.dtype
    assert_array_equal(decoded, values)


def test_huffman_decode_roundtrip_uint8() -> None:
    values = np.array([0, 255, 3, 255, 1, 0, 7, 1], dtype=np.uint8)
    tree, symb, sym_codes, sym_lengths = huffman.build_huffman_tree([values])
    encoded = np.frombuffer(huffman.huffman_compress_array(values, sym_codes, sym_lengths), dtype=np.uint8)
    out = np.empty(values.size, dtype=values.dtype)

    decoded = cpp_utils.huffman_decode(encoded, tree, symb, out)

    assert decoded.dtype == values.dtype
    assert_array_equal(decoded, values)


def test_huffman_decode_requires_out_dtype_to_match_symb() -> None:
    values = np.array([0, 255, 3, 255, 1, 0, 7, 1], dtype=np.uint8)
    tree, symb, sym_codes, sym_lengths = huffman.build_huffman_tree([values])
    encoded = np.frombuffer(huffman.huffman_compress_array(values, sym_codes, sym_lengths), dtype=np.uint8)
    out = np.empty(values.size, dtype=np.int16)

    with pytest.raises(RuntimeError, match="'out' must have the same dtype as 'symb'"):
        cpp_utils.huffman_decode(encoded, tree, symb, out)


def test_psi_digitization_uses_one_based_circular_bins() -> None:
    npsi = 8
    width = 2*np.pi/npsi
    angles = np.array([0.0, 0.9*width, width, 2*np.pi - 0.1*width, 2*np.pi, -0.1*width])

    differences = huffman.preproc_digitize_and_diff(angles, npsi)
    bins = np.cumsum(differences)

    assert_array_equal(bins, [1, 1, 2, 8, 1, 8])
