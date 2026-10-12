import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from commander4.backend import compression as cpp_compression
from commander4.compression import huffman
from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.pointing import PixelPointing


def test_huffman_decode_roundtrip_int8() -> None:
    values = np.array([-8, -3, 7, -3, 2, 7, -8, 2], dtype=np.int8)
    tree, symb, sym_codes, sym_lengths = huffman.build_huffman_tree([values])
    encoded = np.frombuffer(huffman.huffman_compress_array(values, sym_codes, sym_lengths), dtype=np.uint8)
    out = np.empty(values.size, dtype=values.dtype)

    decoded = cpp_compression.huffman_decode(encoded, tree, symb, out)

    assert decoded.dtype == values.dtype
    assert_array_equal(decoded, values)


def test_huffman_decode_roundtrip_uint8() -> None:
    values = np.array([0, 255, 3, 255, 1, 0, 7, 1], dtype=np.uint8)
    tree, symb, sym_codes, sym_lengths = huffman.build_huffman_tree([values])
    encoded = np.frombuffer(huffman.huffman_compress_array(values, sym_codes, sym_lengths), dtype=np.uint8)
    out = np.empty(values.size, dtype=values.dtype)

    decoded = cpp_compression.huffman_decode(encoded, tree, symb, out)

    assert decoded.dtype == values.dtype
    assert_array_equal(decoded, values)


def test_huffman_decode_requires_out_dtype_to_match_symb() -> None:
    values = np.array([0, 255, 3, 255, 1, 0, 7, 1], dtype=np.uint8)
    tree, symb, sym_codes, sym_lengths = huffman.build_huffman_tree([values])
    encoded = np.frombuffer(huffman.huffman_compress_array(values, sym_codes, sym_lengths), dtype=np.uint8)
    out = np.empty(values.size, dtype=np.int16)

    with pytest.raises(RuntimeError, match="'out' must have the same dtype as 'symb'"):
        cpp_compression.huffman_decode(encoded, tree, symb, out)


def test_psi_digitization_uses_one_based_circular_bins() -> None:
    npsi = 8
    width = 2*np.pi/npsi
    angles = np.array([0.0, 0.9*width, width, 2*np.pi - 0.1*width, 2*np.pi, -0.1*width])

    differences = huffman.preproc_digitize_and_diff(angles, npsi)
    bins = np.cumsum(differences)

    assert_array_equal(bins, [1, 1, 2, 8, 1, 8])


def test_flags_decode_from_wrapped_uint8_differences() -> None:
    # As in the SO SAT scan files: the flag differences are stored as uint8, so a step down is 255.
    # A flag that switches off again must decode back to 0, not climb to 256.
    flag = np.array([0, 0, 1, 1, 0, 0, 2, 2, 0, 0], dtype=np.uint8)
    differences = np.diff(flag, prepend=np.uint8(0))  # uint8 arithmetic: 0 - 1 wraps to 255.
    tree, symb, sym_codes, sym_lengths = huffman.build_huffman_tree([differences])
    assert symb.dtype == np.uint8
    ntod = flag.size
    pointing = PixelPointing(np.zeros(ntod, np.int64), np.zeros(ntod), np.array([0], np.int64),
                             None, None, 1, 1, ntod, ntod)
    det = DetectorTOD(
        name="d0", det_idx_fullband=0, tod=np.zeros(ntod, np.float32), pointing=pointing,
        sampling_rate_hz=1.0, orbital_velocity_m_per_s=None, huffman_tree=tree,
        huffman_symbols=symb, default_proc_mask=None, specific_proc_masks={},
        flag_encoded=huffman.huffman_compress_array(differences, sym_codes, sym_lengths),
        bad_data_bitmask=0xFFFFFFFF,
    )

    assert_array_equal(det.flag, flag)
    assert_array_equal(det.good_data_mask, flag == 0)


def test_pointing_decodes_from_wrapped_uint8_differences() -> None:
    npsi = 8
    pix = np.array([10, 12, 11, 190, 3, 3], dtype=np.uint8)
    psi_bins = np.array([1, 8, 2, 2, 7, 1], dtype=np.uint8)  # One-based bins, as the files store.
    pix_differences = np.diff(pix, prepend=np.uint8(0))
    psi_differences = np.diff(psi_bins, prepend=np.uint8(0))
    tree, symb, sym_codes, sym_lengths = huffman.build_huffman_tree([pix_differences,
                                                                     psi_differences])
    pointing = PixelPointing(huffman.huffman_compress_array(pix_differences, sym_codes, sym_lengths),
                             huffman.huffman_compress_array(psi_differences, sym_codes, sym_lengths),
                             tree, symb, npsi, nside=4, data_nside=4, ntod_original=pix.size,
                             ntod=pix.size)

    assert_array_equal(pointing.get_pix(), pix)
    assert_allclose(pointing.get_psi(), (psi_bins - 0.5)*2*np.pi/npsi, rtol=1e-6)
