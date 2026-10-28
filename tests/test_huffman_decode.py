import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from commander4.backend import compression as cpp_compression
from commander4.compression.huffman_decoding import HuffmanCompressedArray
from commander4.compression.huffman_encoding import (build_huffman_tree, huffman_compress_array,
                                                     preproc_diff, preproc_digitize_and_diff)
from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.pointing import PixelPointing


def test_huffman_decode_roundtrip_int8() -> None:
    values = np.array([-8, -3, 7, -3, 2, 7, -8, 2], dtype=np.int8)
    tree, symb, sym_codes, sym_lengths = build_huffman_tree([values])
    encoded = np.frombuffer(huffman_compress_array(values, sym_codes, sym_lengths), dtype=np.uint8)
    out = np.empty(values.size, dtype=values.dtype)

    decoded = cpp_compression.huffman_decode(encoded, tree, symb, out)

    assert decoded.dtype == values.dtype
    assert_array_equal(decoded, values)


def test_huffman_decode_roundtrip_uint8() -> None:
    values = np.array([0, 255, 3, 255, 1, 0, 7, 1], dtype=np.uint8)
    tree, symb, sym_codes, sym_lengths = build_huffman_tree([values])
    encoded = np.frombuffer(huffman_compress_array(values, sym_codes, sym_lengths), dtype=np.uint8)
    out = np.empty(values.size, dtype=values.dtype)

    decoded = cpp_compression.huffman_decode(encoded, tree, symb, out)

    assert decoded.dtype == values.dtype
    assert_array_equal(decoded, values)


def test_huffman_decode_requires_out_dtype_to_match_symb() -> None:
    values = np.array([0, 255, 3, 255, 1, 0, 7, 1], dtype=np.uint8)
    tree, symb, sym_codes, sym_lengths = build_huffman_tree([values])
    encoded = np.frombuffer(huffman_compress_array(values, sym_codes, sym_lengths), dtype=np.uint8)
    out = np.empty(values.size, dtype=np.int16)

    with pytest.raises(RuntimeError, match="'out' must have the same dtype as 'symb'"):
        cpp_compression.huffman_decode(encoded, tree, symb, out)


def test_psi_digitization_uses_one_based_circular_bins() -> None:
    npsi = 8
    width = 2*np.pi/npsi
    angles = np.array([0.0, 0.9*width, width, 2*np.pi - 0.1*width, 2*np.pi, -0.1*width])

    differences = preproc_digitize_and_diff(angles, npsi)
    bins = np.cumsum(differences)

    assert_array_equal(bins, [1, 1, 2, 8, 1, 8])


def test_flags_decode_from_wrapped_uint8_differences() -> None:
    # As in the SO SAT scan files: the flag differences are stored as uint8, so a step down is 255.
    # A flag that switches off again must decode back to 0, not climb to 256.
    flag = np.array([0, 0, 1, 1, 0, 0, 2, 2, 0, 0], dtype=np.uint8)
    differences = np.diff(flag, prepend=np.uint8(0))  # uint8 arithmetic: 0 - 1 wraps to 255.
    tree, symb, sym_codes, sym_lengths = build_huffman_tree([differences])
    assert symb.dtype == np.uint8
    ntod = flag.size
    pointing = PixelPointing(np.zeros(ntod, np.int64), np.zeros(ntod), 1, 1, ntod, ntod)
    det = DetectorTOD(
        name="d0", det_idx_fullband=0, tod=np.zeros(ntod, np.float32), pointing=pointing,
        sampling_rate_hz=1.0, orbital_velocity_m_per_s=None, default_proc_mask=None,
        specific_proc_masks={},
        flag=HuffmanCompressedArray(
            huffman_compress_array(differences, sym_codes, sym_lengths), tree, symb, ntod),
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
    tree, symb, sym_codes, sym_lengths = build_huffman_tree([pix_differences, psi_differences])
    pix_stream = huffman_compress_array(pix_differences, sym_codes, sym_lengths)
    psi_stream = huffman_compress_array(psi_differences, sym_codes, sym_lengths)
    pix_compressed = HuffmanCompressedArray(pix_stream, tree, symb, pix.size)
    psi_compressed = HuffmanCompressedArray(psi_stream, tree, symb, pix.size, nbins=npsi,
                                            vmin=0.0, vmax=2*np.pi)
    pointing = PixelPointing(pix_compressed, psi_compressed, nside=4, data_nside=4,
                             ntod_original=pix.size, ntod=pix.size)

    assert_array_equal(pointing.get_pix(), pix)
    assert_allclose(pointing.get_psi(), (psi_bins - 0.5)*2*np.pi/npsi, rtol=1e-6)


def _encode(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Huffman-encode `values` with a tree built from them; return (bit stream, tree, symbols)."""
    tree, symb, sym_codes, sym_lengths = build_huffman_tree([values])
    stream = huffman_compress_array(values, sym_codes, sym_lengths)
    return np.frombuffer(stream, dtype=np.uint8).copy(), tree, symb


def test_digitized_array_decodes_to_bin_centers_of_its_range() -> None:
    bins = np.array([1, 4, 2, 2, 3], dtype=np.int64)
    stream, tree, symb = _encode(preproc_diff(bins))
    compressed = HuffmanCompressedArray(stream, tree, symb, bins.size, nbins=4, vmin=-1.0,
                                        vmax=1.0)

    decoded = compressed.decode()

    assert decoded.dtype == np.float32
    assert_allclose(decoded, -1.0 + (bins - 0.5)*0.5)


def test_one_symbol_stream_decodes() -> None:
    # simgen writes this for all-zero flags when pix and psi are stored uncompressed: the tree is
    # a single symbol with no internal nodes, and every sample is coded as one 0 bit.
    stream, tree, symb = _encode(np.zeros(13, dtype=np.int64))
    assert tree.size == 1

    assert_array_equal(HuffmanCompressedArray(stream, tree, symb, 13).decode(), np.zeros(13))


def test_stream_without_padding_byte_decodes() -> None:
    # 16 two-bit codes fill 4 bytes exactly. Our writer (like C3's) then appends a byte of 8
    # padding bits; the SO writer appends nothing and stores padding 0. Both must decode.
    values = np.tile(np.array([0, 1, 2, 3], dtype=np.int64), 4)
    stream, tree, symb = _encode(values)
    assert stream[0] == 8 and stream.size == 6
    so_stream = np.concatenate(([0], stream[1:-1])).astype(np.uint8)
    out = np.empty(values.size, dtype=symb.dtype)

    assert_array_equal(cpp_compression.huffman_decode(so_stream, tree, symb, out), values)


def _corrupt(what: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """A bit stream, tree, symbols and output size with one defect, named by `what`."""
    # Four equally common symbols all get 2-bit codes, so no single bit is a whole code word.
    values = np.tile(np.array([0, 1, 2, 3], dtype=np.int64), 4)
    stream, tree, symb = _encode(values)
    nout = values.size
    if what == "child out of range":
        tree[3] = 10**6
    elif what == "wrong root":
        tree[0] -= 1
    elif what == "symbols do not match tree":
        symb = symb[:-1]
    elif what == "padding byte":
        stream[0] = 200
    elif what == "empty stream":
        stream = stream[:0]
    elif what == "ends inside a code word":
        # 32 data bits fill 4 bytes exactly, so the writer adds a full byte of 8 padding bits.
        # Claiming 7 makes the first padding bit data: half of a 2-bit code word.
        assert stream[0] == 8
        stream[0] = 7
    elif what == "out too long":
        nout += 1
    return stream, tree, symb, nout


@pytest.mark.parametrize("what, message", [
    ("child out of range", "tree child index out of range"),
    ("wrong root", "tree root"),
    ("symbols do not match tree", "tree size does not match"),
    ("padding byte", "bad padding byte"),
    ("empty stream", "empty bit stream"),
    ("ends inside a code word", "inside a code word"),
    ("out too long", "out array is too large"),
])
def test_malformed_input_raises_instead_of_reading_out_of_bounds(what: str, message: str) -> None:
    stream, tree, symb, nout = _corrupt(what)
    out = np.empty(nout, dtype=symb.dtype)

    with pytest.raises(RuntimeError, match=message):
        cpp_compression.huffman_decode(stream, tree, symb, out)
