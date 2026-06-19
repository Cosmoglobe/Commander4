"""HDF5 output in the ``litebird_sim`` reader format.

Each scan is written to its own file with the exact layout consumed by
``commander4.experiments.litebird.tod_reader_litebird_sim`` (with ``replace_tod_with_sim: false``);
a per-band ``filelist.txt`` maps scan ids to file paths. Pointing is Huffman-compressed with a
*single per-scan tree* shared across the band's detectors (pix, psi and flag all decode with the
file's ``hufftree``/``huffsymb``), matching how the reader decodes them.
"""
import os
import logging
import numpy as np
import h5py
from numpy.typing import NDArray

import commander4.compression.huffman as huffman

logger = logging.getLogger(__name__)

DEFAULT_NPSI = 4096
# Bitmask the reader ANDs the cumulative flags with for its good-scan check; our flags are all-zero
# (kept here only to document the contract the writer must satisfy).
GOOD_SCAN_BITMASK = 6111232


def write_scan_file(path: str, pid: int, data_nside: int, fsamp: float, npsi: int, ntod: int,
                    vsun: NDArray, det_names: list[str], det_pix: dict[str, NDArray],
                    det_psi: dict[str, NDArray], det_tod: dict[str, NDArray],
                    compress: bool = True) -> None:
    """Write one scan to ``path`` in the litebird per-detector format.

    Args:
        pid: Scan id (the HDF5 group name is its 6-digit form).
        data_nside: nside at which pixel indices are stored (reader remaps to eval_nside).
        npsi: Number of psi digitization bins (only used when ``compress`` is True).
        vsun: Orbital velocity (3,) at the scan midpoint [m/s, Galactic], stored as ``common/vsun``.
        det_pix/det_psi/det_tod: per-detector pixel (int, data_nside), psi (rad) and TOD arrays.
        compress: If True (default), ``pix`` and ``psi`` are Huffman-compressed (opaque payloads the
            reader detects by type). If False they are stored as plain ``int32``/``float32`` arrays,
            which ``tod_reader_litebird_sim`` reads directly (``PixelPointing`` auto-detects). NOTE:
            ``flag`` is always Huffman-compressed because that reader unconditionally decodes it; the
            per-scan ``hufftree``/``huffsymb`` therefore always exist (built from the flag stream).
    """
    # The flag stream (all zeros -> first differences all zero) is always Huffman-encoded; its
    # symbol set (just 0) is what the per-scan tree must cover at minimum.
    flag_diff = huffman.preproc_diff(np.zeros(ntod, dtype=np.int64))
    tree_inputs = [flag_diff]
    pix_diff, psi_diff = {}, {}
    if compress:
        # Diff-encode each detector's pointing too, so one shared tree covers pix, psi and flag.
        for name in det_names:
            pix_diff[name] = huffman.preproc_diff(det_pix[name].astype(np.int64, copy=False))
            psi_diff[name] = huffman.preproc_digitize_and_diff(
                det_psi[name].astype(np.float64, copy=False), npsi)
            tree_inputs += [pix_diff[name], psi_diff[name]]
    hufftree, huffsymb, sym_codes, sym_lengths = huffman.build_huffman_tree(tree_inputs)
    flag_enc = huffman.huffman_compress_array(flag_diff, sym_codes, sym_lengths)

    pidg = f"{pid:06d}"
    with h5py.File(path, "w") as f:
        f["common/nside"] = np.array([data_nside], dtype=np.int32)
        f["common/fsamp"] = np.array([fsamp], dtype=np.float64)
        f["common/npsi"] = np.array([npsi], dtype=np.int32)
        f[f"{pidg}/common/ntod"] = np.array([ntod], dtype=np.int64)
        f[f"{pidg}/common/hufftree"] = hufftree.astype(np.int64, copy=False)
        f[f"{pidg}/common/huffsymb"] = huffsymb
        f[f"{pidg}/common/vsun"] = np.asarray(vsun, dtype=np.float64)
        for name in det_names:
            g = f"{pidg}/{name}"
            f[f"{g}/tod"] = det_tod[name].astype(np.float32, copy=False)
            if compress:
                # Store compressed payloads as opaque scalars so they read back as numpy.void; the
                # reader (PixelPointing) detects compression by this type.
                f[f"{g}/pix"] = np.void(huffman.huffman_compress_array(pix_diff[name], sym_codes,
                                                                       sym_lengths))
                f[f"{g}/psi"] = np.void(huffman.huffman_compress_array(psi_diff[name], sym_codes,
                                                                       sym_lengths))
            else:
                # Plain arrays: int32 pixel indices (at data_nside) and float32 psi in radians.
                f[f"{g}/pix"] = det_pix[name].astype(np.int32, copy=False)
                f[f"{g}/psi"] = det_psi[name].astype(np.float32, copy=False)
            f[f"{g}/flag"] = np.void(flag_enc)


def write_filelist(path: str, entries: list[tuple[int, str]]) -> None:
    """Write a band ``filelist.txt``: a count header then one ``pid "abs_path" 1 0.0 0.0`` per scan.

    Matches the parsing in the litebird/planck readers (first line skipped; 5 whitespace tokens per
    line; the path is quoted and the surrounding quotes are stripped on read).
    """
    entries = sorted(entries, key=lambda e: e[0])
    with open(path, "w") as f:
        f.write(f"{len(entries)}\n")
        for pid, filepath in entries:
            f.write(f'{pid} "{os.path.abspath(filepath)}" 1 0.0 0.0\n')
