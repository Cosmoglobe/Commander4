"""The shared TOD reader: what it reads from a standard scan file, and which detector-scans it drops
(the cuts).

The scan files are written by simgen's `write_scan_file`, with plain (uncompressed) pointing so the
expected pixels and angles are exact. Tests that need flags re-encode them with their own tree.
"""
from pathlib import Path

import h5py
import numpy as np
import pytest
from mpi4py import MPI
from pixell.bunch import Bunch

import commander4.file_io.tod_reader as tod_reader
from commander4.compression import huffman
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.file_io.experiments.base_reader import find_good_fourier_size
from commander4.parameters.parse import params_from_dict
from simgen.writers import write_filelist, write_scan_file

NTOD = 67  # Not a fast FFT size, so the reader trims every scan.
NTOD_FFT = find_good_fourier_size(NTOD)
BAD_FLAG = 2**14  # One of the bits in the standard bad-data bitmask.


def write_scan(directory: Path, scan_id: int, tods: dict[str, np.ndarray],
               flags: dict[str, np.ndarray] | None = None, gain: float = 2.0) -> str:
    """Write one standard-format scan holding the detectors in ``tods``, in that order."""
    names = list(tods)
    pix = {name: np.arange(NTOD) % 12 for name in names}
    psi = {name: np.linspace(0.0, np.pi, NTOD) for name in names}
    # The file stores the gain in micro-units, as the standard format does.
    scalars = {name: np.array([gain*1e6, 1.0, 0.1, -1.0]) for name in names}
    path = str(directory / f"scan_{scan_id:06d}.h5")
    write_scan_file(path, scan_id, 1, 10.0, 4096, NTOD, np.array([1.0, 2.0, 3.0]), names, pix, psi,
                    tods, scalars, compress=False)
    if flags is not None:
        pid = f"{scan_id:06d}"
        diffs = {name: huffman.preproc_diff(flags.get(name, np.zeros(NTOD, dtype=np.int64)))
                 for name in names}
        tree, symbols, codes, lengths = huffman.build_huffman_tree(list(diffs.values()))
        with h5py.File(path, "a") as f:
            del f[f"{pid}/common/hufftree"], f[f"{pid}/common/huffsymb"]
            f[f"{pid}/common/hufftree"] = tree.astype(np.int64)
            f[f"{pid}/common/huffsymb"] = symbols
            for name in names:
                del f[f"{pid}/{name}/flag"]
                f[f"{pid}/{name}/flag"] = np.void(huffman.huffman_compress_array(diffs[name], codes,
                                                                                 lengths))
    return path


def make_params(directory: Path, paths: list[str], det_names: list[str],
                **experiment_keys) -> Bunch:
    """Parameter file with one `general` band reading ``paths`` (scan IDs 1, 2, ...)."""
    write_filelist(str(directory / "filelist.txt"), list(enumerate(paths, start=1)))
    return params_from_dict({"experiments": {"Exp": {
        "experiment_id": "general", **experiment_keys,
        "bands": {"Band": {"filelist": str(directory / "filelist.txt"), "eval_nside": 1,
                           "freq": 100.0, "fwhm": 10.0, "polarization": "IQU",
                           "detectors": {name: {} for name in det_names}}}}},
        "tod_processing": {}})


def read(params: Bunch, comm: MPI.Comm = MPI.COMM_SELF) -> DetectorGroupTOD:
    band = params.experiments.Exp.bands.Band
    return tod_reader.read_tods_from_file(comm, params.experiments.Exp, band,
                                          list(band.detectors), params)


def tod(value: float = 1.0) -> np.ndarray:
    return np.full(NTOD, value, dtype=np.float32) + np.arange(NTOD, dtype=np.float32)*1e-3


def test_standard_scan_is_read_by_detector_name(tmp_path: Path) -> None:
    """The file lists detectors in its own order and lacks one; the band's order decides."""
    path = write_scan(tmp_path, 1, {"b": tod(2.0), "a": tod(1.0)})
    with h5py.File(path, "a") as f:
        del f["common/polang"]
        f["common/polang"] = np.array([0.2, 0.1])  # b, a in file order
    result = read(make_params(tmp_path, [path], ["a", "b", "c"]))

    assert result.nscans == 1 and result.ndet == 3 and result.fsamp == 10.0
    assert (result.scan_idx_start, result.scan_idx_stop, result.nscans_allranks) == (0, 1, 1)
    scan = result.scans[0]
    assert scan.scan_id == 1 and scan.start_time == 0.0
    assert [(det.name, det.det_idx_fullband) for det in scan.detectors] == [("a", 0), ("b", 1)]
    for det, value, polang in zip(scan.detectors, [1.0, 2.0], [0.1, 0.2]):
        assert det.ntod == NTOD_FFT and det.ntod_original == NTOD
        np.testing.assert_array_equal(det.tod, tod(value)[:NTOD_FFT])
        np.testing.assert_array_equal(det.get_pix(), (np.arange(NTOD) % 12)[:NTOD_FFT])
        np.testing.assert_allclose(det.get_psi(), np.linspace(0.0, np.pi, NTOD)[:NTOD_FFT],
                                   rtol=1e-6)
        assert det.init_scalars[0] == pytest.approx(2.0)
        assert det.polang == polang
        np.testing.assert_array_equal(det.orbital_velocity_m_per_s, [1.0, 2.0, 3.0])
        assert det.good_data_mask.all()


def test_bad_detector_scans_and_empty_scans_are_dropped(tmp_path: Path) -> None:
    nan_tod = tod()
    nan_tod[5] = np.nan
    all_flagged = {"flagged": np.full(NTOD, BAD_FLAG, dtype=np.int64)}
    paths = [write_scan(tmp_path, 1, {"good": tod(), "zero": np.zeros(NTOD, np.float32),
                                      "nan": nan_tod, "flagged": tod()}, flags=all_flagged),
             write_scan(tmp_path, 2, {"zero": np.zeros(NTOD, np.float32)})]
    result = read(make_params(tmp_path, paths, ["good", "zero", "nan", "flagged"]))

    assert [scan.scan_id for scan in result.scans] == [1]
    assert [det.name for det in result.scans[0].detectors] == ["good"]


def test_parameter_file_overrides_the_flag_cuts(tmp_path: Path) -> None:
    """``min_unmasked_fraction`` drops a partly flagged detector; a zero bitmask keeps it."""
    half = np.zeros(NTOD, dtype=np.int64)
    half[:NTOD//2] = BAD_FLAG
    path = write_scan(tmp_path, 1, {"good": tod(), "half": tod()}, flags={"half": half})

    kept = read(make_params(tmp_path, [path], ["good", "half"]))
    stricter = read(make_params(tmp_path, [path], ["good", "half"], min_unmasked_fraction=0.9))
    unmasked = read(make_params(tmp_path, [path], ["good", "half"], min_unmasked_fraction=0.9,
                                bad_data_bitmask=0))

    assert [det.name for det in kept.scans[0].detectors] == ["good", "half"]
    assert kept.scans[0].detectors[1].good_data_mask.mean() == pytest.approx(
        1 - (NTOD//2)/NTOD_FFT)
    assert [det.name for det in stricter.scans[0].detectors] == ["good"]
    assert [det.name for det in unmasked.scans[0].detectors] == ["good", "half"]


def test_level_outliers_are_judged_against_their_own_detector(tmp_path: Path) -> None:
    """Scan 3 of detector a has a large offset; b is 1000x louder throughout, but never odd."""
    rng = np.random.default_rng(1)
    paths = []
    for scan_id in range(1, 6):
        a = rng.normal(size=NTOD).astype(np.float32)
        if scan_id == 3:
            a += 100.0
        b = 1000*rng.normal(size=NTOD).astype(np.float32)
        paths.append(write_scan(tmp_path, scan_id, {"a": a, "b": b}))

    result = read(make_params(tmp_path, paths, ["a", "b"], max_rms_ratio=10.0))

    kept = {(scan.scan_id, det.name) for scan in result.scans for det in scan.detectors}
    assert kept == {(scan_id, name) for scan_id in range(1, 6) for name in "ab"} - {(3, "a")}


def test_filelist_slice_and_bad_scan_ids_select_the_scans(tmp_path: Path) -> None:
    paths = [write_scan(tmp_path, scan_id, {"a": tod()}) for scan_id in range(1, 6)]
    np.save(tmp_path / "bad.npy", np.array([3]))
    params = make_params(tmp_path, paths, ["a"], filelist_idx_start=1, filelist_idx_stop=-1,
                         bad_PIDs_path=str(tmp_path / "bad.npy"))

    result = read(params)

    assert [scan.scan_id for scan in result.scans] == [2, 4]
    assert result.nscans_allranks == 2


def test_simulated_tod_skips_the_tod_value_cuts(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """A TOD that is zero on disk survives when the dispatcher replaces it with a simulation."""
    calls = []
    monkeypatch.setattr(tod_reader, "replace_tod_with_sim",
                        lambda comm, band_tod, *args: calls.append(band_tod.nscans))
    path = write_scan(tmp_path, 1, {"a": np.zeros(NTOD, np.float32)})
    params = make_params(tmp_path, [path], ["a"], replace_tod_with_sim=True, sim_params={})

    result = read(params)

    assert result.nscans == 1 and calls == [1]


@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() != 2, reason="requires exactly two MPI ranks")
def test_rank_without_scans_still_gets_the_sample_rate(tmp_path: Path) -> None:
    # Each rank writes its own copy; only rank 0's filelist is read. The one scan goes to rank 0.
    # The level cut is on, so its band-wide gather also runs with an empty rank.
    comm = MPI.COMM_WORLD
    params = make_params(tmp_path, [write_scan(tmp_path, 1, {"a": tod()})], ["a"],
                         max_rms_ratio=10.0)

    result = read(params, comm)

    assert result.nscans == 1 - comm.rank
    assert result.fsamp == 10.0
    assert result.nscans_allranks == 1
