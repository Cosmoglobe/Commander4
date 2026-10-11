"""`c4-scan-weights`: the run times, sky positions and scan lengths in the chains become filelist
columns."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from commander4.standalone_tools.scan_weights import main


def write_chain(path: Path, scan_ids: list[int], runtime: list[float],
                sky_position: list[list[float]] | None = None,
                ntod: list[int] | None = None, dropped: list[int] | None = None) -> str:
    """A band chain file holding only what the tool reads."""
    with h5py.File(path, "w") as f:
        f["scan_ids"] = np.array(scan_ids, dtype=np.int64)
        f["scan_runtime"] = np.array(runtime)
        if sky_position is not None:
            f["scan_sky_position"] = np.array(sky_position)
        if ntod is not None:
            f["scan_ntod"] = np.array(ntod, dtype=np.int64)
        if dropped is not None:
            f["dropped_scan_ids"] = np.array(dropped, dtype=np.int64)
    return str(path)


@pytest.fixture
def filelist(tmp_path: Path) -> str:
    """Scans 10, 20, 30, 40 with weight 1, and theta and phi 0.1*n and 0.2*n in row n."""
    path = tmp_path / "filelist.txt"
    rows = [f'{scan_id} "/data/scan_{scan_id}.h5" 1 {0.1*n:.1f} {0.2*n:.1f}\n'
            for n, scan_id in enumerate((10, 20, 30, 40))]
    path.write_text("4\n" + "".join(rows))
    return str(path)


def test_weights_are_mean_runtimes_and_unmeasured_scans_get_the_median(tmp_path: Path,
                                                                       filelist: str) -> None:
    """Scan 40 is in neither chain file. The first file predates `scan_sky_position` and
    `scan_ntod`."""
    chains = [write_chain(tmp_path / "iter1.h5", [30, 10, 20], [3.0, 1.0, 4.0]),
              write_chain(tmp_path / "iter2.h5", [10, 20, 30], [3.0, 6.0, 5.0],
                          sky_position=[[1.5, 2.5], [1.6, 2.6], [1.7, 2.7]],
                          ntod=[1000, 2000, 3000])]
    output = tmp_path / "weighted.txt"

    main([filelist, *chains, "-o", str(output)])

    lines = output.read_text().splitlines()
    assert lines[0] == "4"
    assert lines[1:] == ['10 "/data/scan_10.h5" 2 1.500000 2.500000 1000 1',
                         '20 "/data/scan_20.h5" 5 1.600000 2.600000 2000 1',
                         '30 "/data/scan_30.h5" 4 1.700000 2.700000 3000 1',
                         '40 "/data/scan_40.h5" 4 0.3 0.6 -1 1']


def test_a_scan_the_run_dropped_is_marked_not_to_be_read(tmp_path: Path, filelist: str) -> None:
    chain = write_chain(tmp_path / "iter1.h5", [10, 30, 40], [1.0, 2.0, 3.0], dropped=[20])
    output = tmp_path / "weighted.txt"

    main([filelist, chain, "-o", str(output)])

    weight_and_read = [(line.split()[2], line.split()[6])
                       for line in output.read_text().splitlines()[1:]]
    assert weight_and_read == [("1", "1"), ("0", "0"), ("2", "1"), ("3", "1")]


def test_clear_read_marks_reads_every_scan_again(tmp_path: Path, filelist: str) -> None:
    """After loosening the cuts, the scan the run dropped gets the median weight instead of 0."""
    chain = write_chain(tmp_path / "iter1.h5", [10, 30, 40], [1.0, 2.0, 3.0], dropped=[20])
    output = tmp_path / "weighted.txt"

    main([filelist, chain, "-o", str(output), "--clear-read-marks"])

    weight_and_read = [(line.split()[2], line.split()[6])
                       for line in output.read_text().splitlines()[1:]]
    assert weight_and_read == [("1", "1"), ("2", "1"), ("2", "1"), ("3", "1")]


def test_a_rerun_on_its_own_output_keeps_what_the_new_chains_lack(tmp_path: Path,
                                                                  filelist: str) -> None:
    """The second run did not read scan 20, which the first marked, and its chain covers only
    scan 40. So scans 10-30 keep their position, ntod and read mark."""
    first = write_chain(tmp_path / "run1.h5", [10, 30], [1.0, 3.0],
                        sky_position=[[1.5, 2.5], [1.7, 2.7]], ntod=[1000, 3000], dropped=[20])
    second = write_chain(tmp_path / "run2.h5", [40], [8.0], sky_position=[[1.8, 2.8]], ntod=[4000])
    main([filelist, first, "-o", str(tmp_path / "weighted1.txt")])

    main([str(tmp_path / "weighted1.txt"), second, "-o", str(tmp_path / "weighted2.txt")])

    lines = (tmp_path / "weighted2.txt").read_text().splitlines()[1:]
    columns = [line.split()[3:] for line in lines]
    assert columns == [["1.500000", "2.500000", "1000", "1"], ["0.1", "0.2", "-1", "0"],
                       ["1.700000", "2.700000", "3000", "1"], ["1.800000", "2.800000", "4000", "1"]]


def test_a_filelist_of_only_ids_and_paths_gets_every_column(tmp_path: Path) -> None:
    """With no positions anywhere, theta and phi are written as 0, so ntod and read stay put."""
    filelist = tmp_path / "filelist.txt"
    filelist.write_text('2\n10 "/data/scan_10.h5"\n20 "/data/scan_20.h5"\n')
    chain = write_chain(tmp_path / "iter1.h5", [10, 20], [1.0, 3.0], ntod=[1000, 2000])
    output = tmp_path / "weighted.txt"

    main([str(filelist), chain, "-o", str(output)])

    assert output.read_text().splitlines()[1:] == ['10 "/data/scan_10.h5" 1 0.0 0.0 1000 1',
                                                   '20 "/data/scan_20.h5" 3 0.0 0.0 2000 1']


def test_chain_scans_the_filelist_does_not_list_are_ignored(tmp_path: Path, filelist: str) -> None:
    """The run read scans 5 and 99 too, e.g. over a wider scan-ID range than this filelist."""
    chain = write_chain(tmp_path / "iter1.h5", [5, 10, 99], [7.0, 1.0, 9.0], dropped=[98])
    output = tmp_path / "weighted.txt"

    main([filelist, chain, "-o", str(output)])

    weights = [line.split()[2] for line in output.read_text().splitlines()[1:]]
    assert weights == ["1", "1", "1", "1"]  # Scan 10's run time, and that median for the others.


def test_a_chain_of_another_band_is_refused(tmp_path: Path, filelist: str) -> None:
    chain = write_chain(tmp_path / "iter1.h5", [98, 99], [1.0, 2.0])

    with pytest.raises(ValueError, match="none of the scans"):
        main([filelist, chain, "-o", str(tmp_path / "weighted.txt")])
