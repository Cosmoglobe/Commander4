"""`c4-scan-weights`: the run times and sky positions in the chains become filelist columns."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from commander4.standalone_tools.scan_weights import main


def write_chain(path: Path, scan_ids: list[int], runtime: list[float],
                sky_position: list[list[float]] | None = None) -> str:
    """A band chain file holding only what the tool reads."""
    with h5py.File(path, "w") as f:
        f["scan_ids"] = np.array(scan_ids, dtype=np.int64)
        f["scan_runtime"] = np.array(runtime)
        if sky_position is not None:
            f["scan_sky_position"] = np.array(sky_position)
    return str(path)


@pytest.fixture
def filelist(tmp_path: Path) -> str:
    """Scans 10, 20, 30, 40 with weight 1, theta and phi 0.1*n and 0.2*n in row n, and a 6th
    column that the tool must pass through."""
    path = tmp_path / "filelist.txt"
    rows = [f'{scan_id} "/data/scan_{scan_id}.h5" 1 {0.1*n:.1f} {0.2*n:.1f} 7\n'
            for n, scan_id in enumerate((10, 20, 30, 40))]
    path.write_text("4\n" + "".join(rows))
    return str(path)


def test_weights_are_mean_runtimes_and_unmeasured_scans_get_the_median(tmp_path: Path,
                                                                       filelist: str) -> None:
    """Scan 40 is in neither chain file. The first file predates `scan_sky_position`."""
    chains = [write_chain(tmp_path / "iter1.h5", [30, 10, 20], [3.0, 1.0, 4.0]),
              write_chain(tmp_path / "iter2.h5", [10, 20, 30], [3.0, 6.0, 5.0],
                          sky_position=[[1.5, 2.5], [1.6, 2.6], [1.7, 2.7]])]
    output = tmp_path / "weighted.txt"

    main([filelist, *chains, "-o", str(output)])

    lines = output.read_text().splitlines()
    assert lines[0] == "4"
    assert lines[1:] == ['10 "/data/scan_10.h5" 2 1.500000 2.500000 7',
                         '20 "/data/scan_20.h5" 5 1.600000 2.600000 7',
                         '30 "/data/scan_30.h5" 4 1.700000 2.700000 7',
                         '40 "/data/scan_40.h5" 4 0.3 0.6 7']


def test_a_chain_of_another_band_is_refused(tmp_path: Path, filelist: str) -> None:
    chain = write_chain(tmp_path / "iter1.h5", [10, 99], [1.0, 2.0])

    with pytest.raises(ValueError, match="scan 99"):
        main([filelist, chain, "-o", str(tmp_path / "weighted.txt")])
