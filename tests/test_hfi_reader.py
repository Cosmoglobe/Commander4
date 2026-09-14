"""HFI instrument efficiencies reach the detector response and sky projection.

Efficiencies are carried through as measured, except that unpolarized bolometers (those below
``UNPOLARIZED_POLEFF_CUTOFF``) are zeroed so they are intensity-only downstream.
"""

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.compression import huffman
from commander4.file_io.experiments.planck_hfi import UNPOLARIZED_POLEFF_CUTOFF, tod_reader
from commander4.parameters.parse import params_from_dict
from commander4.tod.view import TODView
from simgen.writers import write_scan_file


@pytest.fixture
def hfi_inputs(tmp_path: Path) -> tuple[Bunch, list[str]]:
    """Include an absent detector and different configuration, TOD, and instrument orders."""
    scan_path = tmp_path / "scan.h5"
    present_names = ["353-1", "353-3b"]
    det_names = ["353-3a", "353-3b", "353-1"]
    pixels = {}
    angles = {}
    tods = {}
    scalars = {}
    for name in present_names:
        pixels[name] = np.arange(16, dtype=np.int64) % 12
        angles[name] = np.linspace(0.0, np.pi, 16)
        tods[name] = np.arange(1, 17, dtype=np.float32)
        scalars[name] = np.array([1e6, 1.0, 0.1, -1.0])
    write_scan_file(str(scan_path), 42, 1, 180.0, 4096, 16, np.zeros(3), present_names,
                    pixels, angles, tods, scalars)
    differences = huffman.preproc_diff(np.arange(1, 17, dtype=np.int64))
    tree, symbols, codes, lengths = huffman.build_huffman_tree([differences])
    encoded = huffman.huffman_compress_array(differences, codes, lengths)
    with h5py.File(scan_path, "a") as handle:
        handle["000042/common/hufftree2"] = tree
        handle["000042/common/huffsymb2"] = symbols
        for name in present_names:
            handle[f"000042/{name}/ztod"] = np.void(encoded)
    instrument_path = tmp_path / "instrument.h5"
    with h5py.File(instrument_path, "w") as handle:
        handle["353-1/polEff"] = [3.4]
        handle["353-3a/polEff"] = [88.7]
        handle["353-3b/polEff"] = [92.0]
    filelist = tmp_path / "filelist.txt"
    filelist.write_text(f'1\n42 "{scan_path}" 1 0 0\n')
    params = params_from_dict({"experiments": {"HFI": {
        "experiment_id": "planck_hfi", "instrument_file": str(instrument_path),
        "bands": {"Band": {"filelist": str(filelist), "eval_nside": 1, "freq": 353.0,
                           "fwhm": 4.9, "polarization": "IQU"}},
    }}, "tod_processing": {}})
    return params, det_names


def test_instrument_percentages_follow_detector_names_into_sky_projection(
    hfi_inputs: tuple[Bunch, list[str]],
) -> None:
    params, det_names = hfi_inputs
    experiment = params.experiments.HFI
    band = experiment.bands.Band

    result = tod_reader(MPI.COMM_SELF, experiment, band, det_names, params, 0, 1)

    assert result.instrument_filepath == experiment.instrument_file
    assert result.nscans == 1
    assert result.ndet == 3
    detectors = result.scans[0].detectors
    assert [det.name for det in detectors] == ["353-3b", "353-1"]
    assert [det.det_idx_fullband for det in detectors] == [1, 2]
    sky = np.empty((3, 12))
    sky[0], sky[1], sky[2] = 2.0, 3.0, 5.0
    # 353-3b is a polarization-sensitive bolometer; 353-1 is an unpolarized SWB whose 3.4% of
    # leakage is below the cutoff, so it sees intensity alone.
    for det, efficiency in zip(detectors, [0.92, 0.0]):
        assert det.response_I_P == pytest.approx((1.0, efficiency))
        view = TODView(result, SimpleNamespace(), sky).focus(0, det)
        psi = det.get_psi()
        expected = 2.0 + efficiency * (3.0 * np.cos(2 * psi) + 5.0 * np.sin(2 * psi))
        np.testing.assert_allclose(view.get_static_sky_tod(), expected, rtol=1e-6)


def test_efficiency_just_above_the_cutoff_is_kept_as_measured(
    hfi_inputs: tuple[Bunch, list[str]],
) -> None:
    """Only detectors below the cutoff are zeroed; the value itself is never rounded or rescaled."""
    params, det_names = hfi_inputs
    experiment = params.experiments.HFI
    with h5py.File(experiment.instrument_file, "a") as handle:
        del handle["353-1/polEff"]
        handle["353-1/polEff"] = [100 * UNPOLARIZED_POLEFF_CUTOFF + 1.0]

    result = tod_reader(MPI.COMM_SELF, experiment, experiment.bands.Band, det_names, params, 0, 1)

    kept = dict((det.name, det.response_I_P) for det in result.scans[0].detectors)
    assert kept["353-1"] == pytest.approx((1.0, UNPOLARIZED_POLEFF_CUTOFF + 0.01))
    assert kept["353-3b"] == pytest.approx((1.0, 0.92))


def test_intensity_only_band_reads_without_requiring_psi(
    hfi_inputs: tuple[Bunch, list[str]],
) -> None:
    params, det_names = hfi_inputs
    experiment = params.experiments.HFI
    band = experiment.bands.Band
    band.polarization = "I"
    scan_path = Path(band.filelist).parent / "scan.h5"
    with h5py.File(scan_path, "a") as handle:
        del handle["000042/353-1/psi"]
        del handle["000042/353-3b/psi"]

    result = tod_reader(MPI.COMM_SELF, experiment, band, det_names, params, 0, 1)

    for det, efficiency in zip(result.scans[0].detectors, [0.92, 0.0]):
        assert det.response_I_P == pytest.approx((1.0, efficiency))
        sky = np.full((1, 12), 2.0)
        view = TODView(result, SimpleNamespace(), sky).focus(0, det)
        np.testing.assert_array_equal(view.get_static_sky_tod(), np.full(16, 2.0))


def test_missing_instrument_efficiency_is_not_assumed_to_be_unity(
    hfi_inputs: tuple[Bunch, list[str]],
) -> None:
    params, det_names = hfi_inputs
    experiment = params.experiments.HFI
    with h5py.File(experiment.instrument_file, "a") as handle:
        del handle["353-1/polEff"]

    with pytest.raises(KeyError, match="polEff"):
        tod_reader(MPI.COMM_SELF, experiment, experiment.bands.Band, det_names, params, 0, 1)


@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() != 2, reason="requires exactly two MPI ranks")
def test_two_rank_reader_broadcasts_efficiencies_from_an_empty_master(
    hfi_inputs: tuple[Bunch, list[str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    params, det_names = hfi_inputs
    experiment = params.experiments.HFI
    comm = MPI.COMM_WORLD
    original_open = h5py.File
    instrument_opens = []

    def checked_open(filename: str, *args, **kwargs):
        if str(filename) == experiment.instrument_file:
            assert comm.rank == 0
            instrument_opens.append(filename)
        return original_open(filename, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", checked_open)
    # Rank zero has no scans but still supplies the shared detector metadata.
    result = tod_reader(comm, experiment, experiment.bands.Band, det_names, params, 0, comm.rank)

    assert len(instrument_opens) == (1 if comm.rank == 0 else 0)
    assert result.nscans == comm.rank
    if comm.rank == 1:
        assert result.scans[0].detectors[0].response_I_P == pytest.approx((1.0, 0.92))
        assert result.scans[0].detectors[1].response_I_P == pytest.approx((1.0, 0.0))
