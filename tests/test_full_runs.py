"""Full runs of C4 under `mpirun`, on a tiny simgen simulation.

The unit tests leave the joins untested: MPI layout, the hand-over between the TOD and CompSep
sides, chain writing and restart. These tests start the real program. There are three setups, each
a parameter file in `tests/full_runs/`, chosen together so that every TOD step and CompSep solver
that simgen data can drive runs at least once:
    A  joint run: two TOD bands and one file band, both sides active (9 tasks).
    B  CompSep only, on file bands, started from run A's chains (5 tasks).
    C  TOD only, with the TOD options that run A leaves out (4 tasks).
Run A is also restarted from its first iteration, which must reproduce its second exactly.

All commands run in one working directory, so the parameter files use relative paths and can also
be run by hand there: `mpirun -n 9 commander4 -p tests/full_runs/run_A_joint.yml`.
The simulation needs CAMB and PySM3, and PySM3 downloads its templates on first use.

The checks are deliberately loose about the science (they catch a wrong sign or unit, not a small
bias): what they test is that every option runs, writes finite chains, and restarts exactly.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
import pytest
import yaml

FULL_RUNS = Path(__file__).parent / "full_runs"
BIN = Path(sys.executable).parent  # The project venv's `commander4` and `c4-simgen` commands.

# Run every test of this file on one xdist worker, so the simulation and run A are made once.
# Skip them all with `pytest -m "not full_run"`.
pytestmark = [pytest.mark.xdist_group("full_runs"), pytest.mark.full_run]

# Datasets that may hold NaN or inf by design: empty bins of the log-binned TOD power spectra, the
# temporal-gain prior when that step is off, and the glitch templates (the step is a placeholder).
ALLOWED_NONFINITE = ("tod_ps_", "gain_prior", "glitches/")


def mpirun(workdir: Path, ntasks: int, command: str, param_file: Path) -> None:
    """Run `mpirun -n ntasks <command> -p param_file` in `workdir`; fail the test on an error."""
    env = dict(os.environ, OMP_NUM_THREADS="1")
    result = subprocess.run(["mpirun", "-n", str(ntasks), str(BIN / command), "-p", str(param_file)],
                            cwd=workdir, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"{command} -p {param_file.name} failed with exit code {result.returncode}.\n"
                    f"--- stdout (end) ---\n{result.stdout[-6000:]}\n"
                    f"--- stderr (end) ---\n{result.stderr[-6000:]}")


def read_chain_file(path: Path) -> dict[str, np.ndarray]:
    """Every dataset of one chain file, by its full name."""
    datasets = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda name, obj: datasets.update({name: np.asarray(obj[()])})
                     if isinstance(obj, h5py.Dataset) else None)
    return datasets


def assert_finite_chains(run_dir: Path) -> None:
    """Every float dataset in the run's chain files is finite, apart from ALLOWED_NONFINITE.

    A map's rms must be infinite exactly in the pixels without hits.
    """
    for path in sorted(run_dir.glob("chains_*/*.h5")):
        datasets = read_chain_file(path)
        for name, value in datasets.items():
            if value.dtype.kind not in "fc" or name.startswith(ALLOWED_NONFINITE):
                continue
            if name == "maps/rms":
                unobserved = np.broadcast_to(datasets["maps/nhit"] == 0, value.shape)
                assert np.array_equal(~np.isfinite(value), unobserved), f"{path.name}: {name}"
            else:
                assert np.isfinite(value).all(), f"{path.name}: {name} is not finite"


@pytest.fixture(scope="session")
def workdir(tmp_path_factory) -> Path:
    """Simulate the data, then write the file-band maps and masks the runs read from sim_maps/."""
    pytest.importorskip("camb")
    pytest.importorskip("pysm3")
    if shutil.which("mpirun") is None:
        pytest.skip("mpirun is not available.")
    workdir = tmp_path_factory.mktemp("full_runs")
    mpirun(workdir, 3, "c4-simgen", FULL_RUNS / "simgen.yml")

    # File bands: simgen's band sky (beam-smoothed, at eval_nside) plus a white-noise draw from its
    # white-noise rms map. simgen's own binned noise map is not used, as it holds 1/f stripes that
    # the rms map does not describe.
    (workdir / "sim_maps").mkdir()
    rng = np.random.default_rng(5)
    columns = ["TEMPERATURE", "Q-POLARISATION", "U-POLARISATION"]
    for band in ["Band030", "Band100", "Band353", "Band353_16"]:
        with h5py.File(workdir / "sim_debug" / f"{band}_diagnostics.h5", "r") as f:
            rms = f["maps/rms_white_noise"][()]
            signal = f["maps/sky"][()] + rms*rng.standard_normal(rms.shape)
        names = columns[:len(signal)]
        for kind, maps in (("map", signal), ("rms", rms)):
            hp.write_map(str(workdir / "sim_maps" / f"{band}_{kind}.fits"), list(maps),
                         column_names=names, column_units="uK_RJ", dtype=np.float64)

    # Masks, 1 = use: a Galactic cut of |b| > 20 deg (gain, MCMC chi^2) and |b| > 10 deg (n_corr).
    theta, _ = hp.pix2ang(16, np.arange(hp.nside2npix(16)))
    latitude = 90.0 - np.degrees(theta)
    for name, cut in (("chisq_mask", 20.0), ("procmask_gain", 20.0), ("procmask_ncorr", 10.0)):
        hp.write_map(str(workdir / "sim_maps" / f"{name}.fits"), (np.abs(latitude) > cut)*1.0,
                     dtype=np.float64)
    return workdir


@pytest.fixture(scope="session")
def run_A(workdir: Path) -> Path:
    mpirun(workdir, 9, "commander4", FULL_RUNS / "run_A_joint.yml")
    return workdir / "out_A"


def test_joint_run(run_A: Path) -> None:
    for chain in (1, 2):
        for iteration in (1, 2):
            assert (run_A / "chains_compsep" / f"chain{chain:02d}_iter{iteration:04d}.h5").exists()
            for band in ("Band030", "Band100"):
                assert (run_A / "chains_bands" /
                        f"SimSat_{band}_chain{chain:02d}_iter{iteration:04d}.h5").exists()
    assert_finite_chains(run_A)

    # The gains start 3 % away from the truth (1.0); the noise truth is sigma0 = 60, fknee = 0.2.
    for band in ("Band030", "Band100"):
        sample = read_chain_file(run_A / "chains_bands" / f"SimSat_{band}_chain01_iter0002.h5")
        assert abs(sample["abs_gain"] - 1.0) < 0.02, band
        assert abs(np.median(sample["noise_params"][..., 0]) / 60.0 - 1.0) < 0.05, band
        assert 0.1 < np.median(sample["noise_params"][..., 1]) < 0.4, band
        assert sample["accept"].all(), band
    # Below 1 is expected: three components at lmax = 3*nside - 1 have about as many free
    # parameters as the maps have pixels, so the model also fits part of the noise. Band353
    # (nside 8, signal-to-noise ~250) guards the band lmax cut of the realized sky: a sky realized
    # above the band lmax gives it a reduced chi^2 of ~12000.
    sky = read_chain_file(run_A / "chains_compsep" / "chain01_iter0002.h5")
    for polband in ("Band030_I", "Band030_QU", "Band100_I", "Band100_QU", "Band353_I"):
        assert 0.2 < sky[f"chisq/bands/{polband}/reduced"] < 2.0, polband
    assert 1.0 <= sky["comps/dust/sed/beta"] <= 2.5


def test_restart_reproduces_joint_run(run_A: Path, workdir: Path) -> None:
    """Restarting run A after iteration 1 (chain_status: append) gives the same iteration 2."""
    restart_dir = workdir / "out_A_restart"
    shutil.copytree(run_A, restart_dir)
    for path in restart_dir.glob("chains_*/*_iter0002.h5"):
        path.unlink()
    params = yaml.safe_load((FULL_RUNS / "run_A_joint.yml").read_text())
    params["output"]["dir"] = restart_dir.name
    params["gibbs"]["chain_status"] = "append"
    param_file = workdir / "run_A_restart.yml"
    param_file.write_text(yaml.safe_dump(params, sort_keys=False))
    mpirun(workdir, 9, "commander4", param_file)

    original_files = sorted(run_A.glob("chains_*/*_iter0002.h5"))
    assert len(original_files) == 6
    for original_file in original_files:
        original = read_chain_file(original_file)
        restarted = read_chain_file(restart_dir / original_file.relative_to(run_A))
        assert original.keys() == restarted.keys()
        for name, value in original.items():
            if name.startswith("metadata") or name == "scan_runtime":  # Dates and timings.
                continue
            assert np.array_equal(value, restarted[name], equal_nan=value.dtype.kind in "fc"), \
                f"{original_file.name}: {name} differs after the restart"


def test_compsep_only_run(run_A: Path, workdir: Path) -> None:
    mpirun(workdir, 5, "commander4", FULL_RUNS / "run_B_compsep.yml")
    out_B = workdir / "out_B"
    assert sorted(path.name for path in out_B.glob("chains_compsep/*.h5")) == \
        ["chain01_iter0001.h5", "chain01_iter0002.h5"]
    assert not list(out_B.glob("chains_bands/*.h5"))
    assert_finite_chains(out_B)
    sky = read_chain_file(out_B / "chains_compsep" / "chain01_iter0002.h5")
    assert 1.0 <= sky["comps/dust/sed/beta"] <= 2.5
    for polband in ("Band030_I", "Band030_QU", "Band100_I", "Band100_QU", "Band353_I"):
        assert 0.2 < sky[f"chisq/bands/{polband}/reduced"] < 2.0, polband


def test_tod_only_run(workdir: Path) -> None:
    mpirun(workdir, 4, "commander4", FULL_RUNS / "run_C_tod.yml")
    out_C = workdir / "out_C"
    assert not list(out_C.glob("chains_compsep/*.h5"))
    assert_finite_chains(out_C)
    for band in ("Band030", "Band100"):
        first = read_chain_file(out_C / "chains_bands" / f"SimSat_{band}_chain01_iter0001.h5")
        second = read_chain_file(out_C / "chains_bands" / f"SimSat_{band}_chain01_iter0002.h5")
        # Maps are written every second iteration, downgraded to nside 8.
        assert first["maps/nhit"].shape[-1] == hp.nside2npix(8)
        assert not any(name.startswith("maps/") for name in second)
        assert (second["jump_counts"] == 0).all()  # simgen flags no jumps.
    # Band100 reads scans 3 to 22, and takes its start gains from the scan files (truth 1.0).
    assert list(second["scan_ids"]) == list(range(3, 23))
    assert abs(second["abs_gain"] - 1.0) < 0.02
