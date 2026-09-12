"""The order and independent random streams of the two interleaved Gibbs chains.

Both halves of the pipeline walk this same sequence, so it is the one place the chain and
iteration numbering is decided. It used to be four separate modular-arithmetic expressions inside
the main loop, one pair per side, which is how chain 2 came to be left one compsep sample short.
"""
import numpy as np
import pytest
from pixell.bunch import Bunch

from commander4.cli import gibbs_schedule, seed_iteration_rng


def test_chains_alternate_within_each_iteration():
    assert gibbs_schedule(3) == [(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3)]


def _mpi_info(side: str = "tod", rank: int = 3) -> Bunch:
    return Bunch(world=Bunch(side=side, rank=rank))


def test_iteration_seed_is_reproducible_and_ignores_previous_random_draws() -> None:
    params = Bunch(gibbs=Bunch(seed=1234))

    seed_iteration_rng(params, _mpi_info(), chain=1, iteration=2)
    first_draw = np.random.normal(size=5)
    np.random.normal(size=100)
    seed_iteration_rng(params, _mpi_info(), chain=1, iteration=2)
    second_draw = np.random.normal(size=5)

    np.testing.assert_array_equal(first_draw, second_draw)


def test_chain_iteration_rank_and_side_select_independent_streams() -> None:
    params = Bunch(gibbs=Bunch(seed=1234))
    seeds = {
        seed_iteration_rng(params, _mpi_info("tod", 3), chain=1, iteration=2),
        seed_iteration_rng(params, _mpi_info("tod", 3), chain=2, iteration=2),
        seed_iteration_rng(params, _mpi_info("tod", 3), chain=1, iteration=3),
        seed_iteration_rng(params, _mpi_info("tod", 4), chain=1, iteration=2),
        seed_iteration_rng(params, _mpi_info("compsep", 3), chain=1, iteration=2),
    }

    assert len(seeds) == 5


def test_default_root_seed_matches_explicit_1995() -> None:
    default = seed_iteration_rng(Bunch(gibbs=Bunch()), _mpi_info(), chain=1, iteration=1)
    explicit = seed_iteration_rng(
        Bunch(gibbs=Bunch(seed=1995)), _mpi_info(), chain=1, iteration=1,
    )

    assert default == explicit


def test_independent_initial_skies_follow_the_mpi_pipeline(monkeypatch):
    """Exercise the actual two-side schedules with synchronous messages and distinct saved skies."""
    import pytest
    from mpi4py import MPI
    from commander4.cli import run_tod_side, run_compsep_side
    from commander4.compsep import processing as compsep_processing
    from commander4.tod import processing as tod_processing
    from commander4.mpi import transfer

    comm = MPI.COMM_WORLD
    if comm.size != 2:
        pytest.skip("requires exactly two MPI ranks")
    params = Bunch(gibbs=Bunch(num_iterations=2, seed=1995))
    seen = []

    def process_tod(info, data, samples, sky, params, chain, iteration):
        seen.append((chain, iteration, float(sky[0])))
        return (chain, iteration, float(sky[0])), samples

    def process_compsep(info, state, tod, iteration, chain, params, components):
        assert tod == (chain, iteration, float(components[0]))
        components += 100.0 + chain
        return components

    # Synchronous sends ensure an incorrect initial handshake deadlocks even for small payloads.
    monkeypatch.setattr(tod_processing, "process_tod", process_tod)
    monkeypatch.setattr(compsep_processing, "process_compsep", process_compsep)
    monkeypatch.setattr(transfer, "send_tod", lambda info, tod, band, dest: comm.ssend(tod, dest=1))
    monkeypatch.setattr(transfer, "receive_tod", lambda *args: comm.recv(source=0))
    monkeypatch.setattr(transfer, "send_compsep",
                        lambda info, band, sky, dest: comm.ssend(sky, dest=0))
    monkeypatch.setattr(transfer, "receive_compsep", lambda *args: comm.recv(source=1))
    info = Bunch(world=Bunch(comm=comm, rank=comm.rank,
                             side="tod" if comm.rank == 0 else "compsep",
                             tod_band_masters={}, compsep_band_masters={}),
                 band=Bunch(is_master=True), tod=Bunch(rank=0, is_master=True),
                 compsep=Bunch(rank=0))
    if comm.rank == 0:
        first_sky = transfer.receive_compsep(info, None, "B", {})
        run_tod_side(info, params, None, "B", {1: None, 2: None}, first_sky, True,
                     start_iteration=8)
        assert seen == [(1, 8, 11.0), (2, 8, 22.0), (1, 9, 112.0), (2, 9, 124.0)]
    else:
        states = {1: np.full(1024, 11.0), 2: np.full(1024, 22.0)}
        for chain in (1, 2):
            transfer.send_compsep(info, "B_I", states[chain], {})
        first_tod = transfer.receive_tod(info, {}, None, "B_I", None, params)
        run_compsep_side(info, params, None, "B_I", None, states, first_tod, start_iteration=8)
        assert states[1][0] == 213.0
        assert states[2][0] == 226.0


@pytest.mark.parametrize("pipeline_mode", ["combined", "tod_only"])
def test_resume_matches_uninterrupted_sampling_in_the_full_mpi_driver(
        tmp_path, monkeypatch, pipeline_mode):
    """Run real TOD/CompSep steps; replace only raw-data I/O with a tiny deterministic scan."""
    import pytest
    import h5py
    from pathlib import Path
    from copy import deepcopy
    from mpi4py import MPI
    from commander4.cli import run_commander4
    from commander4.data_models.detector_group_tod import DetectorGroupTOD
    from commander4.data_models.detector_tod import DetectorTOD
    from commander4.data_models.pointing import PixelPointing
    from commander4.data_models.scan_tod import ScanTOD
    from commander4.file_io import paths
    from commander4.parameters.parse import load_params, params_from_dict
    from commander4.tod import processing
    from commander4.tod.noise.psd import NoisePSDOof

    comm = MPI.COMM_WORLD
    if comm.size != 2:
        pytest.skip("requires exactly two MPI ranks")
    root = Path(comm.bcast(str(tmp_path) if comm.rank == 0 else None, root=0))
    example = Path(__file__).resolve().parents[1] / "params/sims/simparam_gain.yml"
    _, base, _ = load_params(str(example))
    band = dict(enabled=True, num_tasks=1, eval_nside=1, fwhm=60.0, freq=100.0, fsamp=1.0,
                band_unit="uK_RJ", polarization="I", mapmaker="bin",
                detectors={"det": {"gain": 2.0}})
    base["experiments"]["SimSat"]["bands"] = {"Band100GHz": band}
    base["components"] = {"CMB": base["components"]["CMB"]}
    base["components"]["CMB"]["params"].update(
        polarization="I", lmax=2, Cl_prior_amplitude=None)
    base["compsep"]["bands"] = {
        "Band100GHz": dict(enabled=True, get_from="SimSat", polarization="I")}
    group = base["compsep"]["cg_sampling_groups"]["sample_amps_CG"]
    group.update(comps=["CMB"], bands=["Band100GHz"], max_iter=20, err_tol=1e-8)
    base["tod_processing"]["rel_gain"]["enabled"] = False
    base["tod_processing"]["abs_gain"].update(
        calibrate_against="sky", downsample_time=0.0)
    base["tod_processing"]["corr_noise"]["sample_sigma0"] = False
    base["output"]["chains"].update(write=[1, 2], maps_nside=1, nside_chisq=1,
                                     interval=dict(bands=1, compsep=1, maps=1))
    if pipeline_mode == "tod_only":
        band["num_tasks"] = 2
        base["compsep"]["enabled"] = False
        base["compsep"]["cg_sampling_groups"] = {}
        base["tod_processing"]["abs_gain"]["enabled"] = False

    def read_tods(band_comm, experiment, band, names, params):
        pix = np.tile(np.arange(12, dtype=np.int64), 4)
        ntod = len(pix)
        pointing = PixelPointing(pix, np.zeros(ntod), np.array([0], dtype=np.int64), None,
                                 None, 1, 1, ntod, ntod)
        detector = DetectorTOD(
            name="det", det_idx_fullband=0,
            tod=(2.0 + np.cos(pix) + 0.1 * np.sin(np.arange(ntod))).astype(np.float32),
            pointing=pointing, sampling_rate_hz=1.0, orbital_velocity_m_per_s=np.zeros(3),
            huffman_tree=None, huffman_symbols=None, default_proc_mask=np.ones(12, dtype=bool),
            specific_proc_masks={}, flag_encoded=np.zeros(ntod, dtype=np.int64),
            bad_data_bitmask=1, flag_is_compressed=False, init_scalars=np.array([2., 1., .1, -1.]))
        data = DetectorGroupTOD([ScanTOD([detector], 0.0, band_comm.rank + 1)],
                                 "SimSat", "Band100GHz", 1,
                                 100.0, 60.0, 1.0, 1, "I", NoisePSDOof())
        data.scan_idx_start = band_comm.rank
        data.scan_idx_stop = band_comm.rank + 1
        data.nscans_allranks = band_comm.size
        return data

    monkeypatch.setattr(processing, "read_tods_from_file", read_tods)
    for name, iterations, mode in (("continuous", 3, "new"), ("split", 2, "new"),
                                    ("split", 1, "resume")):
        raw = deepcopy(base)
        raw["gibbs"].update(num_iterations=iterations, start={"mode": mode})
        raw["output"]["dir"] = str(root / name)
        params = params_from_dict(raw)
        if comm.rank == 0:
            paths.create_output_dirs(params.output)
        comm.Barrier()
        assert run_commander4(params, raw) == 0
        # Separate invocations normally have separate MPI queues. Drain the driver's final STOP
        # notification here because this regression executes three invocations in one MPI job.
        if pipeline_mode == "combined" and comm.rank == 1:
            assert comm.recv(source=0) is True
        comm.Barrier()
    if comm.rank == 0:
        for chain in (1, 2):
            filename = f"chain{chain:02d}_iter0003.h5"
            if pipeline_mode == "combined":
                sky_relative_path = Path("chains_compsep") / filename
            else:
                sky_relative_path = Path("initial_state") / f"chain{chain:02d}_iter0000.h5"
            with h5py.File(root / "continuous" / sky_relative_path) as expected:
                with h5py.File(root / "split" / sky_relative_path) as actual:
                    np.testing.assert_allclose(actual["comps/cmb/alms"][:],
                                               expected["comps/cmb/alms"][:], rtol=1e-6, atol=1e-7)
            filename = f"SimSat_Band100GHz_{filename}"
            with h5py.File(root / "continuous/chains_bands" / filename) as expected:
                with h5py.File(root / "split/chains_bands" / filename) as actual:
                    for key in ("abs_gain", "noise_params", "maps/observed_sky"):
                        np.testing.assert_allclose(actual[key][()], expected[key][()],
                                                   rtol=1e-6, atol=1e-7)
