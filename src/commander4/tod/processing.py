"""The TOD-processing side of one Gibbs iteration.

`init_tod_processing` sets up the per-band data and sample containers once, and with them every
sampling step and its settings; `process_tod` runs the sampling steps for a single iteration
(HFI baselines, jumps, glitches, gain, far sidelobes, correlated noise, mapmaking, data selection)
and hands the resulting band maps to component separation.
"""
import logging
import os
import time

from pixell.bunch import Bunch

from commander4.file_io import paths
from commander4.file_io.chain_restart import ChainStart
from commander4.parameters.schema import resolve_param
from commander4.data_models.detector_map import DetectorMap
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.pixel_domain import DistributedMap, PixelDomain
from commander4.data_models.tod_samples import TODSamples
from commander4.tod.view import TODView
from commander4.tod.mapmaking.tod2map import tod2map
from commander4.polarization import get_execution_band_ids
from commander4.file_io.tod_reader import read_tods_from_file
from commander4.file_io.chain_writer import write_band_chain_to_file
from commander4.diagnostics.performance import benchmark, bench_summary, bench_reset

logger = logging.getLogger(__name__)


def init_tod_processing(mpi_info: Bunch, params: Bunch, start: ChainStart) \
        -> tuple[Bunch, str, DetectorGroupTOD, TODSamples, TODSamples]:
    """To be run once before starting TOD processing. Determines what data this rank is responsible
    for, reads other relevant parameter file data, and allocates the TODSamples objects, which
    start from a saved sample where `start` names one for this band. Creating them reads and checks
    the settings of every sampling step, so a bad setting fails here.

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        params (Bunch): The full input parameter file.
        start (ChainStart): Where the two chains start (see `file_io.chain_restart`).

    Output:
        mpi_info (Bunch): The data structure containing all MPI relevant data,
            now also with a 'tod' section as well as the dictionary of band
            master mappings.
        todproc_my_band_id (str): Unique string identifier for the experiment+band this process is
          responsible for, regardless of polarization.
        experiment_data (DetectorGroupTOD): The TOD data for the band of this process.
    """

    experiment_name = mpi_info.experiment.name
    my_band_name = mpi_info.band.name
    my_experiment = params.experiments[experiment_name]
    my_band = my_experiment.bands[my_band_name]
    my_band_pol = my_band.polarization

    mpi_info.tod.comm.Barrier()

    time.sleep(mpi_info.tod.rank*1e-5)  # Small sleep to get prints in nice order.
    band_comm = mpi_info.band.comm
    logger.debug(
        f"TOD-rank {mpi_info.tod.rank:4} (on machine {mpi_info.processor_name}) handles band "
        f"{mpi_info.band.index:4}, local rank {mpi_info.band.rank:4}/{mpi_info.band.size:4}."
    )

    t0 = time.time()
    with benchmark("fileread-tod"):
        experiment_data = read_tods_from_file(band_comm, params, my_experiment, my_band)
    mpi_info.tod.comm.Barrier()
    if mpi_info.tod.is_master:
        logger.summary(f"TOD: Finished reading all files in {time.time()-t0:.1f}s.")

    tod_samples_chain1 = TODSamples(experiment_data, params, my_band, band_comm, 1)
    tod_samples_chain2 = TODSamples(experiment_data, params, my_band, band_comm, 2)
    # Continue from a saved sample, if the run starts from one and it holds this band. A band the
    # saved run never wrote keeps its parameter-file start (`resolve_chain_start` warns about it).
    if start.source_dir is not None:
        for tod_samples in (tod_samples_chain1, tod_samples_chain2):
            chain_file = paths.band_chain_file(
                start.source_dir, experiment_data.experiment_name, experiment_data.band_name,
                start.source_chain[tod_samples.chain], start.source_iteration)
            if os.path.exists(chain_file):
                tod_samples.read_chain(chain_file)

    # Build the band's map-distribution PixelDomain once, here:
    # If we're using dense per-rank maps nothing of note is done here.
    # If we're using sparse maps we here pre-calculate the global->local pixel hash table. This
    # table needs only the pixels hit, so it can be calculated prior to the main Gibbs loop.
    sparse_maps = resolve_param(params, "sparse_maps",
                                (f"experiments.{experiment_data.experiment_name}",),
                                default=True, legal_types=bool)
    experiment_data.pixel_domain = PixelDomain.from_view(
        TODView(experiment_data, tod_samples_chain1), band_comm,
        "sparse" if sparse_maps else "full", experiment_data.nside)

    # Creating "tod_band_masters", an array which maps the band index to the rank of the master
    # of that band.
    todproc_my_band_id = my_band_name
    data_world = (
        todproc_my_band_id,
        mpi_info.world.rank,
        my_band_pol,
    ) if mpi_info.band.is_master else None
    data_tod = (todproc_my_band_id, mpi_info.tod.rank) if mpi_info.band.is_master else None
    all_data_world = mpi_info.tod.comm.allgather(data_world)
    all_data_tod = mpi_info.tod.comm.allgather(data_tod)

    world_band_masters_dict = {
        execution_band_id: item[1]
        for item in all_data_world if item is not None
        for execution_band_id in get_execution_band_ids(item[0], item[2])
    }
    tod_band_masters_dict = {item[0]: item[1] for item in all_data_tod if item is not None}
    mpi_info['world']['tod_band_masters'] = world_band_masters_dict
    mpi_info['tod']['tod_band_masters'] = tod_band_masters_dict

    return mpi_info, todproc_my_band_id, experiment_data, tod_samples_chain1, tod_samples_chain2




def process_tod(mpi_info: Bunch, experiment_data: DetectorGroupTOD, tod_samples: TODSamples,
                compsep_output: DistributedMap, params: Bunch,
                iter: int) -> dict[str, DetectorMap]:
    """Run one TOD iteration for one band and chain, updating `tod_samples` in place.

    The function states the scientific order directly. Each step decides from its own settings
    (``step.cfg``) whether it runs in this iteration. Correlated noise, sigma0, diagnostics, and
    data-selection vetoes run inside the mapmaking scan loop (`tod2map`) because they operate on
    one focused detector-scan and must occur at exact positions relative to map accumulation.

    Args:
        experiment_data: The band's TOD.
        tod_samples: The chain's state of the band.
        compsep_output: The band's realized sky model. The samplers and mapmakers read it along
            the pointing, so they get its local form; the far-sidelobe transform and the chain
            output need the whole sky, so they get the master's full-sky form.

    Returns:
        The detector maps for component separation.
    """
    band_comm = mpi_info.band.comm
    tod_comm = mpi_info.tod.comm
    is_master = mpi_info.band.is_master
    hfi, jumps, glitches = tod_samples.hfi, tod_samples.jumps, tod_samples.glitches
    gain, far_sidelobes = tod_samples.gain, tod_samples.far_sidelobes
    selection = tod_samples.selection

    # HFI baselines use the previous iteration's gain and sigma0, like Commander3. The first pass
    # also determines whether Python's even or odd sample indices carry the positive half-cycle.
    if hfi.cfg.enabled:
        with benchmark("hfi-baselines"):
            hfi.sample(tod_samples, compsep_output.local)

    # Jumps are steps of the demodulated TOD, so they are measured once the HFI phase is known, as
    # Commander3 stitches its HFI DC levels after demodulation.
    if jumps.cfg.enabled and iter >= jumps.cfg.from_iter:
        with benchmark("jump-detect"):
            jumps.sample(tod_samples, compsep_output.local, iter)

    # Glitch detection, performed after HFI demodulation.
    if glitches.cfg.enabled and iter >= glitches.cfg.from_iter:
        with benchmark("glitches"):
            glitches.sample(tod_samples, compsep_output.local)

    # Gain uses the previous iteration's sigma0. The new sigma0 is estimated later, inside the
    # mapmaker scan loop, matching Commander3's gain -> n_corr -> bin_TOD order.
    if gain.cfg.abs_gain.enabled and iter >= gain.cfg.abs_gain.from_iter:
        with benchmark("abs-gain"):
            gain.sample_absolute(tod_samples, compsep_output.local, iter)

    if gain.cfg.rel_gain.enabled and iter >= gain.cfg.rel_gain.from_iter:
        with benchmark("rel-gain"):
            gain.sample_relative(tod_samples, compsep_output.local, iter)

    if gain.cfg.temporal_gain.enabled and iter >= gain.cfg.temporal_gain.from_iter:
        with benchmark("temp-gain"):
            gain.sample_temporal(tod_samples, compsep_output.local, iter)


    # One set of sidelobe cubes per node, shared by every band rank on it, and used by the scan
    # loop of `tod2map`. Freed explicitly at the end of this function; going out of scope does not
    # release an MPI window.
    far_sidelobes_active = far_sidelobes.cfg.enabled and iter >= far_sidelobes.cfg.from_iter
    if far_sidelobes_active:
        with benchmark("far-beam"):
            far_sidelobes.build(tod_samples, compsep_output.full, mpi_info.band.node_comm)


    # What the scan loop below records is rebuilt from scratch every iteration.
    for step in tod_samples.steps:
        step.start_iteration()

    with benchmark("mapmaker"):
        detmap_dict, maps_to_file = tod2map(experiment_data, tod_samples, compsep_output.local,
                                            iter, params)
    if is_master and params.output.chains.include.sky_model_maps:
        maps_to_file["skymodel"] = compsep_output.full

    noise = tod_samples.noise
    if noise.cfg.enabled and iter >= noise.cfg.from_iter:
        with benchmark("log-corr-noise-stats"):
            noise.log_summary(tod_samples, iter)

    # Report during data-selection warm-up as well as active-cut iterations.
    report_selection, apply_cuts = selection.status(iter, noise.cfg)
    if report_selection:
        selection.log_summary(tod_samples, active=apply_cuts, iteration=iter)

    # The samples and the maps go into one band file, written last so it records the final
    # `accept` flags. The gather is collective over band_comm and applies the write gate.
    with benchmark("chain-gather"):
        tod_arrays = tod_samples.gather_chain_arrays(params, iter)
    if is_master and tod_arrays is not None:
        with benchmark("filewrite-band"):
            write_band_chain_to_file(params, tod_samples.chain, iter,
                                     experiment_data.experiment_name, experiment_data.band_name,
                                     tod_arrays, maps_to_file, experiment_data.band_unit_factor)

    with benchmark("end-barrier"):
        tod_comm.Barrier()

    # Free the shared memory of the sidelobe cubes. Every rank of the band evaluates
    # `far_sidelobes_active` identically, so this collective is taken by all of them or by none.
    if far_sidelobes_active:
        far_sidelobes.free()

    bench_summary(tod_comm, label="All bands")
    bench_summary(band_comm, label=f"Band {experiment_data.band_name}")
    bench_reset()
    tod_comm.Barrier()

    return detmap_dict
