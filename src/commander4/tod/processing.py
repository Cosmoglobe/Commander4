"""The TOD-processing side of one Gibbs iteration.

`init_tod_processing` sets up the per-band data and sample containers once, and with them every
sampling step and its settings; `process_tod` runs the sampling steps for a single iteration
(far-sidelobe model, HFI baselines, jumps, glitches, gain, correlated noise, mapmaking, data
selection) and hands the resulting band maps to component separation.
"""
import logging
import os
import time

import numpy as np
from pixell.bunch import Bunch

from commander4.file_io import paths
from commander4.file_io.chain_restart import ChainStart
from commander4.parameters.schema import resolve_param
from commander4.data_models.detector_map import DetectorMap
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.pixel_domain import DistributedMap, PixelDomain
from commander4.tod.samples import TODSamples
from commander4.tod.view import TODView
from commander4.tod.mapmaking.tod2map import tod2map
from commander4.file_io.tod_reader import read_tods_from_file
from commander4.file_io.chain_writer import write_band_chain_to_file
from commander4.mpi.setup import MPIInfo
from commander4.diagnostics.performance import benchmark, bench_summary, bench_reset

logger = logging.getLogger(__name__)


def init_tod_processing(mpi_info: MPIInfo, params: Bunch, start: ChainStart) \
        -> tuple[DetectorGroupTOD, dict[int, TODSamples]]:
    """To be run once before starting TOD processing. Reads this rank's part of its band's TOD, and
    allocates the TODSamples objects, which start from a saved sample where `start` names one for
    this band. Creating them reads and checks the settings of every sampling step, so a bad setting
    fails here.

    Input:
        mpi_info (MPIInfo): Where this rank sits in the run.
        params (Bunch): The full input parameter file.
        start (ChainStart): Where the two chains start (see `file_io.chain_restart`).

    Output:
        experiment_data (DetectorGroupTOD): The TOD data for the band of this process.
        tod_samples_by_chain (dict[int, TODSamples]): Each chain's samples, by chain number.
    """
    my_experiment = params.experiments[mpi_info.tod.experiment_name]
    my_band = my_experiment.bands[mpi_info.tod.band_name]
    band_comm = mpi_info.band.comm

    t0 = time.time()
    experiment_data = read_tods_from_file(band_comm, params, my_experiment, my_band)
    mpi_info.tod.comm.Barrier()
    if mpi_info.tod.rank == 0:
        logger.summary(f"TOD: Finished reading all files in {time.time()-t0:.1f}s.")

    tod_samples_by_chain = {chain: TODSamples(experiment_data, params, my_band, band_comm, chain)
                            for chain in (1, 2)}
    # Continue from a saved sample, if the run starts from one and it holds this band. A band the
    # saved run never wrote keeps its parameter-file start (`resolve_chain_start` warns about it).
    if start.source_dir is not None:
        for tod_samples in tod_samples_by_chain.values():
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
        TODView(experiment_data, tod_samples_by_chain[1]), band_comm,
        "sparse" if sparse_maps else "full", experiment_data.nside)
    return experiment_data, tod_samples_by_chain




def process_tod(mpi_info: MPIInfo, experiment_data: DetectorGroupTOD, tod_samples: TODSamples,
                compsep_output: DistributedMap, params: Bunch,
                iter: int, rng: np.random.Generator) -> dict[str, DetectorMap]:
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
        rng: This rank's random generator for the iteration. Every draw of every step uses it.

    Returns:
        The detector maps for component separation.
    """
    band_comm = mpi_info.band.comm
    tod_comm = mpi_info.tod.comm
    is_master = mpi_info.band.rank == 0
    hfi, jumps, glitches = tod_samples.hfi, tod_samples.jumps, tod_samples.glitches
    gain, far_sidelobes = tod_samples.gain, tod_samples.far_sidelobes
    selection = tod_samples.selection
    noise = tod_samples.noise

    # Run the "start of iteration" cleanup on every TOD step.
    for step in tod_samples.steps:
        step.start_iteration()

    ###############################################################################################
    ### Part 1 : Execute all "stand-alone" TOD sampling steps.                                  ###
    ### These are the ones that can run in isolation from start to finish.                      ###
    ###############################################################################################
    # The far-sidelobe model is built first, since `TODView.get_tod` removes the pickup for every
    # step below, as Commander3 includes `s_sl` in `s_tot`. One set of sidelobe cubes per node,
    # shared by every band rank on it. Freed explicitly at the end of this function; going out of
    # scope does not release an MPI window.
    far_sidelobes_active = far_sidelobes.cfg.enabled and iter >= far_sidelobes.cfg.from_iter
    if far_sidelobes_active:
        far_sidelobes.build(tod_samples, compsep_output.full, mpi_info.band.node_comm)

    # HFI baselines use the previous iteration's gain and sigma0, like Commander3. The first pass
    # also determines whether Python's even or odd sample indices carry the positive half-cycle.
    if hfi.cfg.enabled:
        hfi.sample(tod_samples, compsep_output.local, rng)

    # Jumps are steps of the demodulated TOD, so they are measured once the HFI phase is known, as
    # Commander3 stitches its HFI DC levels after demodulation.
    if jumps.cfg.enabled and iter >= jumps.cfg.from_iter:
        jumps.sample(tod_samples, compsep_output.local, iter)

    # Glitch detection, performed after HFI demodulation.
    if glitches.cfg.enabled and iter >= glitches.cfg.from_iter:
        glitches.sample(tod_samples, compsep_output.local, rng)

    # Gain uses the previous iteration's sigma0. The new sigma0 is estimated later, inside the
    # mapmaker scan loop, matching Commander3's gain -> n_corr -> bin_TOD order.
    if gain.cfg.abs_gain.enabled and iter >= gain.cfg.abs_gain.from_iter:
        gain.sample_absolute(tod_samples, compsep_output.local, iter, rng)

    if gain.cfg.rel_gain.enabled and iter >= gain.cfg.rel_gain.from_iter:
        gain.sample_relative(tod_samples, compsep_output.local, iter, rng)

    if gain.cfg.temporal_gain.enabled and iter >= gain.cfg.temporal_gain.from_iter:
        gain.sample_temporal(tod_samples, compsep_output.local, iter, rng)

    ###############################################################################################
    ### Part 2 : Execute the mapmaking loop, including the sampling steps it contains.          ###
    ### Some sampling steps are inside the mapmaking loop, because they sample a quantity that  ###
    ### is too large to store, or need information only available inside this loop, etc.        ###
    ###############################################################################################
    detmap_dict, maps_to_file = tod2map(experiment_data, tod_samples, compsep_output.local,
                                        iter, params, rng)
    if is_master and params.output.chains.include.sky_model_maps:
        maps_to_file["skymodel"] = compsep_output.full

    ###############################################################################################
    ### Part 3 : Cleanup, logging, and communication                                            ###
    ###############################################################################################
    if noise.cfg.enabled and iter >= noise.cfg.from_iter:
        noise.log_summary(tod_samples, iter)

    # Report during data-selection warm-up as well as active-cut iterations.
    report_selection, apply_cuts = selection.status(iter, noise.cfg)
    if report_selection:
        selection.log_summary(tod_samples, active=apply_cuts, iteration=iter)

    # The samples and the maps go into one band file, written last so it records the final
    # `accept` flags. The gather is collective over band_comm and applies the write gate.
    tod_arrays = tod_samples.gather_chain_arrays(params, iter)
    if is_master and tod_arrays is not None:
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
