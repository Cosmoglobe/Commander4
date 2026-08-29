"""The TOD-processing side of one Gibbs iteration, and the settings that configure it.

`init_tod_processing` sets up the per-band data and sample containers once; `process_tod` runs the
sampling steps for a single iteration (gain, jumps, correlated noise, mapmaking, data selection) and
hands the resulting band maps to component separation. The `*Config` dataclasses here validate the
`tod_processing` block of the parameter file.
"""
import numpy as np
import pixell
from pixell import utils
from mpi4py import MPI
import logging
from scipy.fft import rfftfreq
import time
from dataclasses import dataclass, field, fields
from typing import ClassVar, Self
from numpy.typing import NDArray

from pixell.bunch import Bunch

from commander4.parameters.schema import resolve_param, resolve_band_lmax, split_integer_range
from commander4.data_models.detector_map import DetectorMap
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.tod_samples import TODSamples
from commander4.tod.data_selection import log_dataselect_summary, DataSelectionConfig
from commander4.tod.gain import sample_absolute_gain, sample_relative_gain,\
    sample_temporal_gain_variations, GainConfig
from commander4.tod.jumps import sample_jump_detection, JumpDetectionConfig
from commander4.tod.view import TODView
from commander4.tod.mapmaking.binned import tod2map_bin
from commander4.tod.mapmaking.cg import tod2map_CG
from commander4.tod.mapmaking.config import MapmakingConfig
from commander4.tod.noise.sample_ncorr import CorrelatedNoiseConfig
from commander4.polarization import get_execution_band_ids
from commander4.file_io.tod_reader import read_tods_from_file
from commander4.file_io.chain_writer import write_band_chain_to_file
from commander4.diagnostics.performance import benchmark, bench_summary, bench_reset

logger = logging.getLogger(__name__)


def init_tod_processing(mpi_info: Bunch, params: Bunch) -> tuple[Bunch, str, DetectorGroupTOD,
                                                                 TODSamples, TODSamples]:
    """To be run once before starting TOD processing. Determines what data this rank is responsible
    for, reads other relevant parameter file data, and allocates the TODSamples objects, 

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        params (Bunch): The full input parameter file.

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
    det_names = list(my_band.detectors)

    total_scans = int(resolve_param(params, "num_scans",
                                    (f"experiments.{experiment_name}.bands.{my_band_name}",
                                     f"experiments.{experiment_name}")))
    my_scans_start, my_scans_stop = split_integer_range(total_scans, mpi_info.band.size,
                                                        mpi_info.band.rank)
    mpi_info.tod.comm.Barrier()

    time.sleep(mpi_info.tod.rank*1e-5)  # Small sleep to get prints in nice order.
    band_comm = mpi_info.band.comm
    logger.debug(
        f"TOD-rank {mpi_info.tod.rank:4} (on machine {mpi_info.processor_name}) handles band "
        f"{mpi_info.band.index:4}, local rank {mpi_info.band.rank:4}/{mpi_info.band.size:4}."
    )

    t0 = time.time()
    with benchmark("fileread-tod"):
        experiment_data = read_tods_from_file(band_comm, my_experiment, my_band, det_names, params,
                                              my_scans_start, my_scans_stop)
    mpi_info.tod.comm.Barrier()
    if mpi_info.tod.is_master:
        logger.summary(f"TOD: Finished reading all files in {time.time()-t0:.1f}s.")

    tod_samples_chain1 = TODSamples(experiment_data, params, my_band, band_comm, 1)
    tod_samples_chain2 = TODSamples(experiment_data, params, my_band, band_comm, 2)

    # Build the band's map-distribution PixelDomain once now: the pointing is static, so it is
    # reused across Gibbs iterations by both the mapmakers and the sky-model distribution (which
    # needs it to give each rank only its local pixels). In full mode this is a cheap no-op.
    sparse_maps = resolve_param(params, "sparse_maps",
                                (f"experiments.{experiment_data.experiment_name}",),
                                default=MapmakingConfig.sparse_maps, legal_types=bool)
    experiment_data.get_pixel_domain(TODView(experiment_data, tod_samples_chain1), band_comm,
                                     sparse_maps)

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




def process_tod(mpi_info: Bunch, experiment_data: DetectorGroupTOD,
                tod_samples: TODSamples, compsep_output: NDArray,
                params: Bunch, chain: int, iter: int) -> tuple[dict[str, DetectorMap], TODSamples]:
    """Run one TOD iteration for one band.

    The function states the scientific order directly. Correlated noise, sigma0, diagnostics, and
    data-selection vetoes run inside the selected mapmaker because they operate on one focused
    detector-scan and must occur at exact positions relative to map accumulation.

    Returns:
        The detector maps for component separation and the updated TOD samples.
    """
    band_comm = mpi_info.band.comm
    tod_comm = mpi_info.tod.comm
    is_master = mpi_info.band.is_master

    # Resolve every config before executing any step, so invalid later settings fail before an
    # earlier sampler changes the chain state. Each class owns its own unpacking and validation.
    mapmaking = MapmakingConfig.from_params(params, experiment_data)
    jump_detection = JumpDetectionConfig.from_params(params, experiment_data)
    absolute_gain = GainConfig.from_params(
        params, experiment_data, "abs_gain", "orbital_dipole", iter, is_master,
    )
    relative_gain = GainConfig.from_params(
        params, experiment_data, "rel_gain", "sky", iter, is_master,
    )
    temporal_gain = GainConfig.from_params(
        params, experiment_data, "temporal_gain", "sky", iter, is_master,
    )
    correlated_noise = CorrelatedNoiseConfig.from_params(params, is_master)
    data_selection = DataSelectionConfig.from_params(params)

    # Jump corrections must be stored before any later step reads the TOD.
    if jump_detection.is_active(iter):
        with benchmark("jump-detect"):
            tod_samples = sample_jump_detection(band_comm, experiment_data, tod_samples,
                                                jump_detection, iter)

    # Gain uses the previous iteration's sigma0. The new sigma0 is estimated later, inside the
    # mapmaker scan loop, matching Commander3's gain -> n_corr -> bin_TOD order.
    if absolute_gain.is_active(iter):
        with benchmark("abs-gain"):
            tod_samples = sample_absolute_gain(band_comm, experiment_data, tod_samples,
                                               compsep_output, absolute_gain, iter)

    if relative_gain.is_active(iter):
        with benchmark("rel-gain"):
            tod_samples = sample_relative_gain(band_comm, experiment_data, tod_samples,
                                               compsep_output, relative_gain, iter)

    if temporal_gain.is_active(iter):
        with benchmark("temp-gain"):
            tod_samples = sample_temporal_gain_variations(band_comm, experiment_data, tod_samples,
                                                          compsep_output, temporal_gain, iter)

    # A finite diagnostic means "evaluated this iteration". Previously rejected or absent scans
    # remain NaN and are not counted again by the data-selection summary.
    tod_samples.chisq_z[:] = np.nan
    tod_samples.good_fraction[:] = np.nan

    with benchmark("mapmaker"):
        if mapmaking.mapmaker == "CG":
            detmap_dict, maps_to_file = tod2map_CG(
                band_comm, experiment_data, compsep_output, tod_samples, iter, mapmaking,
                correlated_noise, data_selection,
            )
        else:
            detmap_dict, maps_to_file = tod2map_bin(
                band_comm, experiment_data, compsep_output, tod_samples, iter, mapmaking,
                correlated_noise, data_selection,
            )

    # Report during data-selection warm-up as well as active-cut iterations.
    if data_selection.is_available(iter, correlated_noise):
        log_dataselect_summary(
            band_comm, tod_samples, data_selection,
            active=data_selection.cuts_are_active(iter, correlated_noise),
            iteration=iter,
        )

    # The per-scan samples and the output maps go into one band file, written last so it records
    # the final `accept` flags. The gather is collective over band_comm and also owns the write
    # gate; the writer retains the full parameter tree because it is serialized into file metadata,
    # while the numerical mapmakers only receive the values in their specific configs.
    with benchmark("chain-gather"):
        tod_arrays = tod_samples.gather_chain_arrays(iter)
    if is_master and tod_arrays is not None:
        with benchmark("filewrite-band"):
            write_band_chain_to_file(params, chain, iter, experiment_data.experiment_name,
                                     experiment_data.band_name, tod_arrays, maps_to_file,
                                     tod_samples.band_unit_factor, tod_samples.band_unit)

    with benchmark("end-barrier"):
        tod_comm.Barrier()

    bench_summary(tod_comm, label="All bands")
    bench_summary(band_comm, label=f"Band {experiment_data.band_name}")
    bench_reset()

    return detmap_dict, tod_samples
