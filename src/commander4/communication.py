import numpy as np
import logging
from collections.abc import Mapping
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.data_models.detector_map import DetectorMap
from commander4.data_models.detector_group_TOD import DetGroupTOD
from commander4.maps_from_file import read_data_map_from_file
from commander4.utils.execution_ids import get_execution_band_id
from commander4.sky_models.sky_model import build_initial_sky_model

logger = logging.getLogger(__name__)


def _get_compsep_sender_id_for_tod_band(todproc_my_band_id: str,
                                        senders: Mapping[str, int]) -> str:
    """Return the CompSep execution-view ID that sends the sky model back to a TOD band.

    For bands that have both intensity and QU execution views, the intensity rank sends the
    already reassembled `SkyModel`. QU-only bands fall back to their `_QU` identifier.
    """
    intensity_key = get_execution_band_id(todproc_my_band_id, "I")
    if intensity_key in senders:
        return intensity_key
    pol_key = get_execution_band_id(todproc_my_band_id, "QU")
    if pol_key in senders:
        return pol_key
    raise KeyError(f"No CompSep sender found for TOD band '{todproc_my_band_id}'.")


def _should_send_compsep_result(compsep_my_band_id: str,
                                destinations: Mapping[str, int] | None) -> bool:
    """Return whether this CompSep execution view should send the realized sky model to TOD."""
    if destinations is None or compsep_my_band_id not in destinations:
        return False
    if compsep_my_band_id.endswith("_QU"):
        paired_intensity_id = get_execution_band_id(
            compsep_my_band_id.removesuffix("_QU"),
            "I",
        )
        if paired_intensity_id in destinations:
            return False
    return True

###########################################################
# ON TOD SIDE
###########################################################

# TODO: Communication in this script should be switched from picked (lowercase) to buffered
# (uppercase) whereever possible (e.g. where arrays are communicated).
def receive_compsep(mpi_info: Bunch, experiment_data: DetGroupTOD, todproc_my_band_id: str,
                    senders: dict[str, int]) -> NDArray[np.floating]:
    """Receive the CompSep sky model, realize the local band map, and broadcast it within TOD.

    Args:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        experiment_data (DetGroupTOD): The experiment TOD data, used to determine band frequency,
            resolution, and polarization for evaluating the sky model.
        todproc_my_band_id (str): The string uniquely indentifying the band of this rank
                                  (example: '30GHz').
        senders (dict[str, int]): A dictionary mapping the string in 'todproc_my_band_id' to the
            world rank of the sender task (on the CompSep side), keyed by execution-view band ID.

    Returns:
        NDArray: The sky model realized at the local band frequency and resolution, broadcast to
            all processes on the band communicator.
    """
    world_comm = mpi_info.world.comm
    band_comm = mpi_info.band.comm
    is_band_master = mpi_info.band.is_master
    if is_band_master:
        source_band_id = _get_compsep_sender_id_for_tod_band(todproc_my_band_id, senders)
        sky_model = world_comm.recv(source=senders[source_band_id])
    else:
        sky_model = None
    # Currently all TOD MPI ranks need a copy of the relevant detector map,
    # which is very wasteful - a reason for doing OpenMP for mapmaking.
    sky_model = band_comm.bcast(sky_model, root=0)
    detector_map_arr = sky_model.get_sky_at_nu(experiment_data.nu, experiment_data.nside,
                                experiment_data.pols, fwhm=np.deg2rad(experiment_data.fwhm/60.0))
    return detector_map_arr


def get_local_initial_sky(mpi_info: Bunch, experiment_data: DetGroupTOD,
                          params: Bunch) -> NDArray[np.floating]:
    """Build the initial sky model locally and realize it at this TOD band.

    Used when there are no CompSep ranks: the band master builds the SkyModel from the component
    parameters and init files, broadcasts it within the band communicator, and every rank realizes
    it at the band frequency/resolution. Mirrors `receive_compsep`, minus the cross-world receive.
    """
    if mpi_info.band.is_master:
        sky_model = build_initial_sky_model(params)
    else:
        sky_model = None
    sky_model = mpi_info.band.comm.bcast(sky_model, root=0)
    return sky_model.get_sky_at_nu(experiment_data.nu, experiment_data.nside, experiment_data.pols,
                                   fwhm=np.deg2rad(experiment_data.fwhm/60.0))


def send_tod(mpi_info: Bunch, tod_map_dict: dict[DetectorMap], todproc_my_band_id: str,
             receivers: Bunch) -> None:
    """ MPI-send the results from a single band TOD processing to a task on the CompSep side
        (used in conjunction with receive_tod).

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        tod_map_dict (dict[str, DetectorMap]): The output maps keyed by polarization component
            (e.g. 'I', 'QU') from process_tod for the band belonging to this process.
        todproc_my_band_id (str): The string uniquely indentifying the band of this rank,
                                  regardless of polarization (example: '30GHz').
        receivers (Bunch): Maps a band identifier to the band master on the compsep side.
    """
    if mpi_info.tod.is_master:
        logger.info(f"Compsep band masters: {mpi_info.world.compsep_band_masters}")
    if mpi_info.band.is_master:
        for pol, detector_map in tod_map_dict.items():
            target_band = get_execution_band_id(todproc_my_band_id, pol)
            if target_band in receivers.keys():
                mpi_info.world.comm.send(detector_map, dest=receivers[target_band])
            else:
                logger.info(f"Pol-{pol} TOD-processing result discarded, "\
                            f"as band {todproc_my_band_id} does not require it on compsep side.")


###########################################################
# ON COMPSEP SIDE
###########################################################

def receive_tod(mpi_info: Bunch, senders: dict[str,int], my_band: Bunch, compsep_band_id: str,
                curr_tod_output: DetectorMap|None) -> DetectorMap:
    """ MPI-receive the results from the TOD processing (used in conjunction with send_tod).

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        senders: (dict[str->int]): A dictionary mapping a string uniquely identifying each
                 band to the world rank of the sender task (on the CompSep side).
        my_band (Bunch): The section of the parameter file corresponding to this CompSep band,
                 as a "Bunch" type, it also has an 'identifier' field. 
        compsep_band_id (str): The string uniquely indentifying the band+pol of this
                 rank (example: '30GHz_I').
        curr_tod_output (DetectorMap): The current map output from the TOD process.
                 Should be None unless map is read from file already in a previous iteration.

    Returns:
        data (list of DetectorMaps): nbands (DetectorMap)
    """
    my_compsep_rank = mpi_info.compsep.rank
    if my_band.get_from == "file":
        if curr_tod_output is None:
            logger.info(f"CompSep: Rank {my_compsep_rank} reading static map data from file.")
            curr_tod_output = read_data_map_from_file(my_band)
        else:
            logger.info(f"CompSep: Rank {my_compsep_rank} already has static map data. Continuing.")
    else:
        logger.info(f"CompSep: Rank {my_compsep_rank} receiving TOD data for ({compsep_band_id}) "\
                    f" from TOD process with global rank {senders[compsep_band_id]}")
        curr_tod_output = mpi_info.world.comm.recv(source=senders[compsep_band_id])
    
    
    return curr_tod_output

def send_compsep(mpi_info: Bunch, compsep_my_band_id: str, sky_model,
                 destinations: dict[str, int]|None) -> None:
    """Send the CompSep sky model back to TOD when this execution view owns the return path.

    For split IQU bands, the QU master first transfers its alms to the I master inside the CompSep
    communicator. Only the matching `_I` execution view then sends the fully reassembled sky model
    back to TOD. Pure QU bands send directly from their `_QU` execution view.

    Args:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        compsep_my_band_id (str): The string uniquely indentifying the band+pol of this
                      rank (example: '30GHz_I').
        sky_model: The realized sky model object for this Gibbs sample.
        destinations (dict[str->int]): A dictionary mapping the string in 'compsep_my_band_id' to
                                       the world rank of the destination task (on the TOD side)
                                       (This is the same as is found in mpi_info)
    """
    if _should_send_compsep_result(compsep_my_band_id, destinations):
        mpi_info.world.comm.send(sky_model, dest=destinations[compsep_my_band_id])
