import numpy as np
import logging
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.data_models.detector_map import DetectorMap
from commander4.data_models.detector_TOD import DetectorTOD
from commander4.maps_from_file import read_data_map_from_file
from commander4.output import log


### ON TOD SIDE

# TODO: Communication in this script should be switched from picked (lowercase) to buffered
# (uppercase) whereever possible (e.g. where arrays are communicated).
def receive_compsep(mpi_info: Bunch, experiment_data: DetectorTOD, todproc_my_band_id: str,
                    senders: dict[str, int]) -> NDArray[np.floating]:
    """ MPI-receive the results from compsep (used in conjunction with send_compsep).

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        todproc_my_band_id (str): The string uniquely indentifying the experiment+band of this rank
                                  (example: 'PlanckLFI$$$30GHz').
        senders (dict[str->int]): A dictionary mapping the string in 'todproc_my_band_id' to the
                                  world rank of the senbder task (on the CompSep side).

    Returns:
        detector_map_arr (np.array): The detector map of a single band, distributed to all processes
                                 belonging to the band communicator.
    """
    world_comm = mpi_info.world.comm
    band_comm = mpi_info.band.comm
    is_band_master = mpi_info.band.is_master
    if is_band_master:
        # As only the compsep rank holding I will send everything at once.
        sky_model = world_comm.recv(source=senders[todproc_my_band_id+'_I'])
    else:
        sky_model = None
    # Currently all TOD MPI ranks need a copy of the relevant detector map,
    # which is very wasteful - a reason for doing OpenMP for mapmaking.
    sky_model = band_comm.bcast(sky_model, root=0)
    detector_map_arr = sky_model.get_sky_at_nu(experiment_data.nu, experiment_data.nside,
                                experiment_data.pols, fwhm=np.deg2rad(experiment_data.fwhm/60.0))
    return detector_map_arr


def send_tod(mpi_info: Bunch, tod_map_dict: dict[DetectorMap], todproc_my_band_id: str,
             receivers: Bunch) -> None:
    """ MPI-send the results from a single band TOD processing to a task on the CompSep side
        (used in conjunction with receive_tod).

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        tod_map_list (list[DetectorMap]): The output maps ([I, QU]) from process_tod for the band
                                          belonging to this process.
        todproc_my_band_id (str): The string uniquely indentifying the experiment+band of this rank,
                                  regardless of polarization (example: 'PlanckLFI$$$30GHz').
        receivers (Bunch): Maps a band identifier to the band master on the compsep side.
    """
    logger = logging.getLogger(__name__)
    if mpi_info.tod.is_master:
        logger.info(f"Compsep band masters: {mpi_info.world.compsep_band_masters}")
    if mpi_info.band.is_master:
        # my_pol = mpi_info.world.tod_band_pols[todproc_my_band_id]
        #I
        my_todproc_pols = tod_map_dict.keys()
        for pol in my_todproc_pols:
            target_band = todproc_my_band_id+'_'+pol
            if target_band in receivers.keys(): #Compsep wants an I pol for todproc_my_band_id
                mpi_info.world.comm.send(tod_map_dict[pol], dest=receivers[target_band])
            else:
                logger.info(f"Pol-{pol} TOD-processing result discarded, "\
                            f"as band {todproc_my_band_id} does not require it on compsep side.")

        # target_band = todproc_my_band_id+'_I'
        # if target_band in receivers.keys(): #Compsep wants an I pol for todproc_my_band_id
        #     log.logassert("I" in my_todproc_pols, 
        #                   "Polarization 'I' requested from compsep but not found in TOD processing",
        #                   logger)
        #     mpi_info.world.comm.send(tod_map_dict["I"], dest=receivers[target_band])
        # else:
        #     logger.info(f"Intensity TOD-processing result discarded, as band {todproc_my_band_id} "\
        #                 "is only defined for QU.")
        # #QU
        # target_band = todproc_my_band_id+'_QU'
        # if target_band in receivers.keys(): #Compsep wants a QU pol for todproc_my_band_id
        #     log.logassert("QU" in my_pol, 
        #                   "Polarization 'QU' requested from compsep but not found in TOD processing",
        #                   logger)
        #     mpi_info.world.comm.send(tod_map_dict["QU"], dest=receivers[target_band])
        # else:
        #     logger.info(f"QU TOD-processing result discarded, as band {todproc_my_band_id} is "\
        #                 "only defined for Intensity.")


### ON COMPSEP SIDE

def receive_tod(mpi_info: Bunch, senders: dict[str,int], my_band: Bunch, compsep_band_id: str,
                curr_tod_output: DetectorMap|None) -> DetectorMap:
    """ MPI-receive the results from the TOD processing (used in conjunction with send_tod).

    Input:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        senders: (dict[str->int]): A dictionary mapping a string uniquely identifying each
                 experiment+band to the world rank of the sender task (on the CompSep side).
        my_band (Bunch): The section of the parameter file corresponding to this CompSep band,
                 as a "Bunch" type, it also has an 'identifier' field. 
        compsep_band_id (str): The string uniquely indentifying the experiment+band+pol of this
                 rank (example: 'PlanckLFI$$$30GHz_I').
        curr_tod_output (DetectorMap): The current map output from the TOD process.
                 Should be None unless map is read from file already in a previous iteration.

    Returns:
        data (list of DetectorMaps): nbands (DetectorMap)
    """
    logger = logging.getLogger(__name__)
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

def send_compsep(mpi_info: Bunch, compsep_my_band_id: str, band_sky_map: NDArray[np.floating],
                 destinations: dict[str, int]|None) -> None:
    """ MPI-send the results from compsep to a destinations on the TOD processing side
        (used in conjunction with receive_compsep). Assumes the COMM_WORLD communicator.

    Args:
        mpi_info (Bunch): The data structure containing all MPI relevant data.
        compsep_my_band_id (str): The string uniquely indentifying the experiment+band+pol of this
                                  rank (example: 'PlanckLFI$$$30GHz_I').
        band_sky_map (np.array[float]): A sky realization at a given band frequency.
        destinations (dict[str->int]): A dictionary mapping the string in 'compsep_my_band_id' to
                                       the world rank of the destination task (on the TOD side)
                                       (This is the same as is found in mpi_info)
    """
    if destinations is not None:
        # If the band our rank is holding is not in "destinations", it means it did not come from
        # TOD-processing, and should not be sent back either.
        if compsep_my_band_id in destinations:
            # Currently in the end of compsep_processing the polarization ranks send over to the
            # Intensity ones which then in turn broadcast and evaluate the sky.
            if compsep_my_band_id.endswith('_I'):
                mpi_info.world.comm.send(band_sky_map, dest=destinations[compsep_my_band_id])
