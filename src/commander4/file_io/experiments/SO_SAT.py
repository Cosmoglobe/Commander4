"""TOD reader for the Simons Observatory Small Aperture Telescopes (``experiment_id: SO_SAT``)."""
import numpy as np
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.file_io.experiments.base_reader import TODReader
from commander4.tod.noise.psd import NoisePSDOof


class SOSATReader(TODReader):
    """SO SAT: one boresight path per scan, plus each detector's focal-plane offset.

    `ScanBoresightPointing` rebuilds each detector's pixels and angles from these. The LAT files
    instead carry each detector's pointing directly. The files store the initial gain as is.
    """
    def __init__(self, band_comm: MPI.Comm, experiment: Bunch, band: Bunch, det_names: list[str],
                 params: Bunch):
        super().__init__(band_comm, experiment, band, det_names, params,
                         gain_factor=1.0,
                         boresight_pointing=True,
                         noise_model=NoisePSDOof(P_active_mean=[np.nan, 0.1, -5.0],
                                                 P_uni=[[np.nan, np.nan],  # sigma0
                                                        [0.01, 100.0],     # fknee
                                                        [-5.0, 0.0]]))     # alpha
