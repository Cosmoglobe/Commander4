"""TOD reader for the Simons Observatory Large Aperture Telescope (``experiment_id: SO_LAT``)."""
import numpy as np
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.file_io.experiments.base_reader import TODReader
from commander4.tod.noise.psd import NoisePSDOof


class SOLATReader(TODReader):
    """SO LAT: the standard format, with each detector's pointing Huffman-compressed.

    The files store the initial gain as is.
    """
    def __init__(self, band_comm: MPI.Comm, params: Bunch, experiment: Bunch, band: Bunch):
        super().__init__(band_comm, params, experiment, band,
                         gain_factor=1.0,
                         noise_model=NoisePSDOof(
                             P_active_mean=[np.nan, 10.0, -2.7],
                             P_active_rms=[np.nan, np.inf, np.inf],
                             P_uni=[[np.nan, np.nan], [0.03, 40.0], [-4.0, -2.0]],
                             nu_fit=[[np.nan, np.nan], [0, 10.0], [0, 10.0]]))
