"""TOD reader for AKARI far-infrared survey data (``experiment_id: akari``)."""
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.file_io.experiments.base_reader import TODReader


class AkariReader(TODReader):
    """AKARI: the standard format, intensity-only, with the initial gain stored as is.

    The files carry no orbital velocity, so it is zero.
    """
    def __init__(self, band_comm: MPI.Comm, params: Bunch, experiment: Bunch, band: Bunch):
        super().__init__(band_comm, params, experiment, band, gain_factor=1.0)
