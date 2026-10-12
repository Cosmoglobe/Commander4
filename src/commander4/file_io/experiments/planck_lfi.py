"""TOD reader for Planck LFI flight data (``experiment_id: planck_lfi``)."""
import numpy as np
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.file_io.experiments.base_reader import TODReader
from commander4.tod.noise.psd import NoisePSDOof


class PlanckLFIReader(TODReader):
    """Planck LFI: the standard format, with LFI's noise priors."""
    # TODO: Re-implement the per-detector bandpass shift (C3's `bandpass_shift`).
    def __init__(self, band_comm: MPI.Comm, params: Bunch, experiment: Bunch, band: Bunch):
        super().__init__(band_comm, params, experiment, band,
                         noise_model=NoisePSDOof(
                             P_active_mean=[np.nan, 0.1, -1.0],
                             P_active_rms=[np.nan, np.inf, np.inf],
                             P_uni=[[np.nan, np.nan], [0.01, 0.5], [-2.5, -0.25]],
                             nu_fit=[[np.nan, np.nan], [0, 3.0], [0, 3.0]]))
