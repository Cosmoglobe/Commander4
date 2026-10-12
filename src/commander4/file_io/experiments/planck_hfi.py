"""TOD reader for modulated Planck HFI flight data (``experiment_id: planck_hfi``)."""
import numpy as np
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.file_io.experiments.base_reader import TODReader
from commander4.tod.noise.psd import NoisePSDOof


class PlanckHFIReader(TODReader):
    """Planck HFI: Huffman-compressed, modulated TODs, with responses from the instrument file.

    The experiment's ``instrument_file`` is required. Each detector's ``polEff`` (in percent) sets
    its polarization response, and unpolarized bolometers get exactly zero (see
    `UNPOLARIZED_POLEFF_CUTOFF` in ``base_reader.py``).
    """
    def __init__(self, band_comm: MPI.Comm, params: Bunch, experiment: Bunch, band: Bunch):
        super().__init__(band_comm, params, experiment, band,
                         pol_eff_from_instrument_file=True,
                         hfi_demodulation=True,
                         # The same priors as LFI.
                         noise_model=NoisePSDOof(
                             P_active_mean=[np.nan, 0.1, -1.0],
                             P_active_rms=[np.nan, np.inf, np.inf],
                             P_uni=[[np.nan, np.nan], [0.01, 0.5], [-2.5, -0.25]],
                             nu_fit=[[np.nan, np.nan], [0, 3.0], [0, 3.0]]))
