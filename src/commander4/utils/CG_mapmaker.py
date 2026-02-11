import numpy as np
import ctypes as ct
from mpi4py import MPI
import logging
from numpy.typing import NDArray

from commander4.output.log import logassert
from commander4.utils.ctypes_lib import load_cmdr4_ctypes_lib
from commander4.data_models.detector_TOD import DetectorTOD
from commander4.data_models.detector_samples import DetectorSamples
from commander4.utils.math_operations import inplace_scale

# current_dir_path = os.path.dirname(os.path.realpath(__file__))
# src_dir_path = os.path.abspath(os.path.join(os.path.join(current_dir_path, os.pardir), os.pardir))



# I need to CG-solve P^T T^T N^−1 T P m = P^T T^T N^-1 d
# Where:
# - m is the final map [npix]
# - P is the pointing matrix: [ntod, npix]
# - T is the bolometer trnasfer function operator (check Artem's code)
# - N^-1 is the inverse noise covariance matrix (inv_var in tod_processing.tod2map) which is diagonal in tod space [ntod]
# - d is the calibrated TODs [ntod].
# Notes: T is non-local so each rank must hold all the scans, but only for one detector.


class CG_Mapmaker:
    """
    """
    def __init__(self, detector_tod:DetectorTOD, detector_samples:DetectorSamples, nside:int):
        self.detector_tod = detector_tod
        self.detector_samples = detector_samples
        self.nside = nside


    def compute_RHS(self):
        """
        Compute the RHS of the mapmaking problem:
        P^T T^T N^-1 d
        """
        return NotImplementedError
    
    def apply_N_inv(self):
        """
        Applies N^-1 operator
        """
        for scan, scan_samples in zip(self.detector_tod.scans, self.detector_samples.scans):
            inplace_scale(scan, 1/scan_samples.sigma0**2)
        return self.detector_tod

    def apply_P(self):
        return NotImplementedError

    def apply_P_transpose(self):
        return NotImplementedError


    



