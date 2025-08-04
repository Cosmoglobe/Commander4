import numpy as np
from output import log
import logging
from numpy.testing import assert_ as myassert

class ScanTOD:
    def __init__ (self, value, theta, phi, psi, startTime, scanID):
        logger = logging.getLogger(__name__)
        log.logassert_np(value.ndim==1, "'value' must be a 1D array", logger)
        log.logassert_np(value.dtype in [np.float64,np.float32], "'value' dtype must be np.float64", logger)
        log.logassert_np(theta.shape==value.shape and phi.shape==value.shape and psi.shape==value.shape,
            f"shape mismatch between input arrays. Theta: {theta.shape}, phi: {phi.shape}, psi: {psi.shape}, value: {value.shape}", logger)
        self._value = value
        self._theta = theta
        self._phi = phi
        self._psi = psi
        self._startTime = startTime
        self._scanID = scanID

    @property
    def nsamples(self) -> int:
        return self._value.shape[0]

    @property
    def startTime(self) -> float:
        return self.startTime

    @property
    def data(self):
        return self._value, self._theta, self._phi, self._psi

    @property
    def scanID(self):
        return self._scanID

    # @property
    # def sky_subtracted_tod(self):
    #     if self._component_map is None:
    #         raise ValueError("sky_subtracted_tod property not set. Gibbs iter = {self._sky_subtracted_tod_Gibbs_iter}")
    #     return self._component_map

    # @sky_subtracted_tod.setter
    # def sky_subtracted_tod(self, tod):
    #     if not self._component_map is None:  # Temporarily disabled, might want to add back later.
    #         raise ValueError("DiffuseComponent does not allow for overwriting already set component_map parameter.")
    #     self._component_map = map
