import numpy as np
from output import log
import logging
from numpy.testing import assert_ as myassert

class ScanTOD:
    def __init__ (self, value, theta, phi, psi, startTime, scanID):
        logger = logging.getLogger(__name__)
        log.logassert_np(value.ndim==1, "'value' must be a 1D array", logger)
        log.logassert_np(value.dtype==np.float64, "'value' dtype must be np.float64", logger)
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
