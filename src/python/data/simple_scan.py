from .scan import Scan
import numpy as np
from numpy.testing import assert_ as myassert

class SimpleScan(Scan):
    def __init__ (self, value, theta, phi, psi, startTime):
        myassert(value.ndim==1, "'value' must be a 1D array")
        myassert(value.dtype==np.float64, "'value' dtype must be np.float64")
        myassert(theta.shape==value.shape and phi.shape==value.shape and psi.shape==value.shape,
            "shape mismatch between input arrays")
        self._value = value
        self._theta = theta
        self._phi = phi
        self._psi = psi
        self._startTime = startTime

    @property
    def nsamples(self) -> int:
        return self._value.shape[0]

    @property
    def startTime(self) -> float:
        return self.startTime

    @property
    def data(self):
        return self._value, self._theta, self._phi, self._psi