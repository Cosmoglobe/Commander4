import numpy as np
from output import log
import logging
from numpy.typing import NDArray

class ScanTOD:
    def __init__ (self, tod, pix, psi, startTime, scanID, nside, fsamp, orb_dir_vec):
        logger = logging.getLogger(__name__)
        log.logassert_np(tod.ndim==1, "'value' must be a 1D array", logger)
        log.logassert_np(tod.dtype in [np.float64,np.float32], "TOD dtype must be floating type,"
                         f" is {tod.dtype}", logger)
        log.logassert_np(np.max(pix) < 12*nside**2, f"Largest pixel index {np.max(pix)}"
                         f"is too large for the specified NSIDE ({nside}).", logger)
        log.logassert_np(orb_dir_vec.size == 3, "orb_dir_vec must be a vector of size 3.", logger)
        self._tod = tod
        self._pix = pix
        self._psi = psi
        self._startTime = startTime
        self._scanID = scanID
        self._nside = nside
        self._fsamp = fsamp
        self._orb_dir_vec = orb_dir_vec

    @property
    def nsamples(self) -> int:
        return self._tod.shape[0]

    @property
    def startTime(self) -> float:
        return self.startTime

    @property
    def tod(self) -> NDArray[np.floating]:
        return self._tod

    @property
    def pix(self) -> NDArray[np.integer]:
        return self._pix

    @property
    def psi(self) -> NDArray[np.floating]:
        return self._psi

    @property
    def scanID(self) -> int:
        return self._scanID
    
    @property
    def nside(self) -> int:
        return self._nside
    
    @property
    def fsamp(self) -> float:
        return self._fsamp
    
    @property
    def orb_dir_vec(self) -> NDArray[np.floating]:
        return self._orb_dir_vec