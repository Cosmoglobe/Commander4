import numpy as np
from numpy.typing import NDArray
from src.python.data_models.detector_map import DetectorMap


class Band:
    def __init__(self, alms, nu, fwhm, nside:int):
        self._alms = alms
        self._nu = nu
        self._fwhm = fwhm
        self._nside = nside

    @classmethod
    def init_from_detector(cls, det_map:DetectorMap, double_precision:bool=False):
        """
        Initialize a Band class object with the metadata from det_map and an empty set of alms.
        """
        alm_len_complex = ((det_map.lmax+1)*(det_map.lmax+2))//2
        npol = 2 if det_map.pol else 1
        dtype = np.complex128 if double_precision else np.complex64
        return cls(np.zeros((npol, alm_len_complex), dtype=dtype), det_map.nu, det_map.fwhm, det_map.nside)

    @property
    def alms(self):
        return self._alms
    
    @property
    def nu(self):
        return self._nu
    
    @property
    def fwhm(self):
        return self._fwhm
    
    @property
    def nside(self):
        return self._nside

    @alms.setter
    def alms(self, alms):
        if alms.ndim == 2:
            if alms.shape[0] in [1,2]:
                self._alms = alms
            else:
                raise ValueError(f"Trying to set alms with wrong first axis length {alms.shape[0]} != 1 or 2")
        else:
            raise ValueError("Trying to set alms with unexpected number of dimensions"
                                f"{alms.ndim} != 2")
            
    @property
    def pol(self):
        return False if self._alms.shape[0] == 1 else True
    
    @property
    def lmax(self):
        return int((-3 + np.sqrt(1 + self._alms.shape[1] * 8))/2)
