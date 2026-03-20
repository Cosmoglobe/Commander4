import numpy as np
from numpy.typing import NDArray
from commander4.data_models.detector_map import DetectorMap


class Band:
    """Holds spherical-harmonic coefficients (alms) and metadata for a frequency band.

    The ``alms`` are stored behind a validated property (with setter), while
    ``nu``, ``fwhm``, and ``nside`` are plain public attributes.

    Attributes:
        nu (float): Band centre frequency in GHz.
        fwhm (float): Beam FWHM in arcminutes.
        nside (int): HEALPix nside associated with this band.
    """
    def __init__(self, alms: NDArray[np.complexfloating], nu: float, fwhm: float, nside: int):
        """Construct a Band.

        Args:
            alms: Complex spherical-harmonic coefficients, shape ``(npol, nalm)``.
            nu: Band centre frequency in GHz.
            fwhm: Beam FWHM in arcminutes.
            nside: HEALPix nside for this band.
        """
        self._alms = alms
        self.nu = nu
        self.fwhm = fwhm #stored in arcmin
        self.nside = nside

    @classmethod
    def init_from_detector(cls, det_map:DetectorMap, double_precision:bool=False):
        """
        Initialize a Band class object with the metadata from det_map and an empty set of alms.
        """
        alm_len_complex = ((det_map.lmax+1)*(det_map.lmax+2))//2
        npol = 2 if det_map.pol else 1
        dtype = np.complex128 if double_precision else np.complex64
        return cls(np.zeros((npol, alm_len_complex), dtype=dtype), det_map.nu, det_map.fwhm,
                   det_map.nside)

    @property
    def alms(self):
        """Spherical-harmonic coefficients, shape ``(npol, nalm)``."""
        return self._alms

    @alms.setter
    def alms(self, alms):
        """Set the alms with shape validation.

        Args:
            alms: 2-D complex array with first axis length 1 (I) or 2 (QU).

        Raises:
            ValueError: If ``alms`` does not have the expected shape.
        """
        if alms.ndim == 2:
            if alms.shape[0] in [1,2]:
                self._alms = alms
            else:
                raise ValueError("Trying to set alms with wrong first axis length "
                                 f"{alms.shape[0]} != 1 or 2")
        else:
            raise ValueError("Trying to set alms with unexpected number of dimensions"
                                f"{alms.ndim} != 2")
            
    @property
    def is_pol(self):
        """Whether the band has polarisation components."""
        return False if self._alms.shape[0] == 1 else True
    
    @property
    def lmax(self):
        """Maximum multipole, inferred from the alm array length."""
        return int((-3 + np.sqrt(1 + self._alms.shape[1] * 8))/2)
