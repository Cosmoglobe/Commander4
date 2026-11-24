"""`Band`: what the components give at one polband, as alms and pixels, in compsep."""
import numpy as np
import healpy as hp
from numpy.typing import NDArray
from commander4.data_models.band_map import BandMap


class Band:
    """What `comp2band` projects the components onto, and `band2comp` reads back, at one polband.

    It holds the alms of the diffuse components and the pixel map of the point sources, with the
    polband's frequency, beam and resolution. The polband's data are in its `BandMap`.

    Attributes:
        alms: Complex spherical-harmonic coefficients, shape ``(npol, nalm)``, with one row (I)
            or two (QU).
        lmax (int): Maximum multipole of the alms.
        nu (float): Band centre frequency in GHz.
        fwhm_rad (float): Beam FWHM in radians.
        nside (int): HEALPix nside associated with this band.
        map: The part of the band that lives in pixel space, shape ``(npol, npix)``: point
            sources, which are added as pixels rather than as alms (as in C3). None until a
            point-source component writes to it.
    """
    def __init__(self, alms: NDArray[np.complexfloating], nu: float, fwhm_rad: float, nside: int):
        """Construct a Band from its alms, centre frequency (GHz), FWHM (radians) and nside."""
        self.alms = alms
        self.lmax = hp.Alm.getlmax(alms.shape[1])
        self.nu = nu
        self.fwhm_rad = fwhm_rad
        self.nside = nside
        self.map = None

    @classmethod
    def init_from_band_map(cls, band_map: BandMap, double_precision: bool = False):
        """A Band with the metadata of `band_map` and alms of zeros."""
        dtype = np.complex128 if double_precision else np.complex64
        return cls(np.zeros((band_map.npol, hp.Alm.getsize(band_map.lmax)), dtype=dtype),
                   band_map.nu, band_map.fwhm_rad, band_map.nside)

    @property
    def pol(self) -> str:
        """The polarization of the band: "I" for one alm row, "QU" for two."""
        return "I" if len(self.alms) == 1 else "QU"
