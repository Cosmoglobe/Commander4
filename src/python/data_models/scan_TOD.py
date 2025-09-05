import numpy as np
import healpy as hp
from output import log
import logging
from numpy.typing import NDArray
from cmdr4_support.utils import huffman_decode

class ScanTOD:
    def __init__ (self, tod, pix_encoded, psi_encoded, startTime, scanID, nside, data_nside, fsamp, orb_dir_vec,
                  huffman_tree, huffman_symbols, npsi, processing_mask_map, ntod_original):
        logger = logging.getLogger(__name__)
        log.logassert_np(tod.ndim==1, "'value' must be a 1D array", logger)
        log.logassert_np(tod.dtype in [np.float64,np.float32], "TOD dtype must be floating type,"
                         f" is {tod.dtype}", logger)
        log.logassert_np(orb_dir_vec.size == 3, "orb_dir_vec must be a vector of size 3.", logger)
        log.logassert_np(processing_mask_map.dtype == bool, "Processing mask is not boolean type",
                         logger)
        self._tod = tod
        self.ntod = self._tod.shape[-1]
        self._pix_encoded = pix_encoded
        self._psi_encoded = psi_encoded
        self._startTime = startTime
        self._scanID = scanID
        self._eval_nside = nside
        self._data_nside = data_nside
        self._fsamp = fsamp
        self._orb_dir_vec = orb_dir_vec.astype(np.float32)
        self._huffman_tree = huffman_tree
        self._huffman_symbols = huffman_symbols
        self._ntod_original = ntod_original  # Size of the original TOD before Fourier cropping.
        self._npsi = npsi
        pix = self.pix
        self._processing_mask_TOD = np.packbits(processing_mask_map[pix])
        del(pix)


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
    def nside(self):
        """ The nside this detector should be evalutated at.
        """
        return self._eval_nside

    @property
    def data_nside(self):
        """ The nside the pixel indices are stored at. This should never explicitly be exposed to
            the user, but we use this nside for internal conversions. 
        """
        return self._data_nside

    @property
    def pix(self) -> NDArray[np.integer]:
        pix = np.zeros(self._ntod_original, dtype=np.int64)
        pix = huffman_decode(np.frombuffer(self._pix_encoded, dtype=np.uint8), self._huffman_tree, self._huffman_symbols, pix)
        #TODO: I think cumsum should eventually be wrapped in somewhere, it can be easy to forget.
        pix = np.cumsum(pix)
        # The TOD was cropped to an ideal Fourier length, but because the pix entry is compressed,
        # we need to unpack the entire original array, and then crop it to the correct length.
        pix = pix[:self.ntod]
        if self.nside != self.data_nside:
            pix = hp.ang2pix(self.nside, *hp.pix2ang(self.data_nside, pix))
        return pix

    @property
    def psi(self) -> NDArray[np.floating]:
        psi = np.zeros(self._ntod_original, dtype=np.int64)
        psi = huffman_decode(np.frombuffer(self._psi_encoded, dtype=np.uint8), self._huffman_tree, self._huffman_symbols, psi)
        psi = np.cumsum(psi)
        psi = psi[:self.ntod]
        psi = 2*np.pi * psi.astype(np.float32)/self._npsi
        return psi
        
        
    @property
    def scanID(self) -> int:
        return self._scanID
    
    @property
    def processing_mask_TOD(self):
        mask = np.unpackbits(self._processing_mask_TOD).view(bool)
        if mask.size > self._tod.size + 7 or mask.size < self._tod.size:
            # The bytearray is stored in multiples of 8, so it can be up to 7 elements
            # longer than the TOD. If it's even longer or shorter, something is wrong.
            raise ValueError(f"Mask size {mask.size} doesn't match TOD size {self._tod.size}.")
        return mask[:self._tod.size]

    @property
    def fsamp(self) -> float:
        return self._fsamp
    
    @property
    def orb_dir_vec(self) -> NDArray[np.floating]:
        return self._orb_dir_vec