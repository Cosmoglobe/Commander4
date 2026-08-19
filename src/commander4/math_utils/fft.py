"""Real-to-complex Fourier transforms of time-ordered data, via ducc0.

The mirrored variants reflect the TOD before transforming (length 2N) and keep the first N samples
on the way back. That suppresses the wrap-around a plain FFT introduces at the scan boundaries,
and is what `apply_N_inv` and the correlated-noise CG use.
"""
import os

import ducc0
import numpy as np
from numpy.typing import NDArray


def forward_rfft(data:NDArray[np.floating], nthreads:int = None) -> NDArray:
    """ Forward real Fourier transform, equivalent to scipy.fft.rfft.
        Args:
            data (np.array): Real-valued data array to be Fourier transformed.
            nthreads (int): Number of threads to use.
        Returns:
            data_f (np.array): The Fourier transform of the input.
                               A complex array of length tod.size//2 + 1.
    """
    nthreads = int(os.environ.get("OMP_NUM_THREADS", 1)) if nthreads is None else nthreads
    return ducc0.fft.r2c(data, nthreads=nthreads)

def backward_rfft(data_f:NDArray, ntod:int, nthreads:int = None) -> NDArray:
    """ Backward real Fourier transform, equivalent to scipy.fft.irfft.
        Args:
            data_f (np.array): Complex Fourier coefficients to be converted back to real data.
            ntod (int): The length of the original TOD. This must be provided because a
                           Fourier array of length e.g. 6 could correspond to ntod = 10 or 11.
            nthreads (int): Number of threads to use.
        Returns:
            data (np.array): A real-valued data array of length ntod.
    """
    # If nthreads is not set, put it to how many threads OMP has been given.
    nthreads = int(os.environ.get("OMP_NUM_THREADS", 1)) if nthreads is None else nthreads
    # Forward = False makes ducc correctly order the output, as the output order is not
    # symmetric for forward and reverse Fourier when doing rfft as supposed to regular fft.
    # inorm = 2 tells ducc to normalize by dividing by ntod, which is the same as what scipy does.
    return ducc0.fft.c2r(data_f, lastsize=ntod, forward=False, nthreads=nthreads, inorm=2)


def forward_rfft_mirrored(data: NDArray, nthreads: int = None) -> NDArray:
    """Forward real FFT on a mirrored (reflected) copy of the input.

    The input is mirrored so that ``dt[0:ntod] = data``,
    ``dt[ntod:2*ntod] = data[::-1]``, giving a length-``2*ntod`` symmetric
    array.  This reduces boundary/periodicity artefacts.

    Parameters
    ----------
    data : NDArray
        Real-valued 1-D array of length *ntod*.
    nthreads : int
        Number of FFT threads.

    Returns
    -------
    dv : NDArray (complex)
        Length ``ntod + 1`` complex Fourier coefficients.
    """
    ntod = len(data)
    dt = np.empty(2 * ntod, dtype=data.dtype)
    dt[:ntod] = data
    dt[ntod:] = data[::-1]
    nthreads = int(os.environ.get("OMP_NUM_THREADS", "1")) if nthreads is None else nthreads
    return ducc0.fft.r2c(dt, nthreads=nthreads)


def backward_rfft_mirrored(data_f: NDArray, ntod: int, nthreads: int = None) -> NDArray:
    """Inverse real FFT returning only the first *ntod* samples.

    The inverse is performed on the full ``2*ntod``-length spectrum and
    divided by ``2*ntod``.  Only the first *ntod* samples are returned.

    Parameters
    ----------
    data_f : NDArray (complex)
        Fourier coefficients from :func:`forward_rfft_mirrored`.
    ntod : int
        Original time-domain length.
    nthreads : int
        Number of FFT threads.

    Returns
    -------
    data : NDArray
        Real-valued 1-D array of length *ntod*.
    """
    nfft = 2 * ntod
    nthreads = int(os.environ.get("OMP_NUM_THREADS", "1")) if nthreads is None else nthreads
    dt = ducc0.fft.c2r(data_f, lastsize=nfft, forward=False, nthreads=nthreads, inorm=0)
    return dt[:ntod] / nfft
