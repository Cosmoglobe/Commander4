"""Noise models (swappable).

A ``NoiseModel`` realizes a per-detector noise TOD given the detector's white level ``sigma0``:

    realize(ntod, fsamp, sigma0, rng) -> ndarray

``OofNoise`` produces white + 1/f noise whose PSD is ``sigma0^2 (1 + (f/fknee)^alpha)``, matching
``commander4.noise_sampling.noise_psd.NoisePSDOof`` (the model the TOD processing assumes). The
realization shapes a white series of RMS ``sigma0`` in Fourier space, exactly as
``inplace_litebird_sim`` does, so the white floor is preserved at high frequency.
"""
import logging
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from pixell.bunch import Bunch

from simgen.config import bget

logger = logging.getLogger(__name__)


class NoiseModel(ABC):
    @abstractmethod
    def realize(self, ntod: int, fsamp: float, sigma0: float,
                rng: np.random.Generator) -> NDArray[np.floating]:
        ...


class WhiteNoise(NoiseModel):
    def realize(self, ntod, fsamp, sigma0, rng):
        return rng.normal(0.0, sigma0, ntod).astype(np.float32)


class OofNoise(NoiseModel):
    """White + 1/f noise; PSD = sigma0^2 (1 + (f/fknee)^alpha)."""

    def __init__(self, fknee: float, alpha: float, include_white: bool = True):
        self.fknee = float(fknee)
        self.alpha = float(alpha)
        self.include_white = include_white

    def realize(self, ntod, fsamp, sigma0, rng):
        white = rng.normal(0.0, sigma0, ntod)
        f = np.fft.rfftfreq(ntod, d=1.0 / fsamp)
        # Give the DC bin a large-but-finite power (half the first non-zero frequency) instead of
        # the divergent f=0 value, as in inplace_litebird_sim; the per-scan monopole this produces
        # is handled downstream by the correlated-noise sampler.
        if f.size > 1:
            f[0] = 0.5 * f[1]
        shape = (f / self.fknee) ** self.alpha
        if self.include_white:
            shape = 1.0 + shape
        return np.fft.irfft(np.fft.rfft(white) * np.sqrt(shape), n=ntod).astype(np.float32)


def make_noise_model(band) -> NoiseModel:
    """Build the per-detector noise model for a band from its resolved ``noise`` config."""
    oof: Bunch | None = band.noise.oof
    if oof is not None and bget(oof, "enabled", False):
        return OofNoise(oof.fknee, oof.alpha, include_white=band.noise.white)
    return WhiteNoise()
