"""Helpers shared by every experiment TOD reader: noise priors, processing masks, FFT sizing."""
import logging

import ducc0.fft
from mpi4py import MPI
from pixell.bunch import Bunch
from numpy.typing import NDArray
import healpy as hp
import numpy as np

from commander4.parameters.schema import resolve_param
from commander4.tod.noise.psd import NoisePSD

logger = logging.getLogger(__name__)


def _resolve_noise_prior_block(params: Bunch, key: str, expname: str, bandname: str,
                               param_names: tuple[str, ...], model_name: str) -> dict | None:
    """One noise-prior block from the band, else the experiment, else ``tod_processing``.

    Returns None when no scope sets it. Every key is checked against the model's parameter names,
    because a misspelled name would otherwise silently leave the default in force, and a parameter
    stuck at a wrong default looks exactly like a converged one in the chain.
    """
    block = resolve_param(params, key, (f"experiments.{expname}.bands.{bandname}",
                                        f"experiments.{expname}", "tod_processing"), default=None)
    if block is None:
        return None
    for name in block:
        if name not in param_names:
            raise ValueError(f"{key!r} for band {bandname!r} names {name!r}, which is not a "
                             f"parameter of {model_name}: {list(param_names)}.")
    return block


def apply_noise_priors(noise_model: NoisePSD, params: Bunch, expname: str, bandname: str) -> None:
    """Override the noise model's prior defaults with anything the parameter file specifies.

    Two optional blocks, each a mapping from noise-parameter name to its setting, taken from the
    band block, else the experiment block, else ``tod_processing``. Only the named parameters are
    changed; the rest keep the instrument-appropriate defaults the reader built the model with::

        noise_prior_bounds:          # hard [lo, hi] limits (C3's p_uni)
          fknee: [0.01, 100.0]
          alpha: [-4.5, -0.5]
        noise_prior:                 # informative [mean, rms] (C3's p_active); optional
          fknee: [10.0, 0.5]         # rms in *decades* for log-normal parameters such as fknee
          alpha: [-2.7, 0.3]

    The bounds are the endpoints of the grid the PSD sampler draws on, so a true value outside them
    cannot be recovered: the sample pins against the nearest edge instead. The informative prior
    multiplies the likelihood along that grid (see `NoisePSD.log_prior`); an rms of ``.inf`` leaves
    it uninformative, and an rms ``<= 0`` holds the parameter fixed at its current value entirely.

    Args:
        noise_model: The model to modify in place.
        expname, bandname: Keys of this band's experiment and band blocks in `params`.
    """
    param_names = noise_model.param_names
    model_name = type(noise_model).__name__
    bounds = _resolve_noise_prior_block(params, "noise_prior_bounds", expname, bandname,
                                        param_names, model_name)
    prior = _resolve_noise_prior_block(params, "noise_prior", expname, bandname,
                                       param_names, model_name)
    if bounds is None and prior is None:
        return
    for name, limits in (bounds or {}).items():
        noise_model.P_uni[param_names.index(name)] = limits
    for name, (mean, rms) in (prior or {}).items():
        noise_model.P_active[param_names.index(name)] = (mean, rms)
    # `sampled` is spelled out because an rms of <= 0 switching a parameter off is easy to miss in
    # the [mean, rms] pairs, and a silently unsampled parameter looks exactly like a converged one.
    logger.info(f"Band {bandname}: noise priors overridden from the parameter file. "
                f"bounds={dict(zip(param_names, noise_model.P_uni.tolist()))}, "
                f"[mean, rms]={dict(zip(param_names, noise_model.P_active.tolist()))}, "
                f"sampled={[n for i, n in enumerate(param_names) if noise_model.is_sampled(i)]}.")


def find_good_fourier_size(ntod: int) -> int:
    """Return the largest fast real-FFT size strictly smaller than ``ntod``.

    ``ducc0.fft.good_size(n, True)`` returns the first fast real-FFT size at or above ``n``. Walking
    downward until it returns the candidate itself finds the closest fast size below the scan
    length without a machine-specific timing table.
    """
    if ntod < 2:
        raise ValueError("TOD length must be at least 2 to select a smaller FFT size.")

    candidate = ntod - 1
    while ducc0.fft.good_size(candidate, True) != candidate:
        candidate -= 1
    return candidate


def read_processing_masks(band_comm: MPI.Comm,
                          band_params: Bunch) -> tuple[NDArray | None, dict[str, NDArray]]:
    """Read a band's default and named processing-mask maps once and broadcast them.

    Args:
        band_comm: The band's MPI communicator; only rank 0 touches the filesystem.
        band_params: The band's parameter block (``processing_mask`` and/or ``processing_masks``).

    Returns:
        ``(default_mask, named_masks)``: the default boolean HEALPix map (or ``None`` if the band
        defines none) and a dict of named boolean HEALPix maps (empty if none are defined). Maps are
        kept at their native nside; ``TODView`` handles any nside mismatch with the pointing.
    """
    default_mask = None
    named_masks: dict[str, NDArray] = {}
    if band_comm.Get_rank() == 0:
        filename = getattr(band_params, "processing_mask", None)
        if filename is not None:
            default_mask = hp.read_map(filename, field=0, dtype=bool)
        for name in getattr(band_params, "processing_masks", []) or []:
            named_masks[name] = hp.read_map(band_params.processing_masks[name], field=0, dtype=bool)
    # bcast returns the broadcast object (it does not fill in place), so capture the return value.
    default_mask = band_comm.bcast(default_mask, root=0)
    named_masks = band_comm.bcast(named_masks, root=0)
    return default_mask, named_masks
