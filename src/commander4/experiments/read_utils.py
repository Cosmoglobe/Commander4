import logging

from mpi4py import MPI
from pixell.bunch import Bunch
from numpy.typing import NDArray
import healpy as hp
import numpy as np

from commander4.diagnostics.log import logassert
from commander4.parameters.schema import resolve_param
from commander4.tod.noise.psd import NoisePSD

logger = logging.getLogger(__name__)


def apply_noise_prior_bounds(noise_model: NoisePSD, params: Bunch, expname: str,
                             bandname: str) -> None:
    """Looks through the provided parameter file for specifications of what the noise model priors
       should be. If it finds it, it overwrites the default in the provided `noise_model`.

    Reads ``noise_prior_bounds``, a mapping from noise-parameter name to ``[lo, hi]``, taken from
    the band block, else the experiment block, else ``tod_processing``. Only the named parameters
    are changed; the rest keep the instrument-appropriate defaults the reader built the model with,
    e.g.::
        noise_prior_bounds:
          fknee: [0.01, 100.0]
          alpha: [-4.5, -0.5]
    These bounds are the endpoints of the grid the PSD sampler draws on, so a true value outside
    them cannot be recovered: the sample pins against the nearest edge instead.

    Args:
        noise_model: The model to modify in place.
        expname, bandname: Keys of this band's experiment and band blocks in `params`.
    """
    bounds = resolve_param(params, "noise_prior_bounds",
                           (f"experiments.{expname}.bands.{bandname}",
                            f"experiments.{expname}", "tod_processing"), default=None)
    if bounds is None:
        return
    for name, limits in bounds.items():
        logassert(name in noise_model.param_names,
                  f"'noise_prior_bounds' for band {bandname!r} names {name!r}, which is not a "
                  f"parameter of {type(noise_model).__name__}: {list(noise_model.param_names)}.",
                  logger)
        noise_model.P_uni[noise_model.param_names.index(name)] = limits
    logger.info(f"Band {bandname}: noise prior bounds overridden from the parameter file, "
                f"P_uni is now {dict(zip(noise_model.param_names, noise_model.P_uni.tolist()))}.")


def find_good_Fourier_time(Fourier_times:NDArray, ntod:int) -> int:
    if ntod <= 10_000 or ntod >= 400_000:
        return ntod
    search_start = int(0.99*ntod)  # Consider sizes up to 1% smaller than ntod.
    best_ntod = np.argmin(Fourier_times[search_start:ntod+1])
    best_ntod += search_start
    assert(best_ntod <= ntod)
    return best_ntod


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
