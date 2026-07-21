"""Instrument model: bands and detectors built from the parameter file.

A ``Band`` groups the detectors that share a frequency, beam and pointing (the "shared boresight"
model). Each ``Detector`` carries the small per-detector quantities that make detectors distinct
within a band -- a polarization-angle offset, a focal-plane offset, its own white-noise level and a
gain. Detector-detector cross-talk is represented by a per-band ``crosstalk`` matrix applied across
the band's detector TODs (see ``modifiers.CrossTalk``).
"""
import logging
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from pixell.bunch import Bunch

from simgen.config import bget
from simgen.transfer import TransferFunction, make_detector_transfer

logger = logging.getLogger(__name__)


@dataclass
class Detector:
    name: str
    idx: int                       # Position within the band (column index for cross-talk).
    psi_offset: float = 0.0        # Added to the band polarization angle [rad].
    fp_offset: tuple[float, float] = (0.0, 0.0)  # Focal-plane (xi, eta) offset [rad] (reserved).
    sigma0: float = 0.0            # Per-sample white-noise RMS in the band's TOD unit.
    gain: float = 1.0
    transfer: TransferFunction | None = None  # Bolometer time-response filter (None -> identity).


@dataclass
class Band:
    name: str
    exp_name: str
    freq: float                    # [GHz]
    fwhm_arcmin: float
    fsamp: float                   # [Hz]
    eval_nside: int
    data_nside: int
    units: str
    polarization: str              # "I" | "QU" | "IQU"
    detectors: list[Detector]
    sigma0: float                  # Band default per-sample white-noise RMS [band unit].
    noise: Bunch                   # Resolved noise config (white flag + optional oof block).
    crosstalk: NDArray[np.floating] | None = None  # (ndet, ndet) mixing matrix, or None.

    @property
    def fwhm_rad(self) -> float:
        return np.deg2rad(self.fwhm_arcmin / 60.0)

    @property
    def npol(self) -> int:
        return {"I": 1, "QU": 2, "IQU": 3}[self.polarization]

    @property
    def ndet(self) -> int:
        return len(self.detectors)

    @property
    def det_names(self) -> list[str]:
        return [d.name for d in self.detectors]


def _resolve_sigma0(spec: Bunch, fsamp: float, fallback: float) -> float:
    """Resolve a per-sample sigma0 from a param block.

    Accepts ``sigma0`` (already per-sample, band unit) or ``sigma0_rts`` (per-root-second; converted
    to per-sample by multiplying by sqrt(fsamp)). Falls back to ``fallback`` when neither is set.
    """
    if "sigma0" in spec:
        return float(spec.sigma0)
    if "sigma0_rts" in spec:
        return float(spec.sigma0_rts) * np.sqrt(fsamp)
    return fallback


def _parse_crosstalk(ct: Bunch, ndet: int) -> NDArray[np.floating] | None:
    """Build an (ndet, ndet) cross-talk mixing matrix from a band's ``crosstalk`` block.

    Either an explicit ``matrix`` (ndet x ndet) or a scalar ``coupling`` epsilon (giving
    ``X = I + epsilon*(1 - I)``, i.e. uniform off-diagonal leakage) may be supplied.
    """
    if ct is None or not bget(ct, "enabled", False):
        return None
    if "matrix" in ct:
        X = np.array(ct.matrix, dtype=np.float64)
        if X.shape != (ndet, ndet):
            raise ValueError(f"crosstalk matrix has shape {X.shape}, expected ({ndet}, {ndet}).")
        return X
    if "coupling" in ct:
        eps = float(ct.coupling)
        return np.eye(ndet) + eps * (np.ones((ndet, ndet)) - np.eye(ndet))
    raise ValueError("crosstalk.enabled is true but neither 'matrix' nor 'coupling' was given.")


def _resolve_noise(band_spec: Bunch, global_noise: Bunch) -> Bunch:
    """Merge the global ``simulation.noise`` defaults with a band-level ``noise`` override."""
    noise = Bunch(white=True, oof=None)
    for src in (global_noise, bget(band_spec, "noise", None)):
        if src is None:
            continue
        if "white" in src:
            noise.white = bool(src.white)
        if "oof" in src:
            noise.oof = src.oof
    return noise


def build_bands(params: Bunch) -> list[Band]:
    """Construct the list of enabled ``Band`` objects from the parameter file."""
    global_noise = bget(params.simulation, "noise", Bunch())
    global_transfer = bget(params.simulation, "transfer_function", None)  # run-wide TF default
    bands: list[Band] = []
    for exp_name, exp in params.experiments.items():
        if not bget(exp, "enabled", True):
            continue
        for band_name, bspec in exp.bands.items():
            if not bspec_enabled(bspec):
                continue
            fsamp = float(bspec.fsamp)
            band_sigma0 = _resolve_sigma0(bspec, fsamp, fallback=0.0)
            detectors = []
            for idx, (det_name, dspec) in enumerate(bspec.detectors.items()):
                dspec = dspec if isinstance(dspec, Bunch) else Bunch()
                detectors.append(Detector(
                    name=det_name,
                    idx=idx,
                    psi_offset=np.deg2rad(float(bget(dspec, "psi_offset_deg", 0.0))),
                    fp_offset=tuple(np.deg2rad(np.array(bget(dspec, "fp_offset_deg", [0.0, 0.0]),
                                                        dtype=np.float64))),
                    sigma0=_resolve_sigma0(dspec, fsamp, fallback=band_sigma0),
                    gain=float(bget(dspec, "gain", 1.0)),
                    transfer=make_detector_transfer(dspec, global_transfer),
                ))
            data_nside = int(bget(bspec, "data_nside", bspec.eval_nside))
            bands.append(Band(
                name=band_name, exp_name=exp_name, freq=float(bspec.freq),
                fwhm_arcmin=float(bspec.fwhm), fsamp=fsamp,
                eval_nside=int(bspec.eval_nside), data_nside=data_nside,
                units=bget(params.general, "units", "uK_RJ"),
                polarization=bget(bspec, "polarization", "IQU"),
                detectors=detectors, sigma0=band_sigma0,
                noise=_resolve_noise(bspec, global_noise),
                crosstalk=_parse_crosstalk(bget(bspec, "crosstalk", None), len(detectors)),
            ))
    if not bands:
        raise ValueError("No enabled bands found in the parameter file.")
    return bands


def bspec_enabled(bspec: Bunch) -> bool:
    return bool(bget(bspec, "enabled", True))
