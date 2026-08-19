"""TOD modifiers (swappable): post-processing applied to a scan's per-detector TOD matrix.

A ``TODModifier`` transforms the stacked detector TODs of one scan in place / returns the modified
matrix:

    apply(tod, band, ctx) -> ndarray          # tod has shape (ndet, ntod)

This is where instrumental complications that couple detectors live. ``CrossTalk`` mixes detectors
with a per-band N x N matrix (``d'_i = sum_j X_ij d_j``). Add a new complication by subclassing
``TODModifier`` and registering it in ``MODIFIERS``.
"""
import logging
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from pixell.bunch import Bunch

from simgen.config import bget

logger = logging.getLogger(__name__)


class TODModifier(ABC):
    @abstractmethod
    def apply(self, tod: NDArray[np.floating], band, ctx: Bunch) -> NDArray[np.floating]:
        ...


class CrossTalk(TODModifier):
    """Linear detector-detector cross-talk: ``tod' = X @ tod`` with the band's (ndet, ndet) matrix.

    A no-op for bands without a cross-talk matrix.
    """
    def apply(self, tod, band, ctx):
        if band.crosstalk is None:
            return tod
        return (band.crosstalk @ tod).astype(tod.dtype, copy=False)


MODIFIERS: dict[str, type[TODModifier]] = {
    "crosstalk": CrossTalk,
}


def build_modifiers(params: Bunch) -> list[TODModifier]:
    """Build the active TOD-modifier chain.

    Cross-talk is included whenever the ``simulation.modifiers.crosstalk`` block is enabled (the
    actual matrices are per-band, so this only toggles the pass); any other registered modifiers are
    added when their block under ``simulation.modifiers`` is enabled.
    """
    mod_cfg = bget(params.simulation, "modifiers", Bunch())
    modifiers: list[TODModifier] = []
    for name, cls in MODIFIERS.items():
        block = bget(mod_cfg, name, None)
        if block is not None and bget(block, "enabled", False):
            modifiers.append(cls())
    return modifiers
