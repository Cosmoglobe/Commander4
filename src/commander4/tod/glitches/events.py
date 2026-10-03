"""Containers for the glitch (cosmic-ray) events found in the TOD.

`GlitchEvents` holds the glitch events of one detector-scan. `TODSamples.glitches` holds one per
detector-scan, in an (nscans, ndet) object array made by `empty_glitch_grid`. The sampler that
*finds* the events is `sampling.py` next to this file, which also describes the model and where
each part of it is applied. The events go to the chain file only as optional debug output, and
are not read back on restart.

In the TOD, each event is a pulse: the bolometer's fast rise and slow decay after a cosmic-ray hit.
Two kinds of event share one container, told apart by `type` [C3: `comm_tod_cray_mod.f90` on the
`hfi_cr` branch, whose template type codes 1, 2, 3 count from 1 and are 0, 1, 2 here]:
    * Bright events (`GLITCH_BRIGHT`): each has its own fitted pulse and a baseline step, and its
      worst samples are cut from the data. Their parameters are stored per event in
      `shape_params`.
    * Template events (short, long, slow): every event of one type in one detector shares a pulse
      shape, scaled by the event's own `amplitude`. The shapes are stored per detector on
      `TODSamples`.

The two models (`baseline_step_tod`, `pulse_tod`) are placeholders that return zeros; the storage
and the cuts are complete.
"""
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Type code of bright events. They have no template, so their code is outside the template rows.
GLITCH_BRIGHT = -1
# Template glitch types. The type code of each is its position here, which is also its row in the
# per-detector template arrays on `TODSamples`.
GLITCH_TEMPLATE_TYPES = ("short", "long", "slow")
# Exponentials per template shape, as in the Planck glitch parameters C3's prototype starts from.
GLITCH_TEMPLATE_NEXP = 8

# The arrays of a `GlitchEvents`, which are also its dataset names in the debug chain output.
GLITCH_EVENT_FIELDS = ("start", "length", "type", "amplitude", "cut_start", "cut_stop",
                       "shape_param_counts", "shape_params")


@dataclass(slots=True)
class GlitchEvents:
    """The glitch events of one detector-scan, with one entry per event in every per-event array.

    Attributes:
        start: (nevent,) First sample of each event.
        length: (nevent,) Number of samples each event's model spans.
        type: (nevent,) `GLITCH_BRIGHT`, or 0, 1, 2 for short, long, slow.
        amplitude: (nevent,) Pulse amplitude in detector units; 1 for bright events, whose scale
            is part of their shape parameters.
        cut_start: (nevent,) First sample cut from the data because of each event.
        cut_stop: (nevent,) One past the last cut sample; equal to `cut_start` if none.
        shape_param_counts: (nevent,) Number of shape parameters of each event; 0 for template
            events.
        shape_params: (sum(shape_param_counts),) Shape parameters, concatenated over events. A
            bright event holds (a1, a2, tau1, tau2, t_dep, spline node values...), times in seconds.
    """

    start: NDArray[np.int64]
    length: NDArray[np.int64]
    type: NDArray[np.int8]
    amplitude: NDArray[np.float32]
    cut_start: NDArray[np.int64]
    cut_stop: NDArray[np.int64]
    shape_param_counts: NDArray[np.int64]
    shape_params: NDArray[np.float64]

    def __post_init__(self):
        """Normalize storage and check that every per-event array has one entry per event."""
        self.start = np.asarray(self.start, dtype=np.int64)
        self.length = np.asarray(self.length, dtype=np.int64)
        self.type = np.asarray(self.type, dtype=np.int8)
        self.amplitude = np.asarray(self.amplitude, dtype=np.float32)
        self.cut_start = np.asarray(self.cut_start, dtype=np.int64)
        self.cut_stop = np.asarray(self.cut_stop, dtype=np.int64)
        self.shape_param_counts = np.asarray(self.shape_param_counts, dtype=np.int64)
        self.shape_params = np.asarray(self.shape_params, dtype=np.float64)
        nevent = self.start.size
        for field in GLITCH_EVENT_FIELDS:
            # `shape_params` is the one array without an entry per event; it is checked below.
            if field != "shape_params" and getattr(self, field).shape != (nevent,):
                raise ValueError(f"GlitchEvents.{field} must be 1-D with {nevent} entries.")
        if self.shape_params.shape != (int(np.sum(self.shape_param_counts)),):
            raise ValueError("GlitchEvents.shape_params must hold sum(shape_param_counts) values.")

    @classmethod
    def empty(cls) -> "GlitchEvents":
        """Return a container without events, for detector-scans with no glitches found."""
        no_ints = np.empty(0, dtype=np.int64)
        return cls(no_ints, no_ints, np.empty(0, dtype=np.int8), np.empty(0, dtype=np.float32),
                   no_ints, no_ints, no_ints, np.empty(0, dtype=np.float64))

    @property
    def num_events(self) -> int:
        """Number of glitch events in this detector-scan."""
        return int(self.start.size)

    def good_sample_mask(self, ntod: int) -> NDArray[np.bool_]:
        """Return a (ntod,) boolean mask that is False on every sample cut because of an event."""
        good = np.ones(ntod, dtype=bool)
        for first, stop in zip(self.cut_start, self.cut_stop):
            good[first:stop] = False
        return good

    def baseline_step_tod(self, ntod: int, fsamp: float) -> NDArray[np.float32]:
        """Return the baseline steps of the bright events, to remove from the raw TOD.

        A bright cosmic ray shifts the level of both sample parities alike, so its step must be
        removed before HFI demodulation flips the sign of every other sample. Each step is a cubic
        spline in time after the event's t_dep. The nodes sit at 0, 3 and 6 samples, then every 5
        samples up to 25, then every 10; the last two nodes are held at zero, and the free node
        values are the spline part of the event's `shape_params`.
        [C3: `cray_event%build_baseline_template`, subtracted in
        `comm_tod_hfi_smod.f90::demodulate_tod`.]

        Placeholder: returns zeros.

        Args:
            ntod: Number of samples in the detector-scan.
            fsamp: Sampling rate in Hz.

        Returns:
            (ntod,) The summed steps in detector units, not modulated.
        """
        return np.zeros(ntod, dtype=np.float32)

    def pulse_tod(self, ntod: int, template_amps: NDArray[np.floating],
                  template_taus: NDArray[np.floating], fsamp: float,
                  template_only: bool = False) -> NDArray[np.float32]:
        """Return the glitch pulses, to remove from the demodulated TOD.

        A cosmic ray heats the bolometer like sky signal does, so its pulse is modulated like the
        sky and, after demodulation, has the sign of the sky. A bright event uses its own shape
        ``a1 (exp(-t/tau1) - exp(-t/tau0)) + a2 (exp(-t/tau2) - exp(-t/tau0))``, with t counted from
        t_dep and a fixed rise time tau0 = 2 ms [C3: `cray_event%build_cr_template`]. A template
        event uses the detector's shape for its type, a sum of exponentials scaled to a peak of 1,
        times the event's `amplitude` [C3 prototype: `templates.glitch_model_func`]. Each pulse
        spans `length` samples from `start`. [C3: `comm_tod_cray::generate` -> `s_cray`.]

        Placeholder: returns zeros.

        Args:
            ntod: Number of samples in the detector-scan.
            template_amps: (ntype, nexp) This detector's template amplitudes, one row per type.
            template_taus: (ntype, nexp) The matching time constants in seconds.
            fsamp: Sampling rate in Hz.
            template_only: If True, leave out the bright events.

        Returns:
            (ntod,) The summed pulses in detector units, with the sign of the sky.
        """
        return np.zeros(ntod, dtype=np.float32)


def empty_glitch_grid(nscans: int, ndet: int) -> NDArray[np.object_]:
    """Return an (nscans, ndet) object array with its own empty `GlitchEvents` in every cell.

    The cells are filled one by one: `np.full` would put one shared object in every cell, so
    changing the arrays of one cell in place would change all of them.
    """
    grid = np.empty((nscans, ndet), dtype=object)
    for iscan in range(nscans):
        for idet in range(ndet):
            grid[iscan, idet] = GlitchEvents.empty()
    return grid
