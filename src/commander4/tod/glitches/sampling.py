"""Glitch (cosmic-ray) sampling: find glitch events once, then refit them on every Gibbs pass.

PLACEHOLDER. Every sampling step below is empty, so no glitches are found or removed yet; the
experiment's glitch flag bits still do all the work. The model follows Commander3
[C3: `comm_tod_cray_mod.f90` on the `hfi_cr` branch, and the Python prototype in
`commander3/todscripts/hfi/glitches/src/`].

Two kinds of event, stored together in `Glitches.events` (see `events.py`), with C3's names
[C3: `CR_BRIGHT`, and the faint events of `detect_events`]. `TODView.get_tod` treats them as two
corrections, "bright_glitches" and "faint_glitches", so they can become two steps:
    * Bright events: found once, where the raw residual exceeds 30 sigma. Each gets its own fitted
      pulse and a baseline step, and its worst samples are cut from the data.
    * Faint events (short, long, slow): found once by a matched filter and typed by the lowest
      chi^2. Each detector has one template shape per type. Every pass resamples the per-event
      amplitudes, and then the shapes from all events of the detector.

`process_tod` runs this step after the HFI baselines (detection needs the modulation phase) and
before gain, like C3's cosmic-ray step. `TODView.get_tod` removes the baseline steps before HFI
demodulation (they are not modulated) and the pulses after it (they have the sign of the sky);
`get_mask` adds the cut samples to the flag cut.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.parameters.schema import read_settings
from commander4.tod.glitches.events import GlitchEvents, empty_glitch_grid,\
    GLITCH_TEMPLATE_TYPES, GLITCH_TEMPLATE_NEXP, GLITCH_EVENT_FIELDS
from commander4.tod.step import ChainArrays, TODStep
from commander4.tod.view import TODView

if TYPE_CHECKING:
    from commander4.data_models.tod_samples import TODSamples

logger = logging.getLogger(__name__)


class Glitches(TODStep):
    """The glitch events of one band's detector-scans on one rank, and the template shapes.

    Detection runs once per run, like C3's `first_call`, and `detected` records that. The events
    are not read back from a chain, so a restart detects them again, as C3 does.

    Sampled:
        events: (nscans, ndet) object array holding the `GlitchEvents` of each detector-scan.
        template_amps: (ndet, type, exponential) Amplitudes of the short/long/slow template shapes
            that each detector's faint events share; NaN until the step has run.
        template_taus: The matching time constants, in seconds.

    Attributes:
        cfg: The `tod_processing.glitches` settings: whether and from which iteration glitches are
            sampled. Detection thresholds and the start-template file will belong here.
        detected: Whether event detection has run.
        write_events: Whether the chain file gets the events and the template shapes, as optional
            debug output (`output.chains.include.glitch_events`).
    """

    def __init__(self, experiment_data: DetectorGroupTOD, params: Bunch, my_band: Bunch,
                 band_comm: MPI.Comm, chain: int):
        self.cfg = read_settings(params, "glitches", ("tod_processing",),
                                 {"enabled": False, "from_iter": 1})
        if not self.cfg.enabled:
            return
        if band_comm.Get_rank() == 0 and chain == 1:
            logger.warning(f"Band {experiment_data.band_name}: glitch sampling is only a "
                           "placeholder; no glitches are found or removed.")

        nscans, ndet = experiment_data.nscans, experiment_data.ndet
        self.events = empty_glitch_grid(nscans, ndet)
        template_shape = (ndet, len(GLITCH_TEMPLATE_TYPES), GLITCH_TEMPLATE_NEXP)
        self.template_amps = np.full(template_shape, np.nan)
        self.template_taus = np.full(template_shape, np.nan)
        self.detected = False
        self.write_events = bool(getattr(params.output.chains.include, "glitch_events", False))


    def chain_arrays(self) -> ChainArrays:
        """The events and the template shapes, once detection has run and only when asked for.
        Not read back."""
        arrays = ChainArrays()
        # `detected` is the same on every rank, since every rank runs `sample`.
        if self.cfg.enabled and self.write_events and self.detected:
            # Per-detector shapes, identical on every rank, so no communication is needed.
            arrays.per_band["glitches/template_amps"] = self.template_amps
            arrays.per_band["glitches/template_taus"] = self.template_taus
            events_by_detector_scan = {}
            nscans, ndet = self.events.shape
            for iscan in range(nscans):
                for idet in range(ndet):
                    events = self.events[iscan, idet]
                    if events.num_events > 0:
                        events_by_detector_scan[(iscan, idet)] = {
                            field: getattr(events, field) for field in GLITCH_EVENT_FIELDS}
            arrays.per_detector_scan["glitches"] = events_by_detector_scan
        return arrays


    def pulse_tod(self, view: TODView, bright: bool, faint: bool) -> NDArray[np.float32]:
        """The glitch pulses of the view's focused detector-scan, for `TODView.get_tod`, at the
        full sample rate, in detector units; zeros when it has no events.

        The pulses have the sign of the sky in the demodulated TOD.

        Args:
            bright: Whether to include the pulses of the bright events.
            faint: Whether to include the pulses of the faint events.
        """
        events = self.events[view.iscan, view.idet]
        if events.num_events == 0:
            return np.zeros(view.detector.ntod, dtype=np.float32)
        return events.pulse_tod(view.detector.ntod, self.template_amps[view.idet],
                                self.template_taus[view.idet], view.fsamp, bright, faint)


    def sample(self, tod_samples: TODSamples, compsep_output: NDArray,
               rng: np.random.Generator | None = None) -> None:
        """Find glitch events on the first pass, then resample their amplitudes and shapes.

        Detection runs once per run, like C3's `first_call`, so a restart detects the events
        again. The control flow is final, but the four steps it calls are placeholders, so the
        catalog stays empty.

        Args:
            tod_samples: The band's sampled state, which the TOD views read.
            compsep_output: Current sky model, removed from the TOD before any fit.
            rng: Optional generator for the Gibbs draws.
        """
        # Every TOD this view returns has the stored glitch model removed, as for all other steps.
        scan_view = TODView(tod_samples.experiment_data, tod_samples, compsep_output=compsep_output)

        if not self.detected:
            for view in scan_view.iter_focused(accepted_only=True):
                self.events[view.iscan, view.idet] = _detect_bright_glitches(view)
            # The template shapes will start from Planck's fitted values here, read from a file
            # named in self.cfg [C3 prototype: glitch_params.npy, per detector and type].
            # A second pass, so the view now removes the bright events and the faint ones are
            # searched for without them.
            for view in scan_view.iter_focused(accepted_only=True):
                self.events[view.iscan, view.idet] = _detect_faint_glitches(
                    view, self.events[view.iscan, view.idet], self.template_amps[view.idet],
                    self.template_taus[view.idet])
            self.detected = True

        for view in scan_view.iter_focused(accepted_only=True):
            self._sample_amplitudes(view, rng)

        # Every rank must call this, because one detector's scans are spread over the band's
        # ranks.
        self._sample_templates(tod_samples.band_comm, scan_view)


    def _sample_amplitudes(self, view: TODView, rng: np.random.Generator | None) -> None:
        """Draw new amplitudes for the faint glitches of the focused detector-scan.

        Let T be the (ntod, nevent) matrix whose columns are the faint events' unit-amplitude
        pulses, and r the data with everything except those pulses removed:
        ``view.get_tod(subtract=<sky and dipole>, keep=("faint_glitches",))``. With white noise N
        at the current sigma0, the amplitudes have a Gaussian posterior with mean
        (T^T N^-1 T)^-1 T^T N^-1 r and covariance (T^T N^-1 T)^-1; a Gibbs step draws from it.
        Cut samples are left out. [C3: `comm_tod_cray::fit_amps`; C3 prototype:
        `subtraction.subtract_glitches_from_data`, which takes the mean only.]

        Placeholder: does nothing. It will replace the amplitudes of this detector-scan's events.

        Args:
            view: View focused on one accepted detector-scan.
            rng: Optional generator for the draw.
        """


    def _sample_templates(self, band_comm: MPI.Comm, scan_view: TODView) -> None:
        """Re-estimate each detector's short, long and slow template shapes from all its faint
        events.

        For each type, the residual around every event, with the faint pulses left in
        (``view.get_tod(subtract=<sky and dipole>, keep=("faint_glitches",))``), is cut out and
        stacked over events; the sum of exponentials is then refitted to the stack. One
        detector's scans are spread over the band's ranks, so the stacks are combined over
        `band_comm`.
        [C3 prototype: `templates.stacking`, `templates.glitch_estimation`.]

        Placeholder: does nothing. It will replace `template_amps` and `template_taus`.

        Args:
            band_comm: Communicator of all ranks holding this band's scans.
            scan_view: View over this rank's scans.
        """


def _detect_bright_glitches(view: TODView) -> GlitchEvents:
    """Find the bright glitches of the focused detector-scan and fit each one.

    The search runs on the residual ``view.get_tod(subtract=<sky and dipole>,
    keep=("bright_glitches",))``. Up to the sign of each sample, that is the raw TOD minus the two
    parity baselines and the modulated, gain-scaled sky. A sample above 30 sigma starts a
    candidate, which lasts until the rms over 10 samples falls below 2 sigma. Each candidate gets
    a joint fit of its own pulse and a baseline step (see `GlitchEvents.pulse_tod` and
    `baseline_step_tod`). The cut then grows outward, refitting each time, while the squared
    residual at its edges exceeds 5 sigma^2 or the reduced chi^2 exceeds 2; at the end it widens
    by two samples on each side. The event is kept when the fit lowers the chi^2.
    [C3: `comm_tod_cray::detect_bright_events`, `cray_event::fit_bright_event_with_baseline`.]

    Placeholder: returns no events.

    Args:
        view: View focused on one accepted detector-scan.
    """
    return GlitchEvents.empty()


def _detect_faint_glitches(view: TODView, events: GlitchEvents,
                           template_amps: NDArray[np.floating],
                           template_taus: NDArray[np.floating]) -> GlitchEvents:
    """Find the faint glitches of the focused detector-scan and give each a type.

    A matched filter correlates the residual (the view's TOD, which has the bright events removed
    already, minus sky and dipole) with the template shape. Peaks with a signal-to-noise above 30
    that lie at least 0.05 s apart become events. Each event then gets one fitted amplitude per
    type, and keeps the type with the lowest chi^2. [C3: `comm_tod_cray::detect_events`,
    `select_cr_type`; C3 prototype: `detection.matched_filter`, `classification.classify_glitches`.]

    Placeholder: returns `events` unchanged.

    Args:
        view: View focused on one accepted detector-scan.
        events: The bright events already found. The new events are added to these.
        template_amps: (ntype, nexp) This detector's template amplitudes, one row per type.
        template_taus: (ntype, nexp) The matching time constants in seconds.
    """
    return events
