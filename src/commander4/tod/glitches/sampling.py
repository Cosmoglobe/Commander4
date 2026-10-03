"""Glitch (cosmic-ray) sampling: find glitch events once, then refit them on every Gibbs pass.

PLACEHOLDER. Every sampling step below is empty, so no glitches are found or removed yet; the
experiment's glitch flag bits still do all the work. The module fixes where each part of the model
goes. The model follows Commander3 [C3: `comm_tod_cray_mod.f90` on the `hfi_cr` branch, and the
Python prototype in `commander3/todscripts/hfi/glitches/src/`].

Two kinds of event, stored together in `TODSamples.glitches` (see `events.py` next to this file):
    * Bright events: found once, where the raw residual exceeds 30 sigma. Each gets its own fitted
      pulse and a baseline step, and its worst samples are cut from the data.
    * Template events (short, long, slow): faint events, found once by a matched filter and typed
      by the lowest chi^2. Each detector has one shape per type. Every pass resamples the per-event
      amplitudes, and then the shapes from all events of the detector.

Where the parts go:
    * `process_tod` runs this step after the HFI baselines, whose first pass finds the modulation
      phase that detection needs, and before gain, like C3's cosmic-ray step. The fits use the
      previous pass's noise level, since C4 does not keep n_corr between passes.
    * `TODView` applies the stored model to every later TOD request: the baseline steps in
      `pre_demodulation_tod` (they are not modulated, so they go before HFI demodulation), the
      pulses in `corrected_tod` (after demodulation, where they have the sign of the sky), and the
      cut samples in the flag cut of `get_mask`. `get_glitch_pulse_tod` returns the pulses on
      their own: the HFI baseline fit removes them, modulated, together with the sky, and the steps
      here that fit the template pulses add those back.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.tod.config import GlitchConfig
from commander4.tod.glitches.events import GlitchEvents
from commander4.tod.view import TODView

if TYPE_CHECKING:
    from commander4.data_models.tod_samples import TODSamples

logger = logging.getLogger(__name__)


def sample_glitches(band_comm: MPI.Comm, experiment_data: DetectorGroupTOD,
                    tod_samples: TODSamples, compsep_output: NDArray, glitch_cfg: GlitchConfig,
                    rng: np.random.Generator | None = None) -> TODSamples:
    """Find glitch events on the first pass, then resample their amplitudes and shapes.

    Detection runs once per run, like C3's `first_call`, so a restart detects the events again.
    The control flow is final, but the four steps it calls are placeholders, so the catalog stays
    empty.

    Args:
        band_comm: Communicator of all ranks holding this band's scans.
        experiment_data: Static TOD container for the band.
        tod_samples: Sampled state. Its glitch catalog and template shapes are updated.
        compsep_output: Current sky model, removed from the TOD before any fit.
        glitch_cfg: Glitch settings. Detection thresholds and the start-template file belong here.
        rng: Optional generator for the Gibbs draws.
    """
    if band_comm.Get_rank() == 0:
        logger.warning(f"Band {experiment_data.band_name}: glitch sampling is only a placeholder; "
                       "no glitches are found or removed.")

    # Every TOD this view returns has the stored glitch model removed, as for all other steps.
    scan_view = TODView(experiment_data, tod_samples, compsep_output=compsep_output)

    if not tod_samples.glitch_events_detected:
        for view in scan_view.iter_focused(accepted_only=True):
            tod_samples.glitches[view.iscan, view.idet] = _detect_bright_glitches(view)
        # The template shapes will start from Planck's fitted values here, read from a file named
        # in glitch_cfg [C3 prototype: glitch_params.npy, per detector and type].
        # A second pass, so the view now removes the bright events and the faint ones are searched
        # for without them.
        for view in scan_view.iter_focused(accepted_only=True):
            tod_samples.glitches[view.iscan, view.idet] = _detect_template_glitches(
                view, tod_samples.glitches[view.iscan, view.idet],
                tod_samples.glitch_template_amps[view.idet],
                tod_samples.glitch_template_taus[view.idet])
        tod_samples.glitch_events_detected = True

    for view in scan_view.iter_focused(accepted_only=True):
        _sample_glitch_amplitudes(view, tod_samples, rng)

    # Every rank must call this, because one detector's scans are spread over the band's ranks.
    _sample_glitch_templates(band_comm, scan_view, tod_samples)
    return tod_samples


def _detect_bright_glitches(view: TODView) -> GlitchEvents:
    """Find the bright glitches of the focused detector-scan and fit each one.

    The search runs on the raw residual: the raw TOD minus the two parity baselines and the
    modulated, gain-scaled sky. A sample above 30 sigma starts a candidate, which lasts until the
    rms over 10 samples falls below 2 sigma. Each candidate gets a joint fit of its own pulse and a
    baseline step (see `GlitchEvents.pulse_tod` and `baseline_step_tod`). The cut then
    grows outward, refitting each time, while the squared residual at its edges exceeds 5 sigma^2
    or the reduced chi^2 exceeds 2; at the end it widens by two samples on each side. The event is
    kept when the fit lowers the chi^2. [C3: `comm_tod_cray::detect_bright_events`,
    `cray_event::fit_bright_event_with_baseline`.]

    Placeholder: returns no events.

    Args:
        view: View focused on one accepted detector-scan.
    """
    return GlitchEvents.empty()


def _detect_template_glitches(view: TODView, events: GlitchEvents,
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


def _sample_glitch_amplitudes(view: TODView, tod_samples: TODSamples,
                              rng: np.random.Generator | None) -> None:
    """Draw new amplitudes for the template glitches of the focused detector-scan.

    Let T be the (ntod, nevent) matrix whose columns are the template events' unit-amplitude
    pulses, and r the data with everything except those pulses removed: the view's TOD minus sky
    and dipole, plus `view.get_glitch_pulse_tod(template_only=True)` at the current amplitudes.
    With white noise N at the current sigma0, the amplitudes have a Gaussian posterior with mean
    (T^T N^-1 T)^-1 T^T N^-1 r and covariance (T^T N^-1 T)^-1; a Gibbs step draws from it. Cut
    samples are left out. [C3: `comm_tod_cray::fit_amps`; C3 prototype:
    `subtraction.subtract_glitches_from_data`, which takes the mean only.]

    Placeholder: does nothing.

    Args:
        view: View focused on one accepted detector-scan.
        tod_samples: Sampled state; the amplitudes of this detector-scan's events are replaced.
        rng: Optional generator for the draw.
    """


def _sample_glitch_templates(band_comm: MPI.Comm, scan_view: TODView,
                             tod_samples: TODSamples) -> None:
    """Re-estimate each detector's short, long and slow template shapes from all its events.

    For each type, the residual around every event, with the event's own pulse added back
    (`view.get_glitch_pulse_tod(template_only=True)` holds them all), is cut out and stacked over
    events; the sum of exponentials is then refitted to the stack. One
    detector's scans are spread over the band's ranks, so the stacks are combined over `band_comm`.
    [C3 prototype: `templates.stacking`, `templates.glitch_estimation`.]

    Placeholder: does nothing.

    Args:
        band_comm: Communicator of all ranks holding this band's scans.
        scan_view: View over this rank's scans.
        tod_samples: Sampled state; `glitch_template_amps` and `glitch_template_taus` are replaced.
    """
