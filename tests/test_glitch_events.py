"""Glitch storage, the glitch cuts, and the placeholder hooks in TODView and the sampler."""
import logging
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.detector_tod import DetectorTOD
from commander4.tod.glitches.events import GlitchEvents, GLITCH_BRIGHT, empty_glitch_grid
from commander4.data_models.pointing import PixelPointing
from commander4.data_models.scan_tod import ScanTOD
from commander4.tod.config import GlitchConfig
from commander4.tod.glitches.sampling import sample_glitches
from commander4.tod.view import TODView

_NTOD = 20


def _bright_event() -> GlitchEvents:
    """One bright event: five shape parameters plus three spline node values, samples 3-7 cut."""
    shape_params = [1.0, 0.5, 0.04, 0.02, 0.01, 0.1, 0.2, 0.3]
    return GlitchEvents([2], [12], [GLITCH_BRIGHT], [1.0], [3], [8], [len(shape_params)],
                        shape_params)


def test_good_sample_mask_cuts_half_open_ranges() -> None:
    events = GlitchEvents([2, 9], [5, 5], [GLITCH_BRIGHT, 1], [1.0, 1.0], [3, 9], [6, 9], [0, 0],
                          [])
    expected = np.ones(12, dtype=bool)
    expected[3:6] = False  # The second event has an empty range, so it cuts nothing.
    np.testing.assert_array_equal(events.good_sample_mask(12), expected)


def _band_and_samples(events: GlitchEvents) -> tuple[DetectorGroupTOD, SimpleNamespace]:
    """A one-scan, one-detector band without demodulation, and samples holding `events`."""
    raw_tod = np.arange(_NTOD, dtype=np.float32)
    pointing = PixelPointing(np.zeros(_NTOD, dtype=np.int64), np.zeros(_NTOD),
                             np.array([0], dtype=np.int64), None, None, 1, 1, _NTOD, _NTOD)
    detector = DetectorTOD(
        name="det", det_idx_fullband=0, tod=raw_tod, pointing=pointing, sampling_rate_hz=10.0,
        orbital_velocity_m_per_s=np.zeros(3, dtype=np.float32), huffman_tree=None,
        huffman_symbols=None, default_proc_mask=np.ones(12, dtype=bool), specific_proc_masks={},
        flag_encoded=np.zeros(_NTOD, dtype=np.int64), bad_data_bitmask=1,
        flag_is_compressed=False,
    )
    noise_model = SimpleNamespace(npar=1, params=np.array([1.0]))
    band = DetectorGroupTOD([ScanTOD([detector], 0.0, 1)], "EXP", "BAND", 1, 100.0, 10.0, 10.0, 1,
                            "I", noise_model)
    glitches = empty_glitch_grid(1, 1)
    glitches[0, 0] = events
    samples = SimpleNamespace(
        jumps=SimpleNamespace(get=lambda iscan, idet: SimpleNamespace(is_empty=lambda: True)),
        glitches=glitches, glitch_template_amps=np.zeros((1, 3, 8)),
        glitch_template_taus=np.ones((1, 3, 8)), glitch_events_detected=False,
        accept=np.ones((1, 1), dtype=bool))
    return band, samples


def test_view_applies_glitch_cuts_and_leaves_the_placeholder_models_inert() -> None:
    band, samples = _band_and_samples(_bright_event())
    view = TODView(band, samples).focus(0, band.scans[0].detectors[0])

    # The baseline-step and pulse models are placeholders, so the TOD is unchanged.
    np.testing.assert_array_equal(view.pre_demodulation_tod, view.raw_tod)
    np.testing.assert_array_equal(view.corrected_tod, view.raw_tod)
    np.testing.assert_array_equal(view.get_glitch_pulse_tod(), np.zeros(_NTOD))

    # The stored cut is already applied, through the bad-data cut that mapmaking also uses.
    expected = np.ones(_NTOD, dtype=bool)
    expected[3:8] = False
    np.testing.assert_array_equal(view.get_mask(proc_mask=False), expected)
    np.testing.assert_array_equal(view.get_mask(good_data_mask=False, proc_mask=False),
                                  np.ones(_NTOD, dtype=bool))


def test_placeholder_sampler_marks_detection_done_and_says_it_does_nothing(caplog) -> None:
    band, samples = _band_and_samples(GlitchEvents.empty())

    with caplog.at_level(logging.WARNING, logger="commander4.tod.glitches.sampling"):
        sample_glitches(MPI.COMM_SELF, band, samples, None, GlitchConfig(enabled=True))

    assert samples.glitch_events_detected
    assert samples.glitches[0, 0].num_events == 0
    assert "placeholder" in caplog.text
