"""Planck HFI alternating-sample phase, baseline sampling, and demodulation state.

HFI's raw bolometer stream alternates positive and negative modulation half-cycles. On the first
Gibbs pass the sign of the first sample parity is identified with Commander3's middle-RING-pixel
sky cut. Two independent constant baselines, one per parity, are sampled on every pass. ``TODView``
applies those samples lazily when downstream code requests a processed TOD.
"""
import logging

import healpy as hp
import numpy as np
from numpy.typing import NDArray

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.tod_samples import TODSamples
from commander4.tod.view import TODView

logger = logging.getLogger(__name__)


def _sample_baselines(
    experiment_data: DetectorGroupTOD,
    tod_samples: TODSamples,
    compsep_output: NDArray | None,
    subtract_sky: bool,
    rng: np.random.Generator | None,
) -> None:
    """Draw the two per-parity constant baselines from their white-noise posteriors.

    Before demodulation, one detector-scan follows

    ``d[p] = baseline[parity] + modulation_sign[parity] * gain * sky[p] + noise[p]``,

    where the signs are ``(+phase, -phase)`` for Python's even/odd indices. With independent white
    noise, each constant baseline has posterior mean equal to the masked residual mean and posterior
    standard deviation ``sigma0/sqrt(number of samples)``.
    """
    scan_view = TODView(experiment_data, tod_samples, compsep_output=compsep_output)
    normal = np.random.normal if rng is None else rng.normal

    for view in scan_view.iter_focused(accepted_only=True):
        # Baselines are instrument parameters, so fit the un-demodulated stream rather than
        # TODView.corrected_tod, which would apply the baselines we are currently trying to sample.
        # C3 uses its n_corr processing mask here: flags and bright masked sky regions do not inform
        # the DC levels.
        mask = view.get_mask(proc_mask_type="ncorr")
        raw_tod = view.raw_tod
        phase = tod_samples.modulation_phase[view.iscan, view.idet]
        gain = view.get_gain()

        # The first pass cannot subtract a sky until it knows which parity is positive. Its raw
        # parity means provide the bootstrap baselines used by _set_modulation_phase below. Once
        # the phase is known, later Gibbs passes condition the baselines on the current sky and gain.
        if subtract_sky:
            sky_tod = view.get_static_sky_tod() + view.get_orbital_dipole_tod()
        else:
            sky_tod = np.zeros(raw_tod.size, dtype=raw_tod.dtype)

        for parity in range(2):
            parity_mask = mask[parity::2]
            count = int(np.count_nonzero(parity_mask))
            if count == 0:
                # A missing parity makes demodulation undefined: both half-cycles need their own
                # baseline. Match C3 by rejecting this detector-scan rather than inventing a value.
                tod_samples.accept[view.iscan, view.idet] = False
                break

            # Python parity 0 is Fortran's odd samples. The raw modulation sign is +phase on
            # parity 0 and -phase on parity 1, so subtract that signed sky before fitting the DC
            # level. For white noise, Var(mean) = sigma0^2 / count.
            modulation_sign = phase if parity == 0 else -phase
            residual = raw_tod[parity::2] - modulation_sign * gain * sky_tod[parity::2]
            posterior_mean = float(np.mean(residual[parity_mask]))
            fluctuation = float(normal()) * view.sigma0 / np.sqrt(count)
            tod_samples.baselines[view.iscan, view.idet, parity] = \
                posterior_mean + fluctuation


def _set_modulation_phase(experiment_data: DetectorGroupTOD, tod_samples: TODSamples) -> None:
    """Identify the positive raw parity with Commander3's middle-RING-pixel sky cut.

    After removing the two bootstrap baselines, a bright sky crossing appears with opposite signs in
    adjacent samples. The sign of ``first - second`` therefore says whether Python parity 0 carries
    the positive or negative half-cycle. The phase is fixed after this bootstrap and is not sampled
    again on later Gibbs passes.
    """
    scan_view = TODView(experiment_data, tod_samples)
    rejected = 0

    for view in scan_view.iter_focused(accepted_only=True):
        raw_tod = view.raw_tod
        # Phase identification follows C3's flag cut only, not the processing mask used for baseline
        # fitting. Both members of a pair must be valid. Drop a final unpaired sample for odd-length
        # scans by stopping the first slice at raw_tod.size - 1.
        good = view.get_mask(proc_mask=False)
        first = slice(0, raw_tod.size - 1, 2)
        second = slice(1, raw_tod.size, 2)
        npix = hp.nside2npix(view.detector.nside)
        pair_mask = good[first] & good[second]
        # C3 uses the middle four percent of RING pixel numbers as a strong-signal sky strip. This
        # avoids deciding the sign from noise-dominated samples away from the bright crossing.
        pair_mask &= view._fullres_pix[first] > 0.48 * npix
        pair_mask &= view._fullres_pix[first] < 0.52 * npix

        if not pair_mask.any():
            tod_samples.accept[view.iscan, view.idet] = False
            rejected += 1
            continue

        baseline_first, baseline_second = tod_samples.baselines[view.iscan, view.idet]
        pair_difference = ((raw_tod[first] - baseline_first)
                           - (raw_tod[second] - baseline_second))
        # Phase starts at +1. Only negative pair differences need to flip it; TODView.corrected_tod
        # later applies (+phase, -phase) while subtracting the matching parity baselines.
        if float(np.mean(pair_difference[pair_mask])) < 0.0:
            tod_samples.modulation_phase[view.iscan, view.idet] = -1

    tod_samples.modulation_phase_initialized = True
    if rejected:
        logger.warning(f"Band {experiment_data.band_name}: rejected {rejected} HFI detector-scans "
                       "with no valid samples in the modulation-phase sky cut.")


def sample_hfi_baselines(
    experiment_data: DetectorGroupTOD,
    tod_samples: TODSamples,
    compsep_output: NDArray,
    rng: np.random.Generator | None = None,
) -> TODSamples:
    """Sample HFI baselines for one Gibbs pass and initialize the phase on the first pass.

    The first pass follows Commander3's bootstrap: sample the two raw parity means without a sky
    subtraction, then use their baseline-subtracted difference in the middle four percent of RING
    pixels to identify the positive parity. Later passes subtract the current gain-scaled sky model
    before drawing the two baselines.
    """
    if not tod_samples.hfi_demodulation:
        return tod_samples

    # The first pass is a two-step bootstrap: fit raw DC levels, then determine phase. Subsequent
    # passes keep that phase and resample only the baselines conditional on the latest Gibbs state.
    first_pass = not tod_samples.modulation_phase_initialized
    _sample_baselines(experiment_data, tod_samples, compsep_output, not first_pass, rng)
    if first_pass:
        _set_modulation_phase(experiment_data, tod_samples)
    return tod_samples
