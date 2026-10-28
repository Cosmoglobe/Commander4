"""`HFIModulation`: Planck HFI alternating-sample phase and baselines, and the step sampling them.

HFI's raw bolometer stream alternates positive and negative modulation half-cycles. On the first
Gibbs pass the sign of the first sample parity is identified with Commander3's middle-RING-pixel
sky cut. Two independent constant baselines, one per parity, are sampled on every pass.
`TODView.get_tod` demodulates with them.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import healpy as hp
import numpy as np
from numpy.typing import NDArray
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.tod.step import ChainArrays, SavedBandSample, TODStep
from commander4.tod.view import TODView

if TYPE_CHECKING:
    from commander4.data_models.tod_samples import TODSamples

logger = logging.getLogger(__name__)


class HFIModulation(TODStep):
    """The HFI modulation state of one band's detector-scans on one rank.

    Enabled only for bands whose reader sets `hfi_demodulation`. Each chain keeps its own, so the
    two interleaved chains draw independent baselines, and both arrays are restart state.

    Sampled:
        modulation_phase: (nscans, ndet) +1 or -1: the sign of the half-cycle that Python's even
            sample indices carry. Identified on the first pass, then fixed.
        baselines: (nscans, ndet, 2) Constant baseline of the even and of the odd samples, in
            detector units. Sampled on every pass.

    Attributes:
        cfg: `enabled`, whether the band's TOD is modulated (`DetectorGroupTOD.hfi_demodulation`).
        initialized: Whether the phase has been identified yet.
    """

    def __init__(self, experiment_data: DetectorGroupTOD, params: Bunch, my_band: Bunch,
                 band_comm: MPI.Comm, chain: int):
        self.cfg = Bunch(enabled=experiment_data.hfi_demodulation)
        if not self.cfg.enabled:
            return

        nscans, ndet = experiment_data.nscans, experiment_data.ndet
        # Until the first pass identifies the phase, demodulation flips the odd samples and
        # subtracts no baselines.
        self.modulation_phase = np.ones((nscans, ndet), dtype=np.int8)
        self.baselines = np.zeros((nscans, ndet, 2), dtype=np.float64)
        self.initialized = False


    def chain_arrays(self) -> ChainArrays:
        """The phase and the baselines, both per scan and both restart state."""
        if not self.cfg.enabled:
            return ChainArrays()
        return ChainArrays(per_scan={"modulation_phase": self.modulation_phase,
                                     "baselines": self.baselines})


    def read_chain(self, saved: SavedBandSample) -> None:
        """Continue from the phase and the baselines in a band chain file."""
        if not self.cfg.enabled:
            return
        self.modulation_phase = saved.per_scan("modulation_phase").astype(np.int8)
        self.baselines = saved.per_scan("baselines").astype(self.baselines.dtype)
        self.initialized = True


    def demodulate(self, iscan: int, idet: int, tod: NDArray,
                   subtract_baselines: bool = True) -> NDArray:
        """Undo the alternating modulation of one detector-scan's TOD, for `TODView`.

        Each sample parity has its baseline subtracted and is multiplied by its sign: +phase for
        Python's even sample indices, -phase for the odd ones. Returns a new array.

        Args:
            subtract_baselines: False leaves the baselines in, for the residual the baseline fit
                works on (`TODView.get_tod(keep=("hfi_baselines",))`).
        """
        phase = self.modulation_phase[iscan, idet]
        baseline_first, baseline_second = self.baselines[iscan, idet]
        if not subtract_baselines:
            baseline_first = baseline_second = 0.0
        demodulated = np.array(tod, copy=True)
        demodulated[0::2] = phase * (demodulated[0::2] - baseline_first)
        demodulated[1::2] = -phase * (demodulated[1::2] - baseline_second)
        return demodulated


    def sample(self, tod_samples: TODSamples, compsep_output: NDArray,
               rng: np.random.Generator) -> None:
        """Sample HFI baselines for one Gibbs pass and initialize the phase on the first pass.

        The first pass follows Commander3's bootstrap: sample the two raw parity means without a
        sky subtraction, then demodulate with them and the trial phase +1, and take the sign of the
        sky in the middle four percent of RING pixels as the phase. Later passes subtract the
        current gain-scaled sky signal (sky, orbital dipole and far sidelobes) before drawing the
        two baselines. Commander3 leaves the far sidelobes in at this step (its `oper_minimal`).

        Args:
            tod_samples: The band's sampled state, which the TOD views read. A detector-scan that
                cannot be demodulated is rejected in its `selection`.
            rng: This rank's random generator for the baseline draws.
        """
        band_comm, experiment_data = tod_samples.band_comm, tod_samples.experiment_data
        first_pass = not self.initialized
        stats = self._sample_baselines(tod_samples, compsep_output, not first_pass, rng)
        if first_pass:
            if band_comm.Get_rank() == 0:
                logger.info(f"{experiment_data.band_name}: First time doing HFI baseline fitting: "\
                            "Finding modulation phase.")
            self._set_modulation_phase(tod_samples)

        # Report some stats on the baseline fit.
        totals = band_comm.reduce(stats, op=MPI.SUM)  # None on every rank but the root.
        if band_comm.Get_rank() == 0 and totals[3] > 0:
            common_mode, half_depth, step, nscans = totals
            message = (f"HFI baselines ({experiment_data.band_name}): common mode "
                       f"{common_mode/nscans:.5e}, modulation half-depth {half_depth/nscans:.4e}")
            # If only baselines were sampled they should move by about 1 sigma, but as other
            # things are moving in the chain it will move by much more in practice.
            if not first_pass:
                message += f", moved {step/nscans:.2f} posterior sigma"
            logger.info(message)


    def _sample_baselines(self, tod_samples: TODSamples, compsep_output: NDArray | None,
                          subtract_sky: bool, rng: np.random.Generator) -> NDArray:
        """Draw the two per-parity constant baselines from their white-noise posteriors.

        Before demodulation, one detector-scan follows

        ``d[p] = baseline[parity] + sign[parity] * (gain * sky[p] + pulses[p] + ...) + noise[p]``,

        where the signs are ``(+phase, -phase)`` for Python's even/odd indices. The residual is the
        demodulated TOD with every model removed except the baselines, so each sample holds
        ``sign[parity] * baseline[parity] + noise``. With independent white noise, each baseline
        has posterior mean ``sign[parity]`` times the masked residual mean of its parity, and
        posterior standard deviation ``sigma0/sqrt(number of samples)``.

        Returns:
            Running sums over this rank's accepted detector-scans, for the log line in `sample`:
            the common mode, the modulation half-depth, the step taken in units of the posterior
            standard deviation, and the number of detector-scans summed over.
        """
        scan_view = TODView(tod_samples.experiment_data, tod_samples, compsep_output=compsep_output)
        stats = np.zeros(4)

        for view in scan_view.iter_focused(accepted_only=True):
            # C3 uses its n_corr processing mask here: flags and bright masked sky regions do not
            # inform the DC levels.
            mask = view.get_mask(proc_mask_type="ncorr")
            # No sky can be subtracted before the phase is known (see `sample`).
            keep = ("hfi_baselines",)
            if not subtract_sky:  # The three components that are considered part of the sky.
                keep += ("sky", "orbital_dipole", "far_sidelobes")
            residual_tod = view.get_tod(keep=keep)

            phase = self.modulation_phase[view.iscan, view.idet]
            signs = (phase, -phase)
            previous = self.baselines[view.iscan, view.idet].copy()
            step = 0.0
            for parity in range(2):
                parity_mask = mask[parity::2]
                count = int(np.count_nonzero(parity_mask))
                if count == 0:
                    # A missing parity makes demodulation undefined: both half-cycles need their
                    # own baseline. Match C3 by rejecting this detector-scan rather than inventing
                    # a value.
                    tod_samples.selection.accept[view.iscan, view.idet] = False
                    break

                # Python parity 0 is Fortran's odd samples. For white noise,
                # Var(mean) = sigma0^2 / count.
                parity_mean = float(np.mean(residual_tod[parity::2][parity_mask]))
                posterior_mean = signs[parity] * parity_mean
                posterior_sd = view.sigma0 / np.sqrt(count)
                baseline = posterior_mean + float(rng.normal()) * posterior_sd
                self.baselines[view.iscan, view.idet, parity] = baseline
                step += abs(baseline - previous[parity]) / posterior_sd / 2

            if tod_samples.selection.accept[view.iscan, view.idet]:
                b0, b1 = self.baselines[view.iscan, view.idet]
                stats += [(b0 + b1)/2, abs(b0 - b1)/2, step, 1.0]

        return stats


    def _set_modulation_phase(self, tod_samples: TODSamples) -> None:
        """Identify the positive raw parity with Commander3's middle-RING-pixel sky cut.

        The phase is still the trial value +1 here. Demodulated with it and with the bootstrap
        baselines, the bright sky in the cut comes out positive if the trial is right, and negative
        if Python parity 0 carries the negative half-cycle. The phase is fixed after this bootstrap
        and is not sampled again on later Gibbs passes.
        """
        scan_view = TODView(tod_samples.experiment_data, tod_samples)
        rejected = 0

        for view in scan_view.iter_focused(accepted_only=True):
            tod = view.get_tod(keep=("sky", "orbital_dipole", "far_sidelobes"))
            # Phase identification follows C3's flag cut only, not the processing mask used for
            # baseline fitting. Both members of a pair must be valid. Drop a final unpaired sample
            # for odd-length scans by stopping the first slice at tod.size - 1.
            good = view.get_mask(proc_mask=False)
            first = slice(0, tod.size - 1, 2)
            second = slice(1, tod.size, 2)
            npix = hp.nside2npix(view.detector.nside)
            pair_mask = good[first] & good[second]
            # C3 uses the middle four percent of RING pixel numbers as a strong-signal sky strip.
            # This avoids deciding the sign from noise-dominated samples away from the bright
            # crossing.
            pair_mask &= view.pix[first] > 0.48 * npix
            pair_mask &= view.pix[first] < 0.52 * npix

            if not pair_mask.any():
                tod_samples.selection.accept[view.iscan, view.idet] = False
                rejected += 1
                continue

            # The sum of the two demodulated samples of a pair is C3's difference of the two
            # baseline-subtracted raw samples.
            if float(np.mean((tod[first] + tod[second])[pair_mask])) < 0.0:
                self.modulation_phase[view.iscan, view.idet] = -1

        self.initialized = True
        if rejected:
            logger.warning(f"Band {tod_samples.experiment_data.band_name}: rejected {rejected} HFI "
                           "detector-scans with no valid samples in the modulation-phase sky cut.")
