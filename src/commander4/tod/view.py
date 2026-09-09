"""`TODView`: the read interface to one detector-scan's data and every TOD derived from it.

One view is focused on one detector at a time and rebuilt as the scan loop advances, so a band
never holds more than one detector decoded. It also owns block-averaging, so a step that wants a
coarser sample rate (gain calibration) asks for a view at that rate instead of downsampling by hand.
"""
import numpy as np
import healpy as hp
import logging
from numpy.typing import NDArray

from pixell.bunch import Bunch

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.pointing import remap_pix_nside
from commander4.data_models.tod_samples import TODSamples
from commander4.tod.noise.sample_ncorr import realize_noise_in_gaps
from commander4.tod.sky_projection import project_sky_to_tod, get_s_orb_tod

logger = logging.getLogger(__name__)


class TODView:
    """Materialize one detector at a time and build derived TOD views on demand.

    The view keeps only the currently focused detector decoded in memory. Calling ``focus()``
    always discards the previous detector's cached arrays, which matches the one-detector-at-a-time
    TOD-processing architecture.

    Downsampling. The view carries a single ``downsample_factor``, fixed at construction: a step
    that wants a coarser rate builds its own view at that rate. The ``raw_tod`` / ``corrected_tod``
    and the *internal* full-rate pointing stay at full resolution, because model TODs must be
    integrated over each block rather than sampled at its center. Every quantity exposed
    downstream is at the active resolution:
    ``pix`` / ``psi`` take block centers, the model and data getters (``get_tod``,
    ``get_static_sky_tod``, ``get_orbital_dipole_tod``, ``get_calib_tod``) are block-averaged over
    the block's good samples, and ``get_mask`` keeps a block when more than ``mask_threshold`` of
    its samples pass. ``downsample_factor == 1`` (the default) is a no-op, so callers that never
    downsample see the full-rate arrays unchanged.
    """

    _ALL_GAIN_TERMS = ("abs", "rel", "temp")
    # Calibration targets, mapped onto the model signals they span. Sampling a gain term against
    # one of these reduces the calibration residual to (target gain term) * s_cal + noise.
    _CALIB_TARGET_SIGNALS = {
        "orbital_dipole": ("orbital_dipole",),
        "sky": ("sky", "orbital_dipole"),
        "sky_no_dipole": ("sky",),
    }

    def __init__(
        self,
        experiment_data: DetectorGroupTOD,
        tod_samples: TODSamples,
        compsep_output: NDArray | None = None,
        downsample_factor: int = 1,
        proc_mask_type: str = "",
        mask_threshold: float = 0.5,
    ):
        """Initialize a detector-local view over one band's TOD data.

        Args:
            experiment_data: Static TOD container for the current band.
            tod_samples: Sampled gain and noise parameters for the current chain state.
            compsep_output: Optional default sky model used by sky-subtraction helpers.
            downsample_factor: Block-averaging factor applied to every derived TOD/mask the view
                returns (1 = full resolution). Each operation that needs a coarser rate (e.g. gain
                calibration) constructs its own view at the desired factor.
            proc_mask_type: Processing mask this view works under, in ``get_mask``'s vocabulary.
                It is what ``get_calib_tod`` cuts on and what block-averaging averages over, so the
                block average and the surviving-block count are taken over the same samples. Only
                the calibration path reads it; ``get_mask`` callers still name their own mask.
            mask_threshold: Fraction of a block's samples that must pass the mask for the block to
                survive downsampling. Unused at ``downsample_factor == 1``.
        """
        self.experiment_data = experiment_data
        self.tod_samples = tod_samples
        self.compsep_output = compsep_output
        if downsample_factor < 1:
            raise ValueError(f"downsample_factor must be >= 1, got {downsample_factor}.")
        if not 0.0 <= mask_threshold < 1.0:
            raise ValueError(f"mask_threshold must be in [0, 1), got {mask_threshold}.")
        self._downsample_factor = int(downsample_factor)
        self._proc_mask_type = proc_mask_type
        self._mask_threshold = float(mask_threshold)
        self._iscan: int | None = None
        self._idet: int | None = None
        self._det = None
        self._clear_cache()

    def _clear_cache(self):
        """Drop all arrays materialized for the current detector."""
        self._raw_tod = None
        self._corrected_tod = None
        self._pix = None
        self._psi = None
        self._flag = None
        self._ds_indices = None
        self._ds_good = None
        self._static_sky = None
        self._orbital_dipole = None
        self._gap_noise: dict[str, NDArray] = {}

    def focus(self, iscan: int, det: DetectorTOD) -> "TODView":
        """Focus the view on one present detector and discard any previous materialization.

        Args:
            iscan: Scan index, local to this rank.
            det: The detector to focus on, an element of ``scans[iscan].detectors`` (which holds
                only the detectors actually present in that scan). Its full-band index
                ``det.det_idx_fullband`` is the column used to address every per-detector sample
                array, so detectors absent from a scan are simply skipped rather than misaligning
                the dense ``(nscans, ndet)`` arrays.
        """
        self._det = det
        self._iscan = iscan
        self._idet = det.det_idx_fullband  # full-band column in the (nscans, ndet) sample arrays
        self._clear_cache()
        return self

    def iter_focused(self, *, accepted_only: bool = False):
        """Focus on each present detector-scan in turn, yielding this re-focused view.

        Canonical detector-scan loop for TOD processing. It folds away the per-detector boilerplate
        (``focus`` + the ``accept`` check) and, importantly, never exposes a per-scan detector
        position that could be mistaken for a dense-array column: address the ``(nscans, ndet)``
        sample arrays through ``view.idet`` (the full-band column) and ``view.iscan`` only.

        The same view instance is re-focused and yielded on every iteration (matching the
        one-detector-at-a-time design), so callers must consume each view within the loop body and
        not retain it across iterations.

        Args:
            accepted_only: If True, skip detector-scans whose ``accept`` flag is False (bad data);
                if False (default), every present detector-scan is yielded (e.g. white-noise/jump
                passes).

        Yields:
            TODView: this view, focused on the current present detector-scan.
        """
        accept = self.tod_samples.accept if accepted_only else None
        for iscan, det in self.experiment_data.iter_detector_scans(accept):
            yield self.focus(iscan, det)

    def _require_focus(self):
        """Return the current detector or raise if the view was not focused yet."""
        if self._iscan is None or self._idet is None or self._det is None:
            raise ValueError("Attempted to use TODView before calling TODView.focus().")
        return self._det

    @property
    def iscan(self) -> int:
        self._require_focus()
        return self._iscan

    @property
    def idet(self) -> int:
        """Full-band detector index (``det_idx_fullband``): the column in per-detector arrays."""
        self._require_focus()
        return self._idet

    @property
    def detector(self):
        return self._require_focus()

    @property
    def fsamp(self) -> float:
        return self.detector.fsamp

    @property
    def downsample_factor(self) -> int:
        return self._downsample_factor

    @property
    def response_I_P(self) -> tuple[float, float]:
        return self.detector.response_I_P

    @property
    def noise_params(self) -> NDArray:
        self._require_focus()
        return self.tod_samples.noise_params[self._iscan, self._idet]

    @property
    def sigma0(self) -> float:
        return float(self.noise_params[0])

    @property
    def accept(self) -> bool:
        """Whether the focused detector-scan is accepted, i.e. present *and* not flagged as bad
        data. ``accept`` (data quality) is distinct from ``present`` (data exists at all)."""
        self._require_focus()
        return bool(self.tod_samples.accept[self._iscan, self._idet])

    def get_gain(self, gain_terms: tuple[str, ...] = _ALL_GAIN_TERMS) -> float:
        """Return the sum of the selected gain terms; an empty tuple gives zero."""
        gain = 0.0
        for term in gain_terms:
            if term == "abs":
                gain += self.tod_samples.abs_gain
            elif term == "rel":
                gain += self.tod_samples.rel_gain[self.idet]
            elif term == "temp":
                gain += self.tod_samples.temporal_gain[self.iscan, self.idet]
        return float(gain)


    # ------------------------------------------------------------------ downsampling helpers
    @property
    def _block_indices(self) -> NDArray[np.integer]:
        """Block-center sample indices mapping the active resolution back to the full rate.

        For ``factor == 1`` this is ``arange(ntod)``; otherwise the full-rate stream is cut into
        contiguous blocks of ``factor`` samples and only the leading complete blocks are kept (the
        trailing partial block, if any, is dropped), matching the data block-averaging. Cached per
        detector (reset by ``_clear_cache``).
        """
        if self._ds_indices is None:
            factor, ntod = self._downsample_factor, self.detector.ntod
            if factor == 1:
                self._ds_indices = np.arange(ntod, dtype=np.int64)
            else:
                # Keep every *complete* block of `factor` samples (ntod // factor of them)
                # Trailing partial block is dropped.
                nblock = ntod // factor
                edges = np.arange(nblock + 1, dtype=np.int64) * factor
                self._ds_indices = (edges[1:] + edges[:-1]) // 2
        return self._ds_indices

    @property
    def _block_good(self) -> NDArray[np.bool_]:
        """Full-rate mask of the samples block-averaging is allowed to use. Cached per detector.

        Flagged samples still hold the raw glitch (HFI cosmic rays reach thousands of sigma), and
        processing-masked samples are exactly the sky the calibration must not fit, so both are kept
        out of the block average as well as out of the count that decides whether a block survives.
        """
        if self._ds_good is None:
            self._ds_good = self._fullres_mask(proc_mask_type=self._proc_mask_type)
        return self._ds_good

    def _downsample_mean(self, arr: NDArray[np.floating]) -> NDArray[np.floating]:
        """Block-average a full-rate array over its good samples (identity when factor == 1).

        Averaging over the good samples alone, rather than over the whole block, is what makes the
        fractional keep threshold in ``_downsample_keep`` safe: a surviving block may still contain
        flagged samples, and including those would move its mean by many sigma. Model TODs go
        through here too, so model and data are averaged over the same samples -- Planck sweeps more
        than a degree during a 0.2 s block, so averaging them over different subsets would not cancel.
        """
        factor = self._downsample_factor
        if factor == 1:
            return arr
        n = self.detector.ntod // factor        # complete blocks; a trailing partial one is dropped
        blocks = arr[:n * factor].reshape(n, factor)
        good = self._block_good[:n * factor].reshape(n, factor)
        # A block with no good samples never survives _downsample_keep, so its value is unused;
        # dividing by 1 there just avoids a NaN propagating through later arithmetic.
        return np.where(good, blocks, 0.0).sum(axis=-1) / np.maximum(good.sum(axis=-1), 1)

    def _downsample_keep(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Reduce a full-rate boolean mask onto the active resolution by its kept fraction.

        A block survives when more than ``mask_threshold`` of its samples pass. [C3:
        `comm_tod_mod.f90::downsample_tod`, whose `threshold` argument defaults to the same 0.5.]
        Requiring *every* sample instead is far too strict for HFI, where cosmic-ray glitches leave
        short flagged stretches all over the TOD: at 857 GHz that rejects 78% of the one-second
        calibration blocks, against 3% for the fractional rule.
        """
        factor = self._downsample_factor
        if factor == 1:
            return mask
        n = self.detector.ntod // factor
        return mask[:n * factor].reshape(n, factor).mean(axis=-1) > self._mask_threshold

    # ------------------------------------------------------------------ raw / pointing accessors
    @property
    def raw_tod(self) -> NDArray[np.floating]:
        """The decoded detector TOD exactly as stored, at full resolution.
        See `self.corrected_tod` for the TOD to use for science. """
        if self._raw_tod is None:
            self._raw_tod = self.detector.tod
        return self._raw_tod

    @property
    def corrected_tod(self) -> NDArray[np.floating]:
        """TOD (in detector units) after low-level corrections, such as jumps or demodulation.

        Figures out what low-level corrections are active and applies those to `self.raw_tod`.
        Currently implemented adjustments include:
            - Jump corrections, requiring the jump-finding sampling step.
            - HFI demodulation, requiring the demodulation phase and the baseline sampling steps.
        """
        if self._corrected_tod is None:
            jump = self.tod_samples.jumps.get(self.iscan, self.idet)
            corrected = self.raw_tod if jump.is_empty() else jump.apply(self.raw_tod)

            if getattr(self.experiment_data, "hfi_demodulation", False):
                if not self.tod_samples.modulation_phase_initialized:
                    raise RuntimeError("HFI TOD requested before modulation phase initialization.")
                phase = self.tod_samples.modulation_phase[self.iscan, self.idet]
                baseline_first, baseline_second = self.tod_samples.baselines[self.iscan, self.idet]
                corrected = np.array(corrected, copy=True)
                corrected[0::2] = phase * (corrected[0::2] - baseline_first)
                corrected[1::2] = -phase * (corrected[1::2] - baseline_second)

            self._corrected_tod = corrected
        return self._corrected_tod

    @property
    def _fullres_pix(self) -> NDArray[np.integer]:
        """Full-rate pixel pointing (used internally for model evaluation and mask projection)."""
        if self._pix is None:
            self._pix, self._psi = self.detector.get_pix_psi()
        return self._pix

    @property
    def _fullres_psi(self) -> NDArray[np.floating] | NDArray[np.integer]:
        """Full-rate polarization angle (used internally for model evaluation)."""
        if self._psi is None:
            self._pix, self._psi = self.detector.get_pix_psi()
        return self._psi

    @property
    def pix(self) -> NDArray[np.integer]:
        """Pixel pointing at the active resolution (block centers when downsampled)."""
        if self._downsample_factor == 1:
            return self._fullres_pix
        return self._fullres_pix[self._block_indices]

    @property
    def psi(self) -> NDArray[np.floating] | NDArray[np.integer]:
        """Polarization angle at the active resolution (block centers when downsampled)."""
        if self._downsample_factor == 1:
            return self._fullres_psi
        return self._fullres_psi[self._block_indices]

    @property
    def flag(self) -> NDArray[np.integer]:
        if self._flag is None:
            self._flag = self.detector.flag
        return self._flag

    # ------------------------------------------------------------------ masks
    def _project_processing_mask(self, mask_type: str = "") -> NDArray[np.bool_] | None:
        """Project a processing-mask HEALPix map onto the focused detector's pointing.

        The named ``mask_type`` is used when the band defines one under ``processing_masks:``;
        otherwise the band's default ``processing_mask:`` is used; if neither exists, returns
        ``None`` (no processing cut). A named type the band does not define falls back to the default
        *silently*, so single-mask bands need no per-operation entries (mistyped ``processing_masks:``
        keys are caught at config load, not per sample). The map is looked up at its native nside,
        converting the pointing from the detector's evaluation nside when the two differ.
        """
        specific = getattr(self.detector, "specific_proc_masks", None) or {}
        default = getattr(self.detector, "default_proc_mask", None)
        if mask_type and mask_type in specific:
            mask_map = specific[mask_type]
        elif default is not None:
            mask_map = default
        else:
            return None

        pix = remap_pix_nside(self._fullres_pix, self.detector.nside,
                              hp.npix2nside(mask_map.size))
        return mask_map[pix]


    def _fullres_mask(self, good_data_mask: bool = True, proc_mask: bool = True,
                      proc_mask_type: str = "") -> NDArray[np.bool_]:
        """Combine the bad-data flag cut and a sky processing mask, at the full sampling rate."""
        mask = np.ones(self.detector.ntod, dtype=bool)
        # Datasets without an explicit flag cut behave as if all samples pass it.
        if good_data_mask and getattr(self.detector, "_good_data_mask", None) is not None:
            mask &= self.detector.good_data_mask
        if proc_mask:
            proc = self._project_processing_mask(proc_mask_type)
            if proc is not None:
                mask &= proc
        return mask

    def get_mask(self, good_data_mask: bool = True, proc_mask: bool = True,
                 proc_mask_type: str = "") -> NDArray[np.bool_]:
        """Return a boolean keep-mask for the focused TOD at the active resolution.

        Combines the bad-data flag cut and a sky processing mask; either can be switched off. When
        the view is downsampled, the full-rate cut is reduced by ``_downsample_keep``.

        Args:
            good_data_mask: Whether to exclude samples flagged as bad by the bit-flag cut.
            proc_mask: Whether to apply a sky processing mask.
            proc_mask_type: Which processing mask to apply: a key under ``processing_masks:`` in
                the band's parameter section, or "" to use the default ``processing_mask:`` entry.
        """
        return self._downsample_keep(self._fullres_mask(good_data_mask, proc_mask, proc_mask_type))


    # ------------------------------------------------------------------ model TODs
    def get_static_sky_tod(self, compsep_output: NDArray | None = None) -> NDArray[np.floating]:
        """Evaluate the static sky model along the focused detector pointing.

        The model is evaluated at the full sampling rate and then block-averaged onto the active
        resolution, integrating the model over the scan path within each block rather than sampling
        it at the block-center pixel. Model and data thereby see the same downsampling transfer
        function, which keeps e.g. gain estimates unbiased.

        Args:
            compsep_output: Sky model to use instead of the view's own. Only the view's own model
                is cached, since an override is a one-off.
        """
        if compsep_output is None and self._static_sky is not None:
            return self._static_sky
        sky_model = self.compsep_output if compsep_output is None else compsep_output
        if sky_model is None:
            raise ValueError("A component-separation sky map must be provided for sky subtraction.")

        # The realized sky model is full-sky (ncomp, npix) on the band master and in non-sparse map
        # mode, but only (ncomp, n_local) on workers in sparse mode (see
        # communication._realize_and_distribute_sky). The column count is what tells the two apart.
        if sky_model.shape[-1] == 12 * self.experiment_data.nside**2:
            sky_pix = self._fullres_pix
        else:
            sky_pix = self.experiment_data.pixel_domain.to_local(self._fullres_pix)
        sky = self._downsample_mean(project_sky_to_tod(sky_model, sky_pix, psi=self._fullres_psi,
                                                       response_I_P=self.response_I_P))
        if compsep_output is None:
            self._static_sky = sky
        return sky


    def get_orbital_dipole_tod(self) -> NDArray[np.floating]:
        """Evaluate the orbital dipole for the focused detector at the active resolution.

        Like ``get_static_sky_tod``, the dipole is built at full rate and block-averaged.
        """
        if self._orbital_dipole is None:
            full = get_s_orb_tod(self.detector, self.experiment_data, self._fullres_pix)
            self._orbital_dipole = self._downsample_mean(full)
        return self._orbital_dipole


    def get_tod(
        self,
        *,
        subtract: tuple[tuple[str, tuple[str, ...]], ...] | None = None,
        divide_by_gain: tuple[str, ...] | None = None,
        compsep_output: NDArray | None = None,
    ) -> NDArray[np.floating]:
        """Return a detector-local TOD after doing both optional sky-subtractions, and after
        last-minute corrections like jump offsets and HFI demodulation.

        Args:
            subtract: Sequence of ``(signal_name, gain_terms)`` pairs, where the signal is
                ``"sky"`` or ``"orbital_dipole"``. Each signal is evaluated and subtracted after
                multiplying it by the selected gain subset.
            divide_by_gain: Gain terms to divide the final TOD by, or ``None``.
            compsep_output: Optional sky model override for static-sky subtraction.
        """
        tod = np.array(self._downsample_mean(self.corrected_tod), copy=True)

        if subtract is not None:
            # Every residual this class supports is a linear combination of the two model TODs.
            for signal_name, gain_terms in subtract:
                if signal_name == "sky":
                    signal = self.get_static_sky_tod(compsep_output=compsep_output)
                elif signal_name == "orbital_dipole":
                    signal = self.get_orbital_dipole_tod()
                else:
                    raise ValueError(f"Unknown TOD signal '{signal_name}'.")
                tod -= self.get_gain(gain_terms) * signal

        if divide_by_gain is not None:
            gain = self.get_gain(divide_by_gain)
            if gain == 0:
                raise ValueError("Cannot divide TOD by a zero gain.")
            tod /= gain

        return tod


    # ------------------------------------------------------------------ gain calibration
    def _gap_noise_draw(self, method: str,
                        compsep_output: NDArray | None = None) -> NDArray[np.floating]:
        """Constrained noise realization at the masked samples of the calibration TOD.

        The noise residual ``d - g*(s_sky + s_orb)`` (full gain) is identical for every gain term, so
        the 1/f + white gap draw is computed once and shared across the abs/rel/temporal solves
        (cached per method, at the view's active resolution). ``method`` is ``'fallback'`` or
        ``'full_cg'``.
        """
        cached = self._gap_noise.get(method)
        if cached is not None:
            return cached
        mask = self.get_mask(proc_mask_type=self._proc_mask_type)
        s_sky = self.get_static_sky_tod(compsep_output=compsep_output)
        s_orb = self.get_orbital_dipole_tod()
        data = self._downsample_mean(self.corrected_tod)
        # Noise residual: data minus the full sky model at the full gain (true noise at valid
        # samples; Galactic-plane garbage at masked ones, which the realization replaces).
        noise_resid = data - self.get_gain(self._ALL_GAIN_TERMS) * (s_sky + s_orb)
        samprate = self.fsamp / self._downsample_factor  # block-averaging downsamples fsamp
        draw = realize_noise_in_gaps(noise_resid, mask, self.experiment_data.noise_model,
                                     self.noise_params, samprate, self.fsamp, method)
        self._gap_noise[method] = draw
        return draw


    def _fill_masked_calibration_samples(
        self,
        tod: NDArray[np.floating],
        mask: NDArray[np.bool_],
        signal: NDArray[np.floating],
        gain_terms: tuple[str, ...],
        rng: np.random.Generator | None,
        method: str = "wn",
        compsep_output: NDArray | None = None,
    ) -> NDArray[np.floating]:
        """Fill masked calibration samples with the target signal plus a noise realization.

        The masked regions retain only the target gain term times the calibrator signal, plus a
        noise draw: white (``method='wn'``, sigma0/sqrt(factor)) or a constrained correlated 1/f +
        white draw (``'fallback'``/``'full_cg'``) shared across gain terms via ``_gap_noise_draw``,
        so the masked residual carries the same 1/f structure as the surrounding valid data.
        """
        filled = np.array(tod, copy=True)
        gap = ~mask
        if not gap.any():
            return filled
        target = self.get_gain(gain_terms) * signal[gap]
        if method == "wn":
            sigma0_effective = self.sigma0 * np.sqrt(1.0 / self._downsample_factor)
            normal = np.random.normal if rng is None else rng.normal
            filled[gap] = target + normal(0.0, sigma0_effective, target.shape)
        else:
            draw = self._gap_noise_draw(method, compsep_output=compsep_output)
            filled[gap] = target + draw[gap]
        return filled


    def get_calib_tod(
        self,
        target_term: str,
        calibrate_against: str,
        *,
        compsep_output: NDArray | None = None,
        fill_masked: bool = True,
        gap_fill_method: str = "wn",
        rng: np.random.Generator | None = None,
    ) -> Bunch:
        """Return the residual, calibrator signal, and mask used to sample one gain term.

        Everything is returned at the view's active downsample resolution. The detector model is
        ``d = (g_abs + g_rel + g_temp) * (s_sky + s_orb) + n``. To sample ``target_term`` against a
        calibrator signal ``s_cal`` (the subset of {static sky, orbital dipole} selected by
        ``calibrate_against``), each model signal is subtracted with the appropriate gain terms so
        the residual reduces to ``g_target * s_cal + n``:
            - signals making up the calibrator keep the target term (only the *other* terms are
              subtracted), contributing ``g_target * s`` to the residual;
            - signals outside the calibrator are subtracted in full and thus removed.

        Args:
            target_term: Gain term being sampled, one of ``_ALL_GAIN_TERMS`` ("abs", "rel", "temp").
            calibrate_against: Calibrator, one of "orbital_dipole", "sky", or "sky_no_dipole".
            compsep_output: Optional sky-model override for the static-sky term.
            fill_masked: If True, fill masked samples with ``g_target * s_cal`` plus a noise draw.
            gap_fill_method: How masked samples are filled when ``fill_masked``: ``'wn'`` (white
                noise), ``'fallback'`` (stationary 1/f Wiener draw), or ``'full_cg'`` (masked
                constrained-CG 1/f draw). See ``_fill_masked_calibration_samples``.
            rng: Optional NumPy generator for the masked-sample white noise (``'wn'`` only).

        Returns:
            Bunch with ``tod`` (residual), ``s_cal``, and ``mask``.
        """
        if target_term not in self._ALL_GAIN_TERMS:
            raise ValueError(f"Unknown gain term '{target_term}'; expected one of "
                             f"{self._ALL_GAIN_TERMS}.")
        if calibrate_against not in self._CALIB_TARGET_SIGNALS:
            raise ValueError(f"Unknown calibrate_against '{calibrate_against}'; expected one of "
                             f"{tuple(self._CALIB_TARGET_SIGNALS)}.")

        mask = self.get_mask(proc_mask_type=self._proc_mask_type)
        s_sky = self.get_static_sky_tod(compsep_output=compsep_output)
        s_orb = self.get_orbital_dipole_tod()

        calib_signals = self._CALIB_TARGET_SIGNALS[calibrate_against]
        other_terms = tuple(t for t in self._ALL_GAIN_TERMS if t != target_term)
        # Calibrator signals keep the target gain term (subtract only the others); non-calibrator
        # signals are subtracted in full so they drop out of the residual entirely.
        subtract = tuple((name, other_terms if name in calib_signals else self._ALL_GAIN_TERMS)
                         for name in ("sky", "orbital_dipole"))
        s_cal = np.zeros_like(s_sky)
        if "sky" in calib_signals:
            s_cal = s_cal + s_sky
        if "orbital_dipole" in calib_signals:
            s_cal = s_cal + s_orb

        tod = self.get_tod(subtract=subtract, compsep_output=compsep_output)
        if fill_masked:
            tod = self._fill_masked_calibration_samples(tod, mask, s_cal, (target_term,), rng,
                                                        method=gap_fill_method,
                                                        compsep_output=compsep_output)
        return Bunch(tod=tod, s_cal=s_cal, mask=mask)
