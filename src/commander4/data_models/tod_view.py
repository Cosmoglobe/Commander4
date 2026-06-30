import numpy as np
import healpy as hp
import logging
from numpy.typing import NDArray

from pixell.bunch import Bunch

from commander4.data_models.detector_group_TOD import DetGroupTOD
from commander4.data_models.detector_TOD import DetectorTOD
from commander4.data_models.TOD_samples import TODSamples
from commander4.noise_sampling.sample_ncorr import realize_noise_in_gaps
from commander4.utils.map_utils import get_static_sky_TOD, get_s_orb_TOD

logger = logging.getLogger(__name__)


class TODView:
    """Materialize one detector at a time and build derived TOD views on demand.

    The view keeps only the currently focused detector decoded in memory. Calling ``focus()``
    always discards the previous detector's cached arrays, which matches the one-detector-at-a-time
    TOD-processing architecture.

    Downsampling. The view carries a single ``downsample_factor`` fixed at construction (overridable
    per detector via ``focus``). The raw ``tod`` / ``corrected_tod`` and the *internal* full-rate
    pointing stay at full resolution -- model TODs must be integrated over each block, not sampled at
    its center -- but every quantity exposed for downstream use is returned at the active resolution:
    ``pix`` / ``psi`` take block centers, the model and data getters (``get_tod``,
    ``get_static_sky_tod``, ``get_orbital_dipole_tod``, ``get_calib_tod``) are block-averaged, and
    ``get_mask`` is AND-reduced over each block. ``downsample_factor == 1`` (the default) is a no-op,
    so callers that never downsample see the full-rate arrays unchanged.
    """

    _ALL_GAIN_TERMS = ("abs", "rel", "temp")
    # Calibration targets, mapped onto the model signals they span. Sampling a gain term against
    # one of these reduces the calibration residual to (target gain term) * s_cal + noise.
    _CALIB_TARGET_SIGNALS = {
        "orbital_dipole": ("orbital_dipole",),
        "sky": ("sky",),
        "full_sky": ("sky", "orbital_dipole"),
    }

    def __init__(
        self,
        experiment_data: DetGroupTOD,
        tod_samples: TODSamples,
        compsep_output: NDArray | None = None,
        downsample_factor: int = 1,
    ):
        """Initialize a detector-local view over one band's TOD data.

        Args:
            experiment_data: Static TOD container for the current band.
            tod_samples: Sampled gain and noise parameters for the current chain state.
            compsep_output: Optional default sky model used by sky-subtraction helpers.
            downsample_factor: Block-averaging factor applied to every derived TOD/mask the view
                returns (1 = full resolution). Each operation that needs a coarser rate (e.g. gain
                calibration) constructs its own view at the desired factor.
        """
        self.experiment_data = experiment_data
        self.tod_samples = tod_samples
        self.compsep_output = compsep_output
        self._downsample_factor = self._validate_factor(downsample_factor)
        self._iscan: int | None = None
        self._idet: int | None = None
        self._det = None
        self._clear_cache()

    @staticmethod
    def _validate_factor(downsample_factor: int) -> int:
        factor = 1 if downsample_factor is None else int(downsample_factor)
        if factor < 1:
            raise ValueError("downsample_factor must be >= 1.")
        return factor

    def _clear_cache(self):
        """Drop all arrays materialized for the current detector."""
        self._tod = None
        self._corrected_tod = None
        self._pix = None
        self._psi = None
        self._flag = None
        self._ds_indices = None
        self._static_sky = None
        self._orbital_dipole = None
        self._gap_noise: dict[str, NDArray] = {}

    def focus(self, iscan: int, det: DetectorTOD,
              downsample_factor: int | None = None) -> "TODView":
        """Focus the view on one present detector and discard any previous materialization.

        Args:
            iscan: Scan index, local to this rank.
            det: The detector to focus on -- an element of ``scans[iscan].detectors`` (which holds
                only the detectors actually present in that scan). Its full-band index
                ``det.det_idx_fullband`` is the column used to address every per-detector sample
                array, so detectors absent from a scan are simply skipped rather than misaligning
                the dense ``(nscans, ndet)`` arrays.
            downsample_factor: Optional per-detector override of the view's downsample factor; the
                construction-time factor is kept when omitted.
        """
        self._det = det
        self._iscan = iscan
        self._idet = det.det_idx_fullband  # full-band column in the (nscans, ndet) sample arrays
        if downsample_factor is not None:
            self._downsample_factor = self._validate_factor(downsample_factor)
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
    def det_response(self) -> NDArray | None:
        return self.detector.det_response

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

    def get_gain(self, gain_terms: tuple[str, ...] | None = _ALL_GAIN_TERMS) -> float:
        """Return the selected subset of the current detector gain model."""
        if gain_terms is None:
            gain_terms = ()
        elif "all" in gain_terms:
            gain_terms = self._ALL_GAIN_TERMS

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
    def ntod(self) -> int:
        """Number of samples at the active (downsampled) resolution."""
        return self._block_indices.size

    def _downsample_mean(self, arr: NDArray[np.floating]) -> NDArray[np.floating]:
        """Block-average a full-rate array onto the active resolution (identity when factor == 1)."""
        factor = self._downsample_factor
        if factor == 1:
            return arr
        n = self._block_indices.size
        return arr[:n * factor].reshape(n, factor).mean(axis=-1)

    def _downsample_all(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """AND-reduce a full-rate boolean mask onto the active resolution.

        A downsampled sample is valid only if *every* high-res sample in its block is valid:
        block-averaging smears a single masked sample (e.g. a point source) across the whole averaged
        value, so masks are AND-reduced over each block rather than sampled at its center.
        """
        factor = self._downsample_factor
        if factor == 1:
            return mask
        n = self._block_indices.size
        return mask[:n * factor].reshape(n, factor).all(axis=-1)

    # ------------------------------------------------------------------ raw / pointing accessors
    @property
    def tod(self) -> NDArray[np.floating]:
        """The raw detector TOD, at full resolution."""
        if self._tod is None:
            self._tod = self.detector.tod
        return self._tod

    @property
    def corrected_tod(self) -> NDArray[np.floating]:
        """The raw TOD with the stored jump offsets applied, at full resolution (detector units)."""
        if self._corrected_tod is None:
            jump = self.tod_samples.jumps.get(self.iscan, self.idet)
            self._corrected_tod = self.tod if jump.is_empty() else jump.apply(self.tod)
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

        pix = self._fullres_pix
        map_nside = hp.npix2nside(mask_map.size)
        if map_nside != self.detector.nside:
            pix = hp.ang2pix(map_nside, *hp.pix2ang(self.detector.nside, pix))
        return mask_map[pix]


    def get_mask(self, good_data_mask: bool = True, proc_mask: bool = True,
                 proc_mask_type: str = "") -> NDArray[np.bool_]:
        """Return a boolean keep-mask for the focused TOD at the active resolution.

        Combines the bad-data flag cut and a sky processing mask; either can be switched off. The
        full-rate cut is AND-reduced onto the active downsample resolution.

        Args:
            good_data_mask: Whether to exclude samples flagged as bad by the bit-flag cut.
            proc_mask: Whether to apply a sky processing mask.
            proc_mask_type: Which processing mask to apply -- a key under ``processing_masks:`` in
                the band's parameter section, or "" to use the default ``processing_mask:`` entry.
        """
        mask = np.ones(self.detector.ntod, dtype=bool)
        # Datasets without an explicit flag cut behave as if all samples pass it.
        if good_data_mask and getattr(self.detector, "_good_data_mask", None) is not None:
            mask &= self.detector.good_data_mask
        if proc_mask:
            proc = self._project_processing_mask(proc_mask_type)
            if proc is not None:
                mask &= proc
        return self._downsample_all(mask)


    # ------------------------------------------------------------------ model TODs
    def _require_compsep_output(self, compsep_output: NDArray | None) -> NDArray:
        """Resolve the sky model to use for static-sky subtraction."""
        sky_model = self.compsep_output if compsep_output is None else compsep_output
        if sky_model is None:
            raise ValueError("A component-separation sky map must be provided for sky subtraction.")
        return sky_model


    def _sky_map_pix(self, sky_model: NDArray) -> NDArray[np.integer]:
        """Full-rate pixel indices into a sky map that may be full-sky or restricted to this rank.

        The realized sky model is full-sky ``(ncomp, npix)`` on the band master and in non-sparse
        map mode, but only ``(ncomp, n_local)`` on workers in sparse mode (see
        ``communication._realize_and_distribute_sky``). Distinguish the two by the map's column
        count and return either global HEALPix indices or compact local-buffer indices.
        """
        if sky_model.shape[-1] == 12 * self.experiment_data.nside**2:
            return self._fullres_pix
        return self.experiment_data.pixel_domain.to_local(self._fullres_pix)


    def get_static_sky_tod(self, compsep_output: NDArray | None = None) -> NDArray[np.floating]:
        """Evaluate the static sky model along the focused detector pointing.

        The model is evaluated at the full sampling rate and then block-averaged onto the active
        resolution, integrating the model over the scan path within each block rather than sampling
        it at the block-center pixel. Model and data thereby see the same downsampling transfer
        function, which keeps e.g. gain estimates unbiased.
        """
        sky_model = self._require_compsep_output(compsep_output)
        sky_pix = self._sky_map_pix(sky_model)
        if compsep_output is None:
            if self._static_sky is None:
                full = get_static_sky_TOD(sky_model, sky_pix, psi=self._fullres_psi)
                self._static_sky = self._downsample_mean(full)
            return self._static_sky
        full = get_static_sky_TOD(sky_model, sky_pix, psi=self._fullres_psi)
        return self._downsample_mean(full)


    def get_orbital_dipole_tod(self) -> NDArray[np.floating]:
        """Evaluate the orbital dipole for the focused detector at the active resolution.

        Like ``get_static_sky_tod``, the dipole is built at full rate and block-averaged.
        """
        if self._orbital_dipole is None:
            full = get_s_orb_TOD(self.detector, self.experiment_data, self._fullres_pix)
            self._orbital_dipole = self._downsample_mean(full)
        return self._orbital_dipole


    def _normalize_signal_name(self, signal_name: str) -> str:
        """Map user-facing TOD component names onto internal canonical names."""
        normalized = signal_name.lower()
        aliases = {
            "sky": "static_sky",
            "static_sky": "static_sky",
            "orb": "orbital_dipole",
            "orbital_dipole": "orbital_dipole",
        }
        if normalized not in aliases:
            raise ValueError(f"Unknown TOD signal '{signal_name}'.")
        return aliases[normalized]


    def _get_signal_tod(self, signal_name: str,
                        compsep_output: NDArray | None = None) -> NDArray[np.floating]:
        """Return one named model TOD evaluated for the focused detector at the active resolution."""
        normalized = self._normalize_signal_name(signal_name)
        if normalized == "static_sky":
            return self.get_static_sky_tod(compsep_output=compsep_output)
        if normalized == "orbital_dipole":
            return self.get_orbital_dipole_tod()
        raise ValueError(f"Unhandled TOD signal '{signal_name}'.")


    def get_tod(
        self,
        *,
        subtract: tuple[tuple[str, tuple[str, ...]], ...] | None = None,
        divide_by_gain: tuple[str, ...] | None = None,
        compsep_output: NDArray | None = None,
    ) -> NDArray[np.floating]:
        """Return a jump-corrected detector-local TOD, at the active resolution, after subtracting
        selected model terms.

        Args:
            subtract: Sequence of ``(signal_name, gain_terms)`` pairs. Each signal is evaluated
                and subtracted after multiplying it by the selected gain subset.
            divide_by_gain: Gain terms to divide the final TOD by, or ``None``.
            compsep_output: Optional sky model override for static-sky subtraction.
        """
        tod = np.array(self._downsample_mean(self.corrected_tod), copy=True)

        if subtract is not None:
            for signal_name, gain_terms in subtract:
                # All supported residuals in this class are linear combinations of named model TODs.
                signal = self._get_signal_tod(signal_name, compsep_output=compsep_output)
                tod -= self.get_gain(gain_terms) * signal

        if divide_by_gain is not None:
            gain = self.get_gain(divide_by_gain)
            if gain == 0:
                raise ValueError("Cannot divide TOD by a zero gain.")
            tod /= gain

        return tod


    # ------------------------------------------------------------------ gain calibration
    def _gap_noise_draw(self, method: str, compsep_output: NDArray | None = None,
                        proc_mask_type: str = "") -> NDArray[np.floating]:
        """Constrained noise realization at the masked samples of the calibration TOD.

        The noise residual ``d - g*(s_sky + s_orb)`` (full gain) is identical for every gain term, so
        the 1/f + white gap draw is computed once and shared across the abs/rel/temporal solves
        (cached per method, at the view's active resolution). ``method`` is ``'fallback'`` or
        ``'full_cg'``. ``proc_mask_type`` selects which processing mask defines the gaps.
        """
        cached = self._gap_noise.get(method)
        if cached is not None:
            return cached
        mask = self.get_mask(proc_mask_type=proc_mask_type)
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
        proc_mask_type: str = "",
    ) -> NDArray[np.floating]:
        """Fill masked calibration samples with the target signal plus a noise realization.

        The masked regions retain only the target gain term times the calibrator signal, plus a
        noise draw: white (``method='wn'``, sigma0/sqrt(factor)) or a constrained correlated 1/f +
        white draw (``'fallback'``/``'full_cg'``) shared across gain terms via ``_gap_noise_draw``,
        so the masked residual carries the same 1/f structure as the surrounding valid data.
        ``proc_mask_type`` is forwarded to the 1/f gap draw so it sees the same gaps as ``mask``.
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
            draw = self._gap_noise_draw(method, compsep_output=compsep_output,
                                        proc_mask_type=proc_mask_type)
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
        proc_mask_type: str = "",
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
            calibrate_against: Calibrator, one of "orbital_dipole", "full_sky", or "sky".
            compsep_output: Optional sky-model override for the static-sky term.
            fill_masked: If True, fill masked samples with ``g_target * s_cal`` plus a noise draw.
            gap_fill_method: How masked samples are filled when ``fill_masked``: ``'wn'`` (white
                noise), ``'fallback'`` (stationary 1/f Wiener draw), or ``'full_cg'`` (masked
                constrained-CG 1/f draw). See ``_fill_masked_calibration_samples``.
            rng: Optional NumPy generator for the masked-sample white noise (``'wn'`` only).
            proc_mask_type: Which processing mask defines the calibration gaps (a key under
                ``processing_masks:``; "" uses the default ``processing_mask:``).

        Returns:
            Bunch with ``tod`` (residual), ``s_cal``, and ``mask``.
        """
        if target_term not in self._ALL_GAIN_TERMS:
            raise ValueError(f"Unknown gain term '{target_term}'; expected one of "
                             f"{self._ALL_GAIN_TERMS}.")
        if calibrate_against not in self._CALIB_TARGET_SIGNALS:
            raise ValueError(f"Unknown calibrate_against '{calibrate_against}'; expected one of "
                             f"{tuple(self._CALIB_TARGET_SIGNALS)}.")

        mask = self.get_mask(proc_mask_type=proc_mask_type)
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
                                                        compsep_output=compsep_output,
                                                        proc_mask_type=proc_mask_type)
        return Bunch(tod=tod, s_cal=s_cal, mask=mask)
