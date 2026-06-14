import numpy as np
import logging
from numpy.typing import NDArray
from typing import Literal

from pixell.bunch import Bunch

from commander4.data_models.detector_group_TOD import DetGroupTOD
from commander4.data_models.TOD_samples import TODSamples
from commander4.utils.map_utils import get_static_sky_TOD, get_s_orb_TOD

logger = logging.getLogger(__name__)


class TODView:
    """Materialize one detector at a time and build derived TOD views on demand.

    The view keeps only the currently focused detector decoded in memory. Calling
    ``focus()`` always discards the previous detector's cached arrays, which matches the
    one-detector-at-a-time TOD-processing architecture.
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
    ):
        """Initialize a detector-local view over one band's TOD data.

        Args:
            experiment_data: Static TOD container for the current band.
            tod_samples: Sampled gain and noise parameters for the current chain state.
            compsep_output: Optional default sky model used by sky-subtraction helpers.
        """
        self.experiment_data = experiment_data
        self.tod_samples = tod_samples
        self.compsep_output = compsep_output
        self._iscan: int | None = None
        self._idet: int | None = None
        self._det = None
        self._clear_cache()

    def _clear_cache(self):
        """Drop all arrays materialized for the current detector."""
        self._tod = None
        self._corrected_tod = None
        self._pix = None
        self._psi = None
        self._flag = None
        self._processing_mask = None
        self._good_data_mask = None
        self._full_mask = None
        self._static_sky = None
        self._orbital_dipole = None
        self._downsampled: dict[int, Bunch] = {}

    def focus(self, iscan: int, idet: int) -> "TODView":
        """Focus the view on one detector and discard any previous materialization."""
        self._det = self.experiment_data.scans[iscan].detectors[idet]
        self._iscan = iscan
        self._idet = idet
        self._clear_cache()
        return self

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
        self._require_focus()
        return self._idet

    @property
    def detector(self):
        return self._require_focus()

    @property
    def fsamp(self) -> float:
        return self.detector.fsamp

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
        """Whether the focused detector-scan is accepted (currently always True)."""
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

    @property
    def tod(self) -> NDArray[np.floating]:
        if self._tod is None:
            self._tod = self.detector.tod
        return self._tod

    @property
    def corrected_tod(self) -> NDArray[np.floating]:
        """Return the raw TOD with the stored jump offsets applied in detector units."""
        if self._corrected_tod is None:
            jump = self.tod_samples.jumps.get(self.iscan, self.idet)
            self._corrected_tod = self.tod if jump.is_empty() else jump.apply(self.tod)
        return self._corrected_tod

    def _get_fullres_pix_psi(self) -> tuple[NDArray[np.integer], NDArray[np.floating] | NDArray[np.integer]]:
        """Decode full-resolution pointing once and reuse it for the current detector."""
        if self._pix is None or self._psi is None:
            self._pix, self._psi = self.detector.get_pix_psi()
        return self._pix, self._psi

    @property
    def pix(self) -> NDArray[np.integer]:
        return self._get_fullres_pix_psi()[0]

    @property
    def psi(self) -> NDArray[np.floating] | NDArray[np.integer]:
        return self._get_fullres_pix_psi()[1]

    @property
    def flag(self) -> NDArray[np.integer]:
        if self._flag is None:
            self._flag = self.detector.flag
        return self._flag

    def _unpack_mask(self, attr_name: str) -> NDArray[np.bool_]:
        """Unpack one of DetectorTOD's packed bit masks for the focused detector."""
        det = self.detector
        packed = getattr(det, attr_name, None)
        if packed is None:
            # Some datasets may not define every cut explicitly; fall back to permissive masks.
            if attr_name == "_processing_mask":
                return np.ones(det.ntod, dtype=bool)
            if attr_name == "_good_data_mask":
                return np.ones(det.ntod, dtype=bool)
            if attr_name == "_full_mask":
                return self.processing_mask.copy()
            raise ValueError(f"Detector mask '{attr_name}' is unavailable.")
        mask = np.unpackbits(packed).view(bool)
        if mask.size > det.ntod + 7 or mask.size < det.ntod:
            raise ValueError(f"Mask size {mask.size} doesn't match ntod {det.ntod}.")
        return mask[:det.ntod]

    @property
    def processing_mask(self) -> NDArray[np.bool_]:
        if self._processing_mask is None:
            self._processing_mask = self._unpack_mask("_processing_mask")
        return self._processing_mask

    @property
    def good_data_mask(self) -> NDArray[np.bool_]:
        if self._good_data_mask is None:
            self._good_data_mask = self._unpack_mask("_good_data_mask")
        return self._good_data_mask

    @property
    def full_mask(self) -> NDArray[np.bool_]:
        if self._full_mask is None:
            self._full_mask = self._unpack_mask("_full_mask")
        return self._full_mask

    def _downsample_factor_or_default(self, downsample_factor: int | None) -> int:
        """Validate a downsampling factor and replace ``None`` with unity."""
        if downsample_factor is None:
            downsample_factor = 1
        if downsample_factor < 1:
            raise ValueError("downsample_factor must be >= 1.")
        return int(downsample_factor)

    def _materialize_downsampled(self, downsample_factor: int) -> Bunch:
        """Cache a downsampled detector view for the requested averaging factor."""
        factor = self._downsample_factor_or_default(downsample_factor)
        cached = self._downsampled.get(factor)
        if cached is not None:
            return cached

        if factor == 1:
            data = Bunch(
                tod=self.corrected_tod,
                pix=self.pix,
                psi=self.psi,
                processing_mask=self.processing_mask,
                good_data_mask=self.good_data_mask,
                full_mask=self.full_mask,
                indices=np.arange(self.detector.ntod, dtype=np.int64),
            )
        else:
            # Average the jump-corrected TOD over contiguous blocks. Pointing and masks are kept at
            # the block centers; model TODs are not evaluated at this pointing but block-averaged at
            # full rate (see get_static_sky_tod), so they share the data's downsampling transfer.
            indices_edges = np.arange(0, self.detector.ntod, factor)
            indices = (indices_edges[1:] + indices_edges[:-1]) // 2
            ntod_down = indices.size
            tod = self.corrected_tod[:ntod_down*factor].reshape((ntod_down, factor))
            data = Bunch(
                tod=np.mean(tod, axis=-1),
                pix=self.pix[indices],
                psi=self.psi[indices],
                processing_mask=self.processing_mask[indices],
                good_data_mask=self.good_data_mask[indices],
                full_mask=self.full_mask[indices],
                indices=indices,
            )

        self._downsampled[factor] = data
        return data

    def get_mask(
        self,
        mask: Literal["none", "processing", "good", "full"] = "none",
        downsample_factor: int | None = None,
    ) -> NDArray[np.bool_] | None:
        """Return one of the cached detector masks, optionally at downsampled resolution."""
        if mask == "none":
            return None
        data = self._materialize_downsampled(self._downsample_factor_or_default(downsample_factor))
        if mask == "processing":
            return data.processing_mask
        if mask == "good":
            return data.good_data_mask
        if mask == "full":
            return data.full_mask
        raise ValueError(f"Unknown mask mode '{mask}'.")

    def _require_compsep_output(self, compsep_output: NDArray | None) -> NDArray:
        """Resolve the sky model to use for static-sky subtraction."""
        sky_model = self.compsep_output if compsep_output is None else compsep_output
        if sky_model is None:
            raise ValueError("A component-separation sky map must be provided for sky subtraction.")
        return sky_model

    def _block_average(self, tod: NDArray[np.floating], factor: int) -> NDArray[np.floating]:
        """Average a full-rate array over the same contiguous blocks as the downsampled TOD."""
        ntod_down = self._materialize_downsampled(factor).tod.shape[0]
        return tod[:ntod_down*factor].reshape((ntod_down, factor)).mean(axis=-1)

    def get_static_sky_tod(
        self,
        compsep_output: NDArray | None = None,
        downsample_factor: int | None = None,
    ) -> NDArray[np.floating]:
        """Evaluate the static sky model along the focused detector pointing.

        For ``downsample_factor > 1`` the model is evaluated at the full sampling rate and averaged
        over the same sample blocks as the data, integrating the model over the scan path within
        each block rather than sampling it at the block-center pixel. Model and data thereby see
        the same downsampling transfer function, which keeps e.g. gain estimates unbiased.
        """
        factor = self._downsample_factor_or_default(downsample_factor)
        sky_model = self._require_compsep_output(compsep_output)
        if compsep_output is None:
            if self._static_sky is None:
                # Reuse the full-resolution sky TOD when both pointing and sky model match.
                self._static_sky = get_static_sky_TOD(sky_model, self.pix, psi=self.psi)
            sky_tod = self._static_sky
        else:
            sky_tod = get_static_sky_TOD(sky_model, self.pix, psi=self.psi)
        return sky_tod if factor == 1 else self._block_average(sky_tod, factor)

    def get_orbital_dipole_tod(self, downsample_factor: int | None = None) -> NDArray[np.floating]:
        """Evaluate the orbital dipole for the focused detector.

        Downsampling block-averages the full-rate dipole TOD, mirroring ``get_static_sky_tod``.
        """
        factor = self._downsample_factor_or_default(downsample_factor)
        if self._orbital_dipole is None:
            self._orbital_dipole = get_s_orb_TOD(self.detector, self.experiment_data, self.pix)
        if factor == 1:
            return self._orbital_dipole
        return self._block_average(self._orbital_dipole, factor)

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

    def _get_signal_tod(
        self,
        signal_name: str,
        compsep_output: NDArray | None = None,
        downsample_factor: int | None = None,
    ) -> NDArray[np.floating]:
        """Return one named model TOD evaluated for the focused detector."""
        normalized = self._normalize_signal_name(signal_name)
        if normalized == "static_sky":
            return self.get_static_sky_tod(compsep_output=compsep_output,
                                           downsample_factor=downsample_factor)
        if normalized == "orbital_dipole":
            return self.get_orbital_dipole_tod(downsample_factor=downsample_factor)
        raise ValueError(f"Unhandled TOD signal '{signal_name}'.")

    def get_tod(
        self,
        *,
        subtract: tuple[tuple[str, tuple[str, ...]], ...] | None = None,
        divide_by_gain: tuple[str, ...] | None = None,
        downsample_factor: int = 1,
        mask: Literal["none", "processing", "good", "full"] = "none",
        compsep_output: NDArray | None = None,
    ) -> NDArray[np.floating]:
        """Return a jump-corrected detector-local TOD after subtracting selected model terms.

        Args:
            subtract: Sequence of ``(signal_name, gain_terms)`` pairs. Each signal is evaluated
                and subtracted after multiplying it by the selected gain subset.
            divide_by_gain: Gain terms to divide the final TOD by, or ``None``.
            downsample_factor: Average the TOD in contiguous chunks before subtraction.
            mask: Optional mask to apply to the output TOD.
            compsep_output: Optional sky model override for static-sky subtraction.
        """
        factor = self._downsample_factor_or_default(downsample_factor)
        tod = np.array(self._materialize_downsampled(factor).tod, copy=True)

        if subtract is not None:
            for signal_name, gain_terms in subtract:
                # All supported residuals in this class are linear combinations of named model TODs.
                signal = self._get_signal_tod(signal_name, compsep_output=compsep_output,
                                              downsample_factor=factor)
                tod -= self.get_gain(gain_terms) * signal

        if divide_by_gain is not None:
            gain = self.get_gain(divide_by_gain)
            if gain == 0:
                raise ValueError("Cannot divide TOD by a zero gain.")
            tod /= gain

        # Apply masking last so callers can request either the full residual or the cut samples.
        mask_arr = self.get_mask(mask, downsample_factor=factor)
        return tod if mask_arr is None else tod[mask_arr]

    def _fill_masked_calibration_samples(
        self,
        tod: NDArray[np.floating],
        mask: NDArray[np.bool_],
        signal: NDArray[np.floating],
        gain_terms: tuple[str, ...],
        downsample_factor: int,
        rng: np.random.Generator | None,
    ) -> NDArray[np.floating]:
        """Fill masked calibration samples with signal plus white noise in detector units."""
        sigma0_effective = self.sigma0 * np.sqrt(1.0 / downsample_factor)
        filled = np.array(tod, copy=True)
        if (~mask).any():
            normal = np.random.normal if rng is None else rng.normal
            noise = normal(0.0, sigma0_effective, signal[~mask].shape)
            # The masked regions retain only the target gain term times the calibrator signal.
            filled[~mask] = self.get_gain(gain_terms) * signal[~mask] + noise
        return filled

    def get_calib_tod(
        self,
        target_term: str,
        calibrate_against: str,
        *,
        compsep_output: NDArray | None = None,
        downsample_factor: int | None = None,
        fill_masked: bool = True,
        rng: np.random.Generator | None = None,
    ) -> Bunch:
        """Return the residual, calibrator signal, and mask used to sample one gain term.

        The detector model is ``d = (g_abs + g_rel + g_temp) * (s_sky + s_orb) + n``. To sample
        ``target_term`` against a calibrator signal ``s_cal`` (the subset of {static sky, orbital
        dipole} selected by ``calibrate_against``), each model signal is subtracted with the
        appropriate gain terms so the residual reduces to ``g_target * s_cal + n``:
            - signals making up the calibrator keep the target term (only the *other* terms are
              subtracted), contributing ``g_target * s`` to the residual;
            - signals outside the calibrator are subtracted in full and thus removed.

        Args:
            target_term: Gain term being sampled, one of ``_ALL_GAIN_TERMS`` ("abs", "rel", "temp").
            calibrate_against: Calibrator, one of "orbital_dipole", "full_sky", or "sky".
            compsep_output: Optional sky-model override for the static-sky term.
            downsample_factor: Block-averaging factor applied to both the data and the model
                TODs; defaults to one second (``int(fsamp)``).
            fill_masked: If True, fill masked samples with ``g_target * s_cal`` plus white noise.
            rng: Optional NumPy generator for the masked-sample noise.

        Returns:
            Bunch with ``tod`` (residual), ``s_cal``, ``pix``, ``psi``, and ``mask``.
        """
        if target_term not in self._ALL_GAIN_TERMS:
            raise ValueError(f"Unknown gain term '{target_term}'; expected one of "
                             f"{self._ALL_GAIN_TERMS}.")
        if calibrate_against not in self._CALIB_TARGET_SIGNALS:
            raise ValueError(f"Unknown calibrate_against '{calibrate_against}'; expected one of "
                             f"{tuple(self._CALIB_TARGET_SIGNALS)}.")

        factor = int(self.fsamp) if downsample_factor is None else int(downsample_factor)
        data = self._materialize_downsampled(factor)
        mask = self.get_mask("full", downsample_factor=factor)
        s_sky = self.get_static_sky_tod(compsep_output=compsep_output, downsample_factor=factor)
        s_orb = self.get_orbital_dipole_tod(downsample_factor=factor)

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

        tod = self.get_tod(subtract=subtract, downsample_factor=factor,
                           compsep_output=compsep_output)
        if fill_masked:
            tod = self._fill_masked_calibration_samples(tod, mask, s_cal, (target_term,), factor,
                                                        rng)
        return Bunch(tod=tod, s_cal=s_cal, pix=data.pix, psi=data.psi, mask=mask)


