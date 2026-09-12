"""TOD config objects and their parameter-file lookup rules.

Each independent dataclass owns its parameter reader and validates its values in __post_init__.
`process_tod` calls every reader at the start of each iteration, before any sampling.
Iteration decisions belong to the processing code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pixell.bunch import Bunch

from commander4.parameters.schema import resolve_param, resolve_band_lmax

if TYPE_CHECKING:
    from commander4.data_models.detector_group_tod import DetectorGroupTOD


CALIBRATION_TARGETS = ("orbital_dipole", "sky", "sky_no_dipole")
GAIN_GAP_FILL_METHODS = ("wn", "fallback", "full_cg")
SIGMA0_METHODS = ("pairwise", "binned_psd")


@dataclass(frozen=True)
class GainConfig:
    """Settings for one gain step; downsample_time is in seconds."""

    enabled: bool = False
    from_iter: int = 1
    calibrate_against: str = "sky"
    gap_fill_method: str = "wn"
    downsample_time: float = 1.0
    mask_threshold: float = 0.5

    def __post_init__(self) -> None:
        """Check the settings before they are used by a numerical routine."""
        if not isinstance(self.enabled, bool):
            raise ValueError("gain.enabled must be true or false.")
        if self.calibrate_against not in CALIBRATION_TARGETS:
            raise ValueError(f"gain.calibrate_against must be one of {CALIBRATION_TARGETS}.")
        if self.gap_fill_method not in GAIN_GAP_FILL_METHODS:
            raise ValueError(f"gain.gap_fill_method must be one of {GAIN_GAP_FILL_METHODS}.")
        if self.downsample_time < 0:
            raise ValueError("gain.downsample_time must be non-negative.")
        if not 0.0 <= self.mask_threshold < 1.0:
            raise ValueError("gain.mask_threshold must be in [0, 1).")

    @classmethod
    def from_params(cls, tod: Bunch | dict, band: Bunch, step_name: str,
                    default_calibrator: str) -> GainConfig:
        """Read global gain settings and apply the band's partial override."""
        values = {"calibrate_against": default_calibrator}
        if step_name in tod:
            values.update(tod[step_name])
        if step_name in band:
            values.update(band[step_name])
        return cls(**values)


@dataclass(frozen=True)
class JumpDetectionConfig:
    """Jump-detection window and the experiment's flag bitmask."""

    enabled: bool = False
    from_iter: int = 1
    window: int = 10
    jump_bitmask: int | None = None

    def __post_init__(self) -> None:
        """Check the settings before they are used by a numerical routine."""
        if not isinstance(self.enabled, bool):
            raise ValueError("jump_detection.enabled must be true or false.")
        if not isinstance(self.window, int) or self.window < 1:
            raise ValueError("jump_detection.window must be an integer of at least 1.")
        if self.enabled and self.jump_bitmask is None:
            raise ValueError("Jump detection is enabled, but the experiment has no jump_bitmask.")

    @classmethod
    def from_params(cls, tod: Bunch | dict, experiment: Bunch) -> JumpDetectionConfig:
        """Read the jump settings and obtain the bitmask from experiment metadata."""
        block = tod["jump_detection"] if "jump_detection" in tod else {}
        return cls(**block, jump_bitmask=getattr(experiment, "jump_bitmask", None))


@dataclass(frozen=True)
class CGConfig:
    """CG controls; each solver defines its own error criterion.

    These defaults are for correlated noise: max_iter=0 selects the stationary fallback.
    The CG mapmaker requires both values in the parameter file.
    """

    max_iter: int = 0
    err_tol: float = 1.0e-4

    def __post_init__(self) -> None:
        """Check the settings before they are used by a numerical routine."""
        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("CG max_iter must be a non-negative integer.")
        if self.err_tol < 0:
            raise ValueError("CG err_tol must be non-negative.")


@dataclass(frozen=True)
class CorrelatedNoiseConfig:
    """Noise sampling controls; sample_sigma0 also applies when n_corr is disabled."""

    enabled: bool = False
    from_iter: int = 1
    sample_psd_params: bool = False
    sample_sigma0: bool = True
    sigma0_method: str = "pairwise"
    sigma0_decimation: int = 1
    nomono: bool = False
    onlymono: bool = False
    psd_bin: bool = False
    use_dct: bool = False
    cg: CGConfig = field(default_factory=CGConfig)

    def __post_init__(self) -> None:
        """Check the settings before they are used by a numerical routine."""
        if not isinstance(self.enabled, bool):
            raise ValueError("corr_noise.enabled must be true or false.")
        if self.sample_psd_params and not self.enabled:
            raise ValueError("corr_noise.sample_psd_params requires enabled=True.")
        if self.sigma0_method not in SIGMA0_METHODS:
            raise ValueError(f"corr_noise.sigma0_method must be one of {SIGMA0_METHODS}.")
        if not isinstance(self.sigma0_decimation, int) or self.sigma0_decimation < 1:
            raise ValueError("corr_noise.sigma0_decimation must be an integer of at least 1.")
        if self.nomono and self.onlymono:
            raise ValueError("corr_noise.nomono and onlymono cannot both be true.")

    @classmethod
    def from_params(cls, tod: Bunch | dict) -> CorrelatedNoiseConfig:
        """Read noise settings, including the nested CG controls."""
        values = dict(tod["corr_noise"]) if "corr_noise" in tod else {}
        # The file readers already apply these limits to NoisePSD.nu_fit, before sampling.
        values.pop("psd_fit_nu_min", None)
        values.pop("psd_fit_nu_max", None)
        cg_cfg = CGConfig(**values.pop("cg", {}))
        return cls(**values, cg=cg_cfg)


@dataclass(frozen=True)
class DataSelectionConfig:
    """Detector-scan rejection thresholds; iteration limits are inclusive."""

    enabled: bool = False
    from_iter: int = 1
    until_iter: int | None = None
    chisq_abs_threshold: float = 1.0e4
    min_good_fraction: float = 0.1

    def __post_init__(self) -> None:
        """Check the settings before they are used by a numerical routine."""
        if not isinstance(self.enabled, bool):
            raise ValueError("data_selection.enabled must be true or false.")
        if self.until_iter is not None and self.until_iter < self.from_iter:
            raise ValueError("data_selection.until_iter cannot be before from_iter.")
        if self.chisq_abs_threshold <= 0:
            raise ValueError("data_selection.chisq_abs_threshold must be positive.")
        if not 0.0 <= self.min_good_fraction <= 1.0:
            raise ValueError("data_selection.min_good_fraction must be between 0 and 1.")

    @classmethod
    def from_params(cls, tod: Bunch | dict) -> DataSelectionConfig:
        """Read the global data-selection block, using disabled defaults when it is absent."""
        block = tod["data_selection"] if "data_selection" in tod else {}
        return cls(**block)


@dataclass(frozen=True)
class FarBeamConfig:
    """Sidelobe convolution limits and normalization.

    Commander3 uses lmax=mmax=100 and scales the sidelobe TOD by two by comparison with LevelS.
    """

    enabled: bool = False
    from_iter: int = 1
    lmax: int = 100
    mmax: int = 100
    epsilon: float = 1.0e-4
    beam_norm: float = 2.0

    def __post_init__(self) -> None:
        """Check the settings before they are used by a numerical routine."""
        if not isinstance(self.enabled, bool):
            raise ValueError("far_beam_deconvolution.enabled must be true or false.")
        if self.lmax < 0 or self.mmax < 0:
            raise ValueError("far_beam_deconvolution.lmax and mmax must be non-negative integers.")
        if self.mmax > self.lmax:
            raise ValueError("far_beam_deconvolution.mmax cannot exceed lmax.")

    @classmethod
    def from_params(cls, tod: Bunch | dict) -> FarBeamConfig:
        """Read the global far-beam block, using disabled defaults when it is absent."""
        block = tod["far_beam_deconvolution"] if "far_beam_deconvolution" in tod else {}
        return cls(**block)


@dataclass(frozen=True)
class MapmakingConfig:
    """Mapmaker choice, resources, and output maps; common_res_fwhm is in arcminutes."""

    mapmaker: str
    num_threads: int
    include_orbital_dipole_maps: bool
    include_corr_noise_maps: bool
    include_sky_model_maps: bool
    include_residual_maps: bool = False
    include_sidelobe_maps: bool = False
    include_hit_maps: bool = False
    include_cov_maps: bool = False
    sparse_maps: bool = False
    common_res_fwhm: float = 0.0
    band_lmax: int | None = None
    cg: CGConfig = field(default_factory=CGConfig)

    def __post_init__(self) -> None:
        """Check the settings before they are used by a numerical routine."""
        if self.mapmaker not in ("CG", "bin"):
            raise ValueError("mapmaker must be CG or bin.")
        if self.num_threads < 1:
            raise ValueError("resources.tod.num_threads must be at least 1.")
        if self.common_res_fwhm < 0:
            raise ValueError("compsep.common_res_fwhm cannot be negative.")

    @classmethod
    def from_params(cls, params: Bunch, experiment_data: DetectorGroupTOD) -> MapmakingConfig:
        """Read map settings, choosing the mapmaker from band, experiment, then global values."""
        tod = dict(params.tod_processing)
        exp_name = experiment_data.experiment_name
        band_name = experiment_data.band_name
        include = dict(params.output.chains.include)

        mapmaker = resolve_param(params, "mapmaker",
                                (f"experiments.{exp_name}.bands.{band_name}",
                                 f"experiments.{exp_name}", "tod_processing"))
        cg_block = tod.get("cg_mapmaker", {})
        if mapmaker == "CG" and ("max_iter" not in cg_block or "err_tol" not in cg_block):
            raise ValueError("tod_processing.cg_mapmaker requires max_iter and err_tol for CG.")
        return cls(
            mapmaker=mapmaker,
            num_threads=params.resources.tod.num_threads,
            sparse_maps=resolve_param(params, "sparse_maps", (f"experiments.{exp_name}",),
                                      default=cls.sparse_maps, legal_types=bool),
            common_res_fwhm=float(getattr(params.compsep, "common_res_fwhm", cls.common_res_fwhm)),
            band_lmax=resolve_band_lmax(params, band_name, exp_name, experiment_data.nside),
            include_orbital_dipole_maps=bool(include["orbital_dipole_maps"]),
            include_corr_noise_maps=bool(include["corr_noise_maps"]),
            include_sky_model_maps=bool(include["sky_model_maps"]),
            include_residual_maps=bool(include.get("residual_maps", cls.include_residual_maps)),
            include_sidelobe_maps=bool(include.get("sidelobe_maps", cls.include_sidelobe_maps)),
            include_hit_maps=bool(include.get("hit_maps", cls.include_hit_maps)),
            include_cov_maps=bool(include.get("cov_maps", cls.include_cov_maps)),
            cg=CGConfig(**cg_block),
        )
