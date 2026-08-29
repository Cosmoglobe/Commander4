"""Configuration of the mapmaking step: which mapmaker runs, and with what settings.

`MapmakingConfig` picks between the binned and CG mapmakers and carries the settings they share,
so the choice is made once from the parameter file rather than inside the scan loop.
"""
from dataclasses import dataclass, field

from pixell.bunch import Bunch

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.parameters.schema import resolve_param, resolve_band_lmax
from commander4.tod.step_config import CGConfig


@dataclass(frozen=True)
class MapmakingConfig:
    """Validated mapmaking resources, algorithm controls, and map-output selection."""

    mapmaker: str
    num_threads: int
    include_orbital_dipole_maps: bool
    include_corr_noise_maps: bool
    include_sky_model_maps: bool
    include_residual_maps: bool = False
    include_hit_maps: bool = False
    include_cov_maps: bool = False
    sparse_maps: bool = False
    common_res_fwhm: float = 0.0
    band_lmax: int | None = None
    cg: CGConfig = field(default_factory=CGConfig)

    def __post_init__(self) -> None:
        if self.mapmaker not in ("CG", "bin"):
            raise ValueError(f"mapmaker must be 'CG' or 'bin', got {self.mapmaker!r}.")
        if not isinstance(self.num_threads, int) or self.num_threads < 1:
            raise ValueError("resources.tod.num_threads must be an integer of at least 1.")
        if self.common_res_fwhm < 0:
            raise ValueError("compsep.common_res_fwhm cannot be negative.")

    @classmethod
    def from_params(cls, params: Bunch,
                    experiment_data: DetectorGroupTOD) -> "MapmakingConfig":
        """Build mapmaking settings using band, experiment, and global precedence."""
        exp_name = experiment_data.experiment_name
        band_name = experiment_data.band_name
        mapmaker = resolve_param(
            params, "mapmaker",
            (f"experiments.{exp_name}.bands.{band_name}", f"experiments.{exp_name}",
             "tod_processing"),
            legal_values=("CG", "bin"),
        )
        tod = params.tod_processing
        if "cg_mapmaker" in tod:
            cg = CGConfig.from_block("tod_processing.cg_mapmaker", tod.cg_mapmaker,
                                     require_all=mapmaker == "CG")
        elif mapmaker == "CG":
            raise ValueError("tod_processing.cg_mapmaker is required for the CG mapmaker.")
        include = params.output.chains.include
        resolved = {
            "mapmaker": mapmaker,
            "sparse_maps": resolve_param(params, "sparse_maps", (f"experiments.{exp_name}",),
                                         default=cls.sparse_maps, legal_types=bool),
            "common_res_fwhm": float(resolve_param(params, "common_res_fwhm", ("compsep",),
                                                   default=cls.common_res_fwhm)),
            "band_lmax": resolve_band_lmax(params, band_name, exp_name, experiment_data.nside),
            "num_threads": params.resources.tod.num_threads,
            "include_orbital_dipole_maps": bool(include.orbital_dipole_maps),
            "include_corr_noise_maps": bool(include.corr_noise_maps),
            "include_sky_model_maps": bool(include.sky_model_maps),
            # Off by default: the residual and hit maps each cost an extra map accumulator, and the
            # covariance is six full-sky float64 maps.
            "include_residual_maps": bool(getattr(include, "residual_maps",
                                                  cls.include_residual_maps)),
            "include_hit_maps": bool(getattr(include, "hit_maps", cls.include_hit_maps)),
            "include_cov_maps": bool(getattr(include, "cov_maps", cls.include_cov_maps)),
        }
        if "cg_mapmaker" in tod:
            resolved["cg"] = cg
        return cls(**resolved)
