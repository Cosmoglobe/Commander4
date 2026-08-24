"""Command-line validation of Commander4 parameter files without starting MPI work."""

import argparse
import os
from collections.abc import Iterator

from commander4.file_io.experiments import EXPERIMENT_READER_MODULES
from commander4.parameters.parse import load_params
from commander4.parameters.schema import (
    derive_task_counts,
    enabled_compsep_views,
    enabled_tod_bands,
    task_count_breakdown,
    validate_param_schema,
)


def _iter_enabled_groups(params) -> Iterator[str]:
    if "compsep" not in params:
        return
    for block_name in (
        "cg_sampling_groups", "per_pixel_sampling_groups", "mcmc_sampling_groups",
    ):
        if block_name not in params.compsep:
            continue
        for group_name in params.compsep[block_name]:
            group = params.compsep[block_name][group_name]
            if "enabled" not in group or group.enabled:
                yield f"{block_name}.{group_name}"


def _validate_components(params) -> None:
    import commander4.sky as sky
    from commander4.sky.component import Component

    if "components" not in params:
        raise ValueError("The top-level 'components' block is required.")
    for component_name in params.components:
        component = params.components[component_name]
        if "component_class" not in component or "params" not in component:
            raise ValueError(
                f"components.{component_name} requires 'component_class' and 'params'."
            )
        component_class = getattr(sky, component.component_class, None)
        if not isinstance(component_class, type) or not issubclass(component_class, Component):
            raise ValueError(
                f"components.{component_name}.component_class is {component.component_class!r}; "
                "that class is not supported."
            )
        component_params = component.params
        if "polarization" not in component_params:
            raise ValueError(f"components.{component_name}.params.polarization is required.")
        if "nu0" in component_params:
            raise ValueError(
                f"components.{component_name}.params.nu0 was renamed to 'nu_ref'."
            )
        removed = sorted(
            key for key in component_params if key.startswith("smoothing_prior_")
        )
        if removed:
            raise ValueError(
                f"components.{component_name}.params uses removed keys {removed}; use the "
                "corresponding 'Cl_prior_*' settings."
            )


def _validate_experiments(params) -> None:
    for band_info in enabled_tod_bands(params):
        experiment = params.experiments[band_info.experiment_name]
        if experiment.experiment_id not in EXPERIMENT_READER_MODULES:
            raise ValueError(
                f"experiments.{band_info.experiment_name}.experiment_id is "
                f"{experiment.experiment_id!r}; known readers are "
                f"{sorted(EXPERIMENT_READER_MODULES)}."
            )
        band = experiment.bands[band_info.band_name]
        if "detectors" not in band or len(band.detectors) == 0:
            raise ValueError(
                f"experiments.{band_info.experiment_name}.bands.{band_info.band_name} must "
                "contain at least one detector."
            )


def _check_referenced_paths(value, source_dir: str, location: str = "") -> None:
    """Check explicit path-like fields; URLs and output paths are intentionally ignored."""
    if not isinstance(value, dict):
        return
    path_keys = {"filelist", "path", "processing_mask", "chisq_mask", "init_from"}
    for key, child in value.items():
        child_location = f"{location}.{key}" if location else key
        if isinstance(child, dict):
            _check_referenced_paths(child, source_dir, child_location)
        elif key in path_keys and isinstance(child, str):
            candidate = child if os.path.isabs(child) else os.path.join(source_dir, child)
            if not os.path.exists(candidate):
                raise FileNotFoundError(f"{child_location} points to missing path {child!r}.")


def validate_parameter_file(
    parameter_file: str, check_paths: bool = False,
) -> tuple[str, list[str]]:
    """Validate one file and return its MPI count summary and enabled sampling groups."""
    params, params_dict, _ = load_params(parameter_file)
    validate_param_schema(params_dict)
    _validate_components(params)
    _validate_experiments(params)
    enabled_compsep_views(params)
    if "compsep" in params:
        from commander4.compsep.processing import resolve_sampling_groups

        resolve_sampling_groups(params)
    counts = derive_task_counts(params)
    if counts.total < 1:
        raise ValueError("The parameter file enables no TOD bands or CompSep views.")
    if check_paths:
        _check_referenced_paths(params_dict, os.path.dirname(os.path.abspath(parameter_file)))
    return task_count_breakdown(counts), list(_iter_enabled_groups(params))


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="c4-validate-params",
        description="Validate Commander4 parameters without starting an MPI run.",
    )
    parser.add_argument("parameter_file", help="Path to a Commander4 YAML parameter file.")
    parser.add_argument(
        "--check-paths", action="store_true",
        help="Also require referenced input files to exist on this machine.",
    )
    args = parser.parse_args()

    task_summary, groups = validate_parameter_file(args.parameter_file, args.check_paths)
    print(f"Valid parameter file: {args.parameter_file}")
    print(f"MPI tasks: {task_summary}")
    print(f"Enabled sampling groups: {', '.join(groups) if groups else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
