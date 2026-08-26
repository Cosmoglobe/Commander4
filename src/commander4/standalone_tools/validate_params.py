"""Command-line validation of Commander4 parameter files without starting MPI work."""

import argparse
import os
from collections.abc import Iterator

from pixell.bunch import Bunch

from commander4.file_io.experiments import EXPERIMENT_READER_MODULES
from commander4.parameters.parse import load_params
from commander4.parameters.schema import (
    TOP_LEVEL_BLOCKS,
    derive_task_counts,
    enabled_compsep_views,
    enabled_tod_bands,
    task_count_breakdown,
    validate_param_schema,
)


SHT_NSIDE_FLOOR = 512
SHT_NSIDE_EXPONENT = 2.7
SHT_SPIN2_FACTOR = 2.0


def _validate_structure(params_dict: dict) -> None:
    """Validate blocks that every Commander4 startup reads before accessing data."""
    missing = [name for name in TOP_LEVEL_BLOCKS if name not in params_dict]
    if missing:
        raise ValueError(f"Missing required top-level parameter block(s) {missing}.")

    for name in TOP_LEVEL_BLOCKS:
        if not isinstance(params_dict[name], dict):
            raise ValueError(
                f"Top-level parameter block {name!r} must be a YAML mapping, not "
                f"{type(params_dict[name]).__name__}. Use '{{}}' for an empty block."
            )

    chains = params_dict["output"].get("chains")
    if not isinstance(chains, dict):
        raise ValueError("output.chains is required and must be a YAML mapping.")
    if "write" not in chains:
        raise ValueError("output.chains.write is required.")


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
    from commander4.sky.diffuse_components import (
        CMB,
        DiffuseComponent,
        FreeFree,
        SpinningDust,
        Synchrotron,
        ThermalDust,
    )
    from commander4.sky.point_sources import RadioSources
    from commander4.sky.template_component import TemplateComponent

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
        if issubclass(component_class, TemplateComponent):
            raise ValueError(
                f"components.{component_name}.component_class is {component.component_class!r}, "
                "which is not implemented."
            )
        supported_classes = (CMB, ThermalDust, Synchrotron, FreeFree, SpinningDust, RadioSources)
        if component_class not in supported_classes:
            raise ValueError(
                f"components.{component_name}.component_class is {component.component_class!r}, "
                "which is not implemented as a concrete parameter-file component."
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

        common_fields = {"polarization", "longname", "shortname"}
        diffuse_fields = {
            "lmax", "spatially_varying_MM", "Cl_prior_amplitude", "Cl_prior_beta",
            "Cl_prior_FWHM", "Cl_prior_l_pivot", "Cl_prior_l_apod", "units", "init_from",
            "amp_prior_mean_map",
        }
        spectral_index_fields = {
            "sample_spectral_index", "spectral_index_bounds", "spectral_index_prior",
            "spectral_index_proposal_sigma",
        }
        class_fields = {
            CMB: {"nu_ref"},
            ThermalDust: {"beta", "T", "nu_ref"} | spectral_index_fields,
            Synchrotron: {"beta", "nu_ref"} | spectral_index_fields,
            FreeFree: {"T", "nu_ref"},
            SpinningDust: {"template_path", "nu_peak", "nu_0"},
            RadioSources: {"template_path", "nu_0"},
        }
        allowed_fields = common_fields | class_fields[component_class]
        if issubclass(component_class, DiffuseComponent):
            allowed_fields |= diffuse_fields
        unknown = sorted(set(component_params) - allowed_fields)
        if unknown:
            raise ValueError(
                f"components.{component_name}.params has unused field(s) {unknown}. "
                f"{component.component_class} accepts {sorted(allowed_fields)}."
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


def estimate_compsep_sht_work(nside: int, polarization: str) -> float:
    """Estimate SHT work relative to an intensity transform at nside 512.

    The supplied benchmark is described well by a power law in nside. Resolutions below 512 use
    the nside-512 cost because other CompSep operations dominate there. Polarization performs a
    spin-2 transform and is assigned twice the intensity cost.
    """
    if isinstance(nside, bool) or not isinstance(nside, int) or nside < 1:
        raise ValueError("nside must be an integer of at least 1.")
    if polarization not in {"I", "QU"}:
        raise ValueError(f"Unknown CompSep execution polarization {polarization!r}.")
    effective_nside = max(nside, SHT_NSIDE_FLOOR)
    spin_factor = SHT_SPIN2_FACTOR if polarization == "QU" else 1.0
    return spin_factor * (effective_nside / SHT_NSIDE_FLOOR)**SHT_NSIDE_EXPONENT


def _compsep_view_nside(params: Bunch, band_name: str) -> int:
    """Return the resolution known from parameters before any map is read."""
    band = params.compsep.bands[band_name]
    if "get_from" not in band:
        raise ValueError(
            f"compsep.bands.{band_name}.get_from is required to suggest a thread allocation."
        )

    if band.get_from == "file":
        source = band
        source_name = f"compsep.bands.{band_name}"
    else:
        experiment_name = band.get_from
        if experiment_name not in params.experiments:
            raise ValueError(
                f"compsep.bands.{band_name}.get_from names unknown experiment "
                f"{experiment_name!r}."
            )
        experiment = params.experiments[experiment_name]
        if band_name not in experiment.bands:
            raise ValueError(
                f"Experiment {experiment_name!r} has no band {band_name!r}, referenced by "
                f"compsep.bands.{band_name}.get_from."
            )
        source = experiment.bands[band_name]
        source_name = f"experiments.{experiment_name}.bands.{band_name}"

    if "eval_nside" not in source:
        raise ValueError(
            f"{source_name}.eval_nside is required to suggest a thread allocation."
        )
    nside = source.eval_nside
    if isinstance(nside, bool) or not isinstance(nside, int) or nside < 1:
        raise ValueError(f"{source_name}.eval_nside must be an integer of at least 1.")
    return nside


def _default_compsep_threads(params: Bunch, num_ranks: int) -> int:
    """Return the current total CompSep thread allocation from scalar or list syntax."""
    if "resources" not in params or "compsep" not in params.resources:
        raise ValueError("resources.compsep is required to suggest a thread allocation.")
    resources = params.resources.compsep
    if "num_threads" not in resources:
        raise ValueError(
            "resources.compsep.num_threads is required to suggest a thread allocation."
        )

    configured = resources.num_threads
    if isinstance(configured, int) and not isinstance(configured, bool):
        if configured < 1:
            raise ValueError("resources.compsep.num_threads must be at least 1.")
        return configured * num_ranks
    if not isinstance(configured, list):
        raise ValueError("resources.compsep.num_threads must be an integer or list of integers.")
    if len(configured) != num_ranks:
        raise ValueError(
            f"Length of resources.compsep.num_threads ({len(configured)}) does not match the "
            f"number of CompSep ranks ({num_ranks})."
        )
    for value in configured:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                "Every resources.compsep.num_threads entry must be an integer of at least 1."
            )
    return sum(configured)


def _allocate_threads(weights: list[float], total_threads: int,
                      threads_per_node: int) -> list[int]:
    """Allocate a thread budget proportionally, bounded to one node's threads per rank."""
    num_ranks = len(weights)
    if total_threads < num_ranks:
        raise ValueError(
            f"The CompSep budget is {total_threads} threads for {num_ranks} ranks; at least "
            "one thread per rank is required."
        )
    if total_threads > num_ranks * threads_per_node:
        raise ValueError(
            f"The CompSep budget is {total_threads} threads, but {num_ranks} ranks capped at "
            f"{threads_per_node} threads can use at most {num_ranks * threads_per_node}."
        )

    # Find the common scale giving the requested total after clipping every proportional share to
    # [1, threads_per_node]. The clipped sum increases continuously with the scale.
    scale_low = 0.0
    scale_high = threads_per_node / min(weights)
    for _ in range(60):
        scale = (scale_low + scale_high) / 2.0
        scaled_total = 0.0
        for weight in weights:
            scaled_total += min(threads_per_node, max(1.0, scale * weight))
        if scaled_total < total_threads:
            scale_low = scale
        else:
            scale_high = scale

    quotas = []
    for weight in weights:
        quotas.append(min(threads_per_node, max(1.0, scale_high * weight)))
    allocation = []
    for quota in quotas:
        allocation.append(int(quota))

    # Largest-remainder rounding preserves the exact total and resolves ties by rank order.
    threads_left = total_threads - sum(allocation)
    remainder_order = sorted(
        range(num_ranks), key=lambda rank: (-(quotas[rank] - allocation[rank]), rank)
    )
    for rank in remainder_order:
        if threads_left == 0:
            break
        if allocation[rank] < threads_per_node:
            allocation[rank] += 1
            threads_left -= 1
    if threads_left != 0:
        raise RuntimeError("Could not round the CompSep thread recommendation to its budget.")
    return allocation


def suggest_compsep_thread_counts(
    params: Bunch, threads_per_node: int | None = None, num_nodes: int = 1,
) -> list[int]:
    """Suggest one thread count per CompSep rank from its SHT workload.

    Args:
        params: Parsed Commander4 parameters.
        threads_per_node: Threads on each node dedicated to CompSep. When omitted, the current
            total is used: scalar ``num_threads`` times rank count, or the sum of a list.
        num_nodes: Nodes dedicated to CompSep.

    Returns:
        Thread counts in CompSep rank order, ready for ``resources.compsep.num_threads``.
    """
    views = enabled_compsep_views(params)
    if not views:
        return []
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes < 1:
        raise ValueError("num_nodes must be an integer of at least 1.")
    if threads_per_node is None:
        threads_per_node = _default_compsep_threads(params, len(views))
    if (isinstance(threads_per_node, bool) or not isinstance(threads_per_node, int)
            or threads_per_node < 1):
        raise ValueError("threads_per_node must be an integer of at least 1.")

    weights = []
    for view in views:
        nside = _compsep_view_nside(params, view.band_name)
        weights.append(estimate_compsep_sht_work(nside, view.polarization))
    total_threads = threads_per_node * num_nodes
    return _allocate_threads(weights, total_threads, threads_per_node)


def validate_parameter_file(
    parameter_file: str, check_paths: bool = False,
) -> tuple[str, list[str]]:
    """Validate one file and return its MPI count summary and enabled sampling groups."""
    params, params_dict, _ = load_params(parameter_file)
    validate_param_schema(params_dict)
    _validate_structure(params_dict)
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
    parser.add_argument(
        "--compsep-threads-per-node", type=int,
        help=("Threads per node dedicated to CompSep. By default, use the current total CompSep "
              "allocation from resources.compsep.num_threads."),
    )
    parser.add_argument(
        "--compsep-nodes", type=int, default=1,
        help="Nodes dedicated to CompSep (default: 1).",
    )
    args = parser.parse_args()

    task_summary, groups = validate_parameter_file(args.parameter_file, args.check_paths)
    print(f"Valid parameter file: {args.parameter_file}")
    print(f"MPI tasks: {task_summary}")
    print(f"Enabled sampling groups: {', '.join(groups) if groups else 'none'}")
    params, _, _ = load_params(args.parameter_file)
    suggestion = suggest_compsep_thread_counts(
        params, threads_per_node=args.compsep_threads_per_node, num_nodes=args.compsep_nodes,
    )
    if suggestion:
        print("CompSep SHT scaling: (max(nside, 512) / 512)^2.7, with spin-2 weighted by 2.")
        print(f"Suggested resources.compsep.num_threads: {suggestion}")
    else:
        print("Suggested resources.compsep.num_threads: none (CompSep is disabled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
