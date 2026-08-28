"""Plot the band and component-separation products in a Commander4 run directory."""

import argparse
import csv
import glob
import logging
import os
import re
import time
import warnings
from dataclasses import dataclass

import h5py
import healpy as hp
import numpy as np
import yaml
from pixell.bunch import Bunch

from commander4.diagnostics import plotting
from commander4.file_io import paths
from commander4.parameters.bunch import as_bunch_recursive
from commander4.sky.comp_list import CompList
from commander4.sky.component import Component


CHAIN_ITER_RE = re.compile(r"chain(?P<chain>\d+)_iter(?P<iter>\d+)\.h5$")
LOGGER = logging.getLogger("plot_chain")
PLOT_TYPES = {"maps", "tod", "compsep", "components"}
DETECTOR_PLOT_MODES = {"individual", "summary", "both", "none"}
DETECTOR_SUMMARY_CHUNK_SIZE = 128
SPECTRUM_SUMMARY_CHUNK_SIZE = 16
POWER_SPECTRA = {
    "tod_ps_raw": "raw TOD",
    "tod_ps_ncorr": "correlated noise",
    "tod_ps_ncorrsub": "correlated-noise subtracted",
    "tod_ps_residual": "residual",
}
TOD_SCAN_DATASETS = {
    "temporal_gain": "temporal gain",
    "jump_counts": "jump count",
}
BAND_MAP_DATASETS = {
    "observed_sky",
    "rms",
    "skymodel",
    "res",
    "corrnoise",
    "nhit",
    "cov",
}
MAP_KINDS = {"rms": "rms", "cov": "weight", "nhit": "count"}
MAP_TITLES = {
    "observed_sky": "Observed sky",
    "rms": "White-noise RMS",
    "skymodel": "Sky model",
    "res": "TOD residual",
    "orbdipole": "Orbital dipole",
    "corrnoise": "Correlated noise",
    "nhit": "Hit count",
    "cov": "Inverse-noise covariance",
}
DETECTOR_SUMMARY_GROUPS = {
    "gain": (
        ("detrel_gain", "relative gain", "linear", (0.0,)),
        ("temporal_gain", "median temporal gain", "linear", (0.0,)),
    ),
    "noise": (
        ("noise_sigma0", "median sigma0", "log", ()),
        ("noise_fknee", "median fknee", "log", ()),
        ("noise_alpha", "median alpha", "linear", ()),
    ),
    "data_quality": (
        ("present_fraction", "present fraction", "linear", (1.0,)),
        ("accept_fraction", "accepted fraction when present", "linear", (1.0,)),
        ("good_fraction", "median good fraction", "linear", (1.0,)),
        ("chisq_abs", "median absolute chi-squared z-score", "linear", (0.0,)),
    ),
    "ncorr_solver": (
        ("ncorr_residual", "median relative CG residual", "log", ()),
        ("ncorr_niter", "median CG iterations", "linear", ()),
        ("ncorr_failure_fraction", "CG failure fraction", "linear", (0.0,)),
    ),
    "jumps": (
        ("jump_count", "mean jumps per scan", "linear", (0.0,)),
    ),
}


@dataclass(frozen=True)
class ChainFile:
    """One selected chain file and the labels parsed from its file name."""

    path: str
    chain: int
    iteration: int
    band: str | None = None


def _decode_h5_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_h5_value(value.item())
    return value


def _load_params_from_chain(run_dir: str) -> Bunch | None:
    """Load the embedded parameter file from the first readable chain file."""
    patterns = [
        os.path.join(run_dir, paths.CHAINS_BANDS, "*.h5"),
        os.path.join(run_dir, paths.CHAINS_COMPSEP, "*.h5"),
    ]
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            try:
                with h5py.File(path, "r") as handle:
                    if "metadata/parameter_file_as_string" not in handle:
                        continue
                    raw_yaml = _decode_h5_value(handle["metadata/parameter_file_as_string"][()])
                if not raw_yaml:
                    continue
                params_dict = yaml.safe_load(raw_yaml)
            except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
                LOGGER.debug("Could not read parameters from %s: %s", path, error)
                continue
            params = as_bunch_recursive(params_dict)
            params.parameter_file_as_string = yaml.dump(params_dict)
            params.parameter_file_binary_yaml = raw_yaml
            return params
    return None


def _extract_chain_iter(filename: str) -> tuple[int | None, int | None]:
    match = CHAIN_ITER_RE.search(filename)
    if not match:
        return None, None
    return int(match.group("chain")), int(match.group("iter"))


def _build_component_list(params: Bunch) -> list[Component]:
    """Build the same component execution views used by the live CompSep pipeline."""
    return list(CompList.init_from_params(params.components, params))


def _match_band_info(
    filename: str,
    params: Bunch | None,
) -> tuple[str | None, str | None, Bunch | None]:
    """Match a band-chain file name to its embedded experiment parameters."""
    if params is None or "experiments" not in params:
        return None, None, None
    for exp_name in params.experiments:
        experiment = params.experiments[exp_name]
        for band_name in experiment.bands:
            prefix = f"{exp_name}_{band_name}_"
            if filename.startswith(prefix):
                return exp_name, band_name, experiment.bands[band_name]
    return None, None, None


def _parse_int_set(value: str | None) -> set[int] | None:
    if value is None or value.lower() in {"all", "*"}:
        return None
    result: set[int] = set()
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_string, end_string = chunk.split("-", 1)
            start = int(start_string)
            end = int(end_string)
            for item in range(min(start, end), max(start, end) + 1):
                result.add(item)
        else:
            result.add(int(chunk))
    return result or None


def _parse_name_set(value: str | None) -> set[str] | None:
    if value is None or value.lower() in {"all", "*"}:
        return None
    result = {item.strip() for item in value.split(",") if item.strip()}
    return result or None


def _parse_plot_types(value: str) -> set[str]:
    selected = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not selected or "all" in selected:
        return set(PLOT_TYPES)
    unknown = selected - PLOT_TYPES
    if unknown:
        raise ValueError(f"Unknown plot types {sorted(unknown)}; choose from {sorted(PLOT_TYPES)}.")
    return selected


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", value).strip("_")


def _category_folder(output_dir: str, root: str, *categories: str) -> str:
    """Return a plot folder grouped by kind; the plot writer creates it lazily."""
    safe_categories = [_safe_name(category) for category in categories]
    return os.path.join(output_dir, root, *safe_categories)


def _band_name_from_file(filename: str) -> str | None:
    match = CHAIN_ITER_RE.search(filename)
    if match is None:
        return None
    return filename[:match.start()].rstrip("_")


def _band_is_selected(
    filename: str,
    band_label: str,
    params: Bunch | None,
    band_filter: set[str] | None,
) -> bool:
    if band_filter is None:
        return True
    exp_name, band_name, _ = _match_band_info(filename, params)
    candidates = {band_label}
    if band_name:
        candidates.add(band_name)
    if exp_name and band_name:
        candidates.add(f"{exp_name}_{band_name}")
    return bool(candidates & band_filter)


def _compsep_band_is_selected(view_name: str, band_filter: set[str] | None) -> bool:
    if band_filter is None:
        return True
    band_name = re.sub(r"_(I|QU)$", "", view_name)
    return view_name in band_filter or band_name in band_filter or any(
        name.endswith(f"_{band_name}") for name in band_filter
    )


def _discover_band_files(
    run_dir: str,
    params: Bunch | None,
    chain_filter: set[int] | None,
    iteration_filter: set[int] | None,
    band_filter: set[str] | None,
) -> list[ChainFile]:
    selected: list[ChainFile] = []
    pattern = os.path.join(run_dir, paths.CHAINS_BANDS, "*.h5")
    for path in sorted(glob.glob(pattern)):
        filename = os.path.basename(path)
        chain, iteration = _extract_chain_iter(filename)
        band = _band_name_from_file(filename)
        if chain is None or iteration is None or band is None:
            LOGGER.warning("Ignoring band file with an unrecognized name: %s", filename)
            continue
        if chain_filter is not None and chain not in chain_filter:
            continue
        if iteration_filter is not None and iteration not in iteration_filter:
            continue
        if not _band_is_selected(filename, band, params, band_filter):
            continue
        selected.append(ChainFile(path, chain, iteration, band))
    return selected


def _discover_compsep_files(
    run_dir: str,
    chain_filter: set[int] | None,
    iteration_filter: set[int] | None,
) -> list[ChainFile]:
    selected: list[ChainFile] = []
    pattern = os.path.join(run_dir, paths.CHAINS_COMPSEP, "*.h5")
    for path in sorted(glob.glob(pattern)):
        filename = os.path.basename(path)
        chain, iteration = _extract_chain_iter(filename)
        if chain is None or iteration is None:
            LOGGER.warning("Ignoring compsep file with an unrecognized name: %s", filename)
            continue
        if chain_filter is not None and chain not in chain_filter:
            continue
        if iteration_filter is not None and iteration not in iteration_filter:
            continue
        selected.append(ChainFile(path, chain, iteration))
    return selected


def _read_detector_names(handle: h5py.File) -> list[str]:
    if "det_names" not in handle:
        return []
    names = np.asarray(handle["det_names"][()]).reshape(-1)
    return [str(_decode_h5_value(name)) for name in names]


def _parameter_labels(dataset_name: str, size: int) -> list[str]:
    known = {
        "noise_params": ["sigma0", "fknee", "alpha"],
        "gain_prior": ["sigma0", "fknee", "alpha"],
        "orbital_velocity": ["x", "y", "z"],
    }
    if size == 1 and dataset_name not in known:
        return [""]
    labels = known.get(dataset_name, [])
    result = []
    for index in range(size):
        result.append(labels[index] if index < len(labels) else f"parameter{index}")
    return result


def _map_row_labels(dataset_name: str, row_count: int) -> list[str]:
    if dataset_name == "nhit":
        return ["all"]
    if dataset_name == "cov":
        covariance_labels = ["II", "IQ", "IU", "QQ", "QU", "UU"]
        return covariance_labels[:row_count]
    if row_count == 1:
        return ["I"]
    if row_count == 2:
        return ["Q", "U"]
    if row_count == 3:
        return ["I", "Q", "U"]
    return [f"row{index}" for index in range(row_count)]


def _ensure_map_rows(map_data: np.ndarray) -> np.ndarray:
    array = np.asarray(map_data)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(
            f"Expected a one- or two-dimensional HEALPix map, got shape {array.shape}."
        )
    return array


def _resample_map(map_data: np.ndarray, nside_out: int | None, kind: str) -> np.ndarray:
    """Resample a map while preserving the quantity that is additive over pixels."""
    rows = _ensure_map_rows(map_data)
    if nside_out is None:
        return rows
    nside_in = hp.npix2nside(rows.shape[-1])
    if nside_in == nside_out:
        return rows

    output = []
    for row in rows:
        if kind == "rms":
            weights = np.zeros(row.shape, dtype=float)
            valid = np.isfinite(row) & (row > 0)
            weights[valid] = 1.0 / row[valid] ** 2
            resampled_weights = hp.ud_grade(weights, nside_out)
            resampled = np.full(resampled_weights.shape, np.inf)
            positive = resampled_weights > 0
            resampled[positive] = 1.0 / np.sqrt(resampled_weights[positive])
        elif kind in {"weight", "count"}:
            # hp.ud_grade averages. This factor turns that average into a conserved sum, and also
            # distributes a parent pixel between its children when plotting at a higher nside.
            pixel_ratio = row.size / hp.nside2npix(nside_out)
            resampled = hp.ud_grade(row.astype(float), nside_out) * pixel_ratio
        else:
            resampled = hp.ud_grade(row, nside_out)
        output.append(resampled)
    return np.asarray(output)


def _map_unit(dataset_name: str, band_unit: str) -> str:
    if dataset_name == "nhit":
        return "samples"
    if dataset_name == "cov":
        return f"{band_unit}$^{{-2}}$"
    return band_unit


def _map_is_symmetric(dataset_name: str, row_label: str) -> bool:
    if dataset_name in {"rms", "nhit"}:
        return False
    if dataset_name == "cov" and row_label in {"II", "QQ", "UU"}:
        return False
    return True


def _plot_band_maps(entries: list[ChainFile], output_dir: str, nside_out: int | None) -> int:
    plot_count = 0
    plotted_once: set[tuple[str | None, int, str]] = set()
    for entry in entries:
        try:
            with h5py.File(entry.path, "r") as handle:
                if "maps" not in handle:
                    continue
                band_unit = "uK_RJ"
                if "metadata/band_unit" in handle:
                    band_unit = str(_decode_h5_value(handle["metadata/band_unit"][()]))
                fwhm = None
                if "metadata/map_fwhm_arcmin" in handle:
                    fwhm = float(handle["metadata/map_fwhm_arcmin"][()])
                nhit = None
                if "maps/nhit" in handle:
                    nhit = _resample_map(handle["maps/nhit"][()], nside_out, "count")[0]

                for dataset_name in sorted(handle["maps"].keys()):
                    if dataset_name not in BAND_MAP_DATASETS:
                        continue
                    once_key = (entry.band, entry.chain, dataset_name)
                    if dataset_name == "nhit" and once_key in plotted_once:
                        continue
                    out_folder = _category_folder(output_dir, "maps_bands", dataset_name)
                    dataset = handle[f"maps/{dataset_name}"]
                    if not isinstance(dataset, h5py.Dataset):
                        continue
                    kind = MAP_KINDS.get(dataset_name, "brightness")
                    try:
                        map_rows = _resample_map(dataset[()], nside_out, kind)
                    except ValueError as error:
                        LOGGER.warning("Skipping %s in %s: %s", dataset_name, entry.path, error)
                        continue
                    if nhit is not None and dataset_name != "nhit":
                        map_rows = np.asarray(map_rows, dtype=float)
                        map_rows[:, nhit <= 0] = np.nan
                    row_labels = _map_row_labels(dataset_name, map_rows.shape[0])
                    title = (
                        f"{MAP_TITLES.get(dataset_name, dataset_name)}; {entry.band}, "
                        f"chain {entry.chain}, iteration {entry.iteration}"
                    )
                    if fwhm is not None and dataset_name in {"observed_sky", "rms"}:
                        title += f", FWHM {fwhm:g} arcmin"
                    if dataset_name == "nhit":
                        filename = os.path.join(
                            out_folder,
                            f"chain{entry.chain:02d}_{_safe_name(entry.band or 'band')}.png",
                        )
                    else:
                        filename = os.path.join(
                            out_folder,
                            f"chain{entry.chain:02d}_{_safe_name(entry.band or 'band')}_"
                            f"iter{entry.iteration:04d}.png",
                        )
                    plotting.plot_chain_map_grid(
                        filename,
                        title,
                        map_rows,
                        row_labels,
                        unit=_map_unit(dataset_name, band_unit),
                        symmetric=[
                            _map_is_symmetric(dataset_name, label) for label in row_labels
                        ],
                    )
                    if os.path.isfile(filename):
                        plot_count += 1
                        if dataset_name == "nhit":
                            plotted_once.add(once_key)
        except OSError as error:
            LOGGER.warning("Could not read %s: %s", entry.path, error)
    return plot_count


def _group_band_entries(entries: list[ChainFile]) -> dict[tuple[str, int], list[ChainFile]]:
    groups: dict[tuple[str, int], list[ChainFile]] = {}
    for entry in entries:
        if entry.band is None:
            continue
        groups.setdefault((entry.band, entry.chain), []).append(entry)
    for group_entries in groups.values():
        group_entries.sort(key=lambda item: item.iteration)
    return groups


def _thin_xy(
    x_values: np.ndarray,
    y_values: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if x_values.size <= max_points:
        return x_values, y_values
    indices = np.linspace(0, x_values.size - 1, max_points, dtype=int)
    return x_values[indices], y_values[indices]


def _plot_tod_scalar_traces(
    band: str,
    chain: int,
    entries: list[ChainFile],
    output_dir: str,
    detector_filter: set[str] | None,
    include_detectors: bool = True,
) -> int:
    plot_count = 0
    for dataset_name in ("abs_gain", "detrel_gain"):
        if dataset_name == "detrel_gain" and not include_detectors:
            continue
        out_folder = _category_folder(output_dir, "tod_traces", dataset_name)
        records = []
        for entry in entries:
            try:
                with h5py.File(entry.path, "r") as handle:
                    if dataset_name not in handle:
                        continue
                    records.append((entry.iteration, _read_detector_names(handle),
                                    np.asarray(handle[dataset_name][()])))
            except OSError:
                continue
        if not records:
            continue

        if records[0][2].ndim == 0:
            x_values = np.array([record[0] for record in records])
            y_values = np.array([float(record[2]) for record in records])
            filename = os.path.join(
                out_folder, f"chain{chain:02d}_{_safe_name(band)}.png"
            )
            plotting.plot_chain_lines(
                filename,
                f"{band}, chain {chain}: {dataset_name}",
                "Gibbs iteration",
                dataset_name,
                [("", x_values, y_values)],
            )
            plot_count += int(os.path.isfile(filename))
            continue

        detector_names = sorted({name for _, names, _ in records for name in names})
        for detector_name in detector_names:
            if detector_filter is not None and detector_name not in detector_filter:
                continue
            parameter_count = max(
                record[2].shape[1] if record[2].ndim > 1 else 1 for record in records
                if detector_name in record[1]
            )
            parameter_labels = _parameter_labels(dataset_name, parameter_count)
            for parameter_index, parameter_label in enumerate(parameter_labels):
                x_values = []
                y_values = []
                for iteration, names, values in records:
                    if detector_name not in names:
                        continue
                    detector_index = names.index(detector_name)
                    if values.ndim == 1:
                        value = values[detector_index]
                    elif parameter_index < values.shape[1]:
                        value = values[detector_index, parameter_index]
                    else:
                        continue
                    x_values.append(iteration)
                    y_values.append(value)
                filename = os.path.join(
                    out_folder,
                    f"chain{chain:02d}_{_safe_name(band)}_{_safe_name(detector_name)}_"
                    f"{parameter_label or 'value'}.png",
                )
                plotting.plot_chain_lines(
                    filename,
                    f"{band} {detector_name}, chain {chain}: {dataset_name} {parameter_label}",
                    "Gibbs iteration",
                    f"{dataset_name} {parameter_label}".strip(),
                    [("", np.asarray(x_values), np.asarray(y_values))],
                )
                plot_count += int(os.path.isfile(filename))
    return plot_count


def _load_tod_scan_records(entries: list[ChainFile], dataset_name: str):
    records = []
    for entry in entries:
        try:
            with h5py.File(entry.path, "r") as handle:
                if dataset_name not in handle or "scan_ids" not in handle:
                    continue
                values = np.asarray(handle[dataset_name][()])
                scan_ids = np.asarray(handle["scan_ids"][()])
                detector_names = _read_detector_names(handle)
                present = np.asarray(handle["present"][()]) if "present" in handle else None
                records.append((entry.iteration, scan_ids, detector_names, present, values))
        except OSError:
            continue
    return records


def _plot_tod_scan_series(
    band: str,
    chain: int,
    entries: list[ChainFile],
    output_dir: str,
    detector_filter: set[str] | None,
    max_points: int,
) -> int:
    plot_count = 0

    for dataset_name, ylabel in TOD_SCAN_DATASETS.items():
        out_folder = _category_folder(output_dir, "tod_scans", dataset_name)
        records = _load_tod_scan_records(entries, dataset_name)
        if not records:
            continue
        detector_names = sorted({name for _, _, names, _, _ in records for name in names})
        max_points_per_line = max(2, max_points // max(1, len(records)))
        for detector_name in detector_names:
            if detector_filter is not None and detector_name not in detector_filter:
                continue
            parameter_count = max(
                values.shape[2] if values.ndim == 3 else 1
                for _, _, names, _, values in records if detector_name in names
            )
            parameter_labels = _parameter_labels(dataset_name, parameter_count)
            for parameter_index, parameter_label in enumerate(parameter_labels):
                series = []
                for iteration, scan_ids, names, present, values in records:
                    if detector_name not in names:
                        continue
                    detector_index = names.index(detector_name)
                    if values.ndim == 2:
                        y_values = np.asarray(values[:, detector_index], dtype=float)
                    elif parameter_index < values.shape[2]:
                        y_values = np.asarray(
                            values[:, detector_index, parameter_index], dtype=float
                        )
                    else:
                        continue
                    if present is not None and dataset_name != "present":
                        y_values[~present[:, detector_index].astype(bool)] = np.nan
                    x_values, y_values = _thin_xy(
                        scan_ids, y_values, max_points_per_line
                    )
                    series.append((f"iter {iteration}", x_values, y_values))
                filename = os.path.join(
                    out_folder,
                    f"chain{chain:02d}_{_safe_name(band)}_{_safe_name(detector_name)}_"
                    f"{parameter_label or 'value'}.png",
                )
                yscale = "log" if dataset_name == "ncorr_cg_residual" else "linear"
                horizontal_lines = (0.0,) if dataset_name == "chisq_z" else ()
                plotting.plot_chain_lines(
                    filename,
                    f"{band} {detector_name}, chain {chain}: {ylabel} {parameter_label}",
                    "scan ID",
                    f"{ylabel} {parameter_label}".strip(),
                    series,
                    yscale=yscale,
                    horizontal_lines=horizontal_lines,
                )
                plot_count += int(os.path.isfile(filename))
    return plot_count


def _scan_panel_series(
    records,
    detector_name: str,
    dataset_name: str,
    parameter_index: int,
    max_points: int,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Build per-iteration scan lines for one detector and dataset column."""
    series = []
    max_points_per_line = max(2, max_points // max(1, len(records)))
    for iteration, scan_ids, names, present, values in records:
        if detector_name not in names:
            continue
        detector_index = names.index(detector_name)
        if values.ndim == 2:
            y_values = np.asarray(values[:, detector_index], dtype=float)
        else:
            y_values = np.asarray(values[:, detector_index, parameter_index], dtype=float)
        if present is not None and dataset_name != "present":
            y_values[~present[:, detector_index].astype(bool)] = np.nan
        x_values, y_values = _thin_xy(scan_ids, y_values, max_points_per_line)
        series.append((f"iter {iteration}", x_values, y_values))
    return series


def _detector_scan_values(records, detector_name: str, dataset_name: str) -> np.ndarray:
    """Flatten finite, present scan values across iterations for inclusion decisions."""
    collected = []
    for _, _, names, present, values in records:
        if detector_name not in names:
            continue
        detector_index = names.index(detector_name)
        detector_values = np.asarray(values[:, detector_index], dtype=float)
        if present is not None and dataset_name != "present":
            detector_values = detector_values[present[:, detector_index].astype(bool)]
        collected.append(detector_values.reshape(-1))
    if not collected:
        return np.array([])
    values = np.concatenate(collected)
    return values[np.isfinite(values)]


def _masked_detector_reduce(
    values: np.ndarray,
    present: np.ndarray | None,
    reduction: str = "median",
) -> np.ndarray:
    """Reduce scans to one value per detector, excluding absent and non-finite samples."""
    numeric = np.asarray(values, dtype=float)
    valid = np.isfinite(numeric)
    if present is not None:
        presence = present
        if numeric.ndim == 3:
            presence = presence[..., None]
        valid &= presence
    masked = np.where(valid, numeric, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if reduction == "mean":
            return np.nanmean(masked, axis=0)
        return np.nanmedian(masked, axis=0)


def _read_detector_summary(
    entry: ChainFile,
    detector_filter: set[str] | None,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Read one band-chain file into equal-weight, per-detector diagnostic summaries."""
    with h5py.File(entry.path, "r") as handle:
        detector_names = _read_detector_names(handle)
        detector_count = len(detector_names)
        metrics: dict[str, np.ndarray] = {}
        if "detrel_gain" in handle:
            values = np.asarray(handle["detrel_gain"][()], dtype=float).reshape(-1)
            if values.size == detector_count:
                metrics["detrel_gain"] = values

        two_dimensional = {
            "temporal_gain": ("temporal_gain", "median", False),
            "good_fraction": ("good_fraction", "median", False),
            "chisq_z": ("chisq_abs", "median", True),
            "ncorr_cg_residual": ("ncorr_residual", "median", False),
            "ncorr_cg_niter": ("ncorr_niter", "median", False),
            "jump_counts": ("jump_count", "mean", False),
        }
        for dataset_name, (metric_name, _, _) in two_dimensional.items():
            if dataset_name in handle:
                metrics[metric_name] = np.full(detector_count, np.nan)
        if "present" in handle:
            metrics["present_fraction"] = np.full(detector_count, np.nan)
        if "accept" in handle:
            metrics["accept_fraction"] = np.full(detector_count, np.nan)
        if "ncorr_converged" in handle:
            metrics["ncorr_failure_fraction"] = np.full(detector_count, np.nan)
        noise_metric_names = ("noise_sigma0", "noise_fknee", "noise_alpha")
        noise_parameter_count = 0
        if "noise_params" in handle and handle["noise_params"].ndim == 3:
            noise_parameter_count = min(
                len(noise_metric_names), handle["noise_params"].shape[2]
            )
            for metric_name in noise_metric_names[:noise_parameter_count]:
                metrics[metric_name] = np.full(detector_count, np.nan)

        for start in range(0, detector_count, DETECTOR_SUMMARY_CHUNK_SIZE):
            stop = min(start + DETECTOR_SUMMARY_CHUNK_SIZE, detector_count)
            present = None
            if "present" in handle:
                present = np.asarray(handle["present"][:, start:stop], dtype=bool)
                metrics["present_fraction"][start:stop] = np.mean(present, axis=0)

            if "accept" in handle:
                accepted = np.asarray(handle["accept"][:, start:stop], dtype=bool)
                valid = present if present is not None else np.ones_like(accepted, dtype=bool)
                denominator = np.sum(valid, axis=0)
                numerator = np.sum(accepted & valid, axis=0)
                metrics["accept_fraction"][start:stop] = np.divide(
                    numerator,
                    denominator,
                    out=np.full(stop - start, np.nan),
                    where=denominator > 0,
                )

            for dataset_name, (metric_name, reduction, absolute) in two_dimensional.items():
                if dataset_name not in handle:
                    continue
                values = np.asarray(handle[dataset_name][:, start:stop])
                if absolute:
                    values = np.abs(values)
                metrics[metric_name][start:stop] = _masked_detector_reduce(
                    values, present, reduction
                )

            if noise_parameter_count:
                noise_values = np.asarray(handle["noise_params"][:, start:stop, :])
                reduced = _masked_detector_reduce(noise_values, present)
                for parameter_index, metric_name in enumerate(
                    noise_metric_names[:noise_parameter_count]
                ):
                    metrics[metric_name][start:stop] = reduced[:, parameter_index]

            if "ncorr_converged" in handle:
                convergence = np.asarray(handle["ncorr_converged"][:, start:stop])
                valid = convergence >= 0
                if present is not None:
                    valid &= present
                denominator = np.sum(valid, axis=0)
                failures = np.sum((convergence == 0) & valid, axis=0)
                metrics["ncorr_failure_fraction"][start:stop] = np.divide(
                    failures,
                    denominator,
                    out=np.full(stop - start, np.nan),
                    where=denominator > 0,
                )

    if detector_filter is None:
        return detector_names, metrics
    selected = np.array([name in detector_filter for name in detector_names], dtype=bool)
    filtered_metrics = {name: values[selected] for name, values in metrics.items()}
    return [name for name, keep in zip(detector_names, selected) if keep], filtered_metrics


def _detector_population_quantiles(values: np.ndarray) -> np.ndarray:
    """Return 2.5, 16, 50, 84 and 97.5 percentiles across equally weighted detectors."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.full(5, np.nan)
    return np.percentile(finite, [2.5, 16.0, 50.0, 84.0, 97.5])


def _write_detector_summary_csv(
    band: str,
    chain: int,
    iteration: int,
    detector_names: list[str],
    metrics: dict[str, np.ndarray],
    output_dir: str,
) -> None:
    """Write the latest per-detector metrics, ordered with likely outliers first."""
    if not detector_names or not metrics:
        return
    metric_names = sorted(metrics)

    def finite_value(metric_name: str, index: int, fallback: float) -> float:
        if metric_name not in metrics or not np.isfinite(metrics[metric_name][index]):
            return fallback
        return float(metrics[metric_name][index])

    indices = list(range(len(detector_names)))
    indices.sort(key=lambda index: (
        finite_value("accept_fraction", index, 1.0),
        finite_value("present_fraction", index, 1.0),
        -finite_value("ncorr_failure_fraction", index, 0.0),
        -finite_value("chisq_abs", index, 0.0),
    ))
    folder = _category_folder(output_dir, "tod_summaries", "detector_tables")
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(
        folder,
        f"chain{chain:02d}_{_safe_name(band)}_iter{iteration:04d}.csv",
    )
    with open(filename, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["detector", *metric_names])
        for detector_index in indices:
            writer.writerow([
                detector_names[detector_index],
                *[metrics[name][detector_index] for name in metric_names],
            ])


def _plot_detector_summaries(
    entries: list[ChainFile],
    per_iteration_entries: list[ChainFile],
    output_dir: str,
    detector_filter: set[str] | None,
) -> int:
    """Plot detector-population quantiles and write latest per-detector metric tables."""
    plot_count = 0
    per_iteration_paths = {entry.path for entry in per_iteration_entries}
    for (band, chain), group_entries in _group_band_entries(entries).items():
        records: dict[str, list[tuple[int, np.ndarray]]] = {}
        latest: tuple[int, list[str], dict[str, np.ndarray]] | None = None
        for entry in group_entries:
            try:
                detector_names, metrics = _read_detector_summary(entry, detector_filter)
            except OSError as error:
                LOGGER.warning("Could not summarize detectors from %s: %s", entry.path, error)
                continue
            for metric_name, values in metrics.items():
                records.setdefault(metric_name, []).append((
                    entry.iteration, _detector_population_quantiles(values)
                ))
            if (
                entry.path in per_iteration_paths
                and "noise_fknee" in metrics
                and "noise_alpha" in metrics
            ):
                accepted_fraction = metrics.get(
                    "accept_fraction", np.ones(len(detector_names))
                )
                fully_accepted = np.isfinite(accepted_fraction) & np.isclose(
                    accepted_fraction, 1.0
                )
                folder = _category_folder(
                    output_dir, "tod_summaries", "noise_params_fknee_alpha"
                )
                filename = os.path.join(
                    folder,
                    f"chain{chain:02d}_{_safe_name(band)}_"
                    f"iter{entry.iteration:04d}.png",
                )
                plotting.plot_noise_parameter_density(
                    filename,
                    f"{band}; chain {chain}, iteration {entry.iteration}: "
                    "per-detector median noise parameters",
                    metrics["noise_fknee"],
                    metrics["noise_alpha"],
                    fully_accepted,
                    density_label="detectors per hexagon",
                    rejected_label="detector has rejected scans",
                )
                plot_count += int(os.path.isfile(filename))
            latest = (entry.iteration, detector_names, metrics)

        percentile_labels = ("2.5%", "16%", "median", "84%", "97.5%")
        for group_name, metric_specs in DETECTOR_SUMMARY_GROUPS.items():
            panels = []
            for metric_name, title, yscale, reference_lines in metric_specs:
                metric_records = records.get(metric_name, [])
                if not metric_records:
                    continue
                iterations = np.array([iteration for iteration, _ in metric_records])
                quantiles = np.asarray([values for _, values in metric_records])
                series = [
                    (label, iterations, quantiles[:, index])
                    for index, label in enumerate(percentile_labels)
                ]
                panels.append(plotting.ChainLinePanel(
                    title=title,
                    ylabel=title,
                    series=series,
                    yscale=yscale,
                    horizontal_lines=reference_lines,
                ))
            if not panels:
                continue
            folder = _category_folder(output_dir, "tod_summaries", group_name)
            filename = os.path.join(
                folder, f"chain{chain:02d}_{_safe_name(band)}.png"
            )
            plotting.plot_chain_line_panels(
                filename,
                f"{band}, chain {chain}: detector population",
                "Gibbs iteration",
                panels,
            )
            plot_count += int(os.path.isfile(filename))

        if latest is not None:
            iteration, detector_names, metrics = latest
            _write_detector_summary_csv(
                band, chain, iteration, detector_names, metrics, output_dir
            )
    return plot_count


def _plot_noise_parameter_dashboard(
    band: str,
    chain: int,
    entries: list[ChainFile],
    output_dir: str,
    detector_filter: set[str] | None,
    max_points: int,
) -> int:
    records = _load_tod_scan_records(entries, "noise_params")
    if not records:
        return 0
    out_folder = _category_folder(output_dir, "tod_scans", "noise_params")
    detector_names = sorted({name for _, _, names, _, _ in records for name in names})
    plot_count = 0
    labels = ["sigma0", "fknee", "alpha"]
    for detector_name in detector_names:
        if detector_filter is not None and detector_name not in detector_filter:
            continue
        parameter_count = min(3, max(record[4].shape[2] for record in records))
        panels = []
        for parameter_index in range(parameter_count):
            label = labels[parameter_index]
            panels.append(plotting.ChainLinePanel(
                title=label,
                ylabel=label,
                series=_scan_panel_series(
                    records, detector_name, "noise_params", parameter_index, max_points
                ),
            ))
        filename = os.path.join(
            out_folder,
            f"chain{chain:02d}_{_safe_name(band)}_{_safe_name(detector_name)}.png",
        )
        plotting.plot_chain_line_panels(
            filename,
            f"{band} {detector_name}, chain {chain}: noise parameters",
            "scan ID",
            panels,
        )
        plot_count += int(os.path.isfile(filename))
    return plot_count


def _plot_data_quality_dashboard(
    band: str,
    chain: int,
    entries: list[ChainFile],
    output_dir: str,
    detector_filter: set[str] | None,
    max_points: int,
) -> int:
    dataset_info = {
        "present": ("present", "present", ()),
        "accept": ("accepted", "accepted", ()),
        "good_fraction": ("good fraction", "fraction", (1.0,)),
        "chisq_z": ("white-noise chi-squared", "z-score", (0.0,)),
    }
    records_by_dataset = {
        name: _load_tod_scan_records(entries, name) for name in dataset_info
    }
    detector_names = sorted({
        detector_name
        for records in records_by_dataset.values()
        for _, _, names, _, _ in records
        for detector_name in names
    })
    out_folder = _category_folder(output_dir, "tod_scans", "data_quality")
    plot_count = 0
    for detector_name in detector_names:
        if detector_filter is not None and detector_name not in detector_filter:
            continue
        panels = []
        for dataset_name, (title, ylabel, reference_lines) in dataset_info.items():
            records = records_by_dataset[dataset_name]
            values = _detector_scan_values(records, detector_name, dataset_name)
            if dataset_name in {"present", "accept"} and values.size and np.all(values == 1.0):
                continue
            panels.append(plotting.ChainLinePanel(
                title=title,
                ylabel=ylabel,
                series=_scan_panel_series(
                    records, detector_name, dataset_name, 0, max_points
                ),
                horizontal_lines=reference_lines,
            ))
        filename = os.path.join(
            out_folder,
            f"chain{chain:02d}_{_safe_name(band)}_{_safe_name(detector_name)}.png",
        )
        plotting.plot_chain_line_panels(
            filename,
            f"{band} {detector_name}, chain {chain}: data quality",
            "scan ID",
            panels,
        )
        plot_count += int(os.path.isfile(filename))
    return plot_count


def _plot_ncorr_solver_dashboard(
    band: str,
    chain: int,
    entries: list[ChainFile],
    output_dir: str,
    detector_filter: set[str] | None,
    max_points: int,
) -> int:
    residual_records = _load_tod_scan_records(entries, "ncorr_cg_residual")
    iteration_records = _load_tod_scan_records(entries, "ncorr_cg_niter")
    convergence_records = _load_tod_scan_records(entries, "ncorr_converged")
    all_records = residual_records + iteration_records + convergence_records
    detector_names = sorted({name for _, _, names, _, _ in all_records for name in names})
    out_folder = _category_folder(output_dir, "tod_scans", "ncorr_solver")
    plot_count = 0
    for detector_name in detector_names:
        if detector_filter is not None and detector_name not in detector_filter:
            continue
        panels = [
            plotting.ChainLinePanel(
                title="final relative CG residual",
                ylabel="relative residual",
                series=_scan_panel_series(
                    residual_records, detector_name, "ncorr_cg_residual", 0, max_points
                ),
                yscale="log",
            ),
            plotting.ChainLinePanel(
                title="CG iterations",
                ylabel="iterations",
                series=_scan_panel_series(
                    iteration_records, detector_name, "ncorr_cg_niter", 0, max_points
                ),
            ),
        ]
        convergence = _detector_scan_values(
            convergence_records, detector_name, "ncorr_converged"
        )
        if np.any(convergence == 0.0):
            panels.append(plotting.ChainLinePanel(
                title="convergence failures",
                ylabel="state",
                series=_scan_panel_series(
                    convergence_records, detector_name, "ncorr_converged", 0, max_points
                ),
                horizontal_lines=(1.0,),
            ))
        filename = os.path.join(
            out_folder,
            f"chain{chain:02d}_{_safe_name(band)}_{_safe_name(detector_name)}.png",
        )
        plotting.plot_chain_line_panels(
            filename,
            f"{band} {detector_name}, chain {chain}: correlated-noise solver",
            "scan ID",
            panels,
        )
        plot_count += int(os.path.isfile(filename))
    return plot_count


def _plot_noise_parameter_density(
    band: str,
    chain: int,
    entries: list[ChainFile],
    output_dir: str,
    detector_filter: set[str] | None,
) -> int:
    """Plot the joint fknee--alpha distribution for every detector and iteration."""
    out_folder = _category_folder(output_dir, "tod_scans", "noise_params_fknee_alpha")
    plot_count = 0
    for entry in entries:
        try:
            with h5py.File(entry.path, "r") as handle:
                if "noise_params" not in handle or handle["noise_params"].shape[-1] < 3:
                    continue
                detector_names = _read_detector_names(handle)
                present = np.asarray(handle["present"][()]).astype(bool)
                if "accept" in handle:
                    accept = np.asarray(handle["accept"][()]).astype(bool)
                else:
                    accept = present.copy()
                for detector_index, detector_name in enumerate(detector_names):
                    if detector_filter is not None and detector_name not in detector_filter:
                        continue
                    noise_params = np.asarray(handle["noise_params"][:, detector_index, 1:3])
                    accepted = present[:, detector_index] & accept[:, detector_index]
                    # Absent detector-scans are neither accepted nor useful rejected samples.
                    fknee = np.asarray(noise_params[:, 0], dtype=float)
                    alpha = np.asarray(noise_params[:, 1], dtype=float)
                    fknee[~present[:, detector_index]] = np.nan
                    alpha[~present[:, detector_index]] = np.nan
                    filename = os.path.join(
                        out_folder,
                        f"chain{chain:02d}_{_safe_name(band)}_"
                        f"{_safe_name(detector_name)}_iter{entry.iteration:04d}.png",
                    )
                    plotting.plot_noise_parameter_density(
                        filename,
                        f"{band} {detector_name}; chain {chain}, "
                        f"iteration {entry.iteration}",
                        fknee,
                        alpha,
                        accepted,
                    )
                    plot_count += int(os.path.isfile(filename))
        except OSError as error:
            LOGGER.warning("Could not read noise parameters from %s: %s", entry.path, error)
    return plot_count


def _spectrum_summary(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(values, axis=0)
        lower = np.nanpercentile(values, 16.0, axis=0)
        upper = np.nanpercentile(values, 84.0, axis=0)
    return median, lower, upper


def _plot_tod_power_spectra(
    band: str,
    chain: int,
    entries: list[ChainFile],
    output_dir: str,
    detector_filter: set[str] | None,
) -> int:
    out_folder = _category_folder(output_dir, "tod_power_spectra", "scan_distribution")
    plot_count = 0
    for entry in entries:
        try:
            with h5py.File(entry.path, "r") as handle:
                if "tod_ps_freqs" not in handle:
                    continue
                detector_names = _read_detector_names(handle)
                present = (
                    np.asarray(handle["present"][()]).astype(bool)
                    if "present" in handle else None
                )
                accept = (
                    np.asarray(handle["accept"][()]).astype(bool)
                    if "accept" in handle else None
                )
                for detector_index, detector_name in enumerate(detector_names):
                    if detector_filter is not None and detector_name not in detector_filter:
                        continue
                    valid_scans = np.ones(handle["tod_ps_freqs"].shape[0], dtype=bool)
                    if present is not None:
                        valid_scans &= present[:, detector_index]
                    if accept is not None:
                        valid_scans &= accept[:, detector_index]
                    if not np.any(valid_scans):
                        continue
                    freqs = np.asarray(handle["tod_ps_freqs"][:, detector_index, :])[valid_scans]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        median_freqs = np.nanmedian(freqs, axis=0)
                    spectra = []
                    for dataset_name, label in POWER_SPECTRA.items():
                        if dataset_name not in handle:
                            continue
                        values = np.asarray(handle[dataset_name][:, detector_index, :])[valid_scans]
                        median, lower, upper = _spectrum_summary(values)
                        spectra.append((label, median_freqs, median, lower, upper))
                    filename = os.path.join(
                        out_folder,
                        f"chain{chain:02d}_{_safe_name(band)}_"
                        f"{_safe_name(detector_name)}_iter{entry.iteration:04d}.png",
                    )
                    plotting.plot_chain_spectra(
                        filename,
                        f"{band} {detector_name}, chain {chain}, iteration {entry.iteration}",
                        spectra,
                    )
                    plot_count += int(os.path.isfile(filename))
        except OSError as error:
            LOGGER.warning("Could not read TOD spectra from %s: %s", entry.path, error)
    return plot_count


def _plot_tod_power_spectra_summary(
    entries: list[ChainFile],
    output_dir: str,
    detector_filter: set[str] | None,
) -> int:
    """Plot one equally detector-weighted TOD-spectrum summary per band and iteration."""
    output_folder = _category_folder(output_dir, "tod_summaries", "power_spectra")
    plot_count = 0
    for entry in entries:
        try:
            with h5py.File(entry.path, "r") as handle:
                if "tod_ps_freqs" not in handle:
                    continue
                detector_names = _read_detector_names(handle)
                detector_count = len(detector_names)
                selected = np.array([
                    detector_filter is None or name in detector_filter
                    for name in detector_names
                ], dtype=bool)
                if not np.any(selected):
                    continue

                frequency_summaries = []
                spectra_by_dataset: dict[str, list[np.ndarray]] = {
                    dataset_name: []
                    for dataset_name in POWER_SPECTRA if dataset_name in handle
                }
                for start in range(0, detector_count, SPECTRUM_SUMMARY_CHUNK_SIZE):
                    stop = min(start + SPECTRUM_SUMMARY_CHUNK_SIZE, detector_count)
                    keep = selected[start:stop]
                    if not np.any(keep):
                        continue
                    scan_count = handle["tod_ps_freqs"].shape[0]
                    valid = np.ones((scan_count, stop - start), dtype=bool)
                    if "present" in handle:
                        valid &= np.asarray(handle["present"][:, start:stop], dtype=bool)
                    if "accept" in handle:
                        valid &= np.asarray(handle["accept"][:, start:stop], dtype=bool)
                    valid = valid[:, keep]
                    frequencies = np.asarray(
                        handle["tod_ps_freqs"][:, start:stop, :]
                    )[:, keep, :]
                    frequency_summaries.append(
                        _masked_detector_reduce(frequencies, valid)
                    )
                    for dataset_name in spectra_by_dataset:
                        values = np.asarray(handle[dataset_name][:, start:stop, :])[:, keep, :]
                        spectra_by_dataset[dataset_name].append(
                            _masked_detector_reduce(values, valid)
                        )

                if not frequency_summaries:
                    continue
                detector_frequencies = np.concatenate(frequency_summaries, axis=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    median_frequencies = np.nanmedian(detector_frequencies, axis=0)
                spectra = []
                for dataset_name, chunks in spectra_by_dataset.items():
                    if not chunks:
                        continue
                    detector_spectra = np.concatenate(chunks, axis=0)
                    median, lower, upper = _spectrum_summary(detector_spectra)
                    spectra.append((
                        POWER_SPECTRA[dataset_name],
                        median_frequencies,
                        median,
                        lower,
                        upper,
                    ))
                filename = os.path.join(
                    output_folder,
                    f"chain{entry.chain:02d}_{_safe_name(entry.band or 'band')}_"
                    f"iter{entry.iteration:04d}.png",
                )
                plotting.plot_chain_spectra(
                    filename,
                    f"{entry.band}, chain {entry.chain}, iteration {entry.iteration}: "
                    "detector-population TOD spectra",
                    spectra,
                )
                plot_count += int(os.path.isfile(filename))
        except OSError as error:
            LOGGER.warning("Could not summarize TOD spectra from %s: %s", entry.path, error)
    return plot_count


def _plot_tod_outputs(
    entries: list[ChainFile],
    per_iteration_entries: list[ChainFile],
    output_dir: str,
    detector_filter: set[str] | None,
    max_points: int,
    *,
    include_scalar_traces: bool = True,
    include_detector_dashboards: bool = True,
    include_per_iteration: bool = True,
    include_individual_detectors: bool = True,
) -> int:
    plot_count = 0
    # These figures contain every Gibbs iteration and deliberately ignore --iter.
    for (band, chain), group_entries in _group_band_entries(entries).items():
        if include_scalar_traces:
            plot_count += _plot_tod_scalar_traces(
                band,
                chain,
                group_entries,
                output_dir,
                detector_filter,
                include_detectors=include_individual_detectors,
            )
        if include_detector_dashboards and include_individual_detectors:
            plot_count += _plot_tod_scan_series(
                band, chain, group_entries, output_dir, detector_filter, max_points
            )
            plot_count += _plot_noise_parameter_dashboard(
                band, chain, group_entries, output_dir, detector_filter, max_points
            )
            plot_count += _plot_data_quality_dashboard(
                band, chain, group_entries, output_dir, detector_filter, max_points
            )
            plot_count += _plot_ncorr_solver_dashboard(
                band, chain, group_entries, output_dir, detector_filter, max_points
            )
    # These figures produce one file per iteration, so --iter limits their output volume.
    if include_per_iteration and include_individual_detectors:
        for (band, chain), group_entries in _group_band_entries(per_iteration_entries).items():
            plot_count += _plot_noise_parameter_density(
                band, chain, group_entries, output_dir, detector_filter
            )
            plot_count += _plot_tod_power_spectra(
                band, chain, group_entries, output_dir, detector_filter
            )
    return plot_count


def _group_compsep_entries(entries: list[ChainFile]) -> dict[int, list[ChainFile]]:
    groups: dict[int, list[ChainFile]] = {}
    for entry in entries:
        groups.setdefault(entry.chain, []).append(entry)
    for group_entries in groups.values():
        group_entries.sort(key=lambda item: item.iteration)
    return groups


def _small_dataset_paths(
    entries: list[ChainFile],
    prefix: str,
    *,
    maximum_size: int = 32,
) -> set[str]:
    result: set[str] = set()
    for entry in entries:
        try:
            with h5py.File(entry.path, "r") as handle:
                if prefix not in handle:
                    continue

                def visitor(name: str, obj) -> None:
                    full_name = f"{prefix.rstrip('/')}/{name}" if name else prefix.rstrip("/")
                    if isinstance(obj, h5py.Dataset) and obj.size <= maximum_size:
                        result.add(full_name)

                handle[prefix].visititems(visitor)
        except OSError:
            continue
    return result


def _plot_dataset_trace(
    entries: list[ChainFile],
    dataset_path: str,
    output_folder: str,
    title_prefix: str,
    filename_prefix: str = "",
) -> int:
    records = []
    for entry in entries:
        try:
            with h5py.File(entry.path, "r") as handle:
                if dataset_path not in handle:
                    continue
                values = np.asarray(handle[dataset_path][()])
                if not np.issubdtype(values.dtype, np.number):
                    continue
                records.append((entry.iteration, values.reshape(-1)))
        except OSError:
            continue
    if not records:
        return 0
    value_count = max(values.size for _, values in records)
    plot_count = 0
    for value_index in range(value_count):
        x_values = []
        y_values = []
        for iteration, values in records:
            if value_index < values.size:
                x_values.append(iteration)
                y_values.append(values[value_index])
        suffix = "" if value_count == 1 else f"_value{value_index}"
        filename = os.path.join(
            output_folder,
            f"{filename_prefix}{_safe_name(dataset_path)}{suffix}.png",
        )
        value_label = "" if value_count == 1 else f" value {value_index}"
        plotting.plot_chain_lines(
            filename,
            f"{title_prefix}: {dataset_path}{value_label}",
            "Gibbs iteration",
            dataset_path,
            [("", np.asarray(x_values), np.asarray(y_values))],
            horizontal_lines=(1.0,) if dataset_path.endswith("/reduced") else (),
        )
        plot_count += int(os.path.isfile(filename))
    return plot_count


def _plot_compsep_diagnostics(
    entries: list[ChainFile],
    per_iteration_entries: list[ChainFile],
    output_dir: str,
    band_filter: set[str] | None,
) -> int:
    plot_count = 0
    # Scalar traces contain every iteration and deliberately ignore --iter.
    for chain, group_entries in _group_compsep_entries(entries).items():
        scalar_paths = {"chi2/reduced", "chi2/z"}
        scalar_paths.update(_small_dataset_paths(group_entries, "mcmc"))
        for dataset_path in sorted(scalar_paths):
            category = dataset_path.split("/", 1)[0]
            trace_folder = _category_folder(
                output_dir, "compsep_traces", category
            )
            plot_count += _plot_dataset_trace(
                group_entries,
                dataset_path,
                trace_folder,
                f"Compsep chain {chain}",
                filename_prefix=f"chain{chain:02d}_",
            )

        views = set()
        for entry in group_entries:
            try:
                with h5py.File(entry.path, "r") as handle:
                    if "chi2/bands" in handle:
                        views.update(handle["chi2/bands"].keys())
            except OSError:
                continue
        for polarization in ("I", "QU"):
            series = []
            for view_name in sorted(views):
                if not view_name.endswith(f"_{polarization}"):
                    continue
                if not _compsep_band_is_selected(view_name, band_filter):
                    continue
                iterations = []
                reduced_chi2 = []
                path = f"chi2/bands/{view_name}/reduced"
                for entry in group_entries:
                    try:
                        with h5py.File(entry.path, "r") as handle:
                            if path in handle:
                                iterations.append(entry.iteration)
                                reduced_chi2.append(float(handle[path][()]))
                    except OSError:
                        continue
                series.append((view_name.removesuffix(f"_{polarization}"),
                               np.asarray(iterations), np.asarray(reduced_chi2)))
            chi2_folder = _category_folder(output_dir, "compsep_traces", "chi2")
            filename = os.path.join(
                chi2_folder,
                f"chain{chain:02d}_per_band_reduced_{polarization}.png",
            )
            plotting.plot_chain_lines(
                filename,
                f"Per-band reduced chi-squared {polarization}; chain {chain}",
                "Gibbs iteration",
                "reduced chi-squared",
                series,
                horizontal_lines=(1.0,),
            )
            plot_count += int(os.path.isfile(filename))

    # Every CG history is its own file, so --iter applies here.
    for chain, group_entries in _group_compsep_entries(per_iteration_entries).items():
        for entry in group_entries:
            try:
                with h5py.File(entry.path, "r") as handle:
                    if "amplitude_groups" not in handle:
                        continue
                    residual_paths = []

                    def visitor(name: str, obj) -> None:
                        if isinstance(obj, h5py.Dataset) and name.endswith("cg_residuals"):
                            residual_paths.append(f"amplitude_groups/{name}")

                    handle["amplitude_groups"].visititems(visitor)
                    series_by_group = {}
                    for residual_path in residual_paths:
                        residual = np.asarray(handle[residual_path][()]).reshape(-1)
                        x_values = np.arange(1, residual.size + 1)
                        name = residual_path.removeprefix("amplitude_groups/")
                        name_parts = name.split("/")
                        group_name, polarization = name_parts[:2]
                        niter_path = residual_path.rsplit("/", 1)[0] + "/n_iter"
                        if niter_path in handle:
                            niter = int(handle[niter_path][()])
                        else:
                            niter = residual.size
                        label = f"{polarization} (n={niter})"
                        series_by_group.setdefault(group_name, []).append(
                            (label, x_values, residual)
                        )
                    for group_name, series in series_by_group.items():
                        cg_folder = _category_folder(output_dir, "compsep_cg", group_name)
                        filename = os.path.join(
                            cg_folder,
                            f"chain{chain:02d}_iter{entry.iteration:04d}.png",
                        )
                        plotting.plot_chain_lines(
                            filename,
                            f"Compsep CG; {group_name}, chain {chain}, "
                            f"iteration {entry.iteration}",
                            "CG iteration",
                            "relative residual",
                            series,
                            yscale="log",
                        )
                        plot_count += int(os.path.isfile(filename))
            except OSError:
                continue
    return plot_count


def _plot_compsep_maps(
    entries: list[ChainFile],
    output_dir: str,
    nside_out: int | None,
    band_filter: set[str] | None,
) -> int:
    plot_count = 0
    for entry in entries:
        try:
            with h5py.File(entry.path, "r") as handle:
                map_groups: list[tuple[str, np.ndarray, str, str]] = []
                if "chi2/map" in handle:
                    chi2_map = _resample_map(handle["chi2/map"][()], nside_out, "count")
                    map_groups.append(("chi2", chi2_map, "Compsep chi-squared", "$z^2$"))
                if "residuals" in handle:
                    for view_name in handle["residuals"]:
                        if not _compsep_band_is_selected(view_name, band_filter):
                            continue
                        residual_map = _resample_map(
                            handle[f"residuals/{view_name}"][()], nside_out, "brightness"
                        )
                        map_groups.append((
                            f"residual_{view_name}", residual_map,
                            f"Compsep residual {view_name}", "uK_RJ",
                        ))
                for map_name, map_rows, map_title, unit in map_groups:
                    category = "chi2" if map_name == "chi2" else "residuals"
                    out_folder = _category_folder(output_dir, "maps_compsep", category)
                    row_labels = _map_row_labels("map", map_rows.shape[0])
                    filename = os.path.join(
                        out_folder,
                        f"chain{entry.chain:02d}_{_safe_name(map_name)}_"
                        f"iter{entry.iteration:04d}.png",
                    )
                    plotting.plot_chain_map_grid(
                        filename,
                        f"{map_title}; chain {entry.chain}, iteration {entry.iteration}",
                        map_rows,
                        row_labels,
                        unit=unit,
                        symmetric=[map_name != "chi2"] * len(row_labels),
                    )
                    plot_count += int(os.path.isfile(filename))
        except OSError as error:
            LOGGER.warning("Could not read compsep maps from %s: %s", entry.path, error)
    return plot_count


def _component_map_nside(lmax: int, nside_out: int | None) -> int:
    """Grid to render a component's alms on: `--nside` if given, else the smallest nside that
    carries `lmax`. Components live in alm space, so this is purely a display choice."""
    if nside_out is not None:
        return nside_out
    minimum = max(1, int(np.ceil((lmax + 1) / 3)))
    return 2 ** int(np.ceil(np.log2(minimum)))


def _alms_to_maps(alms: np.ndarray, lmax: int, nside: int) -> tuple[np.ndarray, list[str]]:
    alms = np.asarray(alms, dtype=np.complex128)
    lmax_out = min(lmax, 3 * nside - 1)
    if lmax_out < lmax:
        resized = []
        for alm in alms:
            resized.append(hp.resize_alm(alm, lmax, lmax, lmax_out, lmax_out))
        alms = np.asarray(resized)
        lmax = lmax_out
    if alms.shape[0] == 1:
        component_maps = hp.alm2map(alms[0], nside, lmax=lmax, pol=False)
        return component_maps.reshape(1, -1), ["I"]
    if alms.shape[0] == 2:
        zero_temperature = np.zeros(alms.shape[1], dtype=np.complex128)
        component_maps = hp.alm2map([zero_temperature, alms[0], alms[1]], nside, lmax=lmax)
        return np.asarray(component_maps[1:3]), ["Q", "U"]
    component_maps = hp.alm2map(alms[:3], nside, lmax=lmax)
    return np.asarray(component_maps), ["I", "Q", "U"]


def _zero_monopole_and_dipole(alms: np.ndarray, lmax: int) -> np.ndarray:
    """Copy alms and remove l=0,1 so a sky cut cannot leak them into higher multipoles."""
    cleaned = np.array(alms, dtype=np.complex128, copy=True)
    for ell in range(min(1, lmax) + 1):
        for m in range(ell + 1):
            cleaned[:, hp.Alm.getidx(lmax, ell, m)] = 0.0
    return cleaned


def _galactic_cut_spectra(
    alms: np.ndarray,
    lmax: int,
    nside: int,
    cut_degrees: float,
) -> dict[str, np.ndarray]:
    """Estimate CMB pseudo-spectra outside a symmetric Galactic latitude cut."""
    cleaned_alms = _zero_monopole_and_dipole(alms, lmax)
    component_maps, row_labels = _alms_to_maps(cleaned_alms, lmax, nside)
    lmax_out = min(lmax, 3 * nside - 1)
    theta = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))[0]
    latitude = 90.0 - np.degrees(theta)
    fsky = np.mean(np.abs(latitude) >= cut_degrees)
    if fsky == 0.0:
        return {}

    if row_labels == ["I"]:
        spectrum = hp.anafast(
            component_maps[0], lmax=lmax_out, iter=0, pol=False, gal_cut=cut_degrees
        )
        return {"T": spectrum / fsky}

    if row_labels == ["Q", "U"]:
        temperature = np.zeros(component_maps.shape[1])
        component_maps = np.vstack([temperature, component_maps])
    spectra = hp.anafast(
        component_maps, lmax=lmax_out, iter=0, pol=True, gal_cut=cut_degrees
    )
    return {"T": spectra[0] / fsky, "E": spectra[1] / fsky, "B": spectra[2] / fsky}


def _component_names(entries: list[ChainFile]) -> set[str]:
    result: set[str] = set()
    for entry in entries:
        try:
            with h5py.File(entry.path, "r") as handle:
                if "comps" in handle:
                    result.update(handle["comps"].keys())
        except OSError:
            continue
    return result


def _plot_component_maps(
    entries: list[ChainFile],
    output_dir: str,
    nside_out: int | None,
) -> int:
    plot_count = 0
    for entry in entries:
        try:
            with h5py.File(entry.path, "r") as handle:
                if "comps" not in handle:
                    continue
                for component_name in handle["comps"]:
                    out_folder = _category_folder(
                        output_dir, "maps_components", component_name
                    )
                    component_path = f"comps/{component_name}"
                    if f"{component_path}/alms" not in handle:
                        continue
                    alms = np.asarray(handle[f"{component_path}/alms"][()])
                    if f"{component_path}/lmax" in handle:
                        lmax = int(handle[f"{component_path}/lmax"][()])
                    else:
                        lmax = hp.Alm.getlmax(alms.shape[-1])
                    nside = _component_map_nside(lmax, nside_out)
                    component_maps, row_labels = _alms_to_maps(alms, lmax, nside)
                    beam = 0.0
                    if f"{component_path}/amp_fwhm_arcmin" in handle:
                        beam = float(handle[f"{component_path}/amp_fwhm_arcmin"][()])
                    nu_ref = np.array([])
                    if f"{component_path}/sed/nu_ref" in handle:
                        nu_ref = np.asarray(handle[f"{component_path}/sed/nu_ref"][()]).reshape(-1)
                    plot_labels = []
                    for row_index, row_label in enumerate(row_labels):
                        plot_label = row_label
                        if nu_ref.size:
                            ref_index = (
                                0 if nu_ref.size == 1 or row_index == 0
                                else min(1, nu_ref.size - 1)
                            )
                            plot_label += f", nu_ref {float(nu_ref[ref_index]):g} GHz"
                        plot_labels.append(plot_label)
                    title = (
                        f"{component_name} amplitude; chain {entry.chain}, "
                        f"iteration {entry.iteration}"
                    )
                    if beam > 0:
                        title += f", FWHM {beam:g} arcmin"
                    filename = os.path.join(
                        out_folder,
                        f"chain{entry.chain:02d}_iter{entry.iteration:04d}.png",
                    )
                    plotting.plot_chain_map_grid(
                        filename,
                        title,
                        component_maps,
                        plot_labels,
                        unit="uK_RJ",
                        symmetric=[True] * len(plot_labels),
                    )
                    plot_count += int(os.path.isfile(filename))
        except OSError as error:
            LOGGER.warning("Could not read components from %s: %s", entry.path, error)
    return plot_count


def _plot_component_spectra(entries: list[ChainFile], output_dir: str) -> int:
    plot_count = 0
    for chain, group_entries in _group_compsep_entries(entries).items():
        for component_name in sorted(_component_names(group_entries)):
            out_folder = _category_folder(
                output_dir, "spectra_components", component_name, "full_sky"
            )
            records = []
            for entry in group_entries:
                try:
                    with h5py.File(entry.path, "r") as handle:
                        path = f"comps/{component_name}/sigma_l"
                        if path in handle:
                            records.append((entry.iteration, np.asarray(handle[path][()])))
                except OSError:
                    continue
            if not records:
                continue
            row_count = max(values.shape[0] for _, values in records)
            row_labels = (["T"] if row_count == 1 else ["E", "B"] if row_count == 2
                          else ["T", "E", "B"])
            panels = []
            for row_index, row_label in enumerate(row_labels):
                series = []
                for iteration, values in records:
                    if row_index >= values.shape[0]:
                        continue
                    multipoles = np.arange(values.shape[1])
                    series.append((f"iter {iteration}", multipoles[2:], values[row_index, 2:]))
                panels.append(plotting.ChainLinePanel(
                    title=row_label,
                    ylabel="$C_l$ [uK_RJ$^2$]",
                    series=series,
                    xscale="log",
                    yscale="log",
                ))
            filename = os.path.join(out_folder, f"chain{chain:02d}_Cl.png")
            plotting.plot_chain_line_panels(
                filename,
                f"{component_name} realized spectra; chain {chain}",
                "multipole l",
                panels,
            )
            plot_count += int(os.path.isfile(filename))
    return plot_count


def _plot_cmb_galactic_cut_spectra(
    entries: list[ChainFile],
    output_dir: str,
    nside_out: int | None,
    cut_degrees: float,
) -> int:
    """Plot an additional CMB-only spectrum outside a built-in Galactic latitude cut."""
    if cut_degrees <= 0.0:
        return 0
    plot_count = 0
    for chain, group_entries in _group_compsep_entries(entries).items():
        records: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {}
        for entry in group_entries:
            try:
                with h5py.File(entry.path, "r") as handle:
                    if "comps" not in handle:
                        continue
                    for component_name in handle["comps"]:
                        component_path = f"comps/{component_name}"
                        comp_name = component_name
                        if f"{component_path}/comp_name" in handle:
                            comp_name = str(_decode_h5_value(
                                handle[f"{component_path}/comp_name"][()]
                            ))
                        if comp_name.lower() != "cmb" and component_name.lower() != "cmb":
                            continue
                        if f"{component_path}/alms" not in handle:
                            continue
                        alms = np.asarray(handle[f"{component_path}/alms"][()])
                        if f"{component_path}/lmax" in handle:
                            lmax = int(handle[f"{component_path}/lmax"][()])
                        else:
                            lmax = hp.Alm.getlmax(alms.shape[-1])
                        nside = _component_map_nside(lmax, nside_out)
                        spectra = _galactic_cut_spectra(
                            alms, lmax, nside, cut_degrees
                        )
                        for row_label, spectrum in spectra.items():
                            records.setdefault((component_name, row_label), []).append(
                                (entry.iteration, spectrum)
                            )
            except OSError as error:
                LOGGER.warning("Could not read CMB alms from %s: %s", entry.path, error)

        component_names = sorted({component_name for component_name, _ in records})
        for component_name in component_names:
            panels = []
            for row_label in ("T", "E", "B"):
                row_records = records.get((component_name, row_label), [])
                if not row_records:
                    continue
                series = []
                for iteration, spectrum in row_records:
                    multipoles = np.arange(spectrum.size)
                    series.append((f"iter {iteration}", multipoles[2:], spectrum[2:]))
                panels.append(plotting.ChainLinePanel(
                    title=row_label,
                    ylabel="$C_l/f_{sky}$ [uK_RJ$^2$]",
                    series=series,
                    xscale="log",
                    yscale="log",
                ))
            out_folder = _category_folder(
                output_dir, "spectra_components", component_name, "galactic_cut"
            )
            filename = os.path.join(out_folder, f"chain{chain:02d}_Cl.png")
            plotting.plot_chain_line_panels(
                filename,
                f"{component_name} pseudo-spectra, "
                f"|b| >= {cut_degrees:g} deg; chain {chain}",
                "multipole l",
                panels,
            )
            plot_count += int(os.path.isfile(filename))
    return plot_count


def _plot_source_amplitudes(entries: list[ChainFile], output_dir: str) -> int:
    plot_count = 0
    for chain, group_entries in _group_compsep_entries(entries).items():
        for component_name in sorted(_component_names(group_entries)):
            out_folder = _category_folder(
                output_dir, "source_amplitudes", component_name
            )
            series = []
            for entry in group_entries:
                try:
                    with h5py.File(entry.path, "r") as handle:
                        path = f"comps/{component_name}/source_amps"
                        if path not in handle:
                            continue
                        amplitudes = np.asarray(handle[path][()]).reshape(-1)
                        series.append((
                            f"iter {entry.iteration}", np.arange(amplitudes.size), amplitudes
                        ))
                except OSError:
                    continue
            if not series:
                continue
            filename = os.path.join(
                out_folder, f"chain{chain:02d}.png"
            )
            plotting.plot_chain_lines(
                filename,
                f"{component_name} source amplitudes; chain {chain}",
                "source index",
                "amplitude [uK_RJ]",
                series,
            )
            plot_count += int(os.path.isfile(filename))
    return plot_count


def _dataset_varies(entries: list[ChainFile], dataset_path: str) -> bool:
    """Whether a small numeric chain dataset changes between Gibbs iterations."""
    values = []
    for entry in entries:
        try:
            with h5py.File(entry.path, "r") as handle:
                if dataset_path in handle:
                    value = np.asarray(handle[dataset_path][()])
                    if np.issubdtype(value.dtype, np.number):
                        values.append(value)
        except OSError:
            continue
    if len(values) < 2:
        return False
    first = values[0]
    for value in values[1:]:
        if not np.array_equal(first, value, equal_nan=True):
            return True
    return False


def _plot_component_traces(
    entries: list[ChainFile],
    output_dir: str,
    band_filter: set[str] | None,
) -> int:
    plot_count = 0
    for chain, group_entries in _group_compsep_entries(entries).items():
        for component_name in sorted(_component_names(group_entries)):
            component_prefix = f"comps/{component_name}"
            dataset_paths = set()
            for subgroup in ("sed", "Cl_prior", "mixing"):
                dataset_paths |= _small_dataset_paths(
                    group_entries, f"{component_prefix}/{subgroup}"
                )
            for dataset_path in sorted(dataset_paths):
                if not _dataset_varies(group_entries, dataset_path):
                    continue
                if "/mixing/" in dataset_path:
                    band_name = dataset_path.rsplit("/", 1)[-1]
                    if not _compsep_band_is_selected(band_name, band_filter):
                        continue
                subgroup = dataset_path.split("/")[2]
                out_folder = _category_folder(
                    output_dir,
                    "component_traces",
                    component_name,
                    subgroup,
                )
                plot_count += _plot_dataset_trace(
                    group_entries,
                    dataset_path,
                    out_folder,
                    f"Component {component_name}, chain {chain}",
                    filename_prefix=f"chain{chain:02d}_",
                )
    return plot_count


def _configure_logging(verbose: bool) -> None:
    LOGGER.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[plot_chain] %(levelname)s: %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOGGER.propagate = False
    for noisy_name in ("matplotlib", "healpy", "h5py", "PIL"):
        logging.getLogger(noisy_name).setLevel(logging.WARNING)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make single-panel plots from current-format Commander4 chain outputs."
    )
    parser.add_argument(
        "run_dir",
        help=f"Run output directory containing {paths.CHAINS_BANDS}/ and/or "
        f"{paths.CHAINS_COMPSEP}/.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Plot directory. Defaults to <run_dir>/{paths.PLOTS}.",
    )
    parser.add_argument(
        "--plots",
        default="all",
        help="Comma-separated plot groups: maps, tod, compsep, components, or all.",
    )
    parser.add_argument("--map-types", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--chain",
        default="all",
        help="Chain numbers or inclusive ranges, for example 1,2 or 1-3.",
    )
    parser.add_argument(
        "--iter",
        default="all",
        help=(
            "Iterations for plots that produce one file per iteration, for example 1,5 or "
            "10-20. Multi-iteration traces and spectra always use every iteration."
        ),
    )
    parser.add_argument(
        "--band",
        "--pixel",
        default="all",
        help="Band names to include. --pixel is retained as an alias for older commands.",
    )
    parser.add_argument(
        "--detector",
        default="all",
        help="Detector names included in individual plots or detector summaries.",
    )
    parser.add_argument(
        "--detector-plots",
        choices=sorted(DETECTOR_PLOT_MODES),
        default="individual",
        help=(
            "Detector output mode: individual (current plots), summary (band-level population "
            "summaries), both, or none (default: individual)."
        ),
    )
    parser.add_argument(
        "--nside",
        default="native",
        help="HEALPix nside for maps, or 'native'.",
    )
    parser.add_argument(
        "--cmb-galactic-cut-deg",
        type=float,
        default=20.0,
        help=(
            "Latitude cut for the additional CMB pseudo-spectrum (default: 20 degrees); "
            "use 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50_000,
        help="Maximum scan samples per panel, shared across iteration lines (default: 50000).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        LOGGER.error("Run output directory not found: %s", run_dir)
        return 1
    try:
        chain_filter = _parse_int_set(args.chain)
        iteration_filter = _parse_int_set(args.iter)
        band_filter = _parse_name_set(args.band)
        detector_filter = _parse_name_set(args.detector)
        plot_types = _parse_plot_types(args.plots)
        nside_out = None if args.nside.lower() == "native" else int(args.nside)
    except ValueError as error:
        parser.error(str(error))
    if nside_out is not None and not hp.isnsideok(nside_out):
        parser.error(f"Invalid HEALPix nside: {nside_out}")
    if args.max_points < 2:
        parser.error("--max-points must be at least 2.")
    if not 0.0 <= args.cmb_galactic_cut_deg < 90.0:
        parser.error("--cmb-galactic-cut-deg must be at least 0 and less than 90.")

    if args.map_types is not None:
        legacy = {item.strip().lower() for item in args.map_types.split(",") if item.strip()}
        if not legacy or "all" in legacy:
            plot_types = set(PLOT_TYPES)
        else:
            plot_types = set()
            if "data" in legacy:
                plot_types.add("maps")
            if "components" in legacy:
                plot_types.add("components")
            if "combo" in legacy:
                LOGGER.warning("Combination maps are outside this plotting pass and were skipped.")
        LOGGER.warning("--map-types is deprecated; use --plots instead.")

    output_dir = os.path.abspath(args.output_dir or os.path.join(run_dir, paths.PLOTS))
    params = _load_params_from_chain(run_dir)
    if params is None:
        LOGGER.warning("No embedded parameter file found; using chain metadata only.")

    all_band_entries = _discover_band_files(
        run_dir, params, chain_filter, None, band_filter
    )
    all_compsep_entries = _discover_compsep_files(run_dir, chain_filter, None)
    band_entries = [
        entry for entry in all_band_entries
        if iteration_filter is None or entry.iteration in iteration_filter
    ]
    compsep_entries = [
        entry for entry in all_compsep_entries
        if iteration_filter is None or entry.iteration in iteration_filter
    ]
    LOGGER.info(
        "Found %d band and %d compsep files; selected %d and %d for per-iteration plots "
        "(plots: %s).",
        len(all_band_entries),
        len(all_compsep_entries),
        len(band_entries),
        len(compsep_entries),
        ", ".join(sorted(plot_types)) or "none",
    )

    started = time.time()
    plot_count = 0
    individual_detectors = args.detector_plots in {"individual", "both"}
    summarize_detectors = args.detector_plots in {"summary", "both"}

    # Quick-look phase: small datasets and already-computed spectra, with no map transforms.
    LOGGER.info("Starting fast plotting phase.")
    if "tod" in plot_types:
        plot_count += _plot_tod_outputs(
            all_band_entries,
            band_entries,
            output_dir,
            detector_filter,
            args.max_points,
            include_scalar_traces=True,
            include_detector_dashboards=False,
            include_per_iteration=False,
            include_individual_detectors=individual_detectors,
        )
    if "compsep" in plot_types:
        plot_count += _plot_compsep_diagnostics(
            all_compsep_entries, compsep_entries, output_dir, band_filter
        )
    if "components" in plot_types:
        plot_count += _plot_component_spectra(all_compsep_entries, output_dir)
        plot_count += _plot_source_amplitudes(all_compsep_entries, output_dir)
        plot_count += _plot_component_traces(
            all_compsep_entries, output_dir, band_filter
        )

    if "tod" in plot_types and summarize_detectors:
        LOGGER.info("Writing detector-population summaries.")
        plot_count += _plot_detector_summaries(
            all_band_entries, band_entries, output_dir, detector_filter
        )

    # Detailed detector dashboards combine all iterations but avoid per-iteration plot explosion.
    if "tod" in plot_types and individual_detectors:
        LOGGER.info("Writing individual detector dashboards.")
        plot_count += _plot_tod_outputs(
            all_band_entries,
            band_entries,
            output_dir,
            detector_filter,
            args.max_points,
            include_scalar_traces=False,
            include_detector_dashboards=True,
            include_per_iteration=False,
            include_individual_detectors=True,
        )

    # Expensive phase: full maps, per-iteration detector figures, and spherical transforms.
    LOGGER.info("Starting map and per-iteration plotting phase.")
    if "maps" in plot_types:
        plot_count += _plot_band_maps(band_entries, output_dir, nside_out)
        plot_count += _plot_compsep_maps(compsep_entries, output_dir, nside_out, band_filter)
    if "tod" in plot_types and individual_detectors:
        plot_count += _plot_tod_outputs(
            all_band_entries,
            band_entries,
            output_dir,
            detector_filter,
            args.max_points,
            include_scalar_traces=False,
            include_detector_dashboards=False,
            include_per_iteration=True,
            include_individual_detectors=True,
        )
    if "tod" in plot_types and summarize_detectors:
        plot_count += _plot_tod_power_spectra_summary(
            band_entries, output_dir, detector_filter
        )
    if "components" in plot_types:
        plot_count += _plot_component_maps(compsep_entries, output_dir, nside_out)
        plot_count += _plot_cmb_galactic_cut_spectra(
            all_compsep_entries, output_dir, nside_out, args.cmb_galactic_cut_deg
        )

    LOGGER.info(
        "Wrote %d plots to %s in %.1fs.", plot_count, output_dir, time.time() - started
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
