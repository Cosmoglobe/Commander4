# This program can be run as `c4-plot-chain <path-to-chain>`, as long as Commander4 installed.

import argparse
import glob
import logging
import os
import re
from copy import deepcopy
from dataclasses import dataclass
import time
import h5py
import healpy as hp
import numpy as np
import yaml
from pixell.bunch import Bunch

import commander4.sky_models.component as component_lib
from commander4.output import plotting
from commander4.sky_models.component import Component


CHAIN_ITER_RE = re.compile(r"chain(?P<chain>\d+)_iter(?P<iter>\d+)\.h5$")
LOGGER = logging.getLogger("plot_chain")


def as_bunch_recursive(dict_of_dicts, name=None):
    res = Bunch()
    
    # 1. Inject the name into the instance, bypassing Bunch's data _dict
    if name is not None:
         object.__setattr__(res, "_name", name)
         
    # 2. Recursively populate the bunch
    for key, val in dict_of_dicts.items():
        if isinstance(val, dict):
            # Pass the key down as the name for the child Bunch
            res[key] = as_bunch_recursive(val, name=key)
        else:
            res[key] = val

    return res


# TODO: Figure out a way of not duplicating this. We can't import it, because that triggers the
# reading of the parameter file, requiring it to be provided as a command-line argument.
def as_bunch_recursive(dict_of_dicts, name=None):
    res = Bunch()
    
    # 1. Inject the name into the instance, bypassing Bunch's data _dict
    if name is not None:
         object.__setattr__(res, "_name", name)
         
    # 2. Recursively populate the bunch
    for key, val in dict_of_dicts.items():
        if isinstance(val, dict):
            # Pass the key down as the name for the child Bunch
            res[key] = as_bunch_recursive(val, name=key)
        else:
            res[key] = val

    return res


def _decode_h5_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_h5_value(value.item())
    return value


def _load_params_from_chain(chain_dir: str) -> Bunch | None:
    patterns = [
        os.path.join(chain_dir, "datamaps", "*.h5"),
        os.path.join(chain_dir, "compsep", "*.h5"),
        os.path.join(chain_dir, "tod", "*.h5"),
        os.path.join(chain_dir, "*.h5"),
    ]
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            try:
                with h5py.File(path, "r") as handle:
                    if "metadata/parameter_file_as_string" in handle:
                        raw_yaml = _decode_h5_value(handle["metadata/parameter_file_as_string"][()])
                    else:
                        continue
            except OSError:
                continue

            if not raw_yaml:
                continue
            params_dict = yaml.safe_load(raw_yaml)
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
    comp_list: list[Component] = []
    for component_str in params.components:
        component = params.components[component_str]
        if not component.enabled:
            continue
        comp_shortname = component.params.shortname
        comp_longname = component.params.longname
        base_params = deepcopy(component.params)
        if base_params.lmax == "full":
            base_params.lmax = (params.general.nside * 5) // 2

        if "I" in base_params.polarization:
            params_i = deepcopy(base_params)
            params_i.longname = comp_longname + "_Intensity"
            params_i.shortname = comp_longname + "_I"
            params_i.polarization = "I"
            params_i.polarized = False
            comp_type = getattr(component_lib, component.component_class)
            comp_list.append(comp_type(params_i, params.general))
        if "QU" in base_params.polarization:
            params_qu = deepcopy(base_params)
            params_qu.longname = comp_longname + "_Polarization"
            params_qu.shortname = comp_longname + "_QU"
            params_qu.polarization = "QU"
            params_qu.polarized = True
            comp_type = getattr(component_lib, component.component_class)
            comp_list.append(comp_type(params_qu, params.general))

    return comp_list


def _load_compsep_components(params: Bunch, compsep_path: str) -> list[Component]:
    comp_list = _build_component_list(params)
    if not comp_list:
        return []

    with h5py.File(compsep_path, "r") as handle:
        if "comps" not in handle:
            return []
        comps_group = handle["comps"]
        available = set(comps_group.keys())
        filtered_list: list[Component] = []
        for comp in comp_list:
            if comp.shortname not in available:
                continue
            comp_group = comps_group[comp.shortname]
            if "alms" not in comp_group:
                continue
            comp.alms = comp_group["alms"][()]
            if "longname" in comp_group:
                comp.longname = _decode_h5_value(comp_group["longname"][()])
            filtered_list.append(comp)
        return filtered_list


def _match_band_info(
    filename: str,
    params: Bunch | None,
) -> tuple[str | None, str | None, Bunch | None]:
    if params is None or "experiments" not in params:
        return None, None, None
    for exp_name in params.experiments:
        experiment = params.experiments[exp_name]
        for band_name in experiment.bands:
            prefix = f"{exp_name}_{band_name}_"
            band_prefix = f"{band_name}_"
            if filename.startswith(prefix) or f"_{exp_name}_{band_name}_" in filename:
                return exp_name, band_name, experiment.bands[band_name]
            if filename.startswith(band_prefix) or f"_{band_name}_" in filename:
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
            start_str, end_str = chunk.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            for item in range(min(start, end), max(start, end) + 1):
                result.add(item)
        else:
            result.add(int(chunk))
    return result if result else None


def _parse_map_types(value: str) -> set[str]:
    allowed = {"data", "combo", "components"}
    selected = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not selected or "all" in selected:
        return allowed
    valid = selected & allowed
    unknown = sorted(selected - allowed)
    if unknown:
        LOGGER.warning("Ignoring unknown --map-types entries: %s", ", ".join(unknown))
    return valid if valid else allowed


def _parse_name_set(value: str | None) -> set[str] | None:
    if value is None or value.lower() in {"all", "*"}:
        return None
    selected = {item.strip() for item in value.split(",") if item.strip()}
    return selected if selected else None


def _matches_pixel_filter(
    pixel_filter: set[str] | None,
    filename: str,
    exp_name: str | None,
    band_name: str | None,
) -> bool:
    if pixel_filter is None:
        return True

    stem = filename.rsplit(".h5", 1)[0]
    stem = stem.rsplit("_chain", 1)[0]
    candidates = {stem}
    if band_name:
        candidates.add(band_name)
    if exp_name and band_name:
        candidates.add(f"{exp_name}_{band_name}")

    return any(candidate in pixel_filter for candidate in candidates)


def _ensure_2d_map(map_data: np.ndarray | None) -> np.ndarray | None:
    if map_data is None:
        return None
    map_array = np.asarray(map_data)
    if map_array.ndim == 1:
        return map_array.reshape(1, -1)
    return map_array


def _align_map_rows(map_data: np.ndarray | None, target_rows: int, npix: int) -> np.ndarray | None:
    map_array = _ensure_2d_map(map_data)
    if map_array is None:
        return None
    if map_array.shape[0] == target_rows:
        return map_array
    if map_array.shape[0] > target_rows:
        return map_array[:target_rows]
    pad = np.zeros((target_rows - map_array.shape[0], npix), dtype=map_array.dtype)
    return np.vstack([map_array, pad])


def _ud_grade_map(map_data: np.ndarray | None, nside_out: int | None) -> np.ndarray | None:
    map_array = _ensure_2d_map(map_data)
    if map_array is None or nside_out is None:
        return map_array
    npix = map_array.shape[-1]
    nside_in = hp.npix2nside(npix)
    if nside_in == nside_out:
        return map_array
    return np.vstack([hp.ud_grade(row, nside_out) for row in map_array])


def _optional_slice(arr: np.ndarray | None, s: slice) -> np.ndarray | None:
    return arr[s] if arr is not None else None


@dataclass
class MapBundle:
    signal: np.ndarray
    rms: np.ndarray | None
    corrnoise: np.ndarray | None
    orbdipole: np.ndarray | None
    skymodel: np.ndarray | None
    residual: np.ndarray | None
    nside: int
    npol: int

    def pol_slice(self, s: slice) -> "MapBundle":
        return MapBundle(
            signal=self.signal[s],
            rms=_optional_slice(self.rms, s),
            corrnoise=_optional_slice(self.corrnoise, s),
            orbdipole=_optional_slice(self.orbdipole, s),
            skymodel=_optional_slice(self.skymodel, s),
            residual=_optional_slice(self.residual, s),
            nside=self.nside,
            npol=self.signal[s].shape[0],
        )


def _load_and_prepare_maps(map_path: str, nside_target: int | None) -> MapBundle | None:
    try:
        with h5py.File(map_path, "r") as f:
            raw = {k: f[k][()] if k in f else None for k in (
                "map_observed_sky", "map_rms", "map_orbdipole", "map_corrnoise", "map_skymodel",
            )}
    except OSError:
        return None

    signal = _ud_grade_map(raw["map_observed_sky"], nside_target)
    if signal is None:
        return None
    rms = _ud_grade_map(raw["map_rms"], nside_target)
    corrnoise = _ud_grade_map(raw["map_corrnoise"], nside_target)
    orbdipole = _ud_grade_map(raw["map_orbdipole"], nside_target)
    skymodel = _ud_grade_map(raw["map_skymodel"], nside_target)

    npix = signal.shape[-1]
    nside = hp.npix2nside(npix)
    npol = signal.shape[0]

    rms = _align_map_rows(rms, npol, npix)
    corrnoise = _align_map_rows(corrnoise, npol, npix)
    orbdipole = _align_map_rows(orbdipole, npol, npix)
    skymodel = _align_map_rows(skymodel, npol, npix)

    residual = None
    if skymodel is not None:
        residual = signal.copy()
        residual -= skymodel

    return MapBundle(
        signal=signal, rms=rms, corrnoise=corrnoise, orbdipole=orbdipole,
        skymodel=skymodel, residual=residual, nside=nside, npol=npol,
    )


def _first_scalar(dataset) -> float | None:
    raw = dataset[()]
    arr = np.asarray(raw)
    if arr.size == 0:
        return None
    value = arr.reshape(-1)[0]
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _plot_tod_sampling(chain_dir: str, output_dir: str, chain_filter: set[int] | None) -> None:
    tod_files = sorted(glob.glob(os.path.join(chain_dir, "tod", "*.h5")))
    if not tod_files:
        LOGGER.info("No tod/*.h5 files found; skipping TOD sampling plots.")
        return

    data: dict[str, dict[int, dict[int, dict[str, float]]]] = {}
    for path in tod_files:
        filename = os.path.basename(path)
        chain, iteration = _extract_chain_iter(filename)
        if chain is None or iteration is None:
            continue
        if chain_filter is not None and chain not in chain_filter:
            continue
        detector = filename.rsplit("_chain", 1)[0]
        try:
            with h5py.File(path, "r") as handle:
                for key in handle.keys():
                    if key in {"scanID", "metadata"}:
                        continue
                    scalar = _first_scalar(handle[key])
                    if scalar is None:
                        continue
                    data.setdefault(detector, {}).setdefault(chain, {}).setdefault(
                        iteration, {}
                    )[key] = scalar
        except OSError:
            continue

    out_folder = os.path.join(output_dir, "tod_sampling")
    os.makedirs(out_folder, exist_ok=True)
    combined_folder = os.path.join(out_folder, "combined")
    os.makedirs(combined_folder, exist_ok=True)

    t0 = time.time()
    for detector, det_chains in data.items():
        for chain, iter_map in det_chains.items():
            iterations = sorted(iter_map.keys())
            all_keys = sorted({k for values in iter_map.values() for k in values.keys()})
            for key in all_keys:
                x_vals, y_vals = [], []
                for iter_val in iterations:
                    if key not in iter_map[iter_val]:
                        continue
                    x_vals.append(iter_val)
                    y_vals.append(iter_map[iter_val][key])
                if x_vals:
                    plotting.plot_tod_series(out_folder, detector, chain, key, x_vals, y_vals)

    all_chains = sorted({chain for det_chains in data.values() for chain in det_chains.keys()})
    for chain in all_chains:
        detectors = sorted([det for det, det_chains in data.items() if chain in det_chains])
        if not detectors:
            continue
        all_keys = sorted(
            {k for det in detectors for vals in data[det][chain].values() for k in vals.keys()}
        )
        all_iterations = sorted({it for det in detectors for it in data[det][chain].keys()})
        for key in all_keys:
            series = {}
            for detector in detectors:
                iter_map = data[detector][chain]
                x_vals, y_vals = [], []
                for iter_val in all_iterations:
                    if iter_val in iter_map and key in iter_map[iter_val]:
                        x_vals.append(iter_val)
                        y_vals.append(iter_map[iter_val][key])
                series[detector] = (x_vals, y_vals)
            plotting.plot_tod_combined(combined_folder, chain, key, series)

    LOGGER.info("Finished plotting TOD sampling in %.1fs.", time.time() - t0)


def _plot_chain_maps(
    plot_params: Bunch,
    detector_base: str,
    chain: int,
    iteration: int,
    maps: MapBundle,
    nu: float,
    fwhm_arcmin: float,
    map_types: set[str],
    comp_list: list[Component] | None,
) -> None:
    LOGGER.debug(
        "Start plotting chain=%d iter=%d detector=%s (map_types=%s)",
        chain, iteration, detector_base, ",".join(sorted(map_types)),
    )

    if "data" in map_types:
        t0 = time.time()
        plotting.plot_data_maps(
            plot_params, detector_base, chain, iteration,
            map_signal=maps.signal, map_rms=maps.rms, map_corrnoise=maps.corrnoise,
            map_residual=maps.residual, map_orbdipole=maps.orbdipole,
        )
        LOGGER.debug(
            "Finished data plots chain=%d iter=%d detector=%s in %.1fs.",
            chain, iteration, detector_base, time.time() - t0,
        )

    if not comp_list or not ({"combo", "components"} & map_types):
        return

    if maps.npol == 1:
        pol_slices = [("I", slice(0, 1))]
    elif maps.npol == 2:
        pol_slices = [("QU", slice(0, 2))]
    else:
        pol_slices = [("I", slice(0, 1)), ("QU", slice(1, 3))]

    for pol_label, pol_slice in pol_slices:
        det_label = f"{detector_base}_{pol_label}"
        sub = maps.pol_slice(pol_slice)

        if "combo" in map_types:
            t0 = time.time()
            plotting.plot_combo_maps(
                plot_params, det_label, chain, iteration, comp_list,
                map_signal=sub.signal, map_rms=sub.rms, map_corrnoise=sub.corrnoise,
                map_orbdipole=sub.orbdipole, map_skymodel=sub.skymodel,
                nu=nu, nside=sub.nside, fwhm_arcmin=fwhm_arcmin,
            )
            LOGGER.debug(
                "Finished combo plots chain=%d iter=%d detector=%s in %.1fs.",
                chain, iteration, det_label, time.time() - t0,
            )
        if "components" in map_types:
            t0 = time.time()
            plotting.plot_components(
                plot_params, det_label, chain, iteration, comp_list,
                map_signal=sub.signal, nu=nu, nside=sub.nside, fwhm_arcmin=fwhm_arcmin,
            )
            LOGGER.debug(
                "Finished component plots chain=%d iter=%d detector=%s in %.1fs.",
                chain, iteration, det_label, time.time() - t0,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Commander4 chain outputs from disk.")
    parser.add_argument(
        "chain_dir",
        help="Path to a chain output directory (containing compsep/, datamaps/, tod/).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots. Defaults to <chain_dir>/plots.",
    )
    parser.add_argument(
        "--chain",
        default="all",
        help="Comma-separated chain numbers or ranges (e.g. 1,2 or 1-3). Use 'all' for everything.",
    )
    parser.add_argument(
        "--iter",
        default="all",
        help=(
            "Comma-separated iteration numbers or ranges (e.g. 1,2 or 1-10). "
            "Use 'all' for everything."
        ),
    )
    parser.add_argument(
        "--map-types",
        default="all",
        help="Comma-separated map types: data, combo, components, or all.",
    )
    parser.add_argument(
        "--nside",
        default="native",
        help="Target nside for plotting. Use 'native' to keep input resolution.",
    )
    parser.add_argument(
        "--pixel",
        default="all",
        help=(
            "Comma-separated detector/band names to plot (e.g. LiteBIRD100GHz_L4). "
            "Use 'all' for everything."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose printing/logging during run.",
    )
    args = parser.parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    LOGGER.handlers.clear()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("[plot_chain] %(levelname)s: %(message)s"))
    LOGGER.addHandler(stream_handler)
    LOGGER.setLevel(level)
    LOGGER.propagate = False

    for noisy_name in ("matplotlib", "healpy", "h5py", "PIL"):
        noisy_logger = logging.getLogger(noisy_name)
        noisy_logger.setLevel(logging.WARNING)
    chain_dir = os.path.abspath(args.chain_dir)
    if not os.path.isdir(chain_dir):
        LOGGER.error("Chain directory not found: %s", chain_dir)
        return 1

    output_dir = args.output_dir or os.path.join(chain_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    plot_params = Bunch(output_paths=Bunch(plots=output_dir))

    chain_filter = _parse_int_set(args.chain)
    iter_filter = _parse_int_set(args.iter)
    map_types = _parse_map_types(args.map_types)
    pixel_filter = _parse_name_set(args.pixel)

    nside_target = None if args.nside.lower() == "native" else int(args.nside)
    LOGGER.info("Plotting at nside: %s", "native" if nside_target is None else nside_target)

    params = _load_params_from_chain(chain_dir)
    if params is None:
        LOGGER.warning(
            "No parameter metadata found in chain outputs; component plots may be skipped."
        )

    compsep_files: dict[tuple[int, int], str] = {}
    for path in glob.glob(os.path.join(chain_dir, "compsep", "*.h5")):
        chain, iteration = _extract_chain_iter(os.path.basename(path))
        if chain is None or iteration is None:
            continue
        compsep_files[(chain, iteration)] = path

    maps_files = sorted(glob.glob(os.path.join(chain_dir, "datamaps", "*.h5")))
    if not maps_files:
        LOGGER.warning("No datamaps/*.h5 files found in %s", chain_dir)

    comp_cache: dict[tuple[int, int], list[Component]] = {}
    LOGGER.info("Found %s map files and %s compsep files", len(maps_files), len(compsep_files))

    _plot_tod_sampling(chain_dir, output_dir, chain_filter)

    selected_maps: list[tuple[str, int, int]] = []
    for map_path in maps_files:
        filename = os.path.basename(map_path)
        chain, iteration = _extract_chain_iter(filename)
        if chain is None or iteration is None:
            continue
        if chain_filter is not None and chain not in chain_filter:
            continue
        if iter_filter is not None and iteration not in iter_filter:
            continue
        exp_name, band_name, _ = _match_band_info(filename, params)
        if not _matches_pixel_filter(pixel_filter, filename, exp_name, band_name):
            continue
        selected_maps.append((map_path, chain, iteration))

    LOGGER.info(
        "Selected %d map files after filtering (chains=%s, iters=%s, pixels=%s).",
        len(selected_maps),
        "all" if chain_filter is None else sorted(chain_filter),
        "all" if iter_filter is None else sorted(iter_filter),
        "all" if pixel_filter is None else sorted(pixel_filter),
    )

    total_selected = len(selected_maps)
    t_maps = time.time()
    progress_stride = 25
    for idx, (map_path, chain, iteration) in enumerate(selected_maps, start=1):
        filename = os.path.basename(map_path)
        exp_name, band_name, band_info = _match_band_info(filename, params)
        nu = np.nan
        fwhm_arcmin = np.nan
        if band_info is not None:
            nu = getattr(band_info, "freq", None)
            if nu is None:
                nu = getattr(band_info, "nu", np.nan)
            fwhm_arcmin = getattr(band_info, "fwhm", np.nan)

        maps = _load_and_prepare_maps(map_path, nside_target)
        if maps is None:
            continue

        comp_list = None
        comp_key = (chain, iteration)
        if comp_key in compsep_files and params is not None and ({"combo", "components"} & map_types):
            comp_list = comp_cache.get(comp_key)
            if comp_list is None:
                comp_list = _load_compsep_components(params, compsep_files[comp_key])
                comp_cache[comp_key] = comp_list

        det_label_base = f"{exp_name}_{band_name}" if (exp_name and band_name) else (band_name or "det")
        LOGGER.debug(
            "Processing map %d/%d: chain=%d iter=%d detector=%s file=%s",
            idx, total_selected, chain, iteration, det_label_base, filename,
        )

        _plot_chain_maps(plot_params, det_label_base, chain, iteration, maps, nu, fwhm_arcmin, map_types, comp_list)

        if idx % progress_stride == 0 or idx == total_selected:
            elapsed = time.time() - t_maps
            LOGGER.info(
                "Progress %d/%d (%.1f%%), latest chain=%d iter=%d detector=%s, elapsed %.1fs",
                idx,
                total_selected,
                100.0 * idx / max(total_selected, 1),
                chain,
                iteration,
                det_label_base,
                elapsed,
            )

    LOGGER.info("Plots written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
