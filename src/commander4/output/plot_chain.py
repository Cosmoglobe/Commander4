import argparse
import glob
import logging
import os
import re
from copy import deepcopy

import h5py
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pixell.bunch import Bunch

import commander4.sky_models.component as component_lib
from commander4.data_models.detector_map import DetectorMap
from commander4.output import plotting
from commander4.sky_models.component import Component
from commander4.utils.params import Params


CHAIN_ITER_RE = re.compile(r"chain(?P<chain>\d+)_iter(?P<iter>\d+)\.h5$")


def _decode_h5_value(value):
	if isinstance(value, bytes):
		return value.decode("utf-8")
	if isinstance(value, np.bytes_):
		return value.tobytes().decode("utf-8")
	if isinstance(value, np.ndarray) and value.shape == ():
		return _decode_h5_value(value.item())
	return value


def _load_params_from_chain(chain_dir: str) -> Params | None:
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
					if "metadata/parameter_file_as_binary_yaml" in handle:
						raw_yaml = _decode_h5_value(handle["metadata/parameter_file_as_binary_yaml"][()])
					elif "metadata/parameter_file_as_string" in handle:
						raw_yaml = _decode_h5_value(handle["metadata/parameter_file_as_string"][()])
					else:
						continue
			except OSError:
				continue

			if not raw_yaml:
				continue
			params_dict = yaml.safe_load(raw_yaml)
			params = Params(params_dict)
			params.parameter_file_as_string = yaml.dump(params_dict)
			params.parameter_file_binary_yaml = raw_yaml
			return params
	return None


def _extract_chain_iter(filename: str) -> tuple[int | None, int | None]:
	match = CHAIN_ITER_RE.search(filename)
	if not match:
		return None, None
	return int(match.group("chain")), int(match.group("iter"))


def _build_component_list(params: Params) -> list[Component]:
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

		if base_params.polarizations[0]:
			params_I = deepcopy(base_params)
			params_I.longname = comp_longname + "_Intensity"
			params_I.shortname = comp_shortname + "_I"
			params_I.polarized = False
			comp_list.append(getattr(component_lib, component.component_class)(params_I, params.general))
		if base_params.polarizations[1] and base_params.polarizations[2]:
			params_QU = deepcopy(base_params)
			params_QU.longname = comp_longname + "_Polarization"
			params_QU.shortname = comp_shortname + "_QU"
			params_QU.polarized = True
			comp_list.append(getattr(component_lib, component.component_class)(params_QU, params.general))

	return comp_list


def _load_compsep_components(params: Params, compsep_path: str) -> list[Component]:
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


def _ensure_2d_map(map_data: np.ndarray | None) -> np.ndarray | None:
	if map_data is None:
		return None
	if map_data.ndim == 1:
		return map_data.reshape((1, -1))
	return map_data


def _align_map_rows(map_data: np.ndarray | None, target_rows: int, npix: int) -> np.ndarray:
	if map_data is None:
		return np.zeros((target_rows, npix))
	map_data = _ensure_2d_map(map_data)
	if map_data.shape[0] == target_rows:
		return map_data
	if map_data.shape[0] >= target_rows:
		return map_data[:target_rows]
	pad = np.zeros((target_rows - map_data.shape[0], npix), dtype=map_data.dtype)
	return np.vstack([map_data, pad])


def _build_detector_data(map_sky: np.ndarray,
						 map_rms: np.ndarray | None,
						 nu: float,
						 fwhm: float,
						 nside: int,
						 map_corrnoise: np.ndarray | None = None,
						 map_orbdipole: np.ndarray | None = None,
						 map_skymodel: np.ndarray | None = None) -> DetectorMap:
	npix = map_sky.shape[-1]
	target_rows = map_sky.shape[0] if map_sky.ndim == 2 else 1
	if map_rms is None:
		map_rms = np.ones((target_rows, npix))
	else:
		map_rms = _align_map_rows(map_rms, target_rows, npix)
	map_corrnoise = _align_map_rows(map_corrnoise, target_rows, npix)
	map_orbdipole = _align_map_rows(map_orbdipole, target_rows, npix)
	map_skymodel = _align_map_rows(map_skymodel, target_rows, npix)

	detmap = DetectorMap(map_sky, map_rms, nu, fwhm, nside)
	detmap.map_corrnoise = map_corrnoise
	detmap.map_orbdipole = map_orbdipole
	detmap.map_skymodel = map_skymodel
	return detmap


def _match_band_info(filename: str, params: Params | None) -> tuple[str | None, str | None, Bunch | None]:
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


def _load_maps_file(map_path: str) -> dict:

	with h5py.File(map_path, "r") as handle:
		map_data = {
			"map_observed_sky": handle["map_observed_sky"][()] if "map_observed_sky" in handle else None,
			"map_rms": handle["map_rms"][()] if "map_rms" in handle else None,
			"map_orbdipole": handle["map_orbdipole"][()] if "map_orbdipole" in handle else None,
			"map_corrnoise": handle["map_corrnoise"][()] if "map_corrnoise" in handle else None,
			"map_skymodel": handle["map_skymodel"][()] if "map_skymodel" in handle else None,
		}
	return map_data


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


def _ud_grade_map(map_data: np.ndarray | None, nside_out: int) -> np.ndarray | None:
	if map_data is None:
		return None
	map_data = _ensure_2d_map(map_data)
	if map_data is None:
		return None
	npix = map_data.shape[-1]
	nside_in = hp.npix2nside(npix)
	if nside_in == nside_out:
		return map_data

	graded = []
	for row in map_data:
		graded.append(hp.ud_grade(row, nside_out))
	return np.vstack(graded)


def _plot_tod_sampling(chain_dir: str, output_dir: str, chain_filter: set[int] | None) -> None:
	tod_files = sorted(glob.glob(os.path.join(chain_dir, "tod", "*.h5")))
	if not tod_files:
		logging.info("No tod/*.h5 files found; skipping tod sampling plots.")
		return

	logging.info("Found %s tod files", len(tod_files))
	data: dict[str, dict[int, dict[int, dict[str, float]]]] = {}
	for path in tod_files:
		filename = os.path.basename(path)
		chain, iteration = _extract_chain_iter(filename)
		if chain is None or iteration is None:
			continue
		if chain_filter is not None and chain not in chain_filter:
			continue
		detector = filename.rsplit("_chain", 1)[0]
		with h5py.File(path, "r") as handle:
			for key in handle.keys():
				if key in {"scanID", "metadata"}:
					continue
				dataset = handle[key]
				if dataset.shape[0] == 0:
					continue
				value = dataset[0].item() if hasattr(dataset[0], "item") else float(dataset[0])
				data.setdefault(detector, {}).setdefault(chain, {}).setdefault(iteration, {})[key] = value

	out_folder = os.path.join(output_dir, "tod_sampling")
	os.makedirs(out_folder, exist_ok=True)
	combined_folder = os.path.join(out_folder, "combined")
	os.makedirs(combined_folder, exist_ok=True)
	for detector, det_chains in data.items():
		for chain, iter_map in det_chains.items():
			iterations = sorted(iter_map.keys())
			if not iterations:
				continue
			all_keys = set()
			for iter_values in iter_map.values():
				all_keys.update(iter_values.keys())
			for key in sorted(all_keys):
				x_vals: list[int] = []
				y_vals: list[float] = []
				for iter_val in iterations:
					if key not in iter_map[iter_val]:
						continue
					x_vals.append(iter_val)
					y_vals.append(iter_map[iter_val][key])
				if not x_vals:
					continue
				plt.figure()
				plt.plot(x_vals, y_vals, marker="o")
				plt.title(f"{detector} chain {chain}: {key}")
				plt.xlabel("iteration")
				plt.ylabel(key)
				filename = f"{detector}_chain{chain}_{key}.png"
				plt.savefig(os.path.join(out_folder, filename), bbox_inches="tight")
				plt.close()

	for chain in sorted({chain for det_chains in data.values() for chain in det_chains.keys()}):
		detectors = sorted([det for det, det_chains in data.items() if chain in det_chains])
		if not detectors:
			continue
		all_keys = set()
		all_iterations = set()
		for det in detectors:
			iter_map = data[det][chain]
			all_iterations.update(iter_map.keys())
			for iter_values in iter_map.values():
				all_keys.update(iter_values.keys())
		iterations = sorted(all_iterations)
		for key in sorted(all_keys):
			plt.figure()
			for det in detectors:
				iter_map = data[det][chain]
				x_vals: list[int] = []
				y_vals: list[float] = []
				for iter_val in iterations:
					if iter_val not in iter_map:
						continue
					if key not in iter_map[iter_val]:
						continue
					x_vals.append(iter_val)
					y_vals.append(iter_map[iter_val][key])
				if x_vals:
					plt.plot(x_vals, y_vals, marker="o", label=det)
			if not plt.gca().lines:
				plt.close()
				continue
			plt.title(f"Chain {chain}: {key} (sample 0)")
			plt.xlabel("iteration")
			plt.ylabel(key)
			plt.legend(loc="best")
			filename = f"chain{chain}_{key}_combined.png"
			plt.savefig(os.path.join(combined_folder, filename), bbox_inches="tight")
			plt.close()

	logging.info("Saved tod sampling plots to %s", out_folder)


def main() -> int:
	parser = argparse.ArgumentParser(description="Plot Commander4 chain outputs from disk.")
	parser.add_argument("chain_dir", help="Path to a chain output directory (containing compsep/, datamaps/, tod/).")
	parser.add_argument("--output-dir", default=None, help="Directory for plots. Defaults to <chain_dir>/plots.")
	parser.add_argument(
		"--chains",
		default="all",
		help="Comma-separated chain numbers or ranges (e.g. 1,2 or 1-3). Use 'all' for everything.",
	)
	parser.add_argument(
		"--iterations",
		default="all",
		help="Comma-separated iteration numbers or ranges (e.g. 1,2 or 1-10). Use 'all' for everything.",
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
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO, format="[plot_chain] %(levelname)s: %(message)s")
	chain_dir = os.path.abspath(args.chain_dir)
	if not os.path.isdir(chain_dir):
		logging.error("Chain directory not found: %s", chain_dir)
		return 1

	output_dir = args.output_dir or os.path.join(chain_dir, "plots")
	os.makedirs(output_dir, exist_ok=True)
	plot_params = Bunch(output_paths=Bunch(plots=output_dir))

	chain_filter = _parse_int_set(args.chains)
	iter_filter = _parse_int_set(args.iterations)
	map_types_raw = {item.strip().lower() for item in args.map_types.split(",") if item.strip()}
	if not map_types_raw or "all" in map_types_raw:
		map_types = {"data", "combo", "components"}
	else:
		map_types = map_types_raw

	params = _load_params_from_chain(chain_dir)
	if params is None:
		logging.warning("No parameter metadata found in chain outputs; some plots may be skipped.")
	elif "components" in params:
		_ = _build_component_list(params)
	if args.nside.lower() == "native":
		nside_target = None
	else:
		nside_target = int(args.nside)
	logging.info("Plotting at nside: %s", "native" if nside_target is None else nside_target)

	compsep_files = {}
	for path in glob.glob(os.path.join(chain_dir, "compsep", "*.h5")):
		chain, iteration = _extract_chain_iter(os.path.basename(path))
		if chain is None or iteration is None:
			continue
		compsep_files[(chain, iteration)] = path

	maps_files = sorted(glob.glob(os.path.join(chain_dir, "datamaps", "*.h5")))
	if not maps_files:
		logging.warning("No datamaps/*.h5 files found in %s", chain_dir)

	comp_cache: dict[tuple[int, int], list[Component]] = {}
	logging.info("Found %s map files and %s compsep files", len(maps_files), len(compsep_files))

	_plot_tod_sampling(chain_dir, output_dir, chain_filter)

	for map_path in maps_files:
		filename = os.path.basename(map_path)
		chain, iteration = _extract_chain_iter(filename)
		if chain is None or iteration is None:
			logging.warning("Skipping map file without chain/iter: %s", filename)
			continue
		if chain_filter is not None and chain not in chain_filter:
			continue
		if iter_filter is not None and iteration not in iter_filter:
			continue

		exp_name, band_name, band_info = _match_band_info(filename, params)
		if band_info is not None:
			nu = getattr(band_info, "freq", None)
			if nu is None:
				nu = getattr(band_info, "nu", np.nan)
			fwhm = getattr(band_info, "fwhm", 0.0)
		else:
			nu = np.nan
			fwhm = 0.0

		map_data = _load_maps_file(map_path)
		map_observed_sky = _ensure_2d_map(map_data["map_observed_sky"])
		if map_observed_sky is None:
			logging.warning("Map file missing map_observed_sky: %s", filename)
			continue
		npix = map_observed_sky.shape[-1]
		nside = hp.npix2nside(npix)
		logging.info("Plotting %s (chain %s iter %s)", filename, chain, iteration)
		if nside_target is not None:
			map_observed_sky = _ud_grade_map(map_observed_sky, nside_target)
			nside = nside_target

		map_rms = _ensure_2d_map(map_data["map_rms"])
		map_orbdipole = _ensure_2d_map(map_data["map_orbdipole"])
		map_corrnoise = _ensure_2d_map(map_data["map_corrnoise"])
		map_skymodel = _ensure_2d_map(map_data["map_skymodel"])
		if nside_target is not None:
			map_rms = _ud_grade_map(map_rms, nside_target)
			map_orbdipole = _ud_grade_map(map_orbdipole, nside_target)
			map_corrnoise = _ud_grade_map(map_corrnoise, nside_target)
			map_skymodel = _ud_grade_map(map_skymodel, nside_target)

		map_skysub = None
		if map_skymodel is not None:
			map_skysub = _align_map_rows(map_observed_sky, map_observed_sky.shape[0], npix)
			map_skysub = map_skysub + _align_map_rows(map_corrnoise, map_observed_sky.shape[0], npix)
			map_skysub = map_skysub - _align_map_rows(map_skymodel, map_observed_sky.shape[0], npix)

		det_label_base = band_name or "det"
		if exp_name and band_name:
			det_label_base = f"{exp_name}_{band_name}"

		if "data" in map_types:
			plotting.plot_data_maps(
				plot_params,
				det_label_base,
				chain,
				iteration,
				map_signal=map_observed_sky,
				map_corrnoise=map_corrnoise,
				map_rms=map_rms,
				map_skysub=map_skysub,
				map_orbdipole=map_orbdipole,
			)

		comp_list = None
		comp_key = (chain, iteration)
		if comp_key in compsep_files and params is not None:
			comp_list = comp_cache.get(comp_key)
			if comp_list is None:
				comp_list = _load_compsep_components(params, compsep_files[comp_key])
				comp_cache[comp_key] = comp_list
		if not comp_list or not ({"combo", "components"} & map_types):
			continue

		if map_observed_sky.shape[0] == 1:
			pol_slices = [("I", slice(0, 1))]
		elif map_observed_sky.shape[0] == 2:
			pol_slices = [("QU", slice(0, 2))]
		else:
			pol_slices = [("I", slice(0, 1)), ("QU", slice(1, 3))]

		for pol_label, pol_slice in pol_slices:
			map_sky = map_observed_sky[pol_slice]
			detmap = _build_detector_data(
				map_sky,
				map_rms[pol_slice] if map_rms is not None else None,
				nu,
				fwhm,
				nside,
				map_corrnoise=map_corrnoise[pol_slice] if map_corrnoise is not None else None,
				map_orbdipole=map_orbdipole[pol_slice] if map_orbdipole is not None else None,
				map_skymodel=map_skymodel[pol_slice] if map_skymodel is not None else None,
			)
			det_label = f"{det_label_base}_{pol_label}"
			if "combo" in map_types:
				plotting.plot_combo_maps(plot_params, det_label, chain, iteration, comp_list, detmap)
			if "components" in map_types:
				plotting.plot_components(plot_params, det_label, chain, iteration, comp_list, detmap)
			logging.info("Finished %s plots", det_label)

	logging.info("Plots written to %s", output_dir)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
