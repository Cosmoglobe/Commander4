import os

import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pixell.bunch import Bunch

from commander4.sky_models.component import Component, split_complist

def _ensure_2d_map(map_data: np.ndarray | None) -> np.ndarray | None:
    if map_data is None:
        return None
    map_array = np.asarray(map_data)
    if map_array.ndim == 1:
        return map_array.reshape(1, -1)
    return map_array


def _force_iqu_rows(map_data: np.ndarray | None) -> np.ndarray | None:
    map_array = _ensure_2d_map(map_data)
    if map_array is None:
        return None
    npix = map_array.shape[-1]
    out = np.zeros((3, npix), dtype=map_array.dtype)
    if map_array.shape[0] == 1:
        out[0] = map_array[0]
    elif map_array.shape[0] == 2:
        out[1:3] = map_array[:2]
    else:
        out[:3] = map_array[:3]
    return out


def _safe_percentile(values: np.ndarray, q: float, fallback: float = 0.0) -> float:
    arr = np.asarray(values)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return fallback
    return float(np.nanpercentile(arr[finite], q))


def _sym_limits(values: np.ndarray, percentile: float = 99.0) -> tuple[float, float]:
    lim = _safe_percentile(np.abs(values), percentile, fallback=1.0)
    lim = max(lim, 1e-12)
    return -lim, lim


def _stokes_labels(npol: int) -> list[str]:
    if npol == 1:
        return ["I"]
    if npol == 2:
        return ["Q", "U"]
    if npol >= 3:
        return ["I", "Q", "U"][:npol]
    return [f"P{i}" for i in range(npol)]


def _get_component_map(
    component: Component,
    freq: float,
    nside: int,
    npol: int,
    ipol: int,
    smoothing_scale_radians: float,
) -> np.ndarray:
    npix = 12 * nside**2
    if npol == 1:
        if component.pol:
            return np.zeros((npix,))
        return component.get_sky(freq, nside, fwhm=smoothing_scale_radians)[0]

    if component.pol:
        return component.get_sky(freq, nside, fwhm=smoothing_scale_radians)[ipol]
    return np.zeros((npix,))


def plot_tod_series(
    out_folder: str,
    detector: str,
    chain: int,
    key: str,
    x_vals: list[int],
    y_vals: list[float],
) -> None:
    plt.figure()
    plt.plot(x_vals, y_vals, marker="o")
    plt.title(f"{detector} chain {chain}: {key}")
    plt.xlabel("iteration")
    plt.ylabel(key)
    filename = os.path.join(out_folder, f"{detector}_chain{chain:02d}_{key}.png")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_tod_combined(
    out_folder: str,
    chain: int,
    key: str,
    series_by_detector: dict[str, tuple[list[int], list[float]]],
) -> None:
    plt.figure(figsize=(12,6))
    any_line = False
    for detector, (x_vals, y_vals) in sorted(series_by_detector.items()):
        if not x_vals:
            continue
        any_line = True
        plt.plot(x_vals, y_vals, marker="o", label=detector)
    if not any_line:
        plt.close()
        return
    plt.title(f"Chain {chain}: {key} (sample 0)")
    plt.xlabel("iteration")
    plt.ylabel(key)
    plt.legend(loc="best", ncol=3)
    filename = os.path.join(out_folder, f"chain{chain:02d}_{key}_combined.png")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_combo_maps(
    params: Bunch,
    detector: str,
    chain: int,
    iteration: int,
    comp_list: list[Component],
    *,
    map_signal: np.ndarray,
    nu: float,
    nside: int,
    map_rms: np.ndarray | None = None,
    map_corrnoise: np.ndarray | None = None,
    map_orbdipole: np.ndarray | None = None,
    map_skymodel: np.ndarray | None = None,
    gain: float | None = None,
    g0: float | None = None,
) -> None:
    out_folder = os.path.join(params.output_paths.plots, "combo_maps")
    os.makedirs(out_folder, exist_ok=True)

    map_signal = _ensure_2d_map(map_signal)
    if map_signal is None:
        return
    npol = map_signal.shape[0]
    npix = 12 * nside**2
    pol_names = _stokes_labels(npol)

    comp_sublist = split_complist(comp_list, 1 if npol > 1 else 0)

    for ipol in range(npol):
        signal = map_signal[ipol]
        if map_corrnoise is not None and ipol < map_corrnoise.shape[0]:
            corrnoise = map_corrnoise[ipol]
        else:
            corrnoise = np.zeros((npix,))
        rms = map_rms[ipol] if map_rms is not None and ipol < map_rms.shape[0] else None
        if map_orbdipole is not None and ipol < map_orbdipole.shape[0]:
            orbdipole = map_orbdipole[ipol]
        else:
            orbdipole = np.zeros((npix,))
        if map_skymodel is not None and ipol < map_skymodel.shape[0]:
            skymodel = map_skymodel[ipol]
        else:
            skymodel = np.zeros((npix,))

        map_rawobs = signal + corrnoise + orbdipole
        map_skysub = signal + corrnoise - skymodel

        foreground_subtracted = signal.copy()
        cmb_subtracted = signal.copy()
        residual = signal.copy()
        cmb_map_anisotropies = np.zeros_like(signal)

        fig, ax = plt.subplots(3, 5, figsize=(32, 13.7))
        if gain is not None and g0 is not None:
            title = (
                f"Iter {iteration:04d}. Freq: {nu:.2f} GHz (det {detector}). Chain {chain}. "
                f"Detector gain = {gain:.4e} (Global gain = {g0:.4e})."
            )
        else:
            title = f"Iter {iteration:04d}. Freq: {nu:.2f} GHz (det {detector}). Chain {chain}."
        fig.suptitle(title, fontsize=24)

        max_component_panels = min(len(comp_sublist), 5)
        for i, component in enumerate(comp_sublist[:max_component_panels]):
            smoothing_scale_radians = component.comp_params.smoothing_scale * np.pi / (180 * 60)
            comp_map = _get_component_map(component, nu, nside, npol, ipol, smoothing_scale_radians)
            if "cmb" not in component.shortname:
                foreground_subtracted -= comp_map
            else:
                cmb_subtracted -= comp_map
                cmb_maps = component.get_sky_anisotropies(
                    nu,
                    nside,
                    fwhm=smoothing_scale_radians,
                )
                cmb_map_anisotropies = cmb_maps[ipol if npol > 1 else 0]
            residual -= comp_map
            plt.axes(ax[2, i])
            limdown, limup = _sym_limits(comp_map)
            hp.mollview(
                comp_map,
                hold=True,
                cmap="RdBu_r",
                title=f"{component.longname} {pol_names[ipol]} at {nu:.2f} GHz",
                min=limdown,
                max=limup,
            )

        plt.axes(ax[0, 0])
        limdown, limup = _sym_limits(map_rawobs)
        hp.mollview(
            map_rawobs,
            fig=fig,
            hold=True,
            cmap="RdBu_r",
            title="Raw observed sky",
            min=limdown,
            max=limup,
        )
        plt.axes(ax[0, 1])
        limdown, limup = _sym_limits(cmb_subtracted)
        hp.mollview(
            cmb_subtracted,
            fig=fig,
            hold=True,
            cmap="RdBu_r",
            title="CMB subtracted sky",
            min=limdown,
            max=limup,
        )
        plt.axes(ax[0, 2])
        limdown, limup = _sym_limits(foreground_subtracted)
        hp.mollview(
            foreground_subtracted,
            fig=fig,
            hold=True,
            cmap="RdBu_r",
            title="Foreground subtracted sky",
            min=limdown,
            max=limup,
        )
        plt.axes(ax[0, 3])
        limdown, limup = _sym_limits(map_skysub)
        hp.mollview(
            map_skysub,
            fig=fig,
            hold=True,
            cmap="RdBu_r",
            title="All sky signals subtracted",
            min=limdown,
            max=limup,
        )
        plt.axes(ax[0, 4])
        limdown, limup = _sym_limits(cmb_map_anisotropies)
        hp.mollview(
            cmb_map_anisotropies,
            fig=fig,
            hold=True,
            cmap="RdBu_r",
            title="CMB anisotropies",
            min=limdown,
            max=limup,
        )

        plt.axes(ax[1, 0])
        limdown, limup = _sym_limits(orbdipole)
        hp.mollview(
            orbdipole,
            fig=fig,
            hold=True,
            cmap="RdBu_r",
            title="Orbital dipole",
            min=limdown,
            max=limup,
        )
        plt.axes(ax[1, 1])
        limdown, limup = _sym_limits(corrnoise)
        hp.mollview(
            corrnoise,
            fig=fig,
            hold=True,
            cmap="RdBu_r",
            title="Corr noise",
            min=limdown,
            max=limup,
        )
        plt.axes(ax[1, 2])
        limdown, limup = _sym_limits(residual)
        hp.mollview(
            residual,
            fig=fig,
            hold=True,
            cmap="RdBu_r",
            title="Residual sky",
            min=limdown,
            max=limup,
        )

        plt.axes(ax[1, 3])
        if rms is not None:
            rel = np.abs(np.divide(residual, rms, out=np.zeros_like(residual), where=rms > 0))
            hp.mollview(
                rel,
                fig=fig,
                hold=True,
                cmap="RdBu_r",
                title="Residual/RMS",
                min=0,
                max=_safe_percentile(rel, 99, 1.0),
            )
        else:
            hp.mollview(
                np.zeros_like(residual),
                fig=fig,
                hold=True,
                cmap="RdBu_r",
                title="Residual/RMS (missing RMS)",
            )

        plt.axes(ax[1, 4])
        if rms is not None:
            hp.mollview(
                rms,
                fig=fig,
                hold=True,
                norm="log",
                title="RMS",
                min=max(float(np.min(rms)), 1e-12),
                max=_safe_percentile(rms, 99, 1.0),
            )
        else:
            hp.mollview(
                np.zeros_like(residual),
                fig=fig,
                hold=True,
                cmap="RdBu_r",
                title="RMS (missing)",
            )

        filename = os.path.join(
            out_folder,
            f"{detector}_chain{chain:02d}_iter{iteration:04d}_{pol_names[ipol]}_combo.png",
        )
        plt.savefig(filename, bbox_inches="tight")
        plt.close()


def plot_data_maps(
    params: Bunch,
    detector: str,
    chain: int,
    iteration: int,
    *,
    map_signal: np.ndarray,
    map_rms: np.ndarray | None = None,
    map_corrnoise: np.ndarray | None = None,
    map_residual: np.ndarray | None = None,
    map_orbdipole: np.ndarray | None = None,
) -> None:
    out_folder = os.path.join(params.output_paths.plots, "maps_data")
    os.makedirs(out_folder, exist_ok=True)

    maps = {
        "map_signal": map_signal,
        "map_corrnoise": map_corrnoise,
        "map_rms": map_rms,
        "map_residual": map_residual,
        "map_orbdipole": map_orbdipole,
    }
    desc = {
        "map_signal": "Signal map",
        "map_corrnoise": "Corr noise map",
        "map_rms": "RMS map",
        "map_residual": "Residual map",
        "map_orbdipole": "Orbital dipole map",
    }

    labels = ["I", "Q", "U"]
    for map_type, map_data in maps.items():
        map_iqu = _force_iqu_rows(map_data)
        if map_iqu is None:
            continue

        residual_types = {"map_residual", "map_corrnoise"}
        cmap = None if map_type == "map_rms" else "RdBu_r"
        plt.figure(figsize=(17, 4))
        for i in range(3):
            arr = map_iqu[i]
            if map_type == "map_rms":
                limup = _safe_percentile(arr, 99, 1.0)
                limdown = max(float(np.nanmin(arr)), 0.0)
            elif map_type in residual_types:
                limdown, limup = _sym_limits(arr)
            else:
                limdown = _safe_percentile(arr, 1, -1.0)
                limup = _safe_percentile(arr, 99, 1.0)
                if limup <= limdown:
                    limdown, limup = _sym_limits(arr)
            hp.mollview(arr, cmap=cmap, title=labels[i], sub=(1, 3, i + 1), min=limdown, max=limup)
        plt.suptitle(f"{desc[map_type]}, det {detector}, chain {chain}, iter {iteration}")
        filename = os.path.join(
            out_folder,
            f"{detector}_chain{chain:02d}_iter{iteration:04d}_{map_type}.png",
        )
        plt.savefig(filename, bbox_inches="tight")
        plt.close()


def plot_cg_res(params: Bunch, chain: int, iteration: int, residual: np.ndarray) -> None:
    out_folder = os.path.join(params.output_paths.plots, "CG_res")
    os.makedirs(out_folder, exist_ok=True)
    plt.figure()
    plt.loglog(np.arange(residual.shape[0]), residual)
    plt.axhline(params.CG_err_tol, ls="--", c="k")
    filename = os.path.join(out_folder, f"CG_res_chain{chain}_iter{iteration}.png")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_components(
    params: Bunch,
    detector: str,
    chain: int,
    iteration: int,
    components_list: list[Component],
    *,
    map_signal: np.ndarray,
    nu: float,
    nside: int,
) -> None:
    map_comp_out = os.path.join(params.output_paths.plots, "maps_comps")
    dl_out = os.path.join(params.output_paths.plots, "spectra_comps_Dl")
    cl_out = os.path.join(params.output_paths.plots, "spectra_comps_Cl")
    os.makedirs(map_comp_out, exist_ok=True)
    os.makedirs(dl_out, exist_ok=True)
    os.makedirs(cl_out, exist_ok=True)

    if map_signal is None:
        return

    npol = map_signal.shape[0]
    pol_names = _stokes_labels(npol)
    comp_sublist = split_complist(components_list, 1 if npol > 1 else 0)

    ells = np.arange(3 * nside)
    Z = ells * (ells + 1) / (2 * np.pi)

    for ipol in range(npol):
        signal = map_signal[ipol]
        foreground_subtracted = signal.copy()
        residual = signal.copy()

        for component in comp_sublist:
            smoothing_scale_radians = component.comp_params.smoothing_scale * np.pi / (180 * 60)
            comp_map = _get_component_map(component, nu, nside, npol, ipol, smoothing_scale_radians)
            if component.shortname != "cmb":
                foreground_subtracted -= comp_map
            residual -= comp_map
            if np.all(comp_map == 0):
                continue

            limdown, limup = _sym_limits(comp_map)
            hp.mollview(
                comp_map,
                cmap="RdBu_r",
                title=(
                    f"{component.longname} {pol_names[ipol]} at {nu:.2f} GHz, "
                    f"det {detector}, chain {chain}, iter {iteration}"
                ),
                min=limdown,
                max=limup,
            )
            plt.savefig(
                os.path.join(
                    map_comp_out,
                    (
                        f"{detector}_chain{chain:02d}_iter{iteration:04d}_"
                        f"{pol_names[ipol]}_{component.shortname}.png"
                    ),
                ),
                bbox_inches="tight",
            )
            plt.close()

            Cl = hp.alm2cl(hp.map2alm(comp_map))
            plt.figure()
            plt.plot(ells, Z * Cl, label=component.longname)
            plt.xscale("log")
            plt.yscale("log")
            plt.savefig(
                os.path.join(
                    dl_out,
                    (
                        f"{detector}_chain{chain:02d}_iter{iteration:04d}_"
                        f"{pol_names[ipol]}_{component.shortname}_Dl.png"
                    ),
                ),
                bbox_inches="tight",
            )
            plt.close()

            plt.figure()
            plt.plot(ells, Cl, label=component.longname)
            plt.xscale("log")
            plt.yscale("log")
            plt.savefig(
                os.path.join(
                    cl_out,
                    (
                        f"{detector}_chain{chain:02d}_iter{iteration:04d}_"
                        f"{pol_names[ipol]}_{component.shortname}_Cl.png"
                    ),
                ),
                bbox_inches="tight",
            )
            plt.close()

        limdown, limup = _sym_limits(foreground_subtracted)
        hp.mollview(
            foreground_subtracted,
            cmap="RdBu_r",
            title=(
                f"Foreground subtracted sky {pol_names[ipol]} at {nu:.2f} GHz, "
                f"det {detector}, chain {chain}, iter {iteration}"
            ),
            min=limdown,
            max=limup,
        )
        plt.savefig(
            os.path.join(
                map_comp_out,
                (
                    f"{detector}_chain{chain:02d}_iter{iteration:04d}_"
                    f"{pol_names[ipol]}_foreground_subtracted.png"
                ),
            ),
            bbox_inches="tight",
        )
        plt.close()

        limdown, limup = _sym_limits(residual)
        hp.mollview(
            residual,
            cmap="RdBu_r",
            title=(
                f"Residual sky {pol_names[ipol]} at {nu:.2f} GHz, det {detector}, "
                f"chain {chain}, iter {iteration}"
            ),
            min=limdown,
            max=limup,
        )
        plt.savefig(
            os.path.join(
                map_comp_out,
                (
                    f"{detector}_chain{chain:02d}_iter{iteration:04d}_"
                    f"{pol_names[ipol]}_residual.png"
                ),
            ),
            bbox_inches="tight",
        )
        plt.close()


def alm_plotter(alm, filename="alm_plot.png"):
    alm_len = alm.shape[-1]
    lmax = int(np.sqrt(2 * alm_len + 0.25) - 1.5)
    mesh_real = np.zeros((lmax + 1, lmax + 1))
    mesh_imag = np.zeros((lmax + 1, lmax + 1))
    for ell in range(lmax + 1):
        for m in range(0, ell + 1):
            idx = hp.Alm.getidx(lmax, ell, m)
            mesh_real[ell, m] = alm[idx].real
            mesh_imag[ell, m] = alm[idx].imag
    for ell in range(lmax + 1):
        for m in range(ell + 1, lmax + 1):
            idx = hp.Alm.getidx(lmax, ell, -m)
            mesh_real[ell, m] = np.nan
            mesh_imag[ell, m] = np.nan
    vmax = max(np.nanmax(np.abs(mesh_real.flatten()[1:])), np.nanmax(np.abs(mesh_imag)))
    vmin = -vmax
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    img = ax[0].imshow(mesh_real, cmap="magma", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(img, ax=ax[0])
    ax[0].set_title("Real Part")
    ax[0].set_xlabel("m")
    ax[0].set_ylabel("l")
    img = ax[1].imshow(mesh_imag, cmap="magma", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(img, ax=ax[1])
    ax[1].set_title("Imaginary Part")
    ax[1].set_xlabel("m")
    ax[1].set_ylabel("l")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
