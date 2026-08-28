"""Diagnostic plots written alongside the chain: band maps, components, TODs and CG residuals."""
import os
import warnings
from dataclasses import dataclass

import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from pixell.bunch import Bunch

from commander4.sky.component import Component


@dataclass
class ChainLinePanel:
    """One axis in a multi-panel chain trace figure."""

    title: str
    ylabel: str
    series: list[tuple[str, np.ndarray, np.ndarray]]
    xscale: str = "linear"
    yscale: str = "linear"
    horizontal_lines: tuple[float, ...] = ()


def _ensure_plot_parent(filename: str) -> None:
    """Create the destination directory only when a plot is actually written."""
    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)


def plot_chain_map(
    filename: str,
    map_data: np.ndarray,
    title: str,
    *,
    unit: str = "",
    symmetric: bool = True,
) -> None:
    """Write one full-sky chain map as a single Mollweide panel."""
    values = np.asarray(map_data, dtype=float)
    invalid = ~np.isfinite(values) | np.isclose(values, hp.UNSEEN, rtol=0.0, atol=1e20)
    plotted = np.ma.array(values, mask=invalid)
    finite = values[~invalid]
    if finite.size == 0 or not np.any(finite != 0.0):
        return
    _ensure_plot_parent(filename)

    if symmetric:
        lower, upper = _sym_limits(finite)
        cmap = "RdBu_r"
    else:
        lower = _safe_percentile(finite, 1.0, fallback=0.0)
        upper = _safe_percentile(finite, 99.0, fallback=1.0)
        if lower >= 0.0:
            lower = 0.0
        if upper <= lower:
            upper = lower + 1.0
        cmap = "viridis"

    hp.mollview(
        plotted,
        cmap=cmap,
        title=title,
        unit=unit,
        min=lower,
        max=upper,
    )
    fig = plt.gcf()
    fig.savefig(filename, bbox_inches="tight", dpi=140)
    plt.close(fig)


def plot_chain_map_grid(
    filename: str,
    title: str,
    map_rows: np.ndarray,
    row_labels: list[str],
    *,
    unit: str = "",
    symmetric: list[bool] | None = None,
) -> None:
    """Write all rows of one map quantity in a compact Mollweide grid."""
    rows = np.asarray(map_rows, dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    valid = np.isfinite(rows) & ~np.isclose(rows, hp.UNSEEN, rtol=0.0, atol=1e20)
    if not np.any(valid) or not np.any(rows[valid] != 0.0):
        return
    _ensure_plot_parent(filename)

    if symmetric is None:
        symmetric = [True] * rows.shape[0]
    ncols = min(3, rows.shape[0])
    nrows = int(np.ceil(rows.shape[0] / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows), squeeze=False)
    flat_axes = axes.reshape(-1)
    for index, (row, label, use_symmetric) in enumerate(
        zip(rows, row_labels, symmetric)
    ):
        invalid = ~np.isfinite(row) | np.isclose(row, hp.UNSEEN, rtol=0.0, atol=1e20)
        plotted = np.ma.array(row, mask=invalid)
        finite = row[~invalid]
        if finite.size == 0:
            finite = np.array([0.0])
        if use_symmetric:
            lower, upper = _sym_limits(finite)
            cmap = "RdBu_r"
        else:
            lower = min(_safe_percentile(finite, 1.0, fallback=0.0), 0.0)
            upper = _safe_percentile(finite, 99.0, fallback=1.0)
            if upper <= lower:
                upper = lower + 1.0
            cmap = "viridis"
        plt.sca(flat_axes[index])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PendingDeprecationWarning)
            hp.mollview(
                plotted,
                hold=True,
                cmap=cmap,
                title=label,
                unit=unit,
                min=lower,
                max=upper,
            )
    for axis in flat_axes[rows.shape[0]:]:
        axis.axis("off")
    fig.suptitle(title, fontsize=16)
    fig.savefig(filename, bbox_inches="tight", dpi=140)
    plt.close(fig)


def plot_chain_lines(
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    series: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    xscale: str = "linear",
    yscale: str = "linear",
    horizontal_lines: tuple[float, ...] = (),
) -> None:
    """Write several related line series to one chain-diagnostic panel."""
    usable = []
    for label, x_values, y_values in series:
        x_array = np.asarray(x_values)
        y_array = np.asarray(y_values, dtype=float)
        finite = np.isfinite(x_array) & np.isfinite(y_array)
        if xscale == "log":
            finite &= x_array > 0
        if yscale == "log":
            finite &= y_array > 0
        if np.any(finite):
            usable.append((label, x_array[finite], y_array[finite]))
    if not usable:
        return
    if not any(np.any(y_values != 0.0) for _, _, y_values in usable):
        return
    _ensure_plot_parent(filename)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    many_iterations = len(usable) > 12 and all(label.startswith("iter ") for label, _, _ in usable)
    if many_iterations:
        iterations = np.array([int(label.split()[-1]) for label, _, _ in usable])
        norm = colors.Normalize(vmin=float(np.min(iterations)), vmax=float(np.max(iterations)))
        cmap = plt.get_cmap("viridis")
        for iteration, (_, x_values, y_values) in zip(iterations, usable):
            ax.plot(x_values, y_values, color=cmap(norm(iteration)), linewidth=0.8, alpha=0.8)
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        colorbar.set_label("Gibbs iteration")
    else:
        for label, x_values, y_values in usable:
            ax.plot(x_values, y_values, linewidth=1.0, marker="." if x_values.size < 200 else None,
                    markersize=3, label=label)
        if len(usable) > 1 or usable[0][0]:
            ax.legend(loc="best", fontsize="small")

    for value in horizontal_lines:
        ax.axhline(value, color="0.4", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(filename, dpi=140)
    plt.close(fig)


def plot_chain_line_panels(
    filename: str,
    title: str,
    xlabel: str,
    panels: list[ChainLinePanel],
) -> None:
    """Write related chain traces as vertically stacked axes in one figure."""
    usable_panels = []
    for panel in panels:
        usable_series = []
        for label, x_values, y_values in panel.series:
            x_array = np.asarray(x_values)
            y_array = np.asarray(y_values, dtype=float)
            finite = np.isfinite(x_array) & np.isfinite(y_array)
            if panel.xscale == "log":
                finite &= x_array > 0
            if panel.yscale == "log":
                finite &= y_array > 0
            if np.any(finite):
                usable_series.append((label, x_array[finite], y_array[finite]))
        if usable_series and any(np.any(values != 0.0) for _, _, values in usable_series):
            usable_panels.append((panel, usable_series))
    if not usable_panels:
        return
    _ensure_plot_parent(filename)

    fig, axes = plt.subplots(
        len(usable_panels), 1, figsize=(10, 3.3 * len(usable_panels)), squeeze=False
    )
    axes = axes[:, 0]
    labels = []
    for _, series in usable_panels:
        for label, _, _ in series:
            if label not in labels:
                labels.append(label)
    many_iterations = len(labels) > 12 and all(label.startswith("iter ") for label in labels)
    if many_iterations:
        iterations = np.array([int(label.split()[-1]) for label in labels])
        norm = colors.Normalize(vmin=float(np.min(iterations)), vmax=float(np.max(iterations)))
        cmap = plt.get_cmap("viridis")
        line_colors = {label: cmap(norm(iteration)) for label, iteration in zip(labels, iterations)}
    else:
        cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        line_colors = {
            label: cycle_colors[index % len(cycle_colors)] for index, label in enumerate(labels)
        }

    for axis, (panel, series) in zip(axes, usable_panels):
        for label, x_values, y_values in series:
            axis.plot(
                x_values,
                y_values,
                color=line_colors[label],
                linewidth=1.0,
                marker="." if x_values.size < 200 else None,
                markersize=3,
                label=label,
            )
        for value in panel.horizontal_lines:
            axis.axhline(value, color="0.4", linestyle="--", linewidth=0.8)
        axis.set_title(panel.title)
        axis.set_ylabel(panel.ylabel)
        axis.set_xscale(panel.xscale)
        axis.set_yscale(panel.yscale)
        axis.grid(alpha=0.2)
        if not many_iterations and len(series) > 1:
            axis.legend(loc="best", fontsize="small", ncol=min(4, len(series)))
    axes[-1].set_xlabel(xlabel)
    if many_iterations:
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.tolist())
        colorbar.set_label("Gibbs iteration")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(filename, dpi=140)
    plt.close(fig)


def plot_chain_spectra(
    filename: str,
    title: str,
    spectra: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """Write median power spectra with their 16--84 percentile scan range."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = False
    for label, freqs, median, lower, upper in spectra:
        freqs = np.asarray(freqs)
        median = np.asarray(median)
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        valid = np.isfinite(freqs) & np.isfinite(median) & (freqs > 0) & (median > 0)
        if not np.any(valid):
            continue
        plotted = True
        line, = ax.loglog(freqs[valid], median[valid], label=label, linewidth=1.3)
        band_valid = valid & np.isfinite(lower) & np.isfinite(upper) & (lower > 0) & (upper > 0)
        ax.fill_between(freqs[band_valid], lower[band_valid], upper[band_valid],
                        color=line.get_color(), alpha=0.12, linewidth=0)
    if not plotted:
        plt.close(fig)
        return
    _ensure_plot_parent(filename)
    ax.set_title(title)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("power spectral density")
    ax.grid(alpha=0.2, which="both")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(filename, dpi=140)
    plt.close(fig)


def plot_noise_parameter_density(
    filename: str,
    title: str,
    fknee: np.ndarray,
    alpha: np.ndarray,
    accepted: np.ndarray,
    *,
    density_label: str = "accepted detector-scans per hexagon",
    rejected_label: str = "rejected",
) -> None:
    """Plot the joint scan distribution of the 1/f knee frequency and slope."""
    fknee = np.asarray(fknee, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    accepted = np.asarray(accepted, dtype=bool)
    valid = np.isfinite(fknee) & np.isfinite(alpha) & (fknee > 0)
    if not np.any(valid) or not np.any(alpha[valid] != 0.0):
        return
    _ensure_plot_parent(filename)

    accepted &= valid
    rejected = valid & ~accepted
    fig, ax = plt.subplots(figsize=(8, 6))
    if np.any(accepted):
        density = ax.hexbin(
            fknee[accepted],
            alpha[accepted],
            gridsize=60,
            mincnt=1,
            bins="log",
            xscale="log",
            cmap="viridis",
        )
        colorbar = fig.colorbar(density, ax=ax)
        colorbar.set_label(density_label)
        accepted_indices = np.flatnonzero(accepted)
        if accepted_indices.size > 5000:
            keep = np.linspace(0, accepted_indices.size - 1, 5000, dtype=int)
            accepted_indices = accepted_indices[keep]
        ax.scatter(
            fknee[accepted_indices],
            alpha[accepted_indices],
            s=2,
            color="black",
            alpha=0.08,
            linewidths=0,
        )
    if np.any(rejected):
        rejected_indices = np.flatnonzero(rejected)
        if rejected_indices.size > 5000:
            keep = np.linspace(0, rejected_indices.size - 1, 5000, dtype=int)
            rejected_indices = rejected_indices[keep]
        ax.scatter(
            fknee[rejected_indices],
            alpha[rejected_indices],
            s=12,
            marker="x",
            color="tab:red",
            linewidths=0.7,
            label=rejected_label,
        )
        ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel(r"$f_\mathrm{knee}$ [Hz]")
    ax.set_ylabel(r"$\alpha$")
    ax.set_xscale("log")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(filename, dpi=140)
    plt.close(fig)


def _components_for_pol(comp_list: list[Component], npol: int) -> list[Component]:
    """The components whose polarization view matches the maps being plotted.

    Maps are plotted one execution view at a time, so `npol` is 1 for an I map and 2 for a QU one,
    and the components are filtered to the matching view. This is `CompList.components_for_eval_pol`
    written out, because the chain plotter builds a plain list rather than a `CompList`.
    """
    target_pol = "QU" if npol > 1 else "I"
    return [comp for comp in comp_list if comp.eval_pol == target_pol]


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
    use_raw = not np.isfinite(freq)
    if npol == 1:
        if component.is_pol:
            return np.zeros((npix,))
        if use_raw:
            return component.get_component_map(nside, fwhm=smoothing_scale_radians)[0]
        return component.get_sky(freq, nside, fwhm=smoothing_scale_radians)[0]

    if component.is_pol:
        if use_raw:
            return component.get_component_map(nside, fwhm=smoothing_scale_radians)[ipol]
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
    fwhm_arcmin: float = np.nan,
) -> None:
    out_folder = os.path.join(params.plots_dir, "combo_maps")
    os.makedirs(out_folder, exist_ok=True)

    map_signal = _ensure_2d_map(map_signal)
    if map_signal is None:
        return
    npol = map_signal.shape[0]
    npix = 12 * nside**2
    pol_names = _stokes_labels(npol)

    comp_sublist = _components_for_pol(comp_list, npol)

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
        beam_radians = fwhm_arcmin * np.pi / (180 * 60) if np.isfinite(fwhm_arcmin) else 0.0
        for i, component in enumerate(comp_sublist[:max_component_panels]):
            comp_map = _get_component_map(component, nu, nside, npol, ipol, beam_radians)
            if "cmb" not in component.shortname:
                foreground_subtracted -= comp_map
            else:
                cmb_subtracted -= comp_map
                cmb_maps = component.get_sky_anisotropies(
                    nu,
                    nside,
                    fwhm=beam_radians,
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
    out_folder = os.path.join(params.plots_dir, "maps_data")
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

        symlim_types = {"map_signal", "map_residual", "map_corrnoise"}
        cmap = None if map_type == "map_rms" else "RdBu_r"
        plt.figure(figsize=(17, 4))
        for i in range(3):
            arr = map_iqu[i]
            if map_type == "map_rms":
                limup = _safe_percentile(arr, 99, 1.0)
                limdown = max(float(np.nanmin(arr)), 0.0)
            elif map_type in symlim_types:
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
    out_folder = os.path.join(params.plots_dir, "CG_res")
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
    fwhm_arcmin: float = np.nan,
) -> None:
    map_comp_out = os.path.join(params.plots_dir, "maps_comps")
    dl_out = os.path.join(params.plots_dir, "spectra_comps_Dl")
    cl_out = os.path.join(params.plots_dir, "spectra_comps_Cl")
    os.makedirs(map_comp_out, exist_ok=True)
    os.makedirs(dl_out, exist_ok=True)
    os.makedirs(cl_out, exist_ok=True)

    if map_signal is None:
        return

    npol = map_signal.shape[0]
    pol_names = _stokes_labels(npol)
    comp_sublist = _components_for_pol(components_list, npol)

    ells = np.arange(3 * nside)
    Z = ells * (ells + 1) / (2 * np.pi)
    beam_radians = fwhm_arcmin * np.pi / (180 * 60) if np.isfinite(fwhm_arcmin) else 0.0

    for ipol in range(npol):
        signal = map_signal[ipol]
        foreground_subtracted = signal.copy()
        residual = signal.copy()

        for component in comp_sublist:
            comp_map = _get_component_map(component, nu, nside, npol, ipol, beam_radians)
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
            Dl = Z * Cl
            plt.figure()
            plt.plot(ells, Dl, label=component.longname)
            plt.xscale("log")
            if np.any(Dl > 0):
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
            if np.any(Cl > 0):
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
