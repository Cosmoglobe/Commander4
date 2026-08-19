"""Optional diagnostic maps and PNG previews for simulated bands."""
import os
from typing import TYPE_CHECKING

import h5py
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from simgen.instrument import Band


def white_noise_normal_matrix(band: "Band", det_pix: dict[str, NDArray],
                              det_psi: dict[str, NDArray]) -> NDArray[np.floating]:
    """Accumulate the white-noise pointing normal matrix for one simulated scan.

    The returned rows are the upper triangle of the per-pixel normal matrix. The row order is
    ``II`` for intensity, ``QQ, QU, UU`` for QU, and ``II, IQ, IU, QQ, QU, UU`` for IQU.
    """
    npix = 12 * band.eval_nside**2
    ncoeff = {"I": 1, "QU": 3, "IQU": 6}[band.polarization]
    normal = np.zeros((ncoeff, npix), dtype=np.float64)
    for det in band.detectors:
        pixels = det_pix[det.name]
        inv_variance = 1.0 / det.sigma0**2
        if band.polarization == "I":
            np.add.at(normal[0], pixels, inv_variance)
            continue
        cosine = np.cos(2.0 * det_psi[det.name])
        sine = np.sin(2.0 * det_psi[det.name])
        if band.polarization == "QU":
            np.add.at(normal[0], pixels, inv_variance * cosine**2)
            np.add.at(normal[1], pixels, inv_variance * cosine * sine)
            np.add.at(normal[2], pixels, inv_variance * sine**2)
        else:
            np.add.at(normal[0], pixels, inv_variance)
            np.add.at(normal[1], pixels, inv_variance * cosine)
            np.add.at(normal[2], pixels, inv_variance * sine)
            np.add.at(normal[3], pixels, inv_variance * cosine**2)
            np.add.at(normal[4], pixels, inv_variance * cosine * sine)
            np.add.at(normal[5], pixels, inv_variance * sine**2)
    return normal


def hit_map(band: "Band", det_pix: dict[str, NDArray]) -> NDArray[np.integer]:
    """Count samples per map pixel across all detectors for one simulated scan."""
    hits = np.zeros(12 * band.eval_nside**2, dtype=np.int64)
    for det in band.detectors:
        np.add.at(hits, det_pix[det.name], 1)
    return hits


def noise_map_rhs(band: "Band", det_pix: dict[str, NDArray], det_psi: dict[str, NDArray],
                  det_noise: dict[str, NDArray]) -> NDArray[np.floating]:
    """Accumulate the white-noise-weighted mapmaking RHS of one scan's noise realization."""
    npix = 12 * band.eval_nside**2
    nstokes = {"I": 1, "QU": 2, "IQU": 3}[band.polarization]
    rhs = np.zeros((nstokes, npix), dtype=np.float64)
    for det in band.detectors:
        pixels = det_pix[det.name]
        weighted_noise = det_noise[det.name] / det.sigma0**2
        if band.polarization == "I":
            np.add.at(rhs[0], pixels, weighted_noise)
            continue
        cosine = np.cos(2.0 * det_psi[det.name])
        sine = np.sin(2.0 * det_psi[det.name])
        if band.polarization == "QU":
            np.add.at(rhs[0], pixels, weighted_noise * cosine)
            np.add.at(rhs[1], pixels, weighted_noise * sine)
        else:
            np.add.at(rhs[0], pixels, weighted_noise)
            np.add.at(rhs[1], pixels, weighted_noise * cosine)
            np.add.at(rhs[2], pixels, weighted_noise * sine)
    return rhs


def _normal_matrices(normal: NDArray[np.floating], polarization: str) -> NDArray[np.floating]:
    """Expand packed normal-matrix rows into one square matrix per pixel."""
    nstokes = {"I": 1, "QU": 2, "IQU": 3}[polarization]
    matrices = np.empty((normal.shape[1], nstokes, nstokes), dtype=np.float64)
    if polarization == "I":
        matrices[:, 0, 0] = normal[0]
    elif polarization == "QU":
        matrices[:, 0, 0] = normal[0]
        matrices[:, 0, 1] = matrices[:, 1, 0] = normal[1]
        matrices[:, 1, 1] = normal[2]
    else:
        matrices[:, 0, 0] = normal[0]
        matrices[:, 0, 1] = matrices[:, 1, 0] = normal[1]
        matrices[:, 0, 2] = matrices[:, 2, 0] = normal[2]
        matrices[:, 1, 1] = normal[3]
        matrices[:, 1, 2] = matrices[:, 2, 1] = normal[4]
        matrices[:, 2, 2] = normal[5]
    return matrices


def _well_conditioned_normal_matrices(normal: NDArray[np.floating],
                                      polarization: str) -> tuple[NDArray, NDArray]:
    """Return expanded normal matrices and a mask of pixels with stable inverses."""
    matrices = _normal_matrices(normal, polarization)
    eigvals = np.linalg.eigvalsh(matrices)
    largest = eigvals[:, -1]
    good = (largest > 0.0) & (eigvals[:, 0] > largest * 1e-12)
    return matrices, good


def normal_matrix_rms(normal: NDArray[np.floating], polarization: str) -> NDArray[np.floating]:
    """Return white-noise RMS from a summed pointing normal matrix.

    Polarized RMS values are the diagonal entries of the inverse IQU normal matrix. Singular or
    ill-conditioned pixels are marked as ``NaN`` because their Stokes parameters are unconstrained.
    """
    matrices, good = _well_conditioned_normal_matrices(normal, polarization)
    inverse = np.full_like(matrices, np.nan)
    inverse[good] = np.linalg.inv(matrices[good])
    return np.diagonal(inverse, axis1=1, axis2=2).T


def noise_map(normal: NDArray[np.floating], rhs: NDArray[np.floating],
              polarization: str) -> NDArray[np.floating]:
    """Bin a realized noise-map RHS with the same normal matrix used for the RMS diagnostic."""
    matrices, good = _well_conditioned_normal_matrices(normal, polarization)
    binned_noise = np.full((matrices.shape[0], matrices.shape[1]), np.nan, dtype=np.float64)
    binned_noise[good] = np.linalg.solve(matrices[good], rhs[:, good].T[..., None])[..., 0]
    return binned_noise.T


def _plot_map(path: str, map_data: NDArray, title: str, *, symmetric: bool = False) -> None:
    """Write one deterministic HEALPix Mollweide preview without requiring a display server."""
    import matplotlib
    matplotlib.use("Agg")
    import healpy as hp
    import matplotlib.pyplot as plt

    finite = np.isfinite(map_data)
    if not np.any(finite):
        return
    values = map_data[finite]
    if symmetric:
        limit = max(float(np.percentile(np.abs(values), 99.0)), 1e-12)
        hp.mollview(map_data, title=title, cmap="RdBu_r", min=-limit, max=limit)
    else:
        lo, hi = np.percentile(values, [1.0, 99.0])
        hp.mollview(map_data, title=title, cmap="viridis", min=lo, max=hi)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def write_band_diagnostics(output_dir: str, band: "Band", skymap: NDArray,
                           hits: NDArray[np.integer], normal: NDArray[np.floating],
                           noise_rhs: NDArray[np.floating]) -> None:
    """Write a band's sky, coverage, uncertainty, and realized-noise products."""
    os.makedirs(output_dir, exist_ok=True)
    rms = normal_matrix_rms(normal, band.polarization)
    binned_noise = noise_map(normal, noise_rhs, band.polarization)
    map_path = os.path.join(output_dir, f"{band.name}_diagnostics.h5")
    with h5py.File(map_path, "w") as outfile:
        outfile["metadata/band"] = band.name
        outfile["metadata/frequency_ghz"] = band.freq
        outfile["metadata/fwhm_arcmin"] = band.fwhm_arcmin
        outfile["metadata/nside"] = band.eval_nside
        outfile["metadata/ordering"] = "RING"
        outfile["metadata/units"] = band.units
        outfile["metadata/polarization"] = band.polarization
        outfile["metadata/rms_description"] = (
            "Map-domain white-noise RMS from the inverse per-pixel pointing normal matrix; "
            "does not include correlated noise, transfer functions, or TOD modifiers."
        )
        outfile["metadata/noise_description"] = (
            "Realized detector noise binned with the nominal white-noise pointing normal matrix; "
            "TOD modifiers are applied before binning."
        )
        outfile["maps/sky"] = skymap.astype(np.float32, copy=False)
        outfile["maps/hits"] = hits.astype(np.float64, copy=False)
        outfile["maps/inv_white_noise"] = normal.astype(np.float64, copy=False)
        outfile["maps/rms_white_noise"] = rms.astype(np.float64, copy=False)
        outfile["maps/noise"] = binned_noise.astype(np.float32, copy=False)

    labels = {"I": ["I"], "QU": ["Q", "U"], "IQU": ["I", "Q", "U"]}[band.polarization]
    for index, label in enumerate(labels):
        _plot_map(os.path.join(output_dir, f"{band.name}_sky_{label}.png"), skymap[index],
                  f"{band.name}: sky {label} [{band.units}]", symmetric=label != "I")
        _plot_map(os.path.join(output_dir, f"{band.name}_rms_white_noise_{label}.png"), rms[index],
                  f"{band.name}: white-noise RMS {label} [{band.units}]")
        _plot_map(os.path.join(output_dir, f"{band.name}_noise_{label}.png"), binned_noise[index],
              f"{band.name}: realized noise {label} [{band.units}]", symmetric=True)
    _plot_map(os.path.join(output_dir, f"{band.name}_hits.png"), hits, f"{band.name}: hits")