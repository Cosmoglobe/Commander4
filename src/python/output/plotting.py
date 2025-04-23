import healpy as hp
import matplotlib.pyplot as plt
import os
import numpy as np
from pixell import bunch

from src.python.model.component import Component


def plot_data_maps(master, params, detector, chain, iteration, **kwargs):
    """
    Plots the input maps used for component separation.

    Can plot the signal map, the correlated noise map, and the rms map.
    
    The plots are placed in the [params.output_paths.plots]/maps_data/ folder,
    which will be created if it does not exist.

    Arguments:
        master (bool): Whether this is the master process
        params (bunch): The parameter bundle, at root level.
        detector, chain, iteration (integers): The indices representing the
            detector, chain and iteration, respectively.
    Optional named arguments:
        map_signal (np.array): The signal map for the detector in question.
        map_corr_noise (np.array): The correlated noise map for the detector in question.
        map_rms (np_array): The RMS map for the detector in question.
    """
    os.makedirs(params.output_paths.plots + "maps_data/", exist_ok=True)
    for maptype, mapdesc in zip(['map_signal', 'map_corr_noise', 'map_rms'],
                                ['Signal map', 'Corr noise map', 'RMS map']):
        if maptype not in kwargs:
            continue
        if maptype == 'map_rms':
            cmap = None
        else:
            cmap = 'RdBu_r'
        hp.mollview(kwargs[maptype], cmap=cmap, title=f"{mapdesc}, det {detector}, chain {chain}, iter {iteration}")
        plt.savefig(params.output_paths.plots + f"maps_data/{maptype}_det{detector}_chain{chain}_iter{iteration}.png", bbox_inches='tight')
        plt.close()


def plot_cg_res(params, chain, iteration, residual):
    """
    Plots the CG residuals on a loglog scale, with the tolerance plotted along
    with it.

    The plots are placed in the [params.output_paths.plots]/CG_res/ folder,
    which will be created if it does not exist.

    Arguments:
        master (bool): Whether this is the master process
        params (bunch): The parameter bundle, at root level.
        chain, iteration (integers): The indices representing the, chain and
            iteration, respectively.
        residual (np.array): The CG residual.
    """

    os.makedirs(params.output_paths.plots + "CG_res/", exist_ok=True)
    plt.figure()
    plt.loglog(np.arange(residual.shape[0]), residual)
    plt.axhline(params.CG_err_tol, ls="--", c="k")
    plt.savefig(params.output_paths.plots + f"CG_res/CG_res_chain{chain}_iter{iteration}.png")
    plt.close()


def plot_components(params: bunch, freq: float, detector: int, chain: int,
                    iteration: int, signal_map: np.array, components_list: list[Component]):
    """
    Plots the resulting component maps produced by component separation. It will
    also plot the total sky map minus the foregrounds, as well as the total map
    minus the foregrounds and the CMB.
    
    The plots are placed in the [params.output_paths.plots]/maps_comps/ folder,
    which will be created if it does not exist.

    Arguments:
        params (bunch): The parameter bundle, at root level.
        freq (float): The frequency of the detector.
        detector, chain, iteration (integers): The indices representing the
            detector, chain and iteration, respectively.
        signal_map (np.array): The corr-noise subtracted sky signal map for the detector in question.
        components_list (list[Component]): The list of components to plot.
    """

    os.makedirs(params.output_paths.plots + "maps_comps/", exist_ok=True)
    os.makedirs(params.output_paths.plots + "spectra_comps/", exist_ok=True)
    
    foreground_subtracted = np.zeros_like(signal_map)
    foreground_subtracted[:] = signal_map
    residual = np.zeros_like(signal_map)
    residual[:] = signal_map

    ells = np.arange(3 * params.nside)
    Z = ells * (ells+1) / (2 * np.pi)
    for component in components_list:
        smoothing_scale_radians = component.params.smoothing_scale*np.pi/(180*60)
        comp_map = component.get_sky(freq, params.nside, fwhm=smoothing_scale_radians)
        if component.shortname != "cmb":
            foreground_subtracted -= comp_map
        residual -= comp_map

        hp.mollview(comp_map, title=f"{component.longname} at {freq:.2f} GHz, det {detector}, chain {chain}, iter {iteration}", min=np.percentile(comp_map, 2), max=np.percentile(comp_map, 98))
        plt.savefig(params.output_paths.plots + f"maps_comps/{component.shortname}_realization_det{detector}_chain{chain}_iter{iteration}.png", bbox_inches='tight')
        plt.close()
        
        Cl = hp.alm2cl(hp.map2alm(comp_map))
        plt.figure()
        plt.plot(ells, Z * Cl, label=component.longname)
        plt.xscale("log")
        plt.yscale("log")
        plt.savefig(params.output_paths.plots + f"spectra_comps/{component.shortname}_Cl_det{detector}_chain{chain}_iter{iteration}.png", bbox_inches='tight')
        plt.close()

    hp.mollview(foreground_subtracted, title=f"Foreground subtracted sky at {freq:.2f} GHz, det {detector}, chain {chain}, iter {iteration}", min=np.percentile(foreground_subtracted, 2), max=np.percentile(foreground_subtracted, 98))
    plt.savefig(params.output_paths.plots + f"maps_comps/foreground_subtr_det{detector}_chain{chain}_iter{iteration}.png", bbox_inches="tight")
    plt.close()

    hp.mollview(residual, title=f"Residual sky at {freq:.2f} GHz, det {detector}, chain {chain}, iter {iteration}", min=np.percentile(residual, 2), max=np.percentile(residual, 98))
    plt.savefig(params.output_paths.plots + f"maps_comps/residual_det{detector}_chain{chain}_iter{iteration}.png", bbox_inches="tight")
    plt.close()


def alm_plotter(alm, filename="alm_plot.png"):
    alm_len = alm.shape[-1]
    lmax = int(np.sqrt(2*alm_len + 0.25) - 1.5)
    mesh_real = np.zeros((lmax+1, lmax+1))
    mesh_imag = np.zeros((lmax+1, lmax+1))
    for l in range(lmax + 1):
        for m in range(0, l + 1):
            idx = hp.Alm.getidx(lmax, l, m)
            mesh_real[l, m] = alm[idx].real
            mesh_imag[l, m] = alm[idx].imag
    for l in range(lmax + 1):
        for m in range(l + 1, lmax + 1):
            idx = hp.Alm.getidx(lmax, l, -m)
            mesh_real[l, m] = np.nan
            mesh_imag[l, m] = np.nan
    vmax = max(np.nanmax(np.abs(mesh_real.flatten()[1:])), np.nanmax(np.abs(mesh_imag)))
    vmin = -vmax
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    img = ax[0].imshow(mesh_real, cmap='magma', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(img, ax=ax[0])
    ax[0].set_title("Real Part")
    ax[0].set_xlabel("m")
    ax[0].set_ylabel("l")
    img = ax[1].imshow(mesh_imag, cmap='magma', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(img, ax=ax[1])
    ax[1].set_title("Imaginary Part")
    ax[1].set_xlabel("m")
    ax[1].set_ylabel("l")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()