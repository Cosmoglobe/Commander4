import healpy as hp
import matplotlib.pyplot as plt
import os
import numpy as np
from numpy.typing import NDArray
from pixell.bunch import Bunch

from src.python.model.component import DiffuseComponent
from src.python.data_models.detector_map import DetectorMap


def plot_combo_maps(params: Bunch, detector: int, chain: int, iteration: int, components_list: list[DiffuseComponent], detector_data: DetectorMap):
    os.makedirs(params.output_paths.plots + "combo_maps/", exist_ok=True)
    map_signal = detector_data.map_sky
    map_corr_noise = detector_data.map_corr_noise
    map_rms = detector_data.map_rms
    map_skysub = detector_data.skysub_map
    map_rawobs = detector_data.rawobs_map
    map_orbdipole = detector_data.orbdipole_map
    freq = detector_data.nu
    gain = detector_data.gain

    foreground_subtracted = np.zeros_like(map_signal)
    cmb_subtracted = np.zeros_like(map_signal)
    foreground_subtracted[:] = map_signal
    cmb_subtracted[:] = map_signal
    residual = np.zeros_like(map_signal)
    residual[:] = map_signal

    fig, ax = plt.subplots(3, 4, figsize=(42, 18))
    fig.suptitle(f"Iter {iteration:04d}. Freq: {freq:.2f} GHz (det {detector}). Chain {chain}. gain = {gain:.4e} (g0={detector_data.g0}).", fontsize=24)

    for i, component in enumerate(components_list):
        smoothing_scale_radians = component.params.smoothing_scale*np.pi/(180*60)
        comp_map = component.get_sky(freq, detector_data.nside, fwhm=smoothing_scale_radians)
        if component.shortname != "cmb":
            foreground_subtracted -= comp_map
        else:
            cmb_subtracted -= comp_map
        residual -= comp_map
        plt.axes(ax[2,i])
        hp.mollview(comp_map, hold=True, title=f"{component.longname} at {freq:.2f} GHz, det {detector}, chain {chain}, iter {iteration}", min=np.percentile(comp_map, 2), max=np.percentile(comp_map, 98))

    residual -= map_corr_noise
    plt.axes(ax[0,0])
    hp.mollview(map_rawobs, fig=fig, hold=True, cmap="RdBu_r", title=f"Raw observed sky", min=np.percentile(map_rawobs, 2), max=np.percentile(map_rawobs, 98))
    plt.axes(ax[0,1])
    hp.mollview(map_signal, fig=fig, hold=True, cmap="RdBu_r", title=f"Static sky signals (d - N_corr - s_orb)", min=np.percentile(map_signal, 2), max=np.percentile(map_signal, 98))
    plt.axes(ax[0,2])
    hp.mollview(cmb_subtracted, fig=fig, hold=True, cmap="RdBu_r", title=f"<- + cmb subtracted", min=np.percentile(foreground_subtracted, 2), max=np.percentile(foreground_subtracted, 98))
    # hp.mollview(foreground_subtracted, fig=fig, hold=True, cmap="RdBu_r", title=f"<- + foreground subtracted", min=np.percentile(foreground_subtracted, 2), max=np.percentile(foreground_subtracted, 98))
    plt.axes(ax[0,3])
    hp.mollview(map_skysub, fig=fig, hold=True, cmap="RdBu_r", title=f"All sky components subtracted (incl. orb-dipole)", min=np.percentile(map_skysub, 2), max=np.percentile(map_skysub, 98))

    plt.axes(ax[1,0])
    hp.mollview(map_orbdipole, fig=fig, hold=True, cmap="RdBu_r", title=f"Orbital dipole", min=np.percentile(map_orbdipole, 2), max=np.percentile(map_orbdipole, 98))
    plt.axes(ax[1,1])
    hp.mollview(map_corr_noise, fig=fig, hold=True, cmap="RdBu_r", title=f"Corr noise", min=np.percentile(map_corr_noise, 2), max=np.percentile(map_corr_noise, 98))
    plt.axes(ax[1,2])
    hp.mollview(residual, fig=fig, hold=True, cmap="RdBu_r", title=f"Residual sky", min=np.percentile(residual, 2), max=np.percentile(residual, 98))
    plt.axes(ax[1,3])
    hp.mollview(map_rms, fig=fig, hold=True, norm="log", title=f"RMS")

    plt.savefig(params.output_paths.plots + f"combo_maps/combo_map_det{detector}_chain{chain}_iter{iteration}.png", bbox_inches="tight")
    plt.close()



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
    for maptype, mapdesc in zip(['map_signal', 'map_corr_noise', 'map_rms', 'map_skysub', 'map_orbdip'],
                                ['Signal map', 'Corr noise map', 'RMS map', 'skysub',     'ordip']):

        if maptype not in kwargs:
            continue
        if maptype == 'map_rms':
            cmap = None
        else:
            cmap = 'RdBu_r'
        if kwargs[maptype].ndim == 1:
            hp.mollview(kwargs[maptype], cmap=cmap, title=f"{mapdesc}, det {detector}, chain {chain}, iter {iteration}", min=np.percentile(kwargs[maptype], 2), max=np.percentile(kwargs[maptype], 98))
            plt.savefig(params.output_paths.plots + f"maps_data/{maptype}_det{detector}_chain{chain}_iter{iteration}.png", bbox_inches='tight')
            plt.close()
        elif kwargs[maptype].ndim == 2:
            if kwargs[maptype].shape[0] == 3:
                plt.figure(figsize=(8.5*3, 5.4))
                labs = ["I", "Q", "U"]
                for i in range(3):
                    if maptype == 'map_rms':
                        limup   = None
                        limdown = None
                    else:
                        limup   = 2*kwargs[maptype][i].std()
                        limdown = -2*kwargs[maptype][i].std()
                    hp.mollview(kwargs[maptype][i], cmap=cmap, title=labs[i],
                            sub=(1,3,i+1), min=limdown, max=limup)
                plt.suptitle(f"{mapdesc}, det {detector}, chain {chain}, iter {iteration}")
                plt.savefig(params.output_paths.plots + f"maps_data/{maptype}_IQU_det{detector}_chain{chain}_iter{iteration}.png", bbox_inches='tight')
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


def plot_components(params: Bunch, freq: float, detector: int, chain: int,
                    iteration: int, signal_map: NDArray, components_list: list[DiffuseComponent], nside):
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

    ells = np.arange(3 * nside)
    Z = ells * (ells+1) / (2 * np.pi)
    for component in components_list:
        smoothing_scale_radians = component.params.smoothing_scale*np.pi/(180*60)
        comp_map = component.get_sky(freq, nside, fwhm=smoothing_scale_radians)
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
