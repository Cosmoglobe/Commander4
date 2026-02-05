import healpy as hp
import matplotlib.pyplot as plt
import os
import numpy as np
from numpy.typing import NDArray
from pixell.bunch import Bunch

from src.python.sky_models.component import Component, split_complist
from src.python.data_models.detector_map import DetectorMap


def plot_combo_maps(params: Bunch, detector: int, chain: int, iteration: int,
                    comp_list: list[Component], detector_data: DetectorMap):

    out_folder = os.path.join(params.output_paths.plots, "combo_maps")
    os.makedirs(out_folder, exist_ok=True)

    nside = detector_data.nside
    npix = 12*nside**2
    pol = detector_data.pol

    print("Plotting combo maps with pol ", pol)
    for ipol in range(detector_data.npol):
        map_signal = detector_data.map_sky[ipol]
        if map_signal is None:
            continue
        map_corrnoise = detector_data.map_corrnoise[ipol] if hasattr(detector_data, "map_corrnoise") else np.zeros((npix,))
        map_rms = detector_data.map_rms[ipol] if hasattr(detector_data, "map_rms") else np.zeros((npix,))
        map_skymodel = detector_data.map_skymodel[ipol] if hasattr(detector_data, "map_skymodel") else np.zeros((npix,))
        map_orbdipole = detector_data.map_orbdipole[ipol] if hasattr(detector_data, "map_orbdipole") else np.zeros((npix,))
        freq = detector_data.nu
        gain = detector_data.gain
        
        map_rawobs = map_signal + map_corrnoise + map_orbdipole
        map_skysub = map_signal + map_corrnoise - map_skymodel

        foreground_subtracted = np.zeros_like(map_signal)
        cmb_subtracted = np.zeros_like(map_signal)
        foreground_subtracted[:] = map_signal
        cmb_subtracted[:] = map_signal
        residual = np.zeros_like(map_signal)
        residual[:] = map_signal

        # print("Comps", [comp.pol for comp in comp_list])

        comp_sublist = split_complist(comp_list, 1 if pol else 0) #pick only the relevant stokes

        # print("Len",len(comp_sublist), len(comp_list))

        fig, ax = plt.subplots(3, 5, figsize=(42, 18))
        fig.suptitle(f"Iter {iteration:04d}. Freq: {freq:.2f} GHz (det {detector}). Chain {chain}. Detector gain = {gain:.4e} (Global gain = {detector_data.g0:.4e}).", fontsize=24)

        for i, component in enumerate(comp_sublist):
            smoothing_scale_radians = component.comp_params.smoothing_scale*np.pi/(180*60)
            if pol:
                if component.pol:
                    comp_map = component.get_sky(freq, nside, fwhm=smoothing_scale_radians)[ipol]
                else:
                    comp_map = np.zeros((npix,))
            else:
                comp_map = component.get_sky(freq, nside, fwhm=smoothing_scale_radians)[0]
            if "cmb" not in component.shortname:
                foreground_subtracted -= comp_map
            else:
                cmb_subtracted -= comp_map
            residual -= comp_map
            plt.axes(ax[2,i])
            hp.mollview(comp_map, hold=True, title=f"{component.longname} at {freq:.2f} GHz, det {detector}, chain {chain}, iter {iteration}", min=np.percentile(comp_map, 1), max=np.percentile(comp_map, 99))

        for component in comp_sublist:
            if "cmb" in component.shortname:
                if ipol > 0:
                    cmb_map_anisotropies = component.get_sky_anisotropies(freq, nside, fwhm=smoothing_scale_radians)[ipol]
                else:
                    cmb_map_anisotropies = component.get_sky_anisotropies(freq, nside, fwhm=smoothing_scale_radians)[0]

        plt.axes(ax[0,0])
        sym_lim = np.percentile(np.abs(map_rawobs), 99)
        hp.mollview(map_rawobs, fig=fig, hold=True, cmap="RdBu_r", title="Raw observed sky", min=-sym_lim, max=sym_lim)
        plt.axes(ax[0,1])
        hp.mollview(cmb_subtracted, fig=fig, hold=True, cmap="RdBu_r", title="CMB subtracted sky", min=np.percentile(foreground_subtracted, 1), max=np.percentile(foreground_subtracted, 99))
        plt.axes(ax[0,2])
        sym_lim = np.percentile(np.abs(foreground_subtracted), 99)
        hp.mollview(foreground_subtracted, fig=fig, hold=True, cmap="RdBu_r", title="Foreground subtracted sky", min=-sym_lim, max=sym_lim)
        plt.axes(ax[0,3])
        sym_lim = np.percentile(np.abs(map_skysub), 99)
        hp.mollview(map_skysub, fig=fig, hold=True, cmap="RdBu_r", title="All sky signals subtracted", min=-sym_lim, max=sym_lim)
        plt.axes(ax[0,4])
        sym_lim = np.percentile(np.abs(cmb_map_anisotropies), 99)
        hp.mollview(cmb_map_anisotropies, fig=fig, hold=True, cmap="RdBu_r", title="CMB anisotropies", min=-sym_lim, max=sym_lim)

        plt.axes(ax[1,0])
        sym_lim = np.percentile(np.abs(map_orbdipole), 99)
        hp.mollview(map_orbdipole, fig=fig, hold=True, cmap="RdBu_r", title="Orbital dipole", min=-sym_lim, max=sym_lim)
        plt.axes(ax[1,1])
        sym_lim = np.percentile(np.abs(map_corrnoise), 99)
        hp.mollview(map_corrnoise, fig=fig, hold=True, cmap="RdBu_r", title="Corr noise", min=-sym_lim, max=sym_lim)
        plt.axes(ax[1,2])
        sym_lim = np.percentile(np.abs(residual), 99)
        hp.mollview(residual, fig=fig, hold=True, cmap="RdBu_r", title="Residual sky", min=-sym_lim, max=sym_lim)
        plt.axes(ax[1,3])
        relative_residual = np.abs(residual/map_rms)
        hp.mollview(relative_residual, fig=fig, hold=True, cmap="RdBu_r", title="Residual/RMS", min=0, max=np.percentile(relative_residual, 99))
        plt.axes(ax[1,4])
        hp.mollview(map_rms, fig=fig, hold=True, norm="log", title="RMS", min=np.min(map_rms), max=np.percentile(map_rms, 99))

        plt.savefig(os.path.join(out_folder, f"{ipol}_combo_map_det{detector}_chain{chain}_iter{iteration}.png"), bbox_inches="tight")
        plt.close()



def plot_data_maps(params, detector, chain, iteration, **kwargs):
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
    out_folder = os.path.join(params.output_paths.plots, "maps_data")
    os.makedirs(out_folder, exist_ok=True)
    for maptype, mapdesc in zip(['map_signal', 'map_corr_noise', 'map_rms', 'map_skysub', 'map_orbdip'],
                                ['Signal map', 'Corr noise map', 'RMS map', 'skysub',     'ordip']):

        if maptype not in kwargs:
            continue
        if maptype == 'map_rms':
            cmap = None
        else:
            cmap = 'RdBu_r'
        # if kwargs[maptype].ndim == 1:
        #     hp.mollview(kwargs[maptype], cmap=cmap, title=f"{mapdesc}, det {detector}, chain {chain}, iter {iteration}", min=np.percentile(kwargs[maptype], 1), max=np.percentile(kwargs[maptype], 99))
        #     plt.savefig(params.output_paths.plots + f"maps_data/{maptype}_det{detector}_chain{chain}_iter{iteration}.png", bbox_inches='tight')
        #     plt.close()
        # elif kwargs[maptype].ndim == 2:
        # if kwargs[maptype].shape[0] == 3:
        plt.figure(figsize=(8.5*3, 5.4))
        labs = ["I", "Q", "U"]
        for i in range(3):
            if kwargs[maptype][i] is not None:
                if maptype == 'map_rms':
                    limup   = np.percentile(kwargs[maptype][i], 99)
                    limdown = np.min(kwargs[maptype][i])
                else:
                    limup   = np.percentile(kwargs[maptype][i], 99)
                    limdown = np.percentile(kwargs[maptype][i], 1)
                hp.mollview(kwargs[maptype][i], cmap=cmap, title=labs[i],
                        sub=(1,3,i+1), min=limdown, max=limup)
        plt.suptitle(f"{mapdesc}, det {detector}, chain {chain}, iter {iteration}")
        plt.savefig(os.path.join(out_folder, f"{maptype}_IQU_det{detector}_chain{chain}_iter{iteration}.png", bbox_inches='tight'))
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
    out_folder = os.path.join(params.output_paths.plots, "CG_res")
    os.makedirs(out_folder, exist_ok=True)
    plt.figure()
    plt.loglog(np.arange(residual.shape[0]), residual)
    plt.axhline(params.CG_err_tol, ls="--", c="k")
    plt.savefig(os.path.join(out_folder, f"CG_res_chain{chain}_iter{iteration}.png"))
    plt.close()


def plot_components(params: Bunch, detector: int, chain: int, iteration: int,
                    components_list: list[Component], detector_data: DetectorMap):
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

    map_comp_out = os.path.join(params.output_paths.plots, "maps_comps/")
    dl_out = os.path.join(params.output_paths.plots, "spectra_comps_Dl/")
    cl_out = os.path.join(params.output_paths.plots, "spectra_comps_Cl/")
    os.makedirs(map_comp_out, exist_ok=True)
    os.makedirs(dl_out, exist_ok=True)
    os.makedirs(cl_out, exist_ok=True)
    
    nside = detector_data.nside
    signal_map = detector_data.map_sky
    freq = detector_data.nu
    npix = 12*nside**2
    pol = detector_data.pol
    for ipol in range(detector_data.npol):
        if signal_map[ipol] is None:
            continue
        foreground_subtracted = np.zeros_like(signal_map[ipol])
        foreground_subtracted[:] = signal_map[ipol]
        residual = np.zeros_like(signal_map[ipol])
        residual[:] = signal_map[ipol]

        ells = np.arange(3 * nside)
        Z = ells * (ells+1) / (2 * np.pi)
        for component in components_list:
            smoothing_scale_radians = component.comp_params.smoothing_scale*np.pi/(180*60)
            if pol:
                if component.pol:
                    comp_map = component.get_sky(freq, nside, fwhm=smoothing_scale_radians)[ipol-1]
                else:
                    comp_map = np.zeros((npix,))
            else:
                comp_map = component.get_sky(freq, nside, fwhm=smoothing_scale_radians)[0]
            if component.shortname != "cmb":
                foreground_subtracted -= comp_map
            residual -= comp_map
            if (comp_map == 0).all():
                continue
            
            hp.mollview(comp_map, title=f"{component.longname} at {freq:.2f} GHz, det {detector}, chain {chain}, iter {iteration}", min=np.percentile(comp_map, 1), max=np.percentile(comp_map, 99))
            plt.savefig(os.path.join(map_comp_out, f"pol{ipol}_{component.shortname}_realization_det{detector}_chain{chain}_iter{iteration}.png"), bbox_inches='tight')
            plt.close()
            
            Cl = hp.alm2cl(hp.map2alm(comp_map))
            plt.figure()
            plt.plot(ells, Z * Cl, label=component.longname)
            plt.xscale("log")
            plt.yscale("log")
            plt.savefig(os.path.join(dl_out,f"pol{ipol}_{component.shortname}_Dl_det{detector}_chain{chain}_iter{iteration}.png"), bbox_inches='tight')
            plt.close()

            plt.figure()
            plt.plot(ells, Cl, label=component.longname)
            plt.xscale("log")
            plt.yscale("log")
            plt.savefig(os.path.join(cl_out, f"pol{ipol}_{component.shortname}_Cl_det{detector}_chain{chain}_iter{iteration}.png"), bbox_inches='tight')
            plt.close()

        hp.mollview(foreground_subtracted, title=f"Foreground subtracted sky at {freq:.2f} GHz, det {detector}, chain {chain}, iter {iteration}", min=np.percentile(foreground_subtracted, 1), max=np.percentile(foreground_subtracted, 99))
        plt.savefig(os.path.join(map_comp_out, f"pol{ipol}_foreground_subtr_det{detector}_chain{chain}_iter{iteration}.png"), bbox_inches="tight")
        plt.close()

        hp.mollview(residual, title=f"Residual sky at {freq:.2f} GHz, det {detector}, chain {chain}, iter {iteration}", min=np.percentile(residual, 1), max=np.percentile(residual, 99))
        plt.savefig(os.path.join(map_comp_out, f"pol{ipol}_residual_det{detector}_chain{chain}_iter{iteration}.png"), bbox_inches="tight")
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
