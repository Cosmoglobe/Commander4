import healpy as hp
import matplotlib.pyplot as plt
import os
import numpy as np


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
    if master and not os.path.isdir(params.output_paths.plots + "maps_data/"):
        os.mkdir(params.output_paths.plots + "maps_data/")
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

    if not os.path.isdir(params.output_paths.plots + "CG_res/"):
        os.mkdir(params.output_paths.plots + "CG_res/")
    plt.figure()
    plt.loglog(np.arange(residual.shape[0]), residual)
    plt.axhline(params.CG_err_tol, ls="--", c="k")
    plt.savefig(params.output_paths.plots + f"CG_res/CG_res_chain{chain}_iter{iteration}.png")
    plt.close()


def plot_components(params, freq, detector, chain, iteration, **kwargs):
    """
    Plots the resulting component maps produced by component separation.

    Can plot the total sky map, the cmb, dust, and synchrotron map. It will
    also plot the total sky map minus the foregrounds, as well as the total map
    minus the foregrounds and the CMB. If the CMB or foregrounds are not
    present, these will still be plotted.
    
    The plots are placed in the [params.output_paths.plots]/maps_comps/ folder,
    which will be created if it does not exist.

    Arguments:
        master (bool): Whether this is the master process
        params (bunch): The parameter bundle, at root level.
        freq (float): The frequency of the detector.
        detector, chain, iteration (integers): The indices representing the
            detector, chain and iteration, respectively.
    Optional named arguments:
        sky (np.array): The full sky map for the detector in question.
        cmb (np.array): The cmb map for the detector in question.
        dust (np.array): The dust map for the detector in question.
        sync (np.array): The synchotron map for the detector in question.
        signal (np.array): The data signal map for the detector in question.
    """

    if not os.path.isdir(params.output_paths.plots + "maps_comps/"):
        os.mkdir(params.output_paths.plots + "maps_comps/")
    
    signal_map = kwargs['signal']
    foreground_subtracted = np.zeros_like(signal_map)
    foreground_subtracted[:] = signal_map
    residual = np.zeros_like(signal_map)
    residual[:] = signal_map
    for maptype, mapdesc in zip(['sky', 'cmb', 'dust', 'sync'],
                                ['Full sky realization',
                                 'CMB realization',
                                 'Dust realization',
                                 'Synchrotron realization']):
        if maptype not in kwargs:
            continue
        if maptype not in ('cmb', 'sky'):
            foreground_subtracted -= kwargs[maptype]
        if maptype != 'sky':
            residual -= kwargs[maptype]
        residual -= kwargs[maptype]
        hp.mollview(kwargs[maptype], title=f"{mapdesc} at {freq:.2f} GHz, det {detector}, chain {chain}, iter {iteration}")
        plt.savefig(params.output_paths.plots + f"maps_comps/{maptype}_realization_det{detector}_chain{chain}_iter{iteration}.png", bbox_inches='tight')
        plt.close()

    hp.mollview(foreground_subtracted, title=f"Foreground subtracted sky at {freq:.2f} GHz, det {detector}, chain {chain}, iter {iteration}")
    plt.savefig(params.output_paths.plots + f"maps_comps/foreground_subtr_det{detector}_chain{chain}_iter{iteration}.png", bbox_inches="tight")
    plt.close()

    hp.mollview(residual, title=f"Residual sky at {freq:.2f} GHz, det {detector}, chain {chain}, iter {iteration}")
    plt.savefig(params.output_paths.plots + f"maps_comps/residual_det{detector}_chain{chain}_iter{iteration}.png", bbox_inches="tight")
    plt.close()

def plot_constrained_cmb_results(master, params, detector, chain, iteration, ells,
                                 CMB_mean_field_map, CMB_fluct_map,
                                 signal_map, true_cl):
    """
    Plots the maps produced by the constrained CMB realization procedure.

    Plots the mean field map, the fluctuation map, the signal map (both in map
    and cl space) and the true CLs.
    
    The plots are placed in the [params.output_paths.plots]/maps_CMB/ folder
    (for the CMB maps) and the [params.output_paths.plots]/plots/ folder (for
    the Cl plots) which will be created if they do not exist.

    Arguments:
        master (bool): Whether this is the master process
        params (bunch): The parameter bundle, at root level.
        detector, chain, iteration (integers): The indices representing the
            detector, chain and iteration, respectively.
        ells (np.array): The l range for which to plot.
        CMB_mean_field_map (np.array): The mean field map.
        CMB_fluct_map (np.array): The fluctuation map map.
        signal_map (np.array): The observed CMB sky.
        true_cl (np.array) (currently): The true CLs.
    """

    if master:
        if not os.path.isdir(params.output_paths.plots + "maps_CMB/"):
            os.mkdir(params.output_paths.plots + "maps_CMB/")
        if not os.path.isdir(params["output_paths"]["plots"] + "plots/"):
            os.mkdir(params["output_paths"]["plots"] + "plots/")

    Z = ells * (ells+1) / (2 * np.pi)

    cmap = "RdBu_r"
    title_base = f"CMB realization chain {chain} iter {iteration}"

    hp.mollview(CMB_mean_field_map, cmap=cmap, title=f"Constrained mean field {title_base}")
    plt.savefig(params.output_paths.plots + f"maps_CMB/CMB_mean_field_chain{chain}_iter{iteration}.png", bbox_inches='tight')
    plt.close()

    hp.mollview(CMB_fluct_map, cmap=cmap, title=f"Constrained fluctuation {title_base}")
    plt.savefig(params.output_paths.plots + f"maps_CMB/CMB_fluct_chain{chain}_iter{iteration}.png", bbox_inches='tight')
    plt.close()

    hp.mollview(CMB_mean_field_map+CMB_fluct_map, cmap=cmap, title=f"Joint constrained {title_base}")
    plt.savefig(params.output_paths.plots + f"maps_CMB/CMB_joint_realization_chain{chain}_iter{iteration}.png", bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(ell, Z*CMB_mean_field_Cl, label="Cl CMB mean field")
    plt.plot(ell, Z*CMB_fluct_Cl, label="Cl CMB fluct")
    plt.plot(ell, Z*CMB_mean_field_Cl + CMB_fluct_Cl, label="Cl CMB joint")
    plt.plot(ell, Z*hp.alm2cl(hp.map2alm(signal_map)), label="CL observed sky")
    plt.plot(ell, Z*true_cl, label="True CMB Cl", c="k")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-2, 1e6)
    plt.savefig(params.output_paths.plots + f"plots/Cl_CMB_chain{chain}_iter{iteration}.png", bbox_inches="tight")
