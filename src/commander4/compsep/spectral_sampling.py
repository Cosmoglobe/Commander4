"""Metropolis sampling of the spectral indices (`beta`) of one MCMC sampling group.

`sample_spectral_params` is the whole sampler: the proposal, the amplitude re-solve, the
chi-squared and the accept/reject decision of each step.

C3 has four more samplers with this same loop (band gains, template scales, tabulated SEDs; see
`comm_mh_specind_mod.f90`). They differ only in which numbers are stepped. To add one of them
here, make the lines marked "beta" below general. Do not copy the loop.
"""
import logging

import numpy as np
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.compsep.amplitude_sampling import sample_amplitudes
from commander4.data_models.detector_map import DetectorMap
from commander4.diagnostics.performance import benchmark
from commander4.sky.comp_list import CompList

logger = logging.getLogger(__name__)


def chi2_of_chisq_bands(mpi_info: Bunch, group: Bunch, band_map: DetectorMap,
                        comp_list: CompList) -> float:
    """The chi-squared that an MCMC group judges a proposal by. Every CompSep rank calls it.

    It is the sum of ((data - model)/sigma)^2 over the bands in `group.chisq_bands`, as C3's
    ``compute_chisq``. The model is the sky of all components in `comp_list`, so the components
    that the group does not sample enter it too. Only the pixels in `group.chisq_pixels` count
    (those the group's `chisq_mask` keeps, set at start-up).
    """
    band = mpi_info.band
    chi2 = 0.0
    if band.name in group.chisq_bands:
        model_sky = comp_list.get_sky_at_band(band_map)
        pixels = group.chisq_pixels
        residual = band_map.map_sky[:, pixels] - model_sky[:, pixels]
        # An unobserved pixel has zero inverse variance, and adds nothing.
        whitened_residual = residual*np.sqrt(band_map.map_inv_var[:, pixels])
        chi2 = float(np.sum(whitened_residual**2))
    return float(mpi_info.compsep.comm.allreduce(chi2, op=MPI.SUM))


def sample_spectral_params(mpi_info: Bunch, cfg: Bunch, group: Bunch, band_map: DetectorMap,
                           comp_list: CompList, rng: np.random.Generator) -> dict:
    """Metropolis steps on the spectral indices of one MCMC group. Every CompSep rank calls it.

    Each of the `group.numstep` steps proposes a new `beta` for all components of the group at
    once, re-solves the amplitude groups in `group.update_amplitude_groups` (C3's
    ``UPDATE_CG_GROUPS``), and accepts or rejects the step as a whole. A rejected step restores
    the betas and all amplitudes. An accepted step keeps the re-solved amplitudes. (C3's
    ``sample_specind_mh_pol`` solves them once more there, to get more CG iterations; here that is
    left out on purpose.)

    Args:
        mpi_info: The MPI data of this rank (`compsep` and `band` are used).
        cfg: The settings of component separation, from `init_compsep_processing`.
        group: The settings of the group, one of `cfg.mcmc_sampling_groups`. `group.beta` is the
            `beta` block of the parameter file: it maps the name of each sampled component to its
            `proposal_sigma`, its hard `bounds`, and its Gaussian prior `prior_mean` and
            `prior_rms` (None for no bounds or no prior).
        band_map: This rank's band map.
        comp_list: The full component list of the chain, updated in place.
        rng: This rank's random generator. The master draws the proposals and the decisions from
            it, and the amplitude re-solves draw from it too.

    Returns:
        The acceptance record for the chain file (C3's ``nonlin-samples_*.dat``): the number of
        steps taken and accepted, and the final beta of each sampled component, by its shortname.
    """
    compsep = mpi_info.compsep
    # The component objects of each sampled component: its I view and its QU view, which share
    # one beta.
    views = {name: [comp for comp in comp_list if comp.comp_name == name] for name in group.beta}

    n_accept = 0
    for step in range(group.numstep):
        # The state before the step, for a rejection. All amplitudes are saved, because the
        # re-solve below changes those of other components too. It must change nothing else.
        old_betas = {name: float(views[name][0].beta) for name in group.beta}  # beta
        old_amps = [comp.amps.copy() for comp in comp_list]
        with benchmark("mcmc-loglike"):
            old_chi2 = chi2_of_chisq_bands(mpi_info, group, band_map, comp_list)

        # The master proposes a Gaussian step from the current value of each beta. A step that
        # leaves the hard bounds of any beta is rejected at once, without a solve (C3's p_uni).
        new_betas, in_bounds = None, True
        if compsep.is_master:
            new_betas = {name: float(rng.normal(old_betas[name], group.beta[name].proposal_sigma))
                         for name in group.beta}
            for name in group.beta:
                bounds = group.beta[name].bounds
                if bounds is not None and not bounds[0] <= new_betas[name] <= bounds[1]:
                    in_bounds = False
        new_betas, in_bounds = compsep.comm.bcast((new_betas, in_bounds), root=0)
        if not in_bounds:
            if compsep.is_master:
                logger.verbose(f"MCMC step {step+1}/{group.numstep}: proposal out of bounds, "
                               f"rejected ({new_betas}).")
            continue

        for name in group.beta:
            for comp in views[name]:
                comp.beta = new_betas[name]  # beta
        with benchmark("mcmc-amp-resolve"):
            for group_name in group.update_amplitude_groups:
                sample_amplitudes(mpi_info, cfg, cfg.amplitude_groups[group_name], band_map,
                                  comp_list, rng)
        with benchmark("mcmc-loglike"):
            new_chi2 = chi2_of_chisq_bands(mpi_info, group, band_map, comp_list)

        # The Metropolis rule: accept with probability min(1, P_new/P_old). The proposal is
        # symmetric, so log(P_new/P_old) is the difference of -chi2/2 plus the difference of the
        # log priors. A Gaussian prior is -0.5 ((beta - mean)/rms)^2 (C3's p_gauss); its
        # normalization is the same before and after, and drops out.
        accept = None
        if compsep.is_master:
            old_log_prior, new_log_prior = 0.0, 0.0
            for name in group.beta:
                prior_mean, prior_rms = group.beta[name].prior_mean, group.beta[name].prior_rms
                if prior_rms is not None:
                    old_log_prior += -0.5*((old_betas[name] - prior_mean)/prior_rms)**2
                    new_log_prior += -0.5*((new_betas[name] - prior_mean)/prior_rms)**2
            log_accept = -0.5*(new_chi2 - old_chi2) + new_log_prior - old_log_prior
            accept = log_accept >= 0.0 or np.log(rng.random()) < log_accept
            logger.info(f"MCMC step {step+1}/{group.numstep}: dchi2={new_chi2 - old_chi2:.3e} "
                        f"[{'ACCEPT' if accept else 'reject'}] ({old_betas} -> {new_betas}).")
        accept = compsep.comm.bcast(accept, root=0)

        if accept:
            n_accept += 1
        else:
            for name in group.beta:
                for comp in views[name]:
                    comp.beta = old_betas[name]  # beta
            for comp, saved in zip(comp_list, old_amps):
                np.copyto(comp.amps, saved)

    if compsep.is_master:
        logger.info(f"MCMC group {group.name!r} finished: {n_accept}/{group.numstep} proposals "
                    "accepted.")
    final_betas = {views[name][0].shortname: float(views[name][0].beta) for name in group.beta}
    return {"numstep": group.numstep, "n_accept": n_accept,
            "accept_rate": n_accept/group.numstep if group.numstep > 0 else float("nan"),
            "params": final_betas}
