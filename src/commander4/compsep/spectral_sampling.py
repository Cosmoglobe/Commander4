"""Metropolis sampling of the spectral indices (`beta`) of one MCMC sampling group.

The lines marked "beta" read or set `beta` on the components.
"""
import logging
from dataclasses import dataclass

import healpy as hp
import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.compsep.amplitude_sampling import AmplitudeGroup, sample_amplitudes
from commander4.data_models.detector_map import DetectorMap
from commander4.diagnostics.performance import benchmark
from commander4.parameters.schema import names_or_all, read_settings
from commander4.mpi.setup import MPIInfo
from commander4.sky.comp_list import CompList

logger = logging.getLogger(__name__)


@dataclass
class MCMCGroup:
    """One MCMC sampling group of the parameter file, set up for this CompSep rank."""
    name: str
    cfg: Bunch                             # The group's block of the parameter file, with defaults
                                           # and "all" replaced by the names.
    chisq_pixels: NDArray[np.bool_] | slice  # The pixels of this rank's band in the chi-squared.
    update_amplitude_groups: list[AmplitudeGroup]  # The amplitude groups it re-solves when sampled.

    # The settings of a group in `compsep.mcmc_sampling_groups`.
    DEFAULTS = {"enabled": True,
                "sample": None,                   # Component -> parameter -> settings. Required.
                "update_amplitude_groups": [],    # Amplitude groups re-solved at each step.
                "numstep": 1,                     # Metropolis steps per Gibbs iteration.
                "chisq_bands": None,              # Bands of the chi-squared; None/"all" -> all.
                "chisq_mask": None}               # Mask file of the chi-squared; None = no mask.
    # The parameters of one component in `sample`. Only beta can be sampled yet.
    COMP_DEFAULTS = {"beta": None}
    # The settings of one sampled parameter.
    PARAM_DEFAULTS = {"proposal_sigma": None,     # Step length of the proposal. Required.
                      "bounds": None,             # Hard bounds [lowest, highest]; None = none.
                      "prior_mean": None,         # Mean and rms of a Gaussian prior; None = no
                      "prior_rms": None}          # prior. Set both or neither.

    @classmethod
    def init_from_params(cls, params: Bunch, name: str, amplitude_groups: list[AmplitudeGroup],
                         mpi_info: MPIInfo | None = None) -> "MCMCGroup":
        """Read and check the group `compsep.mcmc_sampling_groups.<name>`.

        `amplitude_groups` are the enabled amplitude groups, which `update_amplitude_groups` may
        name. Every CompSep rank must build every group, because the chi-squared mask is read on
        the master and broadcast. Without `mpi_info` the group is only read and checked, and
        `chisq_pixels` keeps every pixel.
        """
        section = f"mcmc_sampling_groups.{name}"
        cfg = read_settings(params, name, ("compsep.mcmc_sampling_groups",), cls.DEFAULTS)
        comp_names = [comp for comp in params.components if params.components[comp].enabled]
        band_names = [band for band in params.compsep.bands if params.compsep.bands[band].enabled]
        # `sample` with the defaults filled in. A new dict, so the parameter file is left as it is.
        if not cfg.sample:
            raise ValueError(f"'{section}.sample' must name the components to sample.")
        sample = {}
        for comp_name in names_or_all(list(cfg.sample), comp_names, f"{section}.sample"):
            location = f"{section}.sample.{comp_name}"
            sample[comp_name] = read_settings(params, comp_name, (f"compsep.{section}.sample",),
                                              cls.COMP_DEFAULTS)
            if sample[comp_name].beta is None:
                raise ValueError(f"'{location}' must name a parameter to sample: 'beta'.")
            settings = read_settings(params, "beta", (f"compsep.{location}",), cls.PARAM_DEFAULTS)
            if settings.proposal_sigma is None:
                raise ValueError(f"'{location}.beta.proposal_sigma' must be set.")
            # A prior with only one of its two numbers would be left out without a message.
            if (settings.prior_mean is None) != (settings.prior_rms is None):
                raise ValueError(f"'{location}.beta' must set both 'prior_mean' and 'prior_rms', "
                                 "or neither.")
            sample[comp_name].beta = settings
        cfg.sample = sample
        cfg.chisq_bands = names_or_all(cfg.chisq_bands, band_names, f"{section}.chisq_bands")
        enabled_groups = {group.name: group for group in amplitude_groups}
        unknown = sorted(set(cfg.update_amplitude_groups) - set(enabled_groups))
        if unknown:
            raise ValueError(f"'{section}.update_amplitude_groups' names unknown or disabled "
                             f"amplitude group(s) {unknown}. The enabled amplitude groups are "
                             f"{list(enabled_groups)}.")
        update_amplitude_groups = [enabled_groups[n] for n in cfg.update_amplitude_groups]
        # The band map arrives at the band's `eval_nside` (TOD bands: in the experiment block; map
        # bands: in the compsep block).
        chisq_pixels = slice(None)
        if mpi_info is not None and cfg.chisq_mask:
            band = params.compsep.bands[mpi_info.compsep.band_name]
            if band.get_from != "file":
                band = params.experiments[band.get_from].bands[mpi_info.compsep.band_name]
            chisq_pixels = _read_chisq_mask(mpi_info.compsep.comm, cfg.chisq_mask, band.eval_nside)
        return cls(name=name, cfg=cfg, chisq_pixels=chisq_pixels,
                   update_amplitude_groups=update_amplitude_groups)


def _read_chisq_mask(compsep_comm: MPI.Comm, path: str, nside: int) -> NDArray[np.bool_]:
    """The pixels at `nside` that an MCMC group's chi-squared mask keeps (True = kept).

    The mask (C3's `MCMC_SAMPLING_GROUP_CHISQ_MASK`) enters only the MCMC chi-squared, not the
    amplitude solve. The CompSep master reads field 0 of the file and broadcasts it, so one mask
    applies to Q and U alike. It is brought to `nside` and thresholded at 0.5, so a coarse pixel is
    kept only if more than half of it was.
    """
    mask = None
    if compsep_comm.Get_rank() == 0:
        mask = hp.read_map(path, field=0, dtype=np.float64)
        logger.info(f"CompSep: read chisq mask {path} (fsky = {np.mean(mask > 0.5):.3f}).")
    mask = compsep_comm.bcast(mask, root=0)
    if hp.npix2nside(mask.size) != nside:
        mask = hp.ud_grade(mask, nside)
    return mask > 0.5


@benchmark("mcmc-loglike")
def chi2_of_chisq_bands(mpi_info: MPIInfo, group: MCMCGroup, band_map: DetectorMap,
                        comp_list: CompList) -> float:
    """The chi-squared that an MCMC group judges a proposal by. Every CompSep rank must call it.

    The sum of ((data - model)/sigma)^2 over the bands in `group.cfg.chisq_bands` and the pixels in
    `group.chisq_pixels`. The model is the sky of all components, also those the group does not
    sample.
    """
    chi2 = 0.0
    if mpi_info.compsep.band_name in group.cfg.chisq_bands:
        model_sky = comp_list.get_sky_at_band(band_map)
        pixels = group.chisq_pixels
        residual = band_map.map_sky[:, pixels] - model_sky[:, pixels]
        # An unobserved pixel has zero inverse variance, and adds nothing.
        whitened_residual = residual*np.sqrt(band_map.map_inv_var[:, pixels])
        chi2 = float(np.sum(whitened_residual**2))
    return float(mpi_info.compsep.comm.allreduce(chi2, op=MPI.SUM))


def sample_spectral_params(mpi_info: MPIInfo, group: MCMCGroup, band_map: DetectorMap,
                           comp_list: CompList, rng: np.random.Generator) -> dict:
    """Metropolis steps on the spectral indices of one MCMC group. Every CompSep rank must call it.

    Each of the `numstep` steps proposes a new `beta` for all components of the group at once,
    re-solves the amplitude groups in `update_amplitude_groups` (C3's `UPDATE_CG_GROUPS`), and
    accepts or rejects the step as a whole. A rejection restores the betas and all amplitudes; an
    acceptance keeps the re-solved amplitudes. `comp_list` is updated in place.

    Returns:
        The acceptance record for the chain file: the number of steps taken and accepted, and the
        final beta of each sampled component, by its name.
    """
    compsep = mpi_info.compsep
    # The component objects of each sampled component: its I view and its QU view, which share
    # one beta.
    views = {name: [comp for comp in comp_list if comp.comp_name == name]
             for name in group.cfg.sample}

    n_accept = 0
    for step in range(group.cfg.numstep):
        # The state before the step, for a rejection. All amplitudes are saved, because the
        # re-solve changes those of other components too. It must change nothing but amplitudes,
        # or a rejection would not undo it.
        old_betas = {name: float(views[name][0].beta) for name in group.cfg.sample}  # beta
        old_amps = [comp.amps.copy() for comp in comp_list]
        old_chi2 = chi2_of_chisq_bands(mpi_info, group, band_map, comp_list)

        # The master proposes a Gaussian step from the current value of each beta. A step that
        # leaves the hard bounds of any beta is rejected at once, without a solve.
        new_betas, in_bounds = None, True
        if compsep.rank == 0:
            new_betas = {name: float(rng.normal(old_betas[name], settings.beta.proposal_sigma))
                         for name, settings in group.cfg.sample.items()}
            for name, settings in group.cfg.sample.items():
                bounds = settings.beta.bounds
                if bounds is not None and not bounds[0] <= new_betas[name] <= bounds[1]:
                    in_bounds = False
        new_betas, in_bounds = compsep.comm.bcast((new_betas, in_bounds), root=0)
        if not in_bounds:
            if compsep.rank == 0:
                logger.verbose(f"MCMC step {step+1}/{group.cfg.numstep}: proposal out of bounds, "
                               f"rejected ({new_betas}).")
            continue

        for name in group.cfg.sample:
            for comp in views[name]:
                comp.beta = new_betas[name]  # beta
        with benchmark("mcmc-amp-resolve"):
            for amplitude_group in group.update_amplitude_groups:
                sample_amplitudes(mpi_info, amplitude_group, band_map, comp_list, rng)
        new_chi2 = chi2_of_chisq_bands(mpi_info, group, band_map, comp_list)

        # The Metropolis rule: accept with probability min(1, P_new/P_old). The proposal is
        # symmetric, so log(P_new/P_old) is the difference of -chi2/2 plus the difference of the
        # log priors. A Gaussian log prior is -0.5 ((beta - mean)/rms)^2; its normalization is the
        # same before and after, and drops out.
        accept = None
        if compsep.rank == 0:
            old_log_prior, new_log_prior = 0.0, 0.0
            for name, settings in group.cfg.sample.items():
                prior_mean, prior_rms = settings.beta.prior_mean, settings.beta.prior_rms
                if prior_rms is not None:
                    old_log_prior += -0.5*((old_betas[name] - prior_mean)/prior_rms)**2
                    new_log_prior += -0.5*((new_betas[name] - prior_mean)/prior_rms)**2
            log_accept = -0.5*(new_chi2 - old_chi2) + new_log_prior - old_log_prior
            accept = log_accept >= 0.0 or np.log(rng.random()) < log_accept
            logger.info(f"MCMC step {step+1}/{group.cfg.numstep}: dchi2={new_chi2 - old_chi2:.3e} "
                        f"[{'ACCEPT' if accept else 'reject'}] ({old_betas} -> {new_betas}).")
        accept = compsep.comm.bcast(accept, root=0)

        # An acceptance keeps the amplitudes the step was judged by. C3 (`sample_specind_mh_pol`)
        # re-solves them once more here; C4 does not, on purpose.
        if accept:
            n_accept += 1
        else:
            for name in group.cfg.sample:
                for comp in views[name]:
                    comp.beta = old_betas[name]  # beta
            for comp, saved in zip(comp_list, old_amps):
                np.copyto(comp.amps, saved)

    if compsep.rank == 0:
        logger.info(f"MCMC group {group.name!r} finished: {n_accept}/{group.cfg.numstep} proposals "
                    "accepted.")
    final_betas = {name: float(views[name][0].beta) for name in group.cfg.sample}
    return {"numstep": group.cfg.numstep, "n_accept": n_accept,
            "accept_rate": n_accept/group.cfg.numstep if group.cfg.numstep > 0 else float("nan"),
            "params": final_betas}
