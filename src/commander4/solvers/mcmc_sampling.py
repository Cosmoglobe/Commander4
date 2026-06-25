from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from mpi4py import MPI

from commander4.data_models.detector_map import DetectorMap
from commander4.sky_models.component import CompList
from commander4.sky_models.sky_model import SkyModel

logger = logging.getLogger(__name__)


class MCMCSamplingGroup(ABC):
    """General Metropolis-Hastings sampling group -- the non-linear tier of the compsep Gibbs step.

    This base class owns everything that does *not* depend on which parameters are being sampled:

    * the MH step loop and accept/reject decision,
    * the full-sky chi-squared log-likelihood (an allreduce over every participating band),
    * the MPI lock-step (proposal and accept/reject are drawn on ``root`` and broadcast so all
      CompSep ranks stay synchronized),
    * buffering and reverting all component amplitudes around the coupled amplitude re-solve.

    A concrete subclass supplies *only* the parameter-specific behaviour through the abstract hooks
    (`has_parameters`, `capture_state`, `propose`, `apply_state`). This deliberately decouples the
    machinery from the parameter set, so very different parameters -- scalar spectral indices,
    tabulated-SED bins, template scalings, ... -- can all reuse one implementation instead of the
    near-duplicated routines C3 carries (`sample_specind_mh`, `sample_mbbtab_mh`,
    `sample_template_mh`).

    **The parameter "state" is opaque to this base class.** ``capture_state`` produces it, ``propose``
    maps one to a candidate, ``apply_state`` writes it back; this class never inspects it. It must be

    * an independent **snapshot** (so that ``apply_state(previous_state)`` on rejection fully
      restores the model, even after ``apply_state(proposed_state)`` mutated it), and
    * **MPI-picklable** (it is broadcast from ``root`` to every rank).

    A plain ``dict[str, float]`` satisfies both for every current case, but any object that round-trips
    through the three hooks is allowed (e.g. arrays for spatially-varying parameters).

    **Collective contract.** ``run`` and the hooks below are called by *all* CompSep ranks.
    ``capture_state`` and ``apply_state`` run on every rank and must be deterministic (the replicated
    ``comp_list`` is identical everywhere, and the applied state comes from a broadcast), so the model
    stays globally consistent. Only ``propose`` is evaluated on ``root`` alone.
    """

    def __init__(self, compsep_comm: MPI.Comm, detector_data: DetectorMap, comp_list: CompList, *,
                 target_pol: str, chisq_active: bool, root: int = 0):
        """
        Args:
            compsep_comm: Full CompSep communicator (spans all band views and both polarizations).
            detector_data: This rank's band map (one ``(band, pol)`` execution view).
            comp_list: Globally-replicated full component list, updated in place.
            target_pol: This rank's polarization stream, ``"I"`` or ``"QU"``.
            chisq_active: Whether this rank's band contributes to the chi-squared likelihood.
            root: Rank (in ``compsep_comm``) that draws proposals and the accept/reject decision.
        """
        self.comm = compsep_comm
        self.detector_data = detector_data
        self.comp_list = comp_list
        self.target_pol = target_pol
        self.chisq_active = chisq_active
        self.root = root
        self.is_root = compsep_comm.Get_rank() == root

    # ------------------------------------------------------------------ #
    # Parameter-specific hooks (must be implemented by subclasses).
    # ------------------------------------------------------------------ #
    @abstractmethod
    def has_parameters(self) -> bool:
        """Whether this group has any parameters to sample (else ``run`` is a no-op)."""

    @abstractmethod
    def capture_state(self):
        """Return an independent snapshot of the current parameter values, read from the model.

        Called on every rank; must be deterministic and not alias mutable model state.
        """

    @abstractmethod
    def propose(self, current_state) -> tuple[object, bool]:
        """Draw a candidate state from ``current_state`` (called on ``root`` only).

        Returns ``(proposed_state, in_bounds)``. ``in_bounds=False`` makes ``run`` reject the step
        immediately, without applying the state or evaluating the likelihood.
        """

    @abstractmethod
    def apply_state(self, state) -> None:
        """Write ``state`` into the model, recomputing any derived quantities it implies.

        Called on every rank with a broadcast (hence identical) state; must be deterministic.
        """

    def log_prior(self, state) -> float:
        """Log prior density of ``state`` (evaluated on ``root`` only); default 0 = flat prior.

        Override to add a soft prior on the sampled parameters. It enters the acceptance ratio
        next to the likelihood; the proposal is a symmetric random walk so there is no Hastings term,
        and any state-independent normalization cancels between the proposed and current states.
        """
        return 0.0

    # ------------------------------------------------------------------ #
    # General likelihood (override for ridge/marginal/... variants).
    # ------------------------------------------------------------------ #
    def local_loglike(self) -> float:
        """Whitened-residual log-likelihood of this band against the *full* current sky model.

        Mirrors C3's ``compute_chisq`` for a single band: the model is realized at this band's data
        resolution (``detector_data.fwhm_rad``) and compared to the band map, whitened by the band
        RMS, giving ``-chi2/2``. Each component removes its own ``amp_fwhm_rad`` so deconvolved (CG)
        and data-resolution (per-pixel) amplitudes both land at the band resolution. Uses *all*
        components in this rank's polarization stream, so the components a group holds fixed still
        enter the residual -- this is the conditional likelihood of the MH move.
        """
        band_pol = "QU" if self.detector_data.pol else "I"
        model_sky = SkyModel(self.comp_list.split_for_eval_pol(self.target_pol)).get_sky_at_nu(
            self.detector_data.nu, self.detector_data.nside, band_pol,
            fwhm=self.detector_data.fwhm_rad)
        whitened_residual = (self.detector_data.map_sky - model_sky) / self.detector_data.map_rms
        return -0.5 * float(np.sum(whitened_residual**2))

    def global_loglike(self) -> float:
        """Full-sky log-likelihood: each band's ``local_loglike`` summed over all bands."""
        local = self.local_loglike() if self.chisq_active else 0.0
        return float(self.comm.allreduce(local, op=MPI.SUM))

    # ------------------------------------------------------------------ #
    # Template method: the MH loop, identical for every parameter set.
    # ------------------------------------------------------------------ #
    def run(self, *, numstep: int, resolve_amplitudes: Callable[[], None]) -> None:
        """Take ``numstep`` joint MH steps, each coupled to an amplitude re-solve.

        Called collectively by *all* CompSep ranks. ``resolve_amplitudes`` is a collective callback
        that re-solves the amplitude (CG) sampling groups this MCMC group is coupled to (C3's
        ``UPDATE_CG_GROUPS``); it runs after each in-bounds proposal so the likelihood sees
        amplitudes conditioned on the proposed parameters. On rejection, both the parameters and all
        component amplitudes (which ``resolve_amplitudes`` may have changed) are reverted.
        NB: It's currently assumed that the `resolve_amplitudes` callback ONLY touches the component
        amplitudes, which are stored in `comp._data`.
        """
        if not self.has_parameters():
            if self.is_root:
                logger.info("MCMC group has nothing to sample; skipping.")
            return

        n_accept = 0
        for step in range(numstep):
            # Buffer every component's amplitudes (both polarizations) so a rejected step can be
            # undone even though `resolve_amplitudes` re-solves an arbitrary subset of them.
            amp_buffer = [comp._data.copy() for comp in self.comp_list]
            current_state = self.capture_state()
            loglike_current = self.global_loglike()

            if self.is_root:
                proposed_state, in_bounds = self.propose(current_state)
            else:
                proposed_state, in_bounds = None, None
            proposed_state = self.comm.bcast(proposed_state, root=self.root)
            in_bounds = self.comm.bcast(in_bounds, root=self.root)

            if not in_bounds:
                if self.is_root:
                    logger.info(f"MCMC step {step+1}/{numstep}: proposal out of bounds, rejected "
                                f"({proposed_state}).")
                continue

            self.apply_state(proposed_state)
            resolve_amplitudes()  # Collective: re-solve the coupled CG amplitude groups.
            loglike_proposed = self.global_loglike()
            delta_loglike = loglike_proposed - loglike_current

            # Symmetric proposal, so the log acceptance ratio is the likelihood ratio plus the prior
            # ratio (log_prior defaults to 0, i.e. a flat prior).
            if self.is_root:
                log_accept = (delta_loglike
                              + self.log_prior(proposed_state) - self.log_prior(current_state))
                accept = log_accept >= 0.0 or np.log(np.random.random()) < log_accept
            else:
                accept = None
            accept = self.comm.bcast(accept, root=self.root)

            if self.is_root:
                logger.info(f"MCMC step {step+1}/{numstep}: dloglike={delta_loglike:.3e} "\
                            f"[{'ACCEPT' if accept else 'reject'}] ({current_state} -> "\
                            f"{proposed_state}).")

            if accept:
                n_accept += 1
            else:
                self.apply_state(current_state)
                for comp, saved in zip(self.comp_list, amp_buffer):
                    # Copies the previous amplitude data back into `comp`, as the step was rejected.
                    np.copyto(comp._data, saved)

        if self.is_root:
            logger.info(f"MCMC group finished: {n_accept}/{numstep} proposals accepted.")
