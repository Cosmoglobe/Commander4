"""Sampling of harmonic space components.
"""
from abc import ABC
import logging
import numpy as np
from numpy.typing import NDArray

from mpi4py import MPI

from commander4.data_models.detector_map import DetectorMap
from commander4.sky.comp_list import CompList
from commander4.compsep.mcmc import MCMCSamplingGroup

logger = logging.getLogger(__name__)

class ClSamplingGroup(MCMCSamplingGroup):
    def __init__(self, compsep_comm: MPI.Comm, det_map: DetectorMap, comp_list: CompList, *,
                 config: "ClSamplingGroupConfig", target_pol: str, chisq_active: bool,
                 selected_comps: list[str] | None, chisq_mask: NDArray | None = None, root: int = 0):
        super().__init__(compsep_comm, detector_data, comp_list, target_pol=target_pol,
                         chisq_active=chisq_active, chisq_mask=chisq_mask, root=root)

        self.config = config

    def propose(self, current_state) -> tuple[dict[str, NDArray], bool]:
        logger.verbose("Drawing C_ell proposal.")
        proposal = {}
        for group in self.config.groups:
            sigma_l = self.comp_list[group].sigma_l

            # P(C_l | s) = C_l^(-(2l + 1) / 2) exp(- (2l + 1) sigma_l / 2 C_l)
            l = np.arange(len(sigma_l))
            x = np.random.chisquare(df=2.*l-1.)
            cl = (2. * l + 1.) * sigma_l / x
            proposal[group] = cl
            logger.verbose(f"C_ell[{group}] = {cl}")
        return proposal, True

    def apply_state(self, state) -> None:
        for group in self.config.groups:
            self.comp_list[group].Cl_sample = group
