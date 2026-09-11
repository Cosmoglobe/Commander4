"""Sampling of harmonic space components.
"""
import logging
import numpy as np
from numpy.typing import NDArray

from mpi4py import MPI

from commander4.sky.comp_list import CompList

logger = logging.getLogger(__name__)


class ClSamplingGroup:
    def __init__(self, config: "ClSamplingGroupConfig", comp_list: CompList):
        self.config = config
        self.comp_list = comp_list

    def run(self, current_state: dict) -> tuple[dict[str, NDArray], bool]:
        logger.verbose("Drawing C_ell proposal.")
        proposal = {}
        components = {comp.name: comp for comp in self.comp_list}

        for group in self.config.comps:
            sigma_l = components[group]
            logger.verbose(f"Sigma_ell = [{sigma_l.shape}] {sigma_l}")

            # P(C_l | s) = C_l^(-(2l + 1) / 2) exp(- (2l + 1) sigma_l / 2 C_l)
            l = np.arange(2, len(sigma_l))
            x = np.random.chisquare(df=2.*l-1.)
            cl = np.zeros_like(sigma_l)
            cl[l] = (2. * l + 1.) * sigma_l[l] / x
            proposal[group] = cl
            logger.verbose(f"C_ell[{group}] = {cl}")

        for group in components:
            components[group].Cl_sample = proposal[group]

        return proposal
