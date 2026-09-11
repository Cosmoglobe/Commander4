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

    def run(self) -> tuple[dict[str, NDArray], bool]:
        logger.verbose("Drawing Cl samples.")
        proposal = {}
        components = self.comp_list.joined()

        for group in self.config.comps:
            component = components[group.lower()]
            sigma_l = component.sigma_l

            # P(C_l | s) = C_l^(-(2l + 1) / 2) exp(- (2l + 1) sigma_l / 2 C_l)
            l = np.arange(2, sigma_l.shape[1])
            cl = np.zeros_like(sigma_l)
            
            for i in range(sigma_l.shape[0]):
                x = np.random.chisquare(df=2.*l-1.)
                cl[i,l] = (2. * l + 1.) * sigma_l[i, l] / x
            proposal[group] = cl

        return proposal
