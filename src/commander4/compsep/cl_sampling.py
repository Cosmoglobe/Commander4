"""Sampling of harmonic space components.
"""
from abc import ABC
import numpy as np

from mpi4py import MPI

from commander4.data_models.detector_map import DetectorMap
from commander4.sky.comp_list import CompList
from commander4.compsep.mcmc import MCMCSamplingGroup

class ClSamplingGroup(MCMCSamplingGroup):
    def __init__(self, compsep_comm: MPI.Comm, det_map: DetectorMap, comp_list: CompList, *,
                 config: "ClSamplingGroupConfig", target_pol: str, chisq_active: bool,
                 selected_comps: list[str] | None, chisq_mask: NDArray | None = None, root: int = 0):
        super().__init__(compsep_comm, detector_data, comp_list, target_pol=target_pol,
                         chisq_active=chisq_active, chisq_mask=chisq_mask, root=root)

        self.config = config

    def propose(self, current_state) -> tuple[object, bool]:
        sigma_l = self.comp_list["cmb"].sigma_l

        # P(C_l | s) = C_l^(-(2l + 1) / 2) exp(- (2l + 1) sigma_l / 2 C_l)
        l = np.arange(len(sigma_l))
        x = np.random.chisquare(df=2.*l-1.)
        cl = (2. * l + 1.) * sigma_l / x
        return cl, True

    def apply_state(self, state) -> None:
        self.comp_list["cmb"].Cl_sample = state
