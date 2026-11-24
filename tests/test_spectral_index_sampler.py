"""Tests of `sample_spectral_params`, the Metropolis sampler of the spectral indices.

Each test runs the real sampler on one rank, with one synchrotron component (an I and a QU view)
and one intensity band. Only the random generator is a stand-in, so that a test chooses the
proposals and the uniform draw of the accept/reject decision.
"""
import types

import numpy as np
import pytest
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.compsep.amplitude_sampling import AmplitudeGroup
from commander4.compsep.spectral_sampling import MCMCGroup, sample_spectral_params
from commander4.data_models.band_map import BandMap
from commander4.sky.comp_list import CompList

NSIDE, LMAX, NU, TRUE_BETA = 4, 8, 60.0, -3.0


def _fixed_rng(proposals: list[float]) -> types.SimpleNamespace:
    """A stand-in generator: the proposals come in the given order, the uniform draw is 0.5.

    The per-pixel amplitude solver draws its fluctuations from a real generator.
    """
    proposals = iter(proposals)
    return types.SimpleNamespace(normal=lambda loc, scale: next(proposals), random=lambda: 0.5,
                                 standard_normal=np.random.default_rng(0).standard_normal)


def _make_setup(start_beta: float = TRUE_BETA, **beta_settings):
    """One synchrotron component, and a band map that it fits exactly at `TRUE_BETA`.

    Returns `(mpi_info, params, group, amp_group, band_map, comp_list)`. The MCMC group samples
    the component's beta with a step of 0.1, and with `beta_settings` on top. The amplitudes stay
    fixed, unless a test puts the per-pixel amplitude group `amp_group` in
    `group.update_amplitude_groups`.
    """
    sync = Bunch(enabled=True, component_class="Synchrotron",
                 params=Bunch(lmax=LMAX, polarization="IQU", nu_ref=30.0,
                              Cl_prior_amplitude=None, beta=TRUE_BETA))
    params = Bunch(components=Bunch(sync=sync), compsep=Bunch(
        double_precision=True, bands=Bunch(Band60=Bunch(enabled=True, polarization="I")),
        per_pixel_sampling_groups=Bunch(amps=Bunch(comps=["sync"])),
        mcmc_sampling_groups=Bunch(betas=Bunch(
            sample=Bunch(sync=Bunch(beta=Bunch(proposal_sigma=0.1, **beta_settings))),
            chisq_bands=["Band60"]))))
    comp_list = CompList.init_from_params(params)
    comp_list[0].amps[:] = 1.0  # The I view. Real alms, so they are valid at m = 0 too.

    npix = 12*NSIDE**2
    band_map = BandMap(np.zeros((1, npix)), np.full((1, npix), 1.0e4), nu=NU, fwhm_arcmin=0.0,
                       nside=NSIDE, lmax=LMAX)
    band_map.map_sky = comp_list.get_sky_at_band(band_map).astype(np.float64)
    for comp in comp_list:
        comp.beta = start_beta

    mpi_info = Bunch(compsep=Bunch(comm=MPI.COMM_SELF, pol_comm=MPI.COMM_SELF, rank=0,
                                   num_threads=1, band_name="Band60", pol="I"))
    # The one rank solves the I view; no rank holds a QU band.
    amp_group = AmplitudeGroup.init_from_params(params, "per_pixel_sampling_groups", "amps",
                                                mpi_info)
    group = MCMCGroup.init_from_params(params, "betas", [amp_group], mpi_info)
    return mpi_info, params, group, amp_group, band_map, comp_list


def test_a_proposal_out_of_bounds_is_rejected_without_a_solve():
    mpi_info, params, group, amp_group, band_map, comp_list = _make_setup(
        bounds=[-4.0, -2.0])
    # A step that reached the amplitude solve would fail on this stand-in for a group.
    group.update_amplitude_groups = [None]

    stats = sample_spectral_params(mpi_info, group, band_map, comp_list,
                                   _fixed_rng([-1.0]))

    assert stats["n_accept"] == 0
    assert [comp.beta for comp in comp_list] == [TRUE_BETA, TRUE_BETA]


def test_a_better_fit_is_accepted_on_both_views_of_the_component():
    mpi_info, params, group, amp_group, band_map, comp_list = _make_setup(start_beta=-2.8)

    stats = sample_spectral_params(mpi_info, group, band_map, comp_list,
                                   _fixed_rng([TRUE_BETA]))

    assert [comp.pol for comp in comp_list] == ["I", "QU"]
    assert [comp.beta for comp in comp_list] == [TRUE_BETA, TRUE_BETA]
    assert stats == {"numstep": 1, "n_accept": 1, "accept_rate": 1.0,
                     "betas": {"sync": TRUE_BETA}}


def test_a_worse_fit_is_rejected_and_beta_is_restored():
    mpi_info, params, group, amp_group, band_map, comp_list = _make_setup()

    stats = sample_spectral_params(mpi_info, group, band_map, comp_list,
                                   _fixed_rng([-2.8]))

    assert stats["n_accept"] == 0
    assert [comp.beta for comp in comp_list] == [TRUE_BETA, TRUE_BETA]


def test_a_step_starts_from_the_state_the_previous_step_left():
    """Step 1 moves to the true beta and is accepted. Step 2 moves away from it and is rejected,
    which must give back the state after step 1, not the state before it."""
    mpi_info, params, group, amp_group, band_map, comp_list = _make_setup(start_beta=-2.8)
    group.cfg.numstep = 2

    stats = sample_spectral_params(mpi_info, group, band_map, comp_list,
                                   _fixed_rng([TRUE_BETA, -2.9]))

    assert stats["n_accept"] == 1
    assert [comp.beta for comp in comp_list] == [TRUE_BETA, TRUE_BETA]


@pytest.mark.parametrize("proposal, accepted", [(-3.10, True), (-3.13, False)])
def test_the_gaussian_prior_enters_the_decision(proposal: float, accepted: bool):
    """With no band in the chi-squared, the prior alone decides. The step starts at the prior
    mean, so the log acceptance is -0.5 ((proposal - mean)/rms)^2: -0.5 and -0.845 here. The
    uniform draw of 0.5 accepts a step above log(0.5) = -0.693."""
    mpi_info, params, group, amp_group, band_map, comp_list = _make_setup(
        prior_mean=TRUE_BETA, prior_rms=0.1)
    group.cfg.chisq_bands = []

    stats = sample_spectral_params(mpi_info, group, band_map, comp_list,
                                   _fixed_rng([proposal]))

    assert stats["n_accept"] == int(accepted)
    assert comp_list[0].beta == (proposal if accepted else TRUE_BETA)


def test_an_accepted_step_keeps_the_resolved_amplitudes():
    mpi_info, params, group, amp_group, band_map, comp_list = _make_setup(start_beta=-2.8)
    group.update_amplitude_groups = [amp_group]
    start_amps = comp_list[0].amps.copy()

    stats = sample_spectral_params(mpi_info, group, band_map, comp_list,
                                   _fixed_rng([TRUE_BETA]))

    assert stats["n_accept"] == 1
    assert not np.allclose(comp_list[0].amps, start_amps)


def test_a_rejected_step_restores_the_amplitudes_after_the_resolve():
    """The same amplitude re-solve as in the accepted case, but a narrow prior at the starting
    beta rejects the step. The amplitudes must then be those from before the step."""
    mpi_info, params, group, amp_group, band_map, comp_list = _make_setup(
        start_beta=-2.8, prior_mean=-2.8, prior_rms=1.0e-4)
    group.update_amplitude_groups = [amp_group]
    start_amps = comp_list[0].amps.copy()

    stats = sample_spectral_params(mpi_info, group, band_map, comp_list,
                                   _fixed_rng([TRUE_BETA]))

    assert stats["n_accept"] == 0
    assert [comp.beta for comp in comp_list] == [-2.8, -2.8]
    np.testing.assert_array_equal(comp_list[0].amps, start_amps)
