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

from commander4.compsep.spectral_sampling import sample_spectral_params
from commander4.data_models.detector_map import DetectorMap
from commander4.sky.comp_list import CompList
from commander4.sky.sky_model import SkyModel

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

    Returns `(mpi_info, cfg, group, band_map, comp_list)`. The MCMC group samples the component's
    beta with a step of 0.1, and with `beta_settings` on top. The amplitudes stay fixed, unless a
    test names the per-pixel amplitude group "amps" in `group.update_amplitude_groups`.
    """
    sync = Bunch(enabled=True, component_class="Synchrotron",
                 params=Bunch(lmax=LMAX, polarization="IQU", shortname="sync", nu_ref=30.0,
                              spatially_varying_MM=False, Cl_prior_amplitude=None, beta=TRUE_BETA))
    comp_list = CompList.init_from_params(Bunch(components=Bunch(Synchrotron=sync),
                                                compsep=Bunch(double_precision=True)))
    comp_list[0].amps[:] = 1.0  # The I view. Real alms, so they are valid at m = 0 too.

    npix = 12*NSIDE**2
    band_map = DetectorMap(np.zeros((1, npix)), np.full((1, npix), 1.0e4), nu=NU, fwhm=0.0,
                           nside=NSIDE, double_precision=True, lmax=LMAX)
    band_map.map_sky = SkyModel(comp_list).get_sky_at_band(band_map).astype(np.float64)
    for comp in comp_list:
        comp.beta = start_beta

    mpi_info = Bunch(compsep=Bunch(comm=MPI.COMM_SELF, subcomm=MPI.COMM_SELF, rank=0, master=0,
                                   size=1),
                     band=Bunch(name="Band60", identifier="Band60_I", polarization="I"))
    amps = Bunch(name="amps", comps=["Synchrotron"], bands=["Band60"])
    cfg = Bunch(cg_sampling_groups={}, per_pixel_sampling_groups={"amps": amps},
                amplitude_groups={"amps": amps}, double_precision=True, num_threads=1)
    settings = Bunch(proposal_sigma=0.1, bounds=None, prior_mean=None, prior_rms=None)
    settings.update(beta_settings)
    group = Bunch(name="betas", beta={"Synchrotron": settings}, update_amplitude_groups=[],
                  numstep=1, chisq_bands=["Band60"], chisq_mask_map=None)
    return mpi_info, cfg, group, band_map, comp_list


def test_a_proposal_out_of_bounds_is_rejected_without_a_solve():
    mpi_info, cfg, group, band_map, comp_list = _make_setup(bounds=[-4.0, -2.0])
    # A step that reached the amplitude solve would fail on this unknown group.
    group.update_amplitude_groups = ["no_such_group"]

    stats = sample_spectral_params(mpi_info, cfg, group, band_map, comp_list, _fixed_rng([-1.0]))

    assert stats["n_accept"] == 0
    assert [comp.beta for comp in comp_list] == [TRUE_BETA, TRUE_BETA]


def test_a_better_fit_is_accepted_on_both_views_of_the_component():
    mpi_info, cfg, group, band_map, comp_list = _make_setup(start_beta=-2.8)

    stats = sample_spectral_params(mpi_info, cfg, group, band_map, comp_list,
                                   _fixed_rng([TRUE_BETA]))

    assert [comp.pol for comp in comp_list] == ["I", "QU"]
    assert [comp.beta for comp in comp_list] == [TRUE_BETA, TRUE_BETA]
    assert stats == {"numstep": 1, "n_accept": 1, "accept_rate": 1.0,
                     "params": {"sync": TRUE_BETA}}


def test_a_worse_fit_is_rejected_and_beta_is_restored():
    mpi_info, cfg, group, band_map, comp_list = _make_setup()

    stats = sample_spectral_params(mpi_info, cfg, group, band_map, comp_list, _fixed_rng([-2.8]))

    assert stats["n_accept"] == 0
    assert [comp.beta for comp in comp_list] == [TRUE_BETA, TRUE_BETA]


def test_a_step_starts_from_the_state_the_previous_step_left():
    """Step 1 moves to the true beta and is accepted. Step 2 moves away from it and is rejected,
    which must give back the state after step 1, not the state before it."""
    mpi_info, cfg, group, band_map, comp_list = _make_setup(start_beta=-2.8)
    group.numstep = 2

    stats = sample_spectral_params(mpi_info, cfg, group, band_map, comp_list,
                                   _fixed_rng([TRUE_BETA, -2.9]))

    assert stats["n_accept"] == 1
    assert [comp.beta for comp in comp_list] == [TRUE_BETA, TRUE_BETA]


@pytest.mark.parametrize("proposal, accepted", [(-3.10, True), (-3.13, False)])
def test_the_gaussian_prior_enters_the_decision(proposal: float, accepted: bool):
    """With no band in the chi-squared, the prior alone decides. The step starts at the prior
    mean, so the log acceptance is -0.5 ((proposal - mean)/rms)^2: -0.5 and -0.845 here. The
    uniform draw of 0.5 accepts a step above log(0.5) = -0.693."""
    mpi_info, cfg, group, band_map, comp_list = _make_setup(prior_mean=TRUE_BETA, prior_rms=0.1)
    group.chisq_bands = []

    stats = sample_spectral_params(mpi_info, cfg, group, band_map, comp_list,
                                   _fixed_rng([proposal]))

    assert stats["n_accept"] == int(accepted)
    assert comp_list[0].beta == (proposal if accepted else TRUE_BETA)


def test_an_accepted_step_keeps_the_resolved_amplitudes():
    mpi_info, cfg, group, band_map, comp_list = _make_setup(start_beta=-2.8)
    group.update_amplitude_groups = ["amps"]
    start_amps = comp_list[0].amps.copy()

    stats = sample_spectral_params(mpi_info, cfg, group, band_map, comp_list,
                                   _fixed_rng([TRUE_BETA]))

    assert stats["n_accept"] == 1
    assert not np.allclose(comp_list[0].amps, start_amps)


def test_a_rejected_step_restores_the_amplitudes_after_the_resolve():
    """The same amplitude re-solve as in the accepted case, but a narrow prior at the starting
    beta rejects the step. The amplitudes must then be those from before the step."""
    mpi_info, cfg, group, band_map, comp_list = _make_setup(
        start_beta=-2.8, prior_mean=-2.8, prior_rms=1.0e-4)
    group.update_amplitude_groups = ["amps"]
    start_amps = comp_list[0].amps.copy()

    stats = sample_spectral_params(mpi_info, cfg, group, band_map, comp_list,
                                   _fixed_rng([TRUE_BETA]))

    assert stats["n_accept"] == 0
    assert [comp.beta for comp in comp_list] == [-2.8, -2.8]
    np.testing.assert_array_equal(comp_list[0].amps, start_amps)
