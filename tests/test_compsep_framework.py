from copy import deepcopy

import healpy as hp
import numpy as np
import pytest
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.compsep.amplitude_sampling import AmplitudeGroup, sample_amplitudes
from commander4.compsep.processing import (
    _validate_component_lmax,
    process_compsep,
    read_sampling_groups,
)
from commander4.compsep.chisq import collect_fit_diagnostics
from commander4.diagnostics.log import SUMMARY
from commander4.data_models.band_alms import BandAlms
from commander4.data_models.band_map import BandMap
from commander4.math_utils.alm import _dot_complex_alm_1D_arrays
from commander4.sky.comp_list import CompList
from commander4.compsep.perpix_solver import solve_compsep_perpix


def _make_compsep(ntask_compsep_qu: int = 1, ntask_compsep_i: int = 1) -> Bunch:
    """The `compsep` block components read: its nside and float precision."""
    return Bunch(nside=2, double_precision=False)


def _make_component_cfg(polarization: str = "IQU", lmax: int = 2) -> Bunch:
    return Bunch(
        enabled=True,
        component_class="CMB",
        params=Bunch(
            lmax=lmax,
            polarization=polarization,
            Cl_prior_amplitude=None,  # no prior
        ),
    )


def _make_comp_list(polarization: str = "IQU", ntask_compsep_qu: int = 1,
                    lmax: int = 2) -> CompList:
    cmb = _make_component_cfg(polarization, lmax)
    return CompList.init_from_params(Bunch(components=Bunch(cmb=cmb),
                                           compsep=_make_compsep(ntask_compsep_qu)))


def _make_multi_comp_list() -> CompList:
    cmb = _make_component_cfg("IQU")
    dust = Bunch(
        enabled=True,
        component_class="CMB",
        params=Bunch(
            lmax=2,
            polarization="IQU",
            Cl_prior_amplitude=None,  # no prior
        ),
    )
    ff = Bunch(
        enabled=True,
        component_class="CMB",
        params=Bunch(
            lmax=2,
            polarization="I",
            Cl_prior_amplitude=None,  # no prior
        ),
    )
    components = Bunch(
        {
            "CMB": cmb,
            "ThermalDust": dust,
            "FreeFree": ff,
        }
    )
    return CompList.init_from_params(Bunch(components=components, compsep=_make_compsep()))


def test_joined_comp_list_realizes_iqu_components() -> None:
    sky = _make_comp_list("IQU").joined().get_sky_at_nu(30.0, 2, "IQU", fwhm_rad=0.0)

    assert sky.shape == (3, 48)
    assert np.all(np.isfinite(sky))


@pytest.mark.parametrize("nside_chisq", [1, 2], ids=["same nside", "chi2 map finer than band"])
def test_iteration_chi2_uses_all_observed_map_samples(nside_chisq: int) -> None:
    map_sky = np.arange(1.0, 13.0).reshape(1, -1)
    map_inv_var = np.full_like(map_sky, 1.0/2.0**2)
    map_inv_var[0, -1] = 0.0  # An unobserved pixel.
    band_map = BandMap(map_sky, map_inv_var, nu=30.0, fwhm_arcmin=60.0, nside=1)
    mpi_info = Bunch(compsep=Bunch(comm=MPI.COMM_SELF, rank=0,
                                   band_name="Planck30GHz", pol="I"))
    params = Bunch(output=Bunch(chains=Bunch(nside_chisq=nside_chisq,
                                             include=Bunch(chisq_map=True))))

    fit_tree, _ = collect_fit_diagnostics(mpi_info, params, band_map, CompList([]))  # A zero sky.

    chi2 = fit_tree["chi2"]
    assert chi2["total"] == pytest.approx(np.sum((map_sky[0, :-1] / 2.0)**2))
    assert chi2["ndof"] == 11
    # The band's own contribution is the whole chi-squared here (one band), and the chi-squared map
    # only redistributes z^2 over pixels, so it must carry the same total.
    assert chi2["bands"]["Planck30GHz_I"]["chi2"] == pytest.approx(chi2["total"])
    assert chi2["map"].sum() == pytest.approx(chi2["total"])
    assert chi2["map"][1:].sum() == 0.0  # An intensity view fills row 0 only.


def test_compsep_summary_includes_final_chi2_and_z(monkeypatch, caplog) -> None:
    map_sky = np.arange(1.0, 13.0).reshape(1, -1)
    map_inv_var = np.full_like(map_sky, 1.0/2.0**2)
    band_map = BandMap(map_sky, map_inv_var, nu=30.0, fwhm_arcmin=60.0, nside=1)
    mpi_info = Bunch(compsep=Bunch(comm=MPI.COMM_SELF, rank=0,
                                   band_name="Planck30GHz", pol="I"))
    # With no sampling groups the sky stays zero, and its fit is the final fit.
    params = Bunch(output=Bunch(chains=Bunch(nside_chisq=1, include=Bunch(chisq_map=True))))
    monkeypatch.setattr(
        "commander4.compsep.processing.write_compsep_chain_to_file", lambda *args: None)

    with caplog.at_level(SUMMARY, logger="commander4.compsep.processing"):
        process_compsep(mpi_info, params, [], [], band_map, CompList([]), chain=2,
                        iteration=4, rng=np.random.default_rng(0))

    chi2 = np.sum((map_sky / 2.0)**2)
    z_chi2 = (chi2 - map_sky.size) / np.sqrt(2.0 * map_sky.size)
    assert "Chain 2, iteration 4 complete" in caplog.text
    assert f"chi2={chi2:.6e}" in caplog.text
    assert f"z={z_chi2:.3f}" in caplog.text


def _make_group_params(**sampling_group_sections) -> Bunch:
    """A parameter file with three components, one IQU band, and the given sampling groups."""
    components = Bunch({name: Bunch(enabled=True, component_class="CMB",
                                    params=Bunch(polarization="IQU"))
                        for name in ("CMB", "ThermalDust", "FreeFree")})
    bands = Bunch({"Planck30GHz": Bunch(enabled=True, polarization="IQU")})
    return Bunch(components=components, compsep=Bunch(bands=bands, **sampling_group_sections))


def _dust_sample_block() -> Bunch:
    """The smallest `sample` block of an MCMC group: one component's beta, with a step length."""
    return Bunch(ThermalDust=Bunch(beta=Bunch(proposal_sigma=0.01)))


def test_sampling_group_gets_defaults_and_all_names() -> None:
    params = _make_group_params(cg_sampling_groups=Bunch(
        all=Bunch(comps="all", bands="all"), missing=Bunch()))
    groups, _ = read_sampling_groups(params)

    # "all" and a setting that is left out both select everything.
    assert [group.name for group in groups] == ["all", "missing"]
    for group in groups:
        assert group.solver == "CG"
        assert group.cfg.comps == ["CMB", "ThermalDust", "FreeFree"]
        assert group.cfg.bands == ["Planck30GHz"]
        assert group.cfg.max_iter == 200 and group.cfg.optimize is False


def test_sampling_group_names_components_and_bands() -> None:
    params = _make_group_params(cg_sampling_groups=Bunch(
        selected=Bunch(comps=["CMB"], bands=["Planck30GHz"])))
    groups, _ = read_sampling_groups(params)

    assert groups[0].cfg.comps == ["CMB"]
    assert groups[0].cfg.bands == ["Planck30GHz"]
    # A group names a whole band, not one polarization view of it.
    with pytest.raises(ValueError, match="selected.bands' names unknown"):
        read_sampling_groups(_make_group_params(cg_sampling_groups=Bunch(
            selected=Bunch(comps=["CMB"], bands=["Planck30GHz_QU"]))))


def test_sampling_group_with_an_unknown_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="g.comps' names unknown"):
        read_sampling_groups(_make_group_params(
            cg_sampling_groups=Bunch(g=Bunch(comps=["DoesNotExist"]))))
    with pytest.raises(ValueError, match="g.bands' names unknown"):
        read_sampling_groups(_make_group_params(
            per_pixel_sampling_groups=Bunch(g=Bunch(bands=["NoSuchBand"]))))
    with pytest.raises(ValueError, match="g.chisq_bands' names unknown"):
        read_sampling_groups(_make_group_params(
            mcmc_sampling_groups=Bunch(g=Bunch(sample=_dust_sample_block(),
                                               chisq_bands=["NoSuchBand"]))))
    with pytest.raises(ValueError, match="g.sample' names unknown"):
        read_sampling_groups(_make_group_params(mcmc_sampling_groups=Bunch(
            g=Bunch(sample=Bunch(NoSuchComp=Bunch(beta=Bunch(proposal_sigma=0.01)))))))
    with pytest.raises(ValueError, match="must be a list of names"):
        read_sampling_groups(_make_group_params(cg_sampling_groups=Bunch(g=Bunch(comps="CMB"))))


def test_disabled_sampling_groups_are_left_out() -> None:
    params = _make_group_params(cg_sampling_groups=Bunch(
        active=Bunch(enabled=True), disabled=Bunch(enabled=False)))
    assert [group.name for group in read_sampling_groups(params)[0]] == ["active"]


def _make_spectral_comp_list() -> CompList:
    sync = Bunch(
        enabled=True,
        component_class="Synchrotron",
        params=Bunch(
            lmax=2, polarization="IQU", 
            Cl_prior_amplitude=None, beta=-3.1, nu_ref=30.0),
    )
    dust = Bunch(
        enabled=True,
        component_class="ThermalDust",
        params=Bunch(
            lmax=2, polarization="IQU", 
            Cl_prior_amplitude=None, beta=1.56, T=20.0, nu_ref=545.0),
    )
    return CompList.init_from_params(Bunch(
        components=Bunch(Synchrotron=sync, ThermalDust=dust), compsep=_make_compsep()))


def test_gibbs_chains_have_independent_component_state() -> None:
    chain_components = {1: _make_spectral_comp_list()}
    chain_components[2] = deepcopy(chain_components[1])

    chain_1_sync = [comp for comp in chain_components[1] if comp.comp_name == "Synchrotron"]
    chain_2_sync = [comp for comp in chain_components[2] if comp.comp_name == "Synchrotron"]

    # The component objects of the two chains are independent, with their settings.
    assert chain_1_sync[0].cfg is not chain_2_sync[0].cfg
    assert all(comp_1 is not comp_2 for comp_1, comp_2 in zip(chain_components[1],
                                                              chain_components[2]))

    for comp in chain_1_sync:
        comp.beta = -2.8
    chain_1_sync[0].amps[:] = 4.0 + 2.0j
    chain_1_sync[0].amp_fwhm_rad = 0.25

    assert all(comp.beta == -2.8 for comp in chain_1_sync)
    assert all(comp.beta == -3.1 for comp in chain_2_sync)
    assert np.all(chain_2_sync[0].amps == 0.0)
    assert chain_2_sync[0].amp_fwhm_rad == 0.0


def test_sampling_groups_reject_settings_of_other_methods() -> None:
    with pytest.raises(ValueError, match="sample_class"):
        read_sampling_groups(_make_group_params(
            cg_sampling_groups=Bunch(cg=Bunch(sample_class="amplitude_sampler_CG"))))
    with pytest.raises(ValueError, match="max_iter"):
        read_sampling_groups(_make_group_params(
            per_pixel_sampling_groups=Bunch(pixels=Bunch(max_iter=10))))
    with pytest.raises(ValueError, match="parameters"):
        read_sampling_groups(_make_group_params(
            mcmc_sampling_groups=Bunch(mcmc=Bunch(parameters="gain"))))
    with pytest.raises(ValueError, match="must be a int"):
        read_sampling_groups(_make_group_params(
            cg_sampling_groups=Bunch(cg=Bunch(max_iter=2.5))))


def test_a_per_pixel_group_with_point_sources_is_rejected_at_start_up() -> None:
    params = _make_group_params(per_pixel_sampling_groups=Bunch(pixels=Bunch()))
    params.components.FreeFree.component_class = "RadioSources"
    with pytest.raises(ValueError, match="'FreeFree', a RadioSources"):
        read_sampling_groups(params)
    # A per-pixel group that leaves the point sources out is fine.
    params.compsep.per_pixel_sampling_groups.pixels.comps = ["CMB", "ThermalDust"]
    groups, _ = read_sampling_groups(params)
    assert [group.name for group in groups] == ["pixels"] and groups[0].solver == "per_pixel"


def test_unimplemented_component_classes_are_rejected_during_construction() -> None:
    component = Bunch(
        enabled=True,
        component_class="TemplateComponent",
        params=Bunch(polarization="I"),
    )
    with pytest.raises(NotImplementedError, match="not implemented"):
        CompList.init_from_params(Bunch(components=Bunch(Template=component), compsep=Bunch()))


def test_cg_and_per_pixel_groups_are_mutually_exclusive() -> None:
    params = _make_group_params(cg_sampling_groups=Bunch(cg=Bunch()),
                                per_pixel_sampling_groups=Bunch(pixels=Bunch()))
    with pytest.raises(ValueError, match="mutually exclusive"):
        read_sampling_groups(params)


def test_per_pixel_solver_stores_the_solution_in_the_precision_of_the_components() -> None:
    comp_list = _make_comp_list("I")
    band_map = BandMap(
        map_sky=np.ones((1, 48)), map_inv_var=np.ones((1, 48)),
        nu=30.0, fwhm_arcmin=0.0, nside=2,
    )
    solve_compsep_perpix(MPI.COMM_SELF, band_map, comp_list, nthreads=1,
                         rng=np.random.default_rng(0))
    assert comp_list[0].amps.dtype == np.complex64
    assert np.all(np.isfinite(comp_list[0].amps))


def test_mcmc_group_must_name_enabled_amplitude_groups() -> None:
    amplitudes = Bunch(amps=Bunch(comps=["CMB"]), off=Bunch(enabled=False))
    amplitude_groups, mcmc_groups = read_sampling_groups(_make_group_params(
        cg_sampling_groups=amplitudes, mcmc_sampling_groups=Bunch(
            mcmc=Bunch(sample=_dust_sample_block(), update_amplitude_groups=["amps"]))))
    assert mcmc_groups[0].update_amplitude_groups == amplitude_groups

    for group_name in ("does_not_exist", "off"):
        with pytest.raises(ValueError, match="unknown or disabled amplitude group"):
            read_sampling_groups(_make_group_params(
                cg_sampling_groups=amplitudes, mcmc_sampling_groups=Bunch(
                    mcmc=Bunch(sample=_dust_sample_block(), update_amplitude_groups=[group_name]))))


def test_an_amplitude_group_is_solved_against_the_band_minus_the_fixed_components() -> None:
    """A group that names only CMB fits CMB to the band map minus the other components' sky.

    The band map holds the sky of all three components. Solved against that residual, the CG
    gives back the CMB amplitudes it was made from; solved against the full map, it would not.
    The fixed components and the band map itself are left as they were.
    """
    comp_list = _make_multi_comp_list()  # CMB, ThermalDust, FreeFree (all intensity views exist)
    rng = np.random.default_rng(3)
    for comp in comp_list.split_for_pol("I"):  # Real alms, so they are valid at m = 0 too.
        comp.amps[:] = rng.normal(0.0, 1.0, comp.amps.shape)
    npix = 12*2**2
    band_map = BandMap(np.zeros((1, npix)), np.ones((1, npix)), nu=30.0, fwhm_arcmin=0.0, nside=2)
    band_map.map_sky = comp_list.get_sky_at_band(band_map).astype(np.float64)
    band_sky = band_map.map_sky.copy()
    true_amps = [comp.amps.copy() for comp in comp_list]
    cmb = comp_list.split_for_pol("I")[0]
    cmb.amps[:] = 0.0

    # One intensity band on one rank, so that rank solves the group alone.
    params = _make_group_params(cg_sampling_groups=Bunch(cmb=Bunch(
        comps=["CMB"], optimize=True, max_iter=100, err_tol=1.0e-10,
        preconditioner="NoPreconditioner")))
    params.compsep.bands.Planck30GHz.polarization = "I"
    params.compsep.double_precision = False
    mpi_info = Bunch(compsep=Bunch(comm=MPI.COMM_SELF, subcomm=MPI.COMM_SELF, rank=0,
                                   num_threads=1, band_name="Planck30GHz", pol="I"))
    group = AmplitudeGroup.init_from_params(params, "cg_sampling_groups", "cmb", mpi_info)
    assert group.solver_ranks == {"I": [0], "QU": []}

    sample_amplitudes(mpi_info, group, band_map, comp_list, np.random.default_rng(0))

    assert cmb.comp_name == "CMB"
    np.testing.assert_allclose(cmb.amps, true_amps[0], atol=1e-3)
    for comp, amps in zip(comp_list[1:], true_amps[1:]):
        np.testing.assert_array_equal(comp.amps, amps)
    np.testing.assert_array_equal(band_map.map_sky, band_sky)


def test_band_sky_removes_amp_fwhm_by_quadrature() -> None:
    # A band sky applies only the part of the band beam that the amplitudes do not already carry
    # (amp_fwhm_rad, from the per-pixel solver): applied = sqrt(target^2 - amp_fwhm_rad^2).
    comp = next(c for c in _make_multi_comp_list().split_for_pol("I") if c.comp_name == "CMB")
    comp.amps = np.full_like(comp.amps, 2.0 + 0.0j)  # power at all l so smoothing is non-trivial
    nu, nside, target = 30.0, 2, 0.5  # target beam in radians
    sed = comp.get_sed(nu)
    # Sanity: at this target the beam actually changes the map, so the cases below are distinguishable.
    assert not np.allclose(comp.get_component_map(nside, target), comp.get_component_map(nside, 0.0))

    # Deconvolved amplitudes get the full beam, partly smoothed ones the quadrature remainder, and
    # amplitudes already at the band resolution none.
    for amp_fwhm, applied in ((0.0, target), (0.3, np.sqrt(target**2 - 0.3**2)), (target, 0.0)):
        comp.amp_fwhm_rad = amp_fwhm
        np.testing.assert_allclose(
            CompList([comp]).get_sky_at_nu(nu, nside, "I", fwhm_rad=target),
            comp.get_component_map(nside, applied)*sed, rtol=1e-5, atol=1e-5)


def _random_alms(lmax: int, rng: np.random.Generator) -> np.ndarray:
    """Random complex alms of one Stokes row, real at m = 0 (the first lmax + 1 entries)."""
    alms = rng.standard_normal(hp.Alm.getsize(lmax)) + 1j*rng.standard_normal(hp.Alm.getsize(lmax))
    alms[:lmax + 1] = alms[:lmax + 1].real
    return alms[None, :]


def test_band_sky_is_cut_at_the_band_lmax() -> None:
    """The sky realized for a band holds nothing above the band lmax, as in the CG operator
    (C3's evalBand): a component with a higher lmax is cut, not folded into the band."""
    comp = _make_comp_list("I", lmax=23)[0]
    comp.amps = _random_alms(23, np.random.default_rng(1)).astype(comp.amps.dtype)
    nu, nside, band_lmax, fwhm = 30.0, 4, 7, np.deg2rad(5.0)

    sky = CompList([comp]).get_sky_at_nu(nu, nside, "I", fwhm_rad=fwhm, lmax=band_lmax)

    alms = hp.resize_alm(comp.amps[0].astype(complex), 23, 23, band_lmax, band_lmax)
    expected = hp.alm2map(hp.almxfl(alms, hp.gauss_beam(fwhm, lmax=band_lmax)), nside,
                          lmax=band_lmax)*comp.get_sed(nu)
    np.testing.assert_allclose(sky[0], expected, atol=1e-4*expected.std())


@pytest.mark.parametrize("sed_per_pixel", [False, True], ids=["one SED", "SED per pixel"])
@pytest.mark.parametrize("band_lmax", [5, 11])  # Below and above the component lmax of 8.
def test_band2comp_is_the_transpose_of_comp2band(band_lmax: int, sed_per_pixel: bool) -> None:
    """<comp2band(x), y> = <x, band2comp(y)>, with the beam, the lmax cut and the SED included.

    An SED that is a map takes the pixel-space path of both methods."""
    rng = np.random.default_rng(2)
    comp = _make_comp_list("I", lmax=8)[0]
    comp.amps = _random_alms(8, rng).astype(comp.amps.dtype)
    nside = 4
    if sed_per_pixel:
        sed_map = rng.uniform(0.5, 2.0, 12*nside**2)
        comp.get_sed = lambda nu: sed_map
    band_map = BandMap(np.zeros((1, 12*nside**2)), np.ones((1, 12*nside**2)), nu=30.0,
                       fwhm_arcmin=600.0, nside=nside, lmax=band_lmax)
    band = BandAlms.init_from_band_map(band_map, double_precision=True)

    x = comp.amps.astype(complex)
    band.alms[:] = 0
    comp.comp2band(band)
    comp_x = band.alms.astype(complex)
    y = _random_alms(band_lmax, rng)
    band.alms = y.copy()
    comp.band2comp(band)
    transpose_y = comp.amps.astype(complex)

    lhs = _dot_complex_alm_1D_arrays(comp_x[0], y[0], band_lmax)
    rhs = _dot_complex_alm_1D_arrays(x[0], transpose_y[0], 8)
    assert lhs == pytest.approx(rhs, rel=1e-5)


def test_mcmc_group_reads_its_sample_block() -> None:
    params = _make_group_params(mcmc_sampling_groups=Bunch(mcmc=Bunch(sample=Bunch(
        FreeFree=Bunch(beta=Bunch(proposal_sigma=0.02, bounds=[-4.0, -2.0])),
        ThermalDust=Bunch(beta=Bunch(proposal_sigma=0.01, prior_mean=1.5, prior_rms=0.1))))))
    sample = read_sampling_groups(params)[1][0].cfg.sample

    # The components are stepped in the order of the block, not in the order of the components.
    assert list(sample) == ["FreeFree", "ThermalDust"]
    # A setting that is left out is None: no bounds, or no prior.
    free_free, dust = sample["FreeFree"].beta, sample["ThermalDust"].beta
    assert (free_free.proposal_sigma, free_free.bounds, free_free.prior_mean,
            free_free.prior_rms) == (0.02, [-4.0, -2.0], None, None)
    assert (dust.proposal_sigma, dust.bounds, dust.prior_mean, dust.prior_rms) \
        == (0.01, None, 1.5, 0.1)
    # The defaults go into the group's copy, not into the parameter file.
    assert "prior_mean" not in params.compsep.mcmc_sampling_groups.mcmc.sample.FreeFree.beta


@pytest.mark.parametrize("dust, message", [
    (None, "sample' must name the components"),                        # No `sample` block.
    (Bunch(), "must name a parameter to sample"),                       # Dust, but no parameter.
    (Bunch(T=Bunch(proposal_sigma=0.01)), "Unknown setting"),           # Only beta can be sampled.
    (Bunch(beta=Bunch(bounds=[1.0, 2.0])), "proposal_sigma' must be set"),
    (Bunch(beta=Bunch(proposal_sigma=0.01, prior=1.5)), "Unknown setting"),
    (Bunch(beta=Bunch(proposal_sigma=0.01, prior_mean=1.5)), "both 'prior_mean' and"),
])
def test_mcmc_group_with_a_wrong_sample_block_fails_at_startup(dust, message) -> None:
    group = Bunch() if dust is None else Bunch(sample=Bunch(ThermalDust=dust))
    with pytest.raises(ValueError, match=message):
        read_sampling_groups(_make_group_params(mcmc_sampling_groups=Bunch(mcmc=group)))


# --- component lmax vs band lmax ------------------------------------------------------------
# A component multipole above every band's lmax never meets the data, so the C(l) prior decides it
# on its own. C3 leaves this to the user; we report it, either as a warning (the prior is at least
# tapered there by Cl_prior_l_apod) or as an error (nothing suppresses it).

def _lmax_check_params(comp_lmax: int, band_nside: int = 64, l_apod: int | None = None) -> Bunch:
    """One enabled TOD band and one CMB component, with no band `lmax` set (so 3*nside-1)."""
    comp = _make_component_cfg("I")
    comp.params.lmax = comp_lmax
    comp.params.Cl_prior_amplitude = 1.0e+8
    if l_apod is not None:
        comp.params.Cl_prior_l_apod = l_apod
    params = Bunch(
        components=Bunch(CMB=comp),
        compsep=Bunch(nside=band_nside, double_precision=False,
                      bands=Bunch(BandA=Bunch(enabled=True, get_from="EXP", polarization="I"))),
        experiments=Bunch(EXP=Bunch(bands=Bunch(BandA=Bunch(eval_nside=band_nside)))),
    )
    return CompList.init_from_params(params), params


def test_component_lmax_within_the_band_lmax_is_silent(caplog) -> None:
    comp_list, params = _lmax_check_params(comp_lmax=191)   # nside 64 -> band lmax 191
    with caplog.at_level("WARNING"):
        _validate_component_lmax(comp_list, params)
    assert caplog.text == ""


def test_component_lmax_above_every_band_with_no_taper_is_an_error(caplog) -> None:
    comp_list, params = _lmax_check_params(comp_lmax=250)
    with caplog.at_level("WARNING"):
        _validate_component_lmax(comp_list, params)
    assert "ERROR" in caplog.text
    assert "l = 192-250 is constrained by the C(l) prior alone" in caplog.text


def test_component_lmax_above_every_band_but_tapered_is_only_a_warning(caplog) -> None:
    comp_list, params = _lmax_check_params(comp_lmax=250, l_apod=191)
    with caplog.at_level("WARNING"):
        _validate_component_lmax(comp_list, params)
    assert "WARNING" in caplog.text and "ERROR" not in caplog.text
    assert "Cl_prior_l_apod = 191" in caplog.text
