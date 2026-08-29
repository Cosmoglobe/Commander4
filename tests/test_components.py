from copy import deepcopy

import h5py
import healpy as hp
import numpy as np
import pytest
from pixell.bunch import Bunch

from commander4.sky.comp_list import CompList
from commander4.sky.diffuse_components import CMB, ThermalDust
from commander4.sky.point_sources import PointSourcesComponent
from commander4.sky.sky_model import build_initial_sky_model
from commander4.math_utils.alm import gaussian_random_alm
from commander4.math_utils.sht import alm_to_map
from commander4.sky.comp_list import complist_dot, complist_norm


def _make_compsep(ntask_compsep_qu: int = 1, ntask_compsep_i: int = 1) -> Bunch:
    """The `compsep` block components read: its nside and float precision."""
    return Bunch(nside=2, double_precision=False)


def _make_component_cfg(polarization: str = "IQU") -> Bunch:
    return Bunch(
        enabled=True,
        component_class="CMB",
        params=Bunch(
            lmax=1,
            polarization=polarization,
            shortname="cmb",
            spatially_varying_MM=False,
            Cl_prior_amplitude=None,  # identity prior (C_l = 1)
        ),
    )


def _make_comp_list(polarization: str = "IQU", ntask_compsep_qu: int = 1) -> CompList:
    params = Bunch(compsep=_make_compsep(ntask_compsep_qu))
    cmb = _make_component_cfg(polarization)
    object.__setattr__(cmb, "_name", "cmb")
    components = Bunch({"cmb": cmb})
    return CompList.init_from_params(components, params)


def _make_named_component_cfg(shortname: str, polarization: str = "IQU") -> Bunch:
    cfg = _make_component_cfg(polarization)
    cfg.params.shortname = shortname
    return cfg


def _make_multi_comp_list() -> CompList:
    params = Bunch(compsep=_make_compsep())
    cmb = _make_named_component_cfg("cmb", "IQU")
    object.__setattr__(cmb, "_name", "cmb")
    dust = _make_named_component_cfg("dust", "IQU")
    object.__setattr__(dust, "_name", "dust")
    components = Bunch(
        {
            "cmb": cmb,
            "dust": dust,
        }
    )
    return CompList.init_from_params(components, params)


def test_init_from_params_requires_component_name() -> None:
    params = Bunch(compsep=_make_compsep())
    components = Bunch({"cmb": _make_component_cfg("I")})

    with pytest.raises(AttributeError, match="_name"):
        CompList.init_from_params(components, params)


def test_init_from_params_does_not_mutate_component_params_name() -> None:
    params = Bunch(compsep=_make_compsep())
    component_cfg = _make_component_cfg("I")
    object.__setattr__(component_cfg, "_name", "cmb")
    components = Bunch({"cmb": component_cfg})

    CompList.init_from_params(components, params)

    assert "_name" not in component_cfg.params


def test_init_from_params_builds_all_defined_pol_views() -> None:
    # Construction is independent of the MPI/compsep layout: an IQU component always yields both an
    # I and a QU view, even when zero compsep ranks are configured.
    params = Bunch(compsep=_make_compsep(ntask_compsep_qu=0, ntask_compsep_i=0))
    cmb = _make_named_component_cfg("cmb", "IQU")
    object.__setattr__(cmb, "_name", "cmb")
    ff = _make_named_component_cfg("ff", "I")
    object.__setattr__(ff, "_name", "ff")
    components = Bunch({"cmb": cmb, "ff": ff})

    comp_list = CompList.init_from_params(components, params)

    assert [(comp.comp_name, comp.eval_pol) for comp in comp_list] == [
        ("cmb", "I"), ("cmb", "QU"), ("ff", "I")]


def test_complist_split_preserves_names_and_join_restores_logical_component() -> None:
    comp_list = _make_comp_list("IQU")

    assert [comp.comp_name for comp in comp_list] == ["cmb", "cmb"]
    assert [comp.shortname for comp in comp_list] == ["cmb", "cmb"]
    assert [comp.eval_pol for comp in comp_list] == ["I", "QU"]
    assert [comp.is_split_view for comp in comp_list] == [True, True]

    comp_list[0].alms[:] = 1.0 + 0.0j
    comp_list[1].alms[:] = 2.0 + 0.0j
    joined = comp_list.joined()

    assert len(joined) == 1
    assert joined[0].comp_name == "cmb"
    assert joined[0].shortname == "cmb"
    assert joined[0].eval_pol == "IQU"
    assert not joined[0].is_split_view
    assert np.all(joined[0].alms[0] == 1.0 + 0.0j)
    assert np.all(joined[0].alms[1:] == 2.0 + 0.0j)


def test_component_itruediv_divides_data() -> None:
    comp_list = _make_comp_list("I")
    comp = comp_list[0]
    other = deepcopy(comp)
    comp.alms[:] = 6.0 + 0.0j
    other.alms[:] = 3.0 + 0.0j

    comp /= other

    assert np.all(comp.alms == 2.0 + 0.0j)


def test_complist_add_returns_full_complist() -> None:
    comp_list = _make_comp_list("IQU")
    for idx, comp in enumerate(comp_list, start=1):
        comp.alms[:] = idx + 0.0j

    summed = comp_list + comp_list

    assert isinstance(summed, CompList)
    assert len(summed) == 2
    assert np.all(summed[0].alms == 2.0 + 0.0j)
    assert np.all(summed[1].alms == 4.0 + 0.0j)


def test_complist_norm_is_the_square_root_of_the_self_dot_product() -> None:
    """`complist_norm` must return |x|, not |x|^2, so a ratio of two is a relative L2 error."""
    comp_list = _make_comp_list("IQU")
    for idx, comp in enumerate(comp_list, start=1):
        comp.alms[:] = idx + 0.0j

    self_dot = complist_dot(comp_list, comp_list)

    assert self_dot > 0.0
    assert complist_norm(comp_list) == pytest.approx(np.sqrt(self_dot))


def test_complist_norm_ratio_scales_linearly_with_the_perturbation() -> None:
    """The CG diagnostic divides two norms and reports it as an L2 error, so it must be linear."""
    truth = _make_comp_list("IQU")
    for comp in truth:
        comp.alms[:] = 1.0 + 0.0j
    perturbed = deepcopy(truth)
    for comp in perturbed:
        comp.alms[:] = 1.1 + 0.0j  # a uniform 10% deviation

    relative_error = complist_norm(perturbed - truth)/complist_norm(truth)

    assert relative_error == pytest.approx(0.1, rel=1e-6)


def test_complist_ops_require_matching_execution_views() -> None:
    comp_list = _make_comp_list("IQU")
    other = deepcopy(comp_list)
    other.comp_list.reverse()

    with pytest.raises(ValueError, match="same execution views"):
        _ = comp_list + other
    with pytest.raises(ValueError, match="same execution views"):
        _ = complist_dot(comp_list, other)


def test_complist_split_for_eval_pol_rejects_invalid_polarization() -> None:
    comp_list = _make_comp_list("IQU")

    with pytest.raises(ValueError, match="Unsupported polarization string 'bad'"):
        comp_list.split_for_eval_pol("bad")


def test_point_sources_component_rejects_non_intensity_eval_pol() -> None:
    params = Bunch(shortname="ps")

    with pytest.raises(ValueError, match="does not support evaluation polarization 'QU'"):
        PointSourcesComponent(params, _make_compsep(), comp_name="ps", eval_pol="QU")


def test_complist_split_for_eval_pol_returns_requested_execution_view() -> None:
    comp_list = _make_comp_list("IQU")

    qu_only = comp_list.split_for_eval_pol("QU")

    assert len(qu_only) == 1
    assert qu_only[0].eval_pol == "QU"


def test_complist_constructor_rejects_duplicate_unsplit_comp_names() -> None:
    comp = _make_comp_list("I")[0]
    duplicate = deepcopy(comp)

    with pytest.raises(ValueError, match="Duplicate logical component"):
        CompList([comp, duplicate])


def test_complist_constructor_rejects_reused_shortname_for_distinct_comp_names() -> None:
    comp = _make_comp_list("I")[0]
    other = deepcopy(comp)
    other.comp_name = "dust"

    with pytest.raises(ValueError, match="Shortname"):
        CompList([comp, other])


def test_copy_matching_data_from_leaves_omitted_components_unchanged() -> None:
    comp_list = _make_multi_comp_list()
    intensity = comp_list.split_for_eval_pol("I")
    original_other = intensity[1].alms.copy()
    updated_subset = CompList([deepcopy(intensity[0])])
    updated_subset[0].alms[:] = 7.0 + 0.0j

    intensity.copy_matching_data_from(updated_subset)

    assert np.all(intensity[0].alms == 7.0 + 0.0j)
    assert np.array_equal(intensity[1].alms, original_other)


def _write_chain_alms(path, alms_by_shortname: dict) -> None:
    with h5py.File(path, "w") as f:
        for shortname, alms in alms_by_shortname.items():
            f[f"comps/{shortname}/alms"] = alms


def test_load_initial_alms_reads_and_splits_from_chain(tmp_path) -> None:
    nalm = (1 + 1) * (1 + 2) // 2  # lmax == 1, matching the default component config.
    cmb_alms = np.arange(3 * nalm, dtype=np.float64).reshape(3, nalm).astype(np.complex64)
    ff_alms = (np.arange(nalm, dtype=np.float64) + 100).reshape(1, nalm).astype(np.complex64)
    chain = tmp_path / "init_chain.h5"
    _write_chain_alms(chain, {"cmb": cmb_alms, "ff": ff_alms})

    compsep = _make_compsep()
    gibbs = Bunch(init_from_chain=str(chain))
    cmb = _make_named_component_cfg("cmb", "IQU")
    object.__setattr__(cmb, "_name", "cmb")
    ff = _make_named_component_cfg("ff", "I")
    object.__setattr__(ff, "_name", "ff")
    params = Bunch(compsep=compsep, gibbs=gibbs, components=Bunch({"cmb": cmb, "ff": ff}))

    comp_list = CompList.init_from_params(params.components, params)
    comp_list.load_initial_alms(params)

    views = {(comp.comp_name, comp.eval_pol): comp for comp in comp_list}
    # The joined IQU alms get split into the I row and the two QU rows.
    assert np.array_equal(views[("cmb", "I")].alms, cmb_alms[0:1])
    assert np.array_equal(views[("cmb", "QU")].alms, cmb_alms[1:3])
    assert np.array_equal(views[("ff", "I")].alms, ff_alms[0:1])


def test_load_initial_alms_prefers_per_component_init_from(tmp_path) -> None:
    nalm = (1 + 1) * (1 + 2) // 2
    global_chain = tmp_path / "global.h5"
    special_chain = tmp_path / "special.h5"
    _write_chain_alms(global_chain, {"cmb": np.zeros((3, nalm), dtype=np.complex64)})
    _write_chain_alms(special_chain, {"cmb": np.full((3, nalm), 5.0, dtype=np.complex64)})

    compsep = _make_compsep()
    gibbs = Bunch(init_from_chain=str(global_chain))
    cmb = _make_named_component_cfg("cmb", "IQU")
    object.__setattr__(cmb, "_name", "cmb")
    cmb.params.init_from = str(special_chain)  # Per-component path takes precedence over the global one.
    params = Bunch(compsep=compsep, gibbs=gibbs, components=Bunch({"cmb": cmb}))

    comp_list = CompList.init_from_params(params.components, params)
    comp_list.load_initial_alms(params)

    assert all(np.all(comp.alms == 5.0) for comp in comp_list)


def test_load_initial_alms_leaves_zeros_without_a_source() -> None:
    compsep = _make_compsep()  # No gibbs.init_from_chain, and no per-component init_from.
    gibbs = Bunch()
    cmb = _make_named_component_cfg("cmb", "IQU")
    object.__setattr__(cmb, "_name", "cmb")
    params = Bunch(compsep=compsep, gibbs=gibbs, components=Bunch({"cmb": cmb}))

    comp_list = CompList.init_from_params(params.components, params)
    comp_list.load_initial_alms(params)

    assert all(np.all(comp.alms == 0) for comp in comp_list)


def test_load_initial_alms_from_fits_map(tmp_path) -> None:
    """`init_from` a FITS map recovers that map's alms, per polarization view.

    The map is built by synthesizing known alms, so it is band-limited at the component's lmax and
    the expected result is those alms exactly. That keeps the test on the contract ("the loader
    recovers the sky the map represents") rather than on which analysis transform is used: the
    loader inverts the synthesis (`pseudo_alm_to_map_inverse`), which a quadrature `map_to_alm`
    only approximates -- by ~1e-2 here.
    """
    nside = 8
    lmax = 3
    np.random.seed(0)
    # Distinct I and QU alms, so a wrong row selection would be detectable.
    alms_I = gaussian_random_alm(lmax, lmax, 0, 1).astype(np.complex128)
    alms_QU = gaussian_random_alm(lmax, lmax, 2, 2).astype(np.complex128)
    iqu_map = np.vstack([alm_to_map(alms_I, nside, lmax, spin=0),
                         alm_to_map(alms_QU, nside, lmax, spin=2)])
    fits_path = tmp_path / "init_map.fits"
    hp.write_map(str(fits_path), iqu_map, overwrite=True, dtype=np.float64)

    compsep = _make_compsep()
    gibbs = Bunch()
    compsep.double_precision = True  # So component alms keep the map's precision.
    cmb = _make_named_component_cfg("cmb", "IQU")
    cmb.params.lmax = lmax
    object.__setattr__(cmb, "_name", "cmb")
    cmb.params.init_from = str(fits_path)
    params = Bunch(compsep=compsep, gibbs=gibbs, components=Bunch({"cmb": cmb}))

    comp_list = CompList.init_from_params(params.components, params)
    comp_list.load_initial_alms(params)

    views = {(comp.comp_name, comp.eval_pol): comp for comp in comp_list}
    assert np.allclose(views[("cmb", "I")].alms, alms_I, rtol=1e-6, atol=1e-7)
    assert np.allclose(views[("cmb", "QU")].alms, alms_QU, rtol=1e-6, atol=1e-7)


def test_load_initial_alms_rejects_unknown_extension(tmp_path) -> None:
    compsep = _make_compsep()
    gibbs = Bunch()
    cmb = _make_named_component_cfg("cmb", "IQU")
    object.__setattr__(cmb, "_name", "cmb")
    cmb.params.init_from = str(tmp_path / "init_map.txt")
    params = Bunch(compsep=compsep, gibbs=gibbs, components=Bunch({"cmb": cmb}))

    comp_list = CompList.init_from_params(params.components, params)
    with pytest.raises(ValueError, match="expected a .h5/.hd5 chain or a .fits map"):
        comp_list.load_initial_alms(params)


def test_load_initial_alms_partial_source_leaves_missing_pol_zero(tmp_path) -> None:
    # An intensity-only chain initializes the I view; the IQU component's QU view stays at zero
    # rather than erroring (so e.g. I-from-chain + QU-from-zero works).
    nalm = (1 + 1) * (1 + 2) // 2
    cmb_intensity_only = (np.arange(nalm, dtype=np.float64) + 1).reshape(1, nalm).astype(np.complex64)
    chain = tmp_path / "intensity_only.h5"
    _write_chain_alms(chain, {"cmb": cmb_intensity_only})

    compsep = _make_compsep()
    gibbs = Bunch(init_from_chain=str(chain))
    cmb = _make_named_component_cfg("cmb", "IQU")
    object.__setattr__(cmb, "_name", "cmb")
    params = Bunch(compsep=compsep, gibbs=gibbs, components=Bunch({"cmb": cmb}))

    comp_list = CompList.init_from_params(params.components, params)
    comp_list.load_initial_alms(params)

    views = {(comp.comp_name, comp.eval_pol): comp for comp in comp_list}
    assert np.array_equal(views[("cmb", "I")].alms, cmb_intensity_only)
    assert np.all(views[("cmb", "QU")].alms == 0)


def test_load_initial_alms_missing_component_logs_error_and_continues(tmp_path, caplog) -> None:
    nalm = (1 + 1) * (1 + 2) // 2
    chain = tmp_path / "other_components.h5"
    _write_chain_alms(chain, {"dust": np.ones((3, nalm), dtype=np.complex64)})  # no "cmb" entry

    compsep = _make_compsep()
    gibbs = Bunch(init_from_chain=str(chain))
    cmb = _make_named_component_cfg("cmb", "IQU")
    object.__setattr__(cmb, "_name", "cmb")
    params = Bunch(compsep=compsep, gibbs=gibbs, components=Bunch({"cmb": cmb}))

    comp_list = CompList.init_from_params(params.components, params)
    with caplog.at_level("ERROR"):
        comp_list.load_initial_alms(params)  # must not raise

    assert all(np.all(comp.alms == 0) for comp in comp_list)
    assert "not found" in caplog.text


def test_build_initial_sky_model_returns_realizable_model() -> None:
    compsep = _make_compsep()  # No init paths -> zero alms -> zero sky.
    gibbs = Bunch()
    cmb = _make_named_component_cfg("cmb", "IQU")
    cmb.params.lmax = 2  # Spin-2 (QU) synthesis requires lmax >= 2.
    object.__setattr__(cmb, "_name", "cmb")
    params = Bunch(compsep=compsep, gibbs=gibbs, components=Bunch({"cmb": cmb}))

    sky = build_initial_sky_model(params)
    realized = sky.get_sky_at_nu(100.0, 2, "IQU", fwhm=0.0)

    assert realized.shape == (3, 12 * 2**2)
    assert np.all(realized == 0)


def _dust_params(**overrides) -> Bunch:
    params = Bunch(polarization="IQU", shortname="dust", spatially_varying_MM=False,
                   Cl_prior_amplitude=None, lmax=2,
                   beta=1.5, T=20.0, nu_ref=[857.0, 353.0], units="uK_RJ")
    for key, value in overrides.items():
        params[key] = value
    object.__setattr__(params, "_name", "dust")
    return params


def test_diffuse_component_resolves_per_pol_reference_frequency() -> None:
    compsep = _make_compsep()
    gibbs = Bunch()
    dust_I = ThermalDust(_dust_params(), compsep, eval_pol="I", comp_name="dust")
    dust_QU = ThermalDust(_dust_params(), compsep, eval_pol="QU", comp_name="dust")

    # nu_ref = [nu_I, nu_QU]: each view picks its own entry.
    assert dust_I.nu_ref == 857.0
    assert dust_QU.nu_ref == 353.0
    # The SED is normalized to 1 at each view's own reference frequency.
    assert np.isclose(dust_I.get_sed(857.0), 1.0)
    assert np.isclose(dust_QU.get_sed(353.0), 1.0)


def test_scalar_reference_frequency_is_shared_by_both_polarizations() -> None:
    compsep = _make_compsep()
    gibbs = Bunch()
    dust_I = ThermalDust(_dust_params(nu_ref=545.0), compsep, eval_pol="I", comp_name="dust")
    dust_QU = ThermalDust(_dust_params(nu_ref=545.0), compsep, eval_pol="QU", comp_name="dust")

    # A scalar nu_ref applies to both I and QU.
    assert dust_I.nu_ref == 545.0 and dust_QU.nu_ref == 545.0


def test_P_Cl_prior_flat_Dl_falls_as_ell_squared_in_Cl() -> None:
    # C3 convention (comm_cl_mod.f90): the prior is defined in D_l space, so a flat amplitude
    # "roof" corresponds to C_l = 2*pi*amp/(l(l+1)), with C_0 := D_0 := D_1.
    params = _dust_params(lmax=16, Cl_prior_amplitude=100.0, Cl_prior_beta=0.0,
                          Cl_prior_l_pivot=50, Cl_prior_FWHM=0.0)
    comp = ThermalDust(params, _make_compsep(), eval_pol="I", comp_name="dust")

    ells = np.arange(1, 17)
    np.testing.assert_allclose(comp.P_Cl_prior[1:], 100.0 * 2 * np.pi / (ells * (ells + 1)))
    assert comp.P_Cl_prior[0] == 100.0


def test_P_Cl_prior_power_law_pivot_and_tilt() -> None:
    params = _dust_params(lmax=100, Cl_prior_amplitude=7.0, Cl_prior_beta=-0.5,
                          Cl_prior_l_pivot=10, Cl_prior_FWHM=0.0)
    comp = ThermalDust(params, _make_compsep(), eval_pol="I", comp_name="dust")

    # D_l equals the amplitude at the pivot, and scales as (l/l_pivot)^beta away from it.
    assert np.isclose(comp.P_Cl_prior[10], 7.0 * 2 * np.pi / (10 * 11))
    assert np.isclose(comp.P_Cl_prior[40] / comp.P_Cl_prior[10],
                      (40 / 10)**-0.5 * (10 * 11) / (40 * 41))


def test_P_Cl_prior_gaussian_rolloff_floors_at_1e_minus_10() -> None:
    # 600 arcmin FWHM: at l=100 the exponential is ~1e-24, far below C3's relative 1e-10 floor.
    params = _dust_params(lmax=100, Cl_prior_amplitude=1.0, Cl_prior_FWHM=600.0)
    comp = ThermalDust(params, _make_compsep(), eval_pol="I", comp_name="dust")

    sigma = np.deg2rad(10.0) / np.sqrt(8 * np.log(2))
    np.testing.assert_allclose(comp.P_Cl_prior[5], np.exp(-30 * sigma**2) * 2 * np.pi / 30)
    np.testing.assert_allclose(comp.P_Cl_prior[100], 1e-10 * 2 * np.pi / (100 * 101))
    # The floor keeps the prior strictly positive and its inverse finite.
    assert np.all(comp.P_Cl_prior > 0)
    np.testing.assert_allclose(comp.P_Cl_prior * comp.P_Cl_prior_inv, 1.0)


def test_P_Cl_prior_resolves_per_pol_lists() -> None:
    # Like nu_ref, every Cl_prior parameter can be an [I, QU] pair resolved per execution view.
    def make(eval_pol):
        params = _dust_params(lmax=4, Cl_prior_amplitude=[2.0, 8.0], Cl_prior_beta=[0.0, -0.5],
                              Cl_prior_FWHM=[0.0, 10.0])
        return ThermalDust(params, _make_compsep(), eval_pol=eval_pol, comp_name="dust")

    dust_I, dust_QU = make("I"), make("QU")
    assert (dust_I.Cl_prior_amplitude, dust_I.Cl_prior_beta, dust_I.Cl_prior_FWHM) == (2.0, 0.0, 0.0)
    assert (dust_QU.Cl_prior_amplitude, dust_QU.Cl_prior_beta, dust_QU.Cl_prior_FWHM) \
        == (8.0, -0.5, 10.0)


def test_P_Cl_prior_l_apod_defaults_to_no_taper() -> None:
    # C3's own parameter files nearly always set COMP_L_APOD equal to COMP_AMP_LMAX, which makes
    # get_Cl_apod unity everywhere; that is our default too.
    params = _dust_params(lmax=32, Cl_prior_amplitude=1.0)
    comp = ThermalDust(params, _make_compsep(), eval_pol="I", comp_name="dust")

    assert comp.Cl_prior_l_apod == 32
    np.testing.assert_array_equal(comp.Cl_prior_apodization, np.ones(33))


def test_P_Cl_prior_l_apod_tapers_to_1e_minus_6_in_power_at_lmax() -> None:
    # get_Cl_apod (comm_cl_mod.f90): unity up to l_apod, then a Gaussian reaching exp(-ln(1000))
    # in amplitude at lmax, i.e. 1e-6 in power. The prior enters as C_l * f^2.
    params = _dust_params(lmax=100, Cl_prior_amplitude=1.0, Cl_prior_l_apod=50)
    comp = ThermalDust(params, _make_compsep(), eval_pol="I", comp_name="dust")
    unapodized = ThermalDust(_dust_params(lmax=100, Cl_prior_amplitude=1.0), _make_compsep(),
                             eval_pol="I", comp_name="dust")

    f = comp.Cl_prior_apodization
    np.testing.assert_array_equal(f[:51], np.ones(51))
    assert np.all(np.diff(f[50:]) < 0.0)                      # strictly falling above l_apod
    assert np.isclose(f[100], np.exp(-np.log(1e3) * (100 - 50)**2 / (100 - 50 + 1)**2))
    assert 1e-3 < f[100] < 1.5e-3                             # ~1e-3 amplitude, ~1e-6 in power
    # Only the tail is touched, and it is suppressed by f^2.
    np.testing.assert_allclose(comp.P_Cl_prior[:51], unapodized.P_Cl_prior[:51])
    np.testing.assert_allclose(comp.P_Cl_prior[100], unapodized.P_Cl_prior[100] * f[100]**2)
    assert np.all(comp.P_Cl_prior > 0)                        # 1/C_l stays finite


def test_P_Cl_prior_l_apod_resolves_per_pol_like_the_other_prior_parameters() -> None:
    def make(eval_pol):
        params = _dust_params(lmax=16, Cl_prior_amplitude=1.0, Cl_prior_l_apod=[16, 8])
        return ThermalDust(params, _make_compsep(), eval_pol=eval_pol, comp_name="dust")

    assert make("I").Cl_prior_l_apod == 16
    assert make("QU").Cl_prior_l_apod == 8


def test_P_Cl_prior_none_amplitude_gives_identity() -> None:
    # Cl_prior_amplitude=None: C_l = 1, i.e. no S^{1/2} scaling in the CG reparameterization
    # (the C3 CL_TYPE 'none' analogue).
    comp = ThermalDust(_dust_params(lmax=8), _make_compsep(), eval_pol="I", comp_name="dust")

    np.testing.assert_array_equal(comp.P_Cl_prior, np.ones(9))
    np.testing.assert_array_equal(comp.P_Cl_prior_inv, np.ones(9))


def test_old_smoothing_prior_params_are_rejected() -> None:
    # The old C_l-space 'smoothing_prior_*' parameters changed semantics; fail loudly on stale files.
    with pytest.raises(ValueError, match=r"smoothing_prior_\*"):
        ThermalDust(_dust_params(smoothing_prior_amplitude=1.0e7), _make_compsep(),
                    eval_pol="I", comp_name="dust")


def test_init_map_to_amplitude_is_noop_when_units_match() -> None:
    compsep = _make_compsep()
    gibbs = Bunch()
    dust = ThermalDust(_dust_params(units="uK_RJ"), compsep, eval_pol="I", comp_name="dust")
    arr = np.ones((1, 12))

    # uK_RJ already equals the dust amplitude unit, so the same array is returned untouched.
    assert dust.init_map_to_amplitude(arr) is arr


def test_init_map_to_amplitude_converts_to_amplitude_unit() -> None:
    import pysm3.units as pysm3u

    compsep = _make_compsep()
    gibbs = Bunch()
    dust = ThermalDust(_dust_params(units="uK_CMB", nu_ref=100.0), compsep,
                       eval_pol="I", comp_name="dust")
    expected = (1 * pysm3u.Unit("uK_CMB")).to(
        pysm3u.uK_RJ, equivalencies=pysm3u.cmb_equivalencies(100.0 * pysm3u.GHz)).value

    out = dust.init_map_to_amplitude(np.ones((1, 12)))

    assert np.allclose(out, expected)


def test_cmb_init_map_to_amplitude_converts_uK_CMB_at_nu_ref() -> None:
    import pysm3.units as pysm3u

    compsep = _make_compsep()
    gibbs = Bunch()
    cmb_params = _make_component_cfg("IQU").params
    cmb_params.units = "uK_CMB"
    cmb = CMB(cmb_params, compsep, eval_pol="I", comp_name="cmb")

    # CMB amplitudes are stored in uK_RJ referenced to nu_ref (default 1 GHz), so a uK_CMB disk map
    # is converted to uK_RJ at nu_ref.
    assert cmb.nu_ref == 1.0
    expected = (1 * pysm3u.Unit("uK_CMB")).to(
        pysm3u.uK_RJ, equivalencies=pysm3u.cmb_equivalencies(1.0 * pysm3u.GHz)).value
    out = cmb.init_map_to_amplitude(np.ones((1, 12)))
    assert np.allclose(out, expected)


def test_cmb_get_sed_is_unity_at_nu_ref_and_ratio_elsewhere() -> None:
    import pysm3.units as pysm3u

    compsep = _make_compsep()
    gibbs = Bunch()
    cmb_params = _make_component_cfg("IQU").params
    cmb_params.nu_ref = 100.0
    cmb = CMB(cmb_params, compsep, eval_pol="I", comp_name="cmb")

    # The SED is normalized to 1 at the reference frequency and is the ratio of the
    # thermodynamic-to-RJ conversion elsewhere.
    assert np.isclose(cmb.get_sed(100.0), 1.0)
    def g(f):
        return (1 * pysm3u.uK_CMB).to(
            pysm3u.uK_RJ, equivalencies=pysm3u.cmb_equivalencies(f * pysm3u.GHz)).value
    assert np.isclose(cmb.get_sed(353.0), g(353.0) / g(100.0))

# ===================================================================
# SED parameters in the compsep chain (Component.sed_param_names)
# ===================================================================

def _make_dust_cfg(nu_ref, polarization: str = "IQU") -> Bunch:
    cfg = Bunch(
        enabled=True,
        component_class="ThermalDust",
        params=Bunch(
            lmax=1,
            polarization=polarization,
            shortname="dust",
            spatially_varying_MM=False,
            Cl_prior_amplitude=None,
            beta=1.54,
            T=20.0,
            nu_ref=nu_ref,
        ),
    )
    object.__setattr__(cfg, "_name", "dust")
    return cfg


def test_every_component_class_declares_its_sed_parameters() -> None:
    """A component whose SED parameters are undeclared writes an empty `sed/` group to the chain."""
    from commander4.sky import CMB, FreeFree, Synchrotron, ThermalDust, SpinningDust, RadioSources
    expected = {CMB: ("nu_ref",), ThermalDust: ("beta", "T", "nu_ref"),
                Synchrotron: ("beta", "nu_ref"), FreeFree: ("T", "nu_ref"),
                SpinningDust: ("nu_peak_eval", "nu_peak_ref", "nu_0"), RadioSources: ("nu_ref",)}
    for cls, names in expected.items():
        assert cls.sed_param_names == names, cls.__name__


def test_sed_parameters_are_readable_off_a_joined_component() -> None:
    """The names the chain writer stores must all resolve on the component it is handed."""
    params = Bunch(compsep=_make_compsep())
    comp_list = CompList.init_from_params(Bunch({"dust": _make_dust_cfg(353.0)}), params)
    dust = comp_list.joined()[0]

    stored = {name: getattr(dust, name) for name in dust.sed_param_names}
    assert stored == {"beta": 1.54, "T": 20.0, "nu_ref": 353.0}


def test_joining_restores_a_per_polarization_nu_ref() -> None:
    """`nu_ref: [I, QU]` is common; the joined view must not silently keep only the I value.

    Joining deep-copies the intensity view, whose `nu_ref` has already been resolved by `_per_pol`,
    so without care an Akari-style `[857, 353]` would be recorded in the chain as 857.
    """
    params = Bunch(compsep=_make_compsep())
    comp_list = CompList.init_from_params(Bunch({"dust": _make_dust_cfg([857.0, 353.0])}), params)

    assert [comp.nu_ref for comp in comp_list] == [857.0, 353.0]  # the split views
    np.testing.assert_array_equal(comp_list.joined()[0].nu_ref, [857.0, 353.0])
    # Parameters that are not per-polarization stay scalar rather than becoming a degenerate pair.
    assert comp_list.joined()[0].beta == 1.54


def test_restarting_from_a_chain_restores_a_sampled_spectral_index(tmp_path):
    """Continuing a chain must continue its MH index walk, not reset beta to the start value."""
    import h5py
    from commander4.sky.comp_io import _restore_sampled_sed_params_from_chain

    chain = tmp_path / "chain01_iter0007.h5"
    with h5py.File(chain, "w") as f:
        f["comps/dust/sed/beta"] = 1.5311
        f["comps/dust/sed/T"] = 25.0        # not sampled -> must NOT be restored
        f["comps/dust/sed/nu_ref"] = 217.0  # not sampled -> must NOT be restored

    cfg = _make_dust_cfg(353.0)
    cfg.params.sample_spectral_index = True
    comp = CompList.init_from_params(Bunch({"dust": cfg}),
                                     Bunch(compsep=_make_compsep()))[0]
    assert comp.beta == 1.54
    _restore_sampled_sed_params_from_chain(comp, str(chain))

    assert comp.beta == pytest.approx(1.5311)   # sampled: taken from the chain
    assert comp.T == 20.0                       # fixed: the parameter file still rules
    assert comp.nu_ref == 353.0


def test_a_fixed_spectral_index_is_not_restored_from_a_chain(tmp_path):
    """Without `sample_spectral_index`, beta stays a parameter-file setting the chain cannot
    override."""
    import h5py
    from commander4.sky.comp_io import _restore_sampled_sed_params_from_chain

    chain = tmp_path / "chain01_iter0007.h5"
    with h5py.File(chain, "w") as f:
        f["comps/dust/sed/beta"] = 1.20

    comp = CompList.init_from_params(Bunch({"dust": _make_dust_cfg(353.0)}),
                                     Bunch(compsep=_make_compsep()))[0]
    _restore_sampled_sed_params_from_chain(comp, str(chain))
    assert comp.beta == 1.54


# ===================================================================
# Point sources (RadioSources): units, beam painting and flux
# ===================================================================

def _write_radio_source_table(path, glon=0.0, glat=0.0, flux_mjy=1000.0, alpha=-0.7) -> None:
    """A minimal two-line RadioSources template: a header comment plus one source."""
    with open(path, "w") as handle:
        handle.write("# Glon(deg) Glat(deg) I(mJy) alpha_I\n")
        handle.write(f"{glon} {glat} {flux_mjy} {alpha}\n")


def _make_radio_sources(template_path, nu_ref=30.0):
    from commander4.sky import RadioSources
    cfg = Bunch(shortname="radsources", nu_0=nu_ref, template_path=str(template_path))
    return RadioSources(cfg, _make_compsep(), comp_name="radsources")


def _map_integral(sky_map, nside):
    """The map's integral over the sphere, which is the source's total flux in uK_RJ*sr."""
    return float(sky_map.sum()*hp.nside2pixarea(nside))


def test_radio_sources_component_map_is_frequency_independent(tmp_path) -> None:
    """The amplitude map carries no SED, so no evaluation frequency may leak into it."""
    template = tmp_path / "radio.dat"
    _write_radio_source_table(template)
    comp = _make_radio_sources(template, nu_ref=30.0)

    amp_map = comp.get_component_map(nside=64, fwhm=np.deg2rad(2.0))

    assert amp_map.shape == (1, hp.nside2npix(64))
    assert np.isfinite(amp_map).all()
    assert amp_map.max() > 0.0
    assert np.array_equal(amp_map, comp.get_component_map(nside=64, fwhm=np.deg2rad(2.0)))


def test_radio_sources_component_map_scales_with_the_reference_frequency(tmp_path) -> None:
    """Flux to brightness temperature goes as nu_ref^-2, so 2x nu_ref gives a 4x smaller map."""
    template = tmp_path / "radio.dat"
    _write_radio_source_table(template)

    beam = np.deg2rad(2.0)
    low = _make_radio_sources(template, nu_ref=30.0).get_component_map(nside=64, fwhm=beam)
    high = _make_radio_sources(template, nu_ref=60.0).get_component_map(nside=64, fwhm=beam)

    assert low.max() > 0.0
    assert high.max() == pytest.approx(low.max()/4.0, rel=1e-5)


def test_radio_sources_sed_follows_the_commander3_radio_law(tmp_path) -> None:
    """C3's evalSED_ptsrc for 'radio' is (nu/nu_ref)^(-2+alpha); C4 must agree."""
    template = tmp_path / "radio.dat"
    _write_radio_source_table(template, alpha=-0.7)
    comp = _make_radio_sources(template, nu_ref=30.0)

    assert comp.get_sed(30.0)[0] == pytest.approx(1.0)
    assert comp.get_sed(120.0)[0] == pytest.approx((120.0/30.0)**(-2.0 - 0.7))


def test_radio_sources_sky_equals_amplitude_times_sed(tmp_path) -> None:
    """The flux conversion belongs at nu_ref only.

    Applying it at the band frequency as well double-counts the nu^-2 already inside `get_sed`,
    which made `get_sky(nu)` disagree with `get_component_map() * get_sed(nu)` by (nu_ref/nu)^2.
    """
    template = tmp_path / "radio.dat"
    _write_radio_source_table(template)
    comp = _make_radio_sources(template, nu_ref=30.0)

    amplitude = comp.get_component_map(nside=64, fwhm=np.deg2rad(2.0))
    for nu in (30.0, 100.0, 353.0):
        sky = comp.get_sky(nu, nside=64, fwhm=np.deg2rad(2.0))
        expected = amplitude*comp.get_sed(nu)[0]
        assert sky == pytest.approx(expected, rel=1e-5)


def test_radio_sources_conserve_flux_across_resolutions(tmp_path) -> None:
    """The painted beam is normalized, so the source's total flux must not depend on nside."""
    template = tmp_path / "radio.dat"
    _write_radio_source_table(template)

    integrals = [_map_integral(_make_radio_sources(template).get_component_map(
                     nside=n, fwhm=np.deg2rad(1.0)), n)
                 for n in (16, 64, 256)]

    assert integrals[0] > 0.0
    for integral in integrals[1:]:
        assert integral == pytest.approx(integrals[0], rel=1e-4)


def test_radio_sources_beam_narrower_than_a_pixel_still_paints_the_source(tmp_path) -> None:
    """A beam below the pixel scale used to select no pixels at all, silently dropping the source.

    It now collapses to the single pixel containing the source, carrying the whole flux.
    """
    template = tmp_path / "radio.dat"
    _write_radio_source_table(template)
    comp = _make_radio_sources(template)

    narrow = comp.get_component_map(nside=16, fwhm=np.deg2rad(1.0/60))
    resolved = _make_radio_sources(template).get_component_map(nside=16, fwhm=np.deg2rad(1.0))

    assert np.count_nonzero(narrow) == 1
    assert _map_integral(narrow, 16) == pytest.approx(_map_integral(resolved, 16), rel=1e-4)


def test_radio_sources_take_fwhm_in_radians_like_diffuse_components(tmp_path) -> None:
    """`SkyModel.get_sky_at_nu` passes the band's `fwhm_rad` to whatever components it holds, so
    both families have to read that argument the same way.

    `RadioSources` used to take arcmin here. Given radians it then painted every source through a
    beam ~3400x too narrow, collapsing each to a single pixel; the resulting ragged arrays crashed
    the numba projection kernel rather than quietly returning a wrong sky.
    """
    template = tmp_path / "radio.dat"
    _write_radio_source_table(template)
    one_degree = np.deg2rad(1.0)
    painted = _make_radio_sources(template).get_component_map(nside=64, fwhm=one_degree)

    # A one-degree beam at nside 64 (55' pixels) must cover more than the source's own pixel.
    assert np.count_nonzero(painted) > 1
    # And it must be broader than what the old arcmin reading of the same number would give.
    as_arcmin = _make_radio_sources(template).get_component_map(nside=64,
                                                                fwhm=np.deg2rad(1.0/60))
    assert np.count_nonzero(painted) > np.count_nonzero(as_arcmin)


def _sky_model_at_amp_fwhm(amp_fwhm_rad):
    """A two-component sky model whose amplitudes carry `amp_fwhm_rad` of smoothing."""
    from commander4.sky.sky_model import SkyModel
    comp_list = _make_comp_list("I")   # intensity only: lmax here is 1, too low for spin-2 QU
    for comp in comp_list:
        comp.amp_fwhm_rad = amp_fwhm_rad
    return SkyModel(comp_list)


def test_sky_model_amp_fwhm_is_the_coarsest_component() -> None:
    """A model is only as sharp as its blurriest component, or the sky mixes two resolutions."""
    sky = _sky_model_at_amp_fwhm(0.0)
    assert sky.amp_fwhm_rad == 0.0

    sky._components[0].amp_fwhm_rad = np.deg2rad(100.0/60)
    assert sky.amp_fwhm_rad == pytest.approx(np.deg2rad(100.0/60))


def test_deconvolved_amplitudes_can_be_realized_at_any_beam() -> None:
    """The CG solver leaves `amp_fwhm_rad` zero, so nothing is out of reach."""
    sky = _sky_model_at_amp_fwhm(0.0)
    for fwhm in (0.0, np.deg2rad(1.0), None):
        assert sky.get_sky_at_nu(100.0, 2, "I", fwhm=fwhm).shape == (1, 12*2**2)


def test_asking_for_a_sharper_sky_than_the_amplitudes_hold_is_refused() -> None:
    """Per-pixel amplitudes carry the common beam; a finer sky simply does not exist, and
    silently handing back a coarser one hides that from whoever asked."""
    sky = _sky_model_at_amp_fwhm(np.deg2rad(100.0/60))

    with pytest.raises(ValueError, match="cannot be realized any sharper"):
        sky.get_sky_at_nu(100.0, 2, "I", fwhm=np.deg2rad(30.0/60))


def test_none_means_the_sharpest_this_model_can_give() -> None:
    """The way for a caller to say "best available" without naming a number."""
    sky = _sky_model_at_amp_fwhm(np.deg2rad(100.0/60))

    at_none = sky.get_sky_at_nu(100.0, 2, "I", fwhm=None)
    at_amp = sky.get_sky_at_nu(100.0, 2, "I", fwhm=sky.amp_fwhm_rad)

    assert np.array_equal(at_none, at_amp)


def test_the_amplitudes_own_beam_is_not_treated_as_too_sharp() -> None:
    """`amp_fwhm_rad` is set from a band's own fwhm, so equality here must not trip the check."""
    sky = _sky_model_at_amp_fwhm(np.deg2rad(100.0/60))

    assert sky.get_sky_at_nu(100.0, 2, "I", fwhm=np.deg2rad(100.0/60)).shape == (1, 12*2**2)
