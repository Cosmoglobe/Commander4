from copy import deepcopy

import h5py
import healpy as hp
import numpy as np
import pytest
from pixell.bunch import Bunch

from commander4.sky_models.component import CMB, CompList, PointSourcesComponent, ThermalDust
from commander4.sky_models.sky_model import build_initial_sky_model
from commander4.utils.math_operations import complist_dot, map_to_alm


def _make_general(ntask_compsep_qu: int = 1, ntask_compsep_i: int = 1) -> Bunch:
    return Bunch(
        nside=2,
        CG_float_precision="single",
        MPI_config=Bunch(ntask_compsep_I=ntask_compsep_i, ntask_compsep_QU=ntask_compsep_qu),
    )


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
    params = Bunch(general=_make_general(ntask_compsep_qu))
    cmb = _make_component_cfg(polarization)
    object.__setattr__(cmb, "_name", "cmb")
    components = Bunch({"cmb": cmb})
    return CompList.init_from_params(components, params)


def _make_named_component_cfg(shortname: str, polarization: str = "IQU") -> Bunch:
    cfg = _make_component_cfg(polarization)
    cfg.params.shortname = shortname
    return cfg


def _make_multi_comp_list() -> CompList:
    params = Bunch(general=_make_general())
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
    params = Bunch(general=_make_general())
    components = Bunch({"cmb": _make_component_cfg("I")})

    with pytest.raises(AttributeError, match="_name"):
        CompList.init_from_params(components, params)


def test_init_from_params_does_not_mutate_component_params_name() -> None:
    params = Bunch(general=_make_general())
    component_cfg = _make_component_cfg("I")
    object.__setattr__(component_cfg, "_name", "cmb")
    components = Bunch({"cmb": component_cfg})

    CompList.init_from_params(components, params)

    assert "_name" not in component_cfg.params


def test_init_from_params_builds_all_defined_pol_views() -> None:
    # Construction is independent of the MPI/compsep layout: an IQU component always yields both an
    # I and a QU view, even when zero compsep ranks are configured.
    params = Bunch(general=_make_general(ntask_compsep_qu=0, ntask_compsep_i=0))
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


def test_complist_ops_require_matching_execution_views() -> None:
    comp_list = _make_comp_list("IQU")
    other = deepcopy(comp_list)
    other.comp_list.reverse()

    with pytest.raises(ValueError, match="same execution views"):
        _ = comp_list + other
    with pytest.raises(ValueError, match="same execution views"):
        _ = complist_dot(comp_list, other)


def test_complist_split_for_eval_pol_rejects_invalid_polarization(caplog) -> None:
    comp_list = _make_comp_list("IQU")

    with pytest.raises(AssertionError):
        comp_list.split_for_eval_pol("bad")
    assert "Unsupported polarization string bad" in caplog.text


def test_point_sources_component_rejects_non_intensity_eval_pol(caplog) -> None:
    params = Bunch(shortname="ps")

    with pytest.raises(AssertionError):
        PointSourcesComponent(params, _make_general(), comp_name="ps", eval_pol="QU")
    assert "PointSourcesComponent does not support evaluation polarization 'QU'" in caplog.text


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

    general = _make_general()
    general.init_chain_path = str(chain)
    cmb = _make_named_component_cfg("cmb", "IQU")
    object.__setattr__(cmb, "_name", "cmb")
    ff = _make_named_component_cfg("ff", "I")
    object.__setattr__(ff, "_name", "ff")
    params = Bunch(general=general, components=Bunch({"cmb": cmb, "ff": ff}))

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

    general = _make_general()
    general.init_chain_path = str(global_chain)
    cmb = _make_named_component_cfg("cmb", "IQU")
    object.__setattr__(cmb, "_name", "cmb")
    cmb.params.init_from = str(special_chain)  # Per-component path takes precedence over the global one.
    params = Bunch(general=general, components=Bunch({"cmb": cmb}))

    comp_list = CompList.init_from_params(params.components, params)
    comp_list.load_initial_alms(params)

    assert all(np.all(comp.alms == 5.0) for comp in comp_list)


def test_load_initial_alms_leaves_zeros_without_a_source() -> None:
    general = _make_general()  # No init_chain_path, and no per-component init_from.
    cmb = _make_named_component_cfg("cmb", "IQU")
    object.__setattr__(cmb, "_name", "cmb")
    params = Bunch(general=general, components=Bunch({"cmb": cmb}))

    comp_list = CompList.init_from_params(params.components, params)
    comp_list.load_initial_alms(params)

    assert all(np.all(comp.alms == 0) for comp in comp_list)


def test_load_initial_alms_from_fits_map(tmp_path) -> None:
    nside = 2
    lmax = 3
    npix = 12 * nside**2
    iqu_map = np.zeros((3, npix), dtype=np.float64)
    iqu_map[0] = 1.0 + np.arange(npix)  # Distinct I, Q, U so a wrong row would be detectable.
    iqu_map[1] = 2.0
    iqu_map[2] = 3.0
    fits_path = tmp_path / "init_map.fits"
    hp.write_map(str(fits_path), iqu_map, overwrite=True, dtype=np.float64)

    general = _make_general()
    general.CG_float_precision = "double"  # So component alms match map_to_alm output exactly.
    cmb = _make_named_component_cfg("cmb", "IQU")
    cmb.params.lmax = lmax
    object.__setattr__(cmb, "_name", "cmb")
    cmb.params.init_from = str(fits_path)
    params = Bunch(general=general, components=Bunch({"cmb": cmb}))

    comp_list = CompList.init_from_params(params.components, params)
    comp_list.load_initial_alms(params)

    views = {(comp.comp_name, comp.eval_pol): comp for comp in comp_list}
    expected_I = map_to_alm(np.ascontiguousarray(iqu_map[0:1]), nside, lmax, spin=0)
    expected_QU = map_to_alm(np.ascontiguousarray(iqu_map[1:3]), nside, lmax, spin=2)
    assert np.allclose(views[("cmb", "I")].alms, expected_I)
    assert np.allclose(views[("cmb", "QU")].alms, expected_QU)


def test_load_initial_alms_rejects_unknown_extension(tmp_path) -> None:
    general = _make_general()
    cmb = _make_named_component_cfg("cmb", "IQU")
    object.__setattr__(cmb, "_name", "cmb")
    cmb.params.init_from = str(tmp_path / "init_map.txt")
    params = Bunch(general=general, components=Bunch({"cmb": cmb}))

    comp_list = CompList.init_from_params(params.components, params)
    with pytest.raises(AssertionError):
        comp_list.load_initial_alms(params)


def test_load_initial_alms_partial_source_leaves_missing_pol_zero(tmp_path) -> None:
    # An intensity-only chain initializes the I view; the IQU component's QU view stays at zero
    # rather than erroring (so e.g. I-from-chain + QU-from-zero works).
    nalm = (1 + 1) * (1 + 2) // 2
    cmb_intensity_only = (np.arange(nalm, dtype=np.float64) + 1).reshape(1, nalm).astype(np.complex64)
    chain = tmp_path / "intensity_only.h5"
    _write_chain_alms(chain, {"cmb": cmb_intensity_only})

    general = _make_general()
    general.init_chain_path = str(chain)
    cmb = _make_named_component_cfg("cmb", "IQU")
    object.__setattr__(cmb, "_name", "cmb")
    params = Bunch(general=general, components=Bunch({"cmb": cmb}))

    comp_list = CompList.init_from_params(params.components, params)
    comp_list.load_initial_alms(params)

    views = {(comp.comp_name, comp.eval_pol): comp for comp in comp_list}
    assert np.array_equal(views[("cmb", "I")].alms, cmb_intensity_only)
    assert np.all(views[("cmb", "QU")].alms == 0)


def test_load_initial_alms_missing_component_logs_error_and_continues(tmp_path, caplog) -> None:
    nalm = (1 + 1) * (1 + 2) // 2
    chain = tmp_path / "other_components.h5"
    _write_chain_alms(chain, {"dust": np.ones((3, nalm), dtype=np.complex64)})  # no "cmb" entry

    general = _make_general()
    general.init_chain_path = str(chain)
    cmb = _make_named_component_cfg("cmb", "IQU")
    object.__setattr__(cmb, "_name", "cmb")
    params = Bunch(general=general, components=Bunch({"cmb": cmb}))

    comp_list = CompList.init_from_params(params.components, params)
    with caplog.at_level("ERROR"):
        comp_list.load_initial_alms(params)  # must not raise

    assert all(np.all(comp.alms == 0) for comp in comp_list)
    assert "not found" in caplog.text


def test_build_initial_sky_model_returns_realizable_model() -> None:
    general = _make_general()  # No init paths -> zero alms -> zero sky.
    cmb = _make_named_component_cfg("cmb", "IQU")
    cmb.params.lmax = 2  # Spin-2 (QU) synthesis requires lmax >= 2.
    object.__setattr__(cmb, "_name", "cmb")
    params = Bunch(general=general, components=Bunch({"cmb": cmb}))

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
    general = _make_general()
    dust_I = ThermalDust(_dust_params(), general, eval_pol="I", comp_name="dust")
    dust_QU = ThermalDust(_dust_params(), general, eval_pol="QU", comp_name="dust")

    # nu_ref = [nu_I, nu_QU]: each view picks its own entry.
    assert dust_I.nu_ref == 857.0
    assert dust_QU.nu_ref == 353.0
    # The SED is normalized to 1 at each view's own reference frequency.
    assert np.isclose(dust_I.get_sed(857.0), 1.0)
    assert np.isclose(dust_QU.get_sed(353.0), 1.0)


def test_scalar_reference_frequency_is_shared_by_both_polarizations() -> None:
    general = _make_general()
    dust_I = ThermalDust(_dust_params(nu_ref=545.0), general, eval_pol="I", comp_name="dust")
    dust_QU = ThermalDust(_dust_params(nu_ref=545.0), general, eval_pol="QU", comp_name="dust")

    # A scalar nu_ref applies to both I and QU.
    assert dust_I.nu_ref == 545.0 and dust_QU.nu_ref == 545.0


def test_P_Cl_prior_flat_Dl_falls_as_ell_squared_in_Cl() -> None:
    # C3 convention (comm_cl_mod.f90): the prior is defined in D_l space, so a flat amplitude
    # "roof" corresponds to C_l = 2*pi*amp/(l(l+1)), with C_0 := D_0 := D_1.
    params = _dust_params(lmax=16, Cl_prior_amplitude=100.0, Cl_prior_beta=0.0,
                          Cl_prior_l_pivot=50, Cl_prior_FWHM=0.0)
    comp = ThermalDust(params, _make_general(), eval_pol="I", comp_name="dust")

    ells = np.arange(1, 17)
    np.testing.assert_allclose(comp.P_Cl_prior[1:], 100.0 * 2 * np.pi / (ells * (ells + 1)))
    assert comp.P_Cl_prior[0] == 100.0


def test_P_Cl_prior_power_law_pivot_and_tilt() -> None:
    params = _dust_params(lmax=100, Cl_prior_amplitude=7.0, Cl_prior_beta=-0.5,
                          Cl_prior_l_pivot=10, Cl_prior_FWHM=0.0)
    comp = ThermalDust(params, _make_general(), eval_pol="I", comp_name="dust")

    # D_l equals the amplitude at the pivot, and scales as (l/l_pivot)^beta away from it.
    assert np.isclose(comp.P_Cl_prior[10], 7.0 * 2 * np.pi / (10 * 11))
    assert np.isclose(comp.P_Cl_prior[40] / comp.P_Cl_prior[10],
                      (40 / 10)**-0.5 * (10 * 11) / (40 * 41))


def test_P_Cl_prior_gaussian_rolloff_floors_at_1e_minus_10() -> None:
    # 600 arcmin FWHM: at l=100 the exponential is ~1e-24, far below C3's relative 1e-10 floor.
    params = _dust_params(lmax=100, Cl_prior_amplitude=1.0, Cl_prior_FWHM=600.0)
    comp = ThermalDust(params, _make_general(), eval_pol="I", comp_name="dust")

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
        return ThermalDust(params, _make_general(), eval_pol=eval_pol, comp_name="dust")

    dust_I, dust_QU = make("I"), make("QU")
    assert (dust_I.Cl_prior_amplitude, dust_I.Cl_prior_beta, dust_I.Cl_prior_FWHM) == (2.0, 0.0, 0.0)
    assert (dust_QU.Cl_prior_amplitude, dust_QU.Cl_prior_beta, dust_QU.Cl_prior_FWHM) \
        == (8.0, -0.5, 10.0)


def test_P_Cl_prior_none_amplitude_gives_identity() -> None:
    # Cl_prior_amplitude=None: C_l = 1, i.e. no S^{1/2} scaling in the CG reparameterization
    # (the C3 CL_TYPE 'none' analogue).
    comp = ThermalDust(_dust_params(lmax=8), _make_general(), eval_pol="I", comp_name="dust")

    np.testing.assert_array_equal(comp.P_Cl_prior, np.ones(9))
    np.testing.assert_array_equal(comp.P_Cl_prior_inv, np.ones(9))


def test_old_smoothing_prior_params_are_rejected() -> None:
    # The old C_l-space 'smoothing_prior_*' parameters changed semantics; fail loudly on stale files.
    with pytest.raises(AssertionError):
        ThermalDust(_dust_params(smoothing_prior_amplitude=1.0e7), _make_general(),
                    eval_pol="I", comp_name="dust")


def test_init_map_to_amplitude_is_noop_when_units_match() -> None:
    general = _make_general()
    dust = ThermalDust(_dust_params(units="uK_RJ"), general, eval_pol="I", comp_name="dust")
    arr = np.ones((1, 12))

    # uK_RJ already equals the dust amplitude unit, so the same array is returned untouched.
    assert dust.init_map_to_amplitude(arr) is arr


def test_init_map_to_amplitude_converts_to_amplitude_unit() -> None:
    import pysm3.units as pysm3u

    general = _make_general()
    dust = ThermalDust(_dust_params(units="uK_CMB", nu_ref=100.0), general,
                       eval_pol="I", comp_name="dust")
    expected = (1 * pysm3u.Unit("uK_CMB")).to(
        pysm3u.uK_RJ, equivalencies=pysm3u.cmb_equivalencies(100.0 * pysm3u.GHz)).value

    out = dust.init_map_to_amplitude(np.ones((1, 12)))

    assert np.allclose(out, expected)


def test_cmb_init_map_to_amplitude_converts_uK_CMB_at_nu_ref() -> None:
    import pysm3.units as pysm3u

    general = _make_general()
    cmb_params = _make_component_cfg("IQU").params
    cmb_params.units = "uK_CMB"
    cmb = CMB(cmb_params, general, eval_pol="I", comp_name="cmb")

    # CMB amplitudes are stored in uK_RJ referenced to nu_ref (default 1 GHz), so a uK_CMB disk map
    # is converted to uK_RJ at nu_ref.
    assert cmb.nu_ref == 1.0
    expected = (1 * pysm3u.Unit("uK_CMB")).to(
        pysm3u.uK_RJ, equivalencies=pysm3u.cmb_equivalencies(1.0 * pysm3u.GHz)).value
    out = cmb.init_map_to_amplitude(np.ones((1, 12)))
    assert np.allclose(out, expected)


def test_cmb_get_sed_is_unity_at_nu_ref_and_ratio_elsewhere() -> None:
    import pysm3.units as pysm3u

    general = _make_general()
    cmb_params = _make_component_cfg("IQU").params
    cmb_params.nu_ref = 100.0
    cmb = CMB(cmb_params, general, eval_pol="I", comp_name="cmb")

    # The SED is normalized to 1 at the reference frequency and is the ratio of the
    # thermodynamic-to-RJ conversion elsewhere.
    assert np.isclose(cmb.get_sed(100.0), 1.0)
    def g(f):
        return (1 * pysm3u.uK_CMB).to(
            pysm3u.uK_RJ, equivalencies=pysm3u.cmb_equivalencies(f * pysm3u.GHz)).value
    assert np.isclose(cmb.get_sed(353.0), g(353.0) / g(100.0))