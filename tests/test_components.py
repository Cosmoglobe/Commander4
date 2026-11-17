import warnings

import h5py
import healpy as hp
import numpy as np
import pytest
from pixell.bunch import Bunch

from commander4.sky.comp_list import CompList
from commander4.sky.diffuse_components import CMB, ThermalDust
from commander4.sky.point_sources import PointSourcesComponent
from commander4.math_utils.alm import gaussian_random_alm
from commander4.math_utils.sht import alm_to_map


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
            Cl_prior_amplitude=None,  # no prior
        ),
    )


def _make_params(**components: Bunch) -> Bunch:
    """A parameter file with the given component blocks, by component name."""
    return Bunch(components=Bunch(**components), compsep=_make_compsep())


def _make_comp_list(polarization: str = "IQU") -> CompList:
    return CompList.init_from_params(_make_params(cmb=_make_component_cfg(polarization)))


def _make_named_component_cfg(shortname: str, polarization: str = "IQU") -> Bunch:
    cfg = _make_component_cfg(polarization)
    cfg.params.shortname = shortname
    return cfg


def test_init_from_params_builds_all_defined_pol_views() -> None:
    # Construction is independent of the MPI/compsep layout: an IQU component always yields both an
    # I and a QU view, even when zero compsep ranks are configured.
    comp_list = CompList.init_from_params(_make_params(
        cmb=_make_named_component_cfg("cmb", "IQU"), ff=_make_named_component_cfg("ff", "I")))

    assert [(comp.comp_name, comp.pol) for comp in comp_list] == [
        ("cmb", "I"), ("cmb", "QU"), ("ff", "I")]


def test_complist_split_preserves_names_and_join_restores_logical_component() -> None:
    comp_list = _make_comp_list("IQU")

    assert [comp.comp_name for comp in comp_list] == ["cmb", "cmb"]
    assert [comp.shortname for comp in comp_list] == ["cmb", "cmb"]
    assert [comp.pol for comp in comp_list] == ["I", "QU"]

    comp_list[0].amps[:] = 1.0 + 0.0j
    comp_list[1].amps[:] = 2.0 + 0.0j
    joined = comp_list.joined()

    assert len(joined) == 1
    assert joined[0].comp_name == "cmb"
    assert joined[0].shortname == "cmb"
    assert joined[0].pol == "IQU"
    assert np.all(joined[0].amps[0] == 1.0 + 0.0j)
    assert np.all(joined[0].amps[1:] == 2.0 + 0.0j)


def test_complist_copy_has_its_own_amplitudes_and_shares_the_rest() -> None:
    """A CG vector is a copy of the list. It must not change the original's amplitudes, and it
    must not duplicate what the components hold besides them."""
    comp_list = _make_comp_list("IQU")
    for comp in comp_list:
        comp.amps[:] = 1.0 + 0.0j

    copied = comp_list.copy()
    zeros = comp_list.zeros_like()
    for comp in copied:
        comp.amps[:] = 5.0 + 0.0j

    assert [comp.pol for comp in copied] == ["I", "QU"]
    assert all(np.all(comp.amps == 1.0 + 0.0j) for comp in comp_list)
    assert all(np.all(comp.amps == 0.0) for comp in zeros)
    assert all(new.cfg is old.cfg for new, old in zip(copied, comp_list))


def test_complist_vector_arithmetic_acts_on_every_component() -> None:
    x = _make_comp_list("IQU")
    y = x.copy()
    for value, (comp_x, comp_y) in enumerate(zip(x, y), start=1):
        comp_x.amps[:] = value + 0.0j        # 1 in the I view, 2 in the QU view.
        comp_y.amps[:] = 10.0*value + 0.0j

    x.add_scaled(y, 0.5)                     # x = x + 0.5*y = 6*value
    assert np.all(x[0].amps == 6.0 + 0.0j) and np.all(x[1].amps == 12.0 + 0.0j)
    x.scale_and_add(y, 2.0)                  # x = 2*x + y = 22*value
    assert np.all(x[0].amps == 22.0 + 0.0j) and np.all(x[1].amps == 44.0 + 0.0j)


def test_complist_dot_counts_each_real_number_of_the_alms_once() -> None:
    """With every alm equal to 1, the dot product is the number of real numbers behind the alms
    that the solver uses: (lmax+1)^2 per polarization row. Here lmax = 1, and the QU view has two
    rows, so I and QU together give 4 + 2*4."""
    comp_list = _make_comp_list("IQU")
    for comp in comp_list:
        comp.amps[:] = 1.0 + 0.0j

    assert comp_list.dot(comp_list) == pytest.approx(12.0)


def test_point_sources_component_rejects_non_intensity_pol() -> None:
    params = _make_params(ps=Bunch(params=Bunch(polarization="I", shortname="ps")))

    with pytest.raises(ValueError, match="a PointSourcesComponent has no 'QU' view"):
        PointSourcesComponent(params, "ps", "QU")


def test_complist_split_for_pol_returns_requested_execution_view() -> None:
    comp_list = _make_comp_list("IQU")

    qu_only = comp_list.split_for_pol("QU")

    assert len(qu_only) == 1
    assert qu_only[0].pol == "QU"


def test_init_from_params_rejects_one_shortname_for_two_components() -> None:
    """The chain stores each component under its shortname."""
    params = _make_params(cmb=_make_named_component_cfg("same", "IQU"),
                          dust=_make_named_component_cfg("same", "I"))

    with pytest.raises(ValueError, match="Shortname 'same' is used for both 'cmb' and 'dust'"):
        CompList.init_from_params(params)


def _make_list(*configs: tuple[str, str], **settings) -> CompList:
    """A component list of CMB-class components, one per (name, polarization), with lmax 1.

    `settings` are further settings for the `params` block of every component.
    """
    components = Bunch()
    for name, polarization in configs:
        components[name] = _make_named_component_cfg(name, polarization)
        for key, value in settings.items():
            components[name].params[key] = value
    return CompList.init_from_params(Bunch(components=components, compsep=_make_compsep()))


def _write_chain(path, comp_list: CompList) -> str:
    """Write `comp_list` the way a CompSep chain file holds it, and return the path."""
    with h5py.File(path, "w") as f:
        comp_list.write_chain(f, {})
    return str(path)


def test_load_initial_state_reads_and_splits_from_chain(tmp_path) -> None:
    source = _make_list(("cmb", "IQU"), ("ff", "I"))
    for i, comp in enumerate(source):
        comp.amps[:] = np.arange(comp.amps.size).reshape(comp.amps.shape) + 10*i
    chain = _write_chain(tmp_path / "init_chain.h5", source)

    comp_list = _make_list(("cmb", "IQU"), ("ff", "I"))
    for comp in comp_list:
        comp.load_initial_state(chain, apply_init_from=True, restore_beta=False)

    # The chain holds the joined IQU alms; each view gets back exactly its own rows.
    for comp, saved in zip(comp_list, source):
        assert (comp.comp_name, comp.pol) == (saved.comp_name, saved.pol)
        assert np.array_equal(comp.amps, saved.amps)


def test_per_component_init_from_wins_only_for_a_new_chain(tmp_path) -> None:
    """A component's `init_from` is an initial guess: it beats the chain a new run starts from,
    but not the saved state a chain continues from (`chain_status: append`)."""
    source = _make_list(("cmb", "IQU"))
    global_chain = _write_chain(tmp_path / "global.h5", source)
    for comp in source:
        comp.amps[:] = 5.0
    special_chain = _write_chain(tmp_path / "special.h5", source)

    comp_list = _make_list(("cmb", "IQU"), init_from=special_chain)
    for comp in comp_list:
        comp.load_initial_state(global_chain, apply_init_from=True, restore_beta=False)
    assert all(np.all(comp.amps == 5.0) for comp in comp_list)

    for comp in comp_list:
        comp.load_initial_state(global_chain, apply_init_from=False, restore_beta=False)
    assert all(np.all(comp.amps == 0.0) for comp in comp_list)


def test_load_initial_state_from_fits_map(tmp_path) -> None:
    """`init_from` a FITS map recovers that map's alms, per polarization view.

    The map is built by synthesizing known alms, so it is band-limited at the component's lmax and
    the expected result is those alms exactly. That keeps the test on the contract ("the loader
    recovers the sky the map represents") rather than on which analysis transform is used: the
    loader inverts the synthesis (`pseudo_alm_to_map_inverse`), which a quadrature `map_to_alm`
    only approximates -- by ~1e-2 here.
    """
    nside = 8
    lmax = 3
    rng = np.random.default_rng(0)
    # Distinct I and QU alms, so a wrong row selection would be detectable.
    alms_I = gaussian_random_alm(lmax, lmax, 0, 1, rng).astype(np.complex128)
    alms_QU = gaussian_random_alm(lmax, lmax, 2, 2, rng).astype(np.complex128)
    iqu_map = np.vstack([alm_to_map(alms_I, nside, lmax, spin=0),
                         alm_to_map(alms_QU, nside, lmax, spin=2)])
    fits_path = tmp_path / "init_map.fits"
    hp.write_map(str(fits_path), iqu_map, overwrite=True, dtype=np.float64)

    compsep = _make_compsep()
    compsep.double_precision = True  # So component alms keep the map's precision.
    cmb = _make_named_component_cfg("cmb", "IQU")
    cmb.params.lmax = lmax
    cmb.params.init_from = str(fits_path)
    params = Bunch(compsep=compsep, components=Bunch({"cmb": cmb}))

    comp_list = CompList.init_from_params(params)
    for comp in comp_list:
        comp.load_initial_state(None, apply_init_from=True, restore_beta=False)

    views = {(comp.comp_name, comp.pol): comp for comp in comp_list}
    assert np.allclose(views[("cmb", "I")].amps, alms_I, rtol=1e-6, atol=1e-7)
    assert np.allclose(views[("cmb", "QU")].amps, alms_QU, rtol=1e-6, atol=1e-7)


def test_load_initial_state_rejects_unknown_extension(tmp_path) -> None:
    comp_list = _make_list(("cmb", "IQU"), init_from=str(tmp_path / "init_map.txt"))
    with pytest.raises(ValueError, match="expected a .h5/.hd5 chain or a .fits map"):
        for comp in comp_list:
            comp.load_initial_state(None, apply_init_from=True, restore_beta=False)


def test_load_initial_state_partial_source_leaves_missing_pol_zero(tmp_path) -> None:
    # An intensity-only chain initializes the I view; the IQU component's QU view stays at zero
    # rather than erroring (so e.g. I-from-chain + QU-from-zero works).
    source = _make_list(("cmb", "I"))
    source[0].amps[:] = np.arange(source[0].amps.size) + 1
    chain = _write_chain(tmp_path / "intensity_only.h5", source)

    comp_list = _make_list(("cmb", "IQU"))
    for comp in comp_list:
        comp.load_initial_state(chain, apply_init_from=True, restore_beta=False)

    views = {(comp.comp_name, comp.pol): comp for comp in comp_list}
    assert np.array_equal(views[("cmb", "I")].amps, source[0].amps)
    assert np.all(views[("cmb", "QU")].amps == 0)


def test_load_initial_state_missing_component_logs_error_and_continues(tmp_path, caplog) -> None:
    source = _make_list(("dust", "IQU"))  # no "cmb" entry
    chain = _write_chain(tmp_path / "other_components.h5", source)

    comp_list = _make_list(("cmb", "IQU"))
    with caplog.at_level("ERROR"):
        for comp in comp_list:
            # Must not raise.
            comp.load_initial_state(chain, apply_init_from=True, restore_beta=False)

    assert all(np.all(comp.amps == 0) for comp in comp_list)
    assert "not found" in caplog.text


def test_init_from_null_keeps_the_parameter_file_start(tmp_path) -> None:
    """`init_from: null` opts one component out of the chain a new run starts from."""
    source = _make_list(("cmb", "IQU"))
    for comp in source:
        comp.amps[:] = 5.0
    chain = _write_chain(tmp_path / "chain.h5", source)

    comp_list = _make_list(("cmb", "IQU"), init_from=None)
    for comp in comp_list:
        comp.load_initial_state(chain, apply_init_from=True, restore_beta=False)

    assert all(np.all(comp.amps == 0) for comp in comp_list)


def _dust_params(**overrides) -> Bunch:
    params = Bunch(polarization="IQU", shortname="dust", 
                   Cl_prior_amplitude=None, lmax=2,
                   beta=1.5, T=20.0, nu_ref=[857.0, 353.0], units="uK_RJ")
    for key, value in overrides.items():
        params[key] = value
    return params


def _dust(pol: str, **overrides) -> ThermalDust:
    """One view of a thermal dust component. `overrides` are settings of its `params` block."""
    params = _make_params(dust=Bunch(params=_dust_params(**overrides)))
    return ThermalDust(params, "dust", pol)


def test_diffuse_component_resolves_per_pol_reference_frequency() -> None:
    dust_I = _dust("I")
    dust_QU = _dust("QU")

    # nu_ref = [nu_I, nu_QU]: each view picks its own entry.
    assert dust_I.nu_ref == 857.0
    assert dust_QU.nu_ref == 353.0
    # The SED is normalized to 1 at each view's own reference frequency.
    assert np.isclose(dust_I.get_sed(857.0), 1.0)
    assert np.isclose(dust_QU.get_sed(353.0), 1.0)


def test_scalar_reference_frequency_is_shared_by_both_polarizations() -> None:
    dust_I = _dust("I", nu_ref=545.0)
    dust_QU = _dust("QU", nu_ref=545.0)

    # A scalar nu_ref applies to both I and QU.
    assert dust_I.nu_ref == 545.0 and dust_QU.nu_ref == 545.0


def test_P_Cl_prior_flat_Dl_falls_as_ell_squared_in_Cl() -> None:
    # C3 convention (comm_cl_mod.f90): the prior is defined in D_l space, so a flat amplitude
    # "roof" corresponds to C_l = 2*pi*amp/(l(l+1)), with C_0 := D_0 := D_1.
    comp = _dust("I", lmax=16, Cl_prior_amplitude=100.0, Cl_prior_beta=0.0, Cl_prior_l_pivot=50,
                 Cl_prior_FWHM=0.0)

    ells = np.arange(1, 17)
    np.testing.assert_allclose(comp.P_Cl_prior[1:], 100.0 * 2 * np.pi / (ells * (ells + 1)))
    assert comp.P_Cl_prior[0] == 100.0


def test_P_Cl_prior_power_law_pivot_and_tilt() -> None:
    comp = _dust("I", lmax=100, Cl_prior_amplitude=7.0, Cl_prior_beta=-0.5, Cl_prior_l_pivot=10,
                 Cl_prior_FWHM=0.0)

    # D_l equals the amplitude at the pivot, and scales as (l/l_pivot)^beta away from it.
    assert np.isclose(comp.P_Cl_prior[10], 7.0 * 2 * np.pi / (10 * 11))
    assert np.isclose(comp.P_Cl_prior[40] / comp.P_Cl_prior[10],
                      (40 / 10)**-0.5 * (10 * 11) / (40 * 41))


def test_P_Cl_prior_gaussian_rolloff_floors_at_1e_minus_10() -> None:
    # 600 arcmin FWHM: at l=100 the exponential is ~1e-24, far below C3's relative 1e-10 floor.
    comp = _dust("I", lmax=100, Cl_prior_amplitude=1.0, Cl_prior_FWHM=600.0)

    sigma = np.deg2rad(10.0) / np.sqrt(8 * np.log(2))
    np.testing.assert_allclose(comp.P_Cl_prior[5], np.exp(-30 * sigma**2) * 2 * np.pi / 30)
    np.testing.assert_allclose(comp.P_Cl_prior[100], 1e-10 * 2 * np.pi / (100 * 101))
    # The floor keeps the prior strictly positive and its inverse finite.
    assert np.all(comp.P_Cl_prior > 0)
    np.testing.assert_allclose(comp.P_Cl_prior * comp.P_Cl_prior_inv, 1.0)


def test_P_Cl_prior_resolves_per_pol_lists() -> None:
    # Like nu_ref, every Cl_prior parameter can be an [I, QU] pair resolved per execution view.
    def make(pol):
        return _dust(pol, lmax=4, Cl_prior_amplitude=[2.0, 8.0], Cl_prior_beta=[0.0, -0.5],
                     Cl_prior_FWHM=[0.0, 10.0])

    dust_I, dust_QU = make("I"), make("QU")
    assert (dust_I.Cl_prior_amplitude, dust_I.Cl_prior_beta, dust_I.Cl_prior_FWHM) == (2.0, 0.0, 0.0)
    assert (dust_QU.Cl_prior_amplitude, dust_QU.Cl_prior_beta, dust_QU.Cl_prior_FWHM) \
        == (8.0, -0.5, 10.0)


def test_P_Cl_prior_l_apod_defaults_to_no_taper() -> None:
    # C3's own parameter files nearly always set COMP_L_APOD equal to COMP_AMP_LMAX, which makes
    # get_Cl_apod unity everywhere; that is our default too.
    comp = _dust("I", lmax=32, Cl_prior_amplitude=1.0)

    assert comp.Cl_prior_l_apod == 32
    np.testing.assert_array_equal(comp.Cl_prior_apodization, np.ones(33))


def test_P_Cl_prior_l_apod_tapers_to_1e_minus_6_in_power_at_lmax() -> None:
    # get_Cl_apod (comm_cl_mod.f90): unity up to l_apod, then a Gaussian reaching exp(-ln(1000))
    # in amplitude at lmax, i.e. 1e-6 in power. The prior enters as C_l * f^2.
    comp = _dust("I", lmax=100, Cl_prior_amplitude=1.0, Cl_prior_l_apod=50)
    unapodized = _dust("I", lmax=100, Cl_prior_amplitude=1.0)

    f = comp.Cl_prior_apodization
    np.testing.assert_array_equal(f[:51], np.ones(51))
    assert np.all(np.diff(f[50:]) < 0.0)                      # strictly falling above l_apod
    assert np.isclose(f[100], np.exp(-np.log(1e3) * (100 - 50)**2 / (100 - 50 + 1)**2))
    assert 1e-3 < f[100] < 1.5e-3                             # ~1e-3 amplitude, ~1e-6 in power
    # Only the tail is touched, and it is suppressed by f^2.
    np.testing.assert_allclose(comp.P_Cl_prior[:51], unapodized.P_Cl_prior[:51])
    np.testing.assert_allclose(comp.P_Cl_prior[100], unapodized.P_Cl_prior[100] * f[100]**2)
    assert np.all(comp.P_Cl_prior > 0)                        # 1/C_l stays finite


def test_P_Cl_prior_l_tighten_shrinks_the_prior_below_it() -> None:
    # get_Cl_apod (comm_cl_mod.f90) with COMP_AMP_PRIOR_LMAX: below that multipole the factor on
    # S^{1/2} is (0.5 (cos(pi (max(l, 1) - l_tighten) / l_tighten) + 1))^2. The prior is C_l * f^2.
    comp = _dust("I", lmax=100, Cl_prior_amplitude=1.0, Cl_prior_l_tighten=60)
    untightened = _dust("I", lmax=100, Cl_prior_amplitude=1.0)
    assert untightened.Cl_prior_l_tighten == 0                # The default: no tightening.

    ells = np.arange(101)
    c3_factor = (0.5*(np.cos(np.pi*(np.maximum(ells, 1) - 60)/60) + 1))**2
    c3_factor[60:] = 1.0
    np.testing.assert_allclose(comp.P_Cl_prior, untightened.P_Cl_prior * c3_factor**2)
    f = comp.Cl_prior_apodization
    np.testing.assert_array_equal(f[60:], np.ones(41))        # Untouched from l_tighten up.
    assert np.all(np.diff(f[1:61]) > 0.0)                     # Tighter towards the large scales.
    assert np.isclose(f[30]**2, 1.0/16.0)                     # The variance at half of l_tighten.
    assert f[0] == f[1]
    assert np.all(comp.P_Cl_prior > 0)                        # 1/C_l stays finite


def test_P_Cl_prior_none_amplitude_gives_identity() -> None:
    # Cl_prior_amplitude=None: no prior (C3's CL_TYPE 'none'). The S^{1/2} scaling of the CG
    # reparameterization then does nothing; the CG solver leaves out the prior terms itself.
    comp = _dust("I", lmax=8)

    np.testing.assert_array_equal(comp.P_Cl_prior, np.ones(9))
    np.testing.assert_array_equal(comp.P_Cl_prior_inv, np.ones(9))


def test_P_Cl_prior_amplitude_is_given_in_the_component_units() -> None:
    """As C3's COMP_UNIT: an amplitude in uK_CMB^2 is converted to uK_RJ^2 at the view's nu_ref."""
    in_rj = _dust("QU", lmax=8, Cl_prior_amplitude=100.0)
    in_cmb = _dust("QU", lmax=8, Cl_prior_amplitude=100.0, units="uK_CMB")

    # dT_RJ/dT_CMB = x^2 e^x/(e^x - 1)^2, with x = h nu/(k T_CMB), at the QU nu_ref of 353 GHz.
    x = 6.62607015e-34*353.0e9/(1.380649e-23*2.7255)
    cmb_to_rj = x**2*np.exp(x)/np.expm1(x)**2
    np.testing.assert_allclose(in_cmb.P_Cl_prior, in_rj.P_Cl_prior*cmb_to_rj**2, rtol=1e-4)


def test_old_smoothing_prior_params_are_rejected() -> None:
    # The old C_l-space 'smoothing_prior_*' parameters changed semantics; fail loudly on stale files.
    with pytest.raises(ValueError, match="Unknown setting 'components.dust.params.smoothing_prior"):
        _dust("I", smoothing_prior_amplitude=1.0e7)


def test_a_component_setting_that_is_left_out_or_misspelled_fails_at_startup() -> None:
    """The settings of a component are those in the `DEFAULTS` of its class."""
    with pytest.raises(ValueError, match="'components.dust.params.T' must be set"):
        _dust("I", T=None)
    with pytest.raises(ValueError, match="Unknown setting 'components.dust.params.betta'"):
        _dust("I", betta=1.6)
    with pytest.raises(ValueError, match="'components.dust.params.polarization' must be one of"):
        CompList.init_from_params(_make_params(dust=Bunch(
            enabled=True, component_class="ThermalDust", params=_dust_params(polarization="P"))))


def test_init_map_to_amplitude_is_noop_when_units_match() -> None:
    dust = _dust("I", units="uK_RJ")
    arr = np.ones((1, 12))

    # uK_RJ already equals the dust amplitude unit, so the same array is returned untouched.
    assert dust.init_map_to_amplitude(arr) is arr


def test_init_map_to_amplitude_converts_to_amplitude_unit() -> None:
    import pysm3.units as pysm3u

    dust = _dust("I", units="uK_CMB", nu_ref=100.0)
    expected = (1 * pysm3u.Unit("uK_CMB")).to(
        pysm3u.uK_RJ, equivalencies=pysm3u.cmb_equivalencies(100.0 * pysm3u.GHz)).value

    out = dust.init_map_to_amplitude(np.ones((1, 12)))

    assert np.allclose(out, expected)


def test_cmb_init_map_to_amplitude_converts_uK_CMB_at_nu_ref() -> None:
    import pysm3.units as pysm3u

    cmb_cfg = _make_component_cfg("IQU")
    cmb_cfg.params.units = "uK_CMB"
    cmb = CMB(_make_params(cmb=cmb_cfg), "cmb", "I")

    # CMB amplitudes are stored in uK_RJ referenced to nu_ref (default 1 GHz), so a uK_CMB disk map
    # is converted to uK_RJ at nu_ref.
    assert cmb.nu_ref == 1.0
    expected = (1 * pysm3u.Unit("uK_CMB")).to(
        pysm3u.uK_RJ, equivalencies=pysm3u.cmb_equivalencies(1.0 * pysm3u.GHz)).value
    out = cmb.init_map_to_amplitude(np.ones((1, 12)))
    assert np.allclose(out, expected)


def test_spinning_dust_converts_its_units_at_nu_0(tmp_path) -> None:
    """The parameter `nu_0` is the reference frequency (`nu_ref`) for the unit conversion, as for
    every other component. A prior given in uK_CMB^2 then works too."""
    import pysm3.units as pysm3u
    from commander4.sky import SpinningDust

    template = tmp_path / "spdust.dat"
    np.savetxt(template, np.column_stack([np.logspace(0, 3, 50), np.ones(50)]))
    params = Bunch(polarization="I", shortname="ame", 
                   Cl_prior_amplitude=100.0, lmax=4, units="uK_CMB", template_path=str(template),
                   nu_peak=30.0, nu_0=22.0)
    ame = SpinningDust(_make_params(ame=Bunch(params=params)), "ame", "I")

    cmb_to_rj = (1*pysm3u.uK_CMB).to(pysm3u.uK_RJ,
                                     equivalencies=pysm3u.cmb_equivalencies(22.0*pysm3u.GHz)).value
    assert ame.init_map_to_amplitude(1.0) == pytest.approx(cmb_to_rj)
    assert np.all(np.isfinite(ame.P_Cl_prior))


def test_spinning_dust_rejects_a_reference_frequency_outside_its_template(tmp_path) -> None:
    """The SED divides by the template at `nu_0`, which is zero outside the tabulated range."""
    from commander4.sky import SpinningDust

    template = tmp_path / "spdust.dat"
    np.savetxt(template, np.column_stack([np.logspace(0, 2, 50), np.ones(50)]))  # 1-100 GHz.
    params = Bunch(polarization="I", shortname="ame", lmax=4, template_path=str(template),
                   nu_peak=30.0, nu_0=200.0)
    with pytest.raises(ValueError, match="outside the template range"):
        SpinningDust(_make_params(ame=Bunch(params=params)), "ame", "I")


def test_cmb_get_sed_is_unity_at_nu_ref_and_ratio_elsewhere() -> None:
    import pysm3.units as pysm3u

    cmb_cfg = _make_component_cfg("IQU")
    cmb_cfg.params.nu_ref = 100.0
    cmb = CMB(_make_params(cmb=cmb_cfg), "cmb", "I")

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
            Cl_prior_amplitude=None,
            beta=1.54,
            T=20.0,
            nu_ref=nu_ref,
        ),
    )
    return cfg


def test_every_component_class_declares_its_sed_parameters() -> None:
    """A component whose SED parameters are undeclared writes an empty `sed/` group to the chain."""
    from commander4.sky import CMB, FreeFree, Synchrotron, ThermalDust, SpinningDust, RadioSources
    expected = {CMB: ("nu_ref",), ThermalDust: ("beta", "T", "nu_ref"),
                Synchrotron: ("beta", "nu_ref"), FreeFree: ("T", "nu_ref"),
                SpinningDust: ("nu_peak_eval", "nu_peak_ref", "nu_ref"), RadioSources: ("nu_ref",)}
    for cls, names in expected.items():
        assert cls.sed_param_names == names, cls.__name__


def test_sed_parameters_are_readable_off_a_joined_component() -> None:
    """The names the chain writer stores must all resolve on the component it is handed."""
    comp_list = CompList.init_from_params(_make_params(dust=_make_dust_cfg(353.0)))
    dust = comp_list.joined()[0]

    stored = {name: getattr(dust, name) for name in dust.sed_param_names}
    assert stored == {"beta": 1.54, "T": 20.0, "nu_ref": 353.0}


def test_joining_restores_a_per_polarization_nu_ref() -> None:
    """`nu_ref: [I, QU]` is common; the joined view must not silently keep only the I value.

    Joining deep-copies the intensity view, whose `nu_ref` has already been resolved by `_per_pol`,
    so without care an Akari-style `[857, 353]` would be recorded in the chain as 857.
    """
    comp_list = CompList.init_from_params(_make_params(dust=_make_dust_cfg([857.0, 353.0])))

    assert [comp.nu_ref for comp in comp_list] == [857.0, 353.0]  # the split views
    np.testing.assert_array_equal(comp_list.joined()[0].nu_ref, [857.0, 353.0])
    # Parameters that are not per-polarization stay scalar rather than becoming a degenerate pair.
    assert comp_list.joined()[0].beta == 1.54


def _saved_dust_chain(tmp_path) -> tuple[CompList, str]:
    """A dust chain whose beta, T and nu_ref all differ from `_make_dust_cfg`, and a fresh list
    to read it into."""
    params = _make_params(dust=_make_dust_cfg(353.0))
    source = CompList.init_from_params(params)
    for comp in source:
        comp.beta, comp.T, comp.nu_ref = 1.5311, 25.0, 217.0
    chain = _write_chain(tmp_path / "chain01_iter0007.h5", source)
    return CompList.init_from_params(params), chain


def test_restarting_from_a_chain_restores_a_sampled_spectral_index(tmp_path):
    """Continuing a chain must continue its MH index walk, not reset beta to the start value."""
    comp_list, chain = _saved_dust_chain(tmp_path)
    assert all(comp.beta == 1.54 for comp in comp_list)
    for comp in comp_list:
        comp.load_initial_state(chain, apply_init_from=False, restore_beta=True)

    for comp in comp_list:
        assert comp.beta == pytest.approx(1.5311)   # sampled: taken from the chain
        assert comp.T == 20.0                       # fixed: the parameter file still rules
        assert comp.nu_ref == 353.0


def test_a_fixed_spectral_index_is_not_restored_from_a_chain(tmp_path):
    """A beta that the run does not sample stays a parameter-file setting the chain cannot
    override."""
    comp_list, chain = _saved_dust_chain(tmp_path)
    for comp in comp_list:
        comp.load_initial_state(chain, apply_init_from=False, restore_beta=False)
    assert all(comp.beta == 1.54 for comp in comp_list)


def test_the_amplitudes_beam_is_restored_from_a_chain(tmp_path):
    """A per-pixel solver's amplitudes carry its common-resolution beam, and must keep it."""
    source = _make_list(("cmb", "IQU"))
    for comp in source:
        comp.amp_fwhm_rad = np.radians(1.0)
    chain = _write_chain(tmp_path / "chain.h5", source)

    comp_list = _make_list(("cmb", "IQU"))
    for comp in comp_list:
        comp.load_initial_state(chain, apply_init_from=False, restore_beta=False)
    assert all(comp.amp_fwhm_rad == pytest.approx(np.radians(1.0)) for comp in comp_list)


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
    cfg = Bunch(polarization="I", nu_0=nu_ref, template_path=str(template_path))
    return RadioSources(_make_params(radsources=Bunch(params=cfg)), "radsources", "I")


def test_radio_source_amplitudes_round_trip_through_a_chain(tmp_path) -> None:
    template = tmp_path / "radio.dat"
    _write_radio_source_table(template)
    source = CompList([_make_radio_sources(template)])
    source[0].amps[:] = 1234.0
    chain = _write_chain(tmp_path / "chain.h5", source)

    comp_list = CompList([_make_radio_sources(template)])
    for comp in comp_list:
        comp.load_initial_state(chain, apply_init_from=False, restore_beta=False)
    assert np.all(comp_list[0].amps == 1234.0)


def _map_integral(sky_map, nside):
    """The map's integral over the sphere, which is the source's total flux in uK_RJ*sr."""
    return float(sky_map.sum()*hp.nside2pixarea(nside))


def test_radio_sources_component_map_scales_with_the_reference_frequency(tmp_path) -> None:
    """Flux to brightness temperature goes as nu_ref^-2, so 2x nu_ref gives a 4x smaller map."""
    template = tmp_path / "radio.dat"
    _write_radio_source_table(template)

    beam = np.deg2rad(2.0)
    low = _make_radio_sources(template, nu_ref=30.0).get_component_map(nside=64, fwhm_rad=beam)
    high = _make_radio_sources(template, nu_ref=60.0).get_component_map(nside=64, fwhm_rad=beam)

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

    amplitude = comp.get_component_map(nside=64, fwhm_rad=np.deg2rad(2.0))
    for nu in (30.0, 100.0, 353.0):
        sky = comp.get_sky(nu, nside=64, fwhm_rad=np.deg2rad(2.0))
        expected = amplitude*comp.get_sed(nu)[0]
        assert sky == pytest.approx(expected, rel=1e-5)


def test_radio_sources_conserve_flux_across_resolutions(tmp_path) -> None:
    """The painted beam is normalized, so the source's total flux must not depend on nside."""
    template = tmp_path / "radio.dat"
    _write_radio_source_table(template)

    integrals = [_map_integral(_make_radio_sources(template).get_component_map(
                     nside=n, fwhm_rad=np.deg2rad(1.0)), n)
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

    narrow = comp.get_component_map(nside=16, fwhm_rad=np.deg2rad(1.0/60))
    resolved = _make_radio_sources(template).get_component_map(nside=16, fwhm_rad=np.deg2rad(1.0))

    assert np.count_nonzero(narrow) == 1
    assert _map_integral(narrow, 16) == pytest.approx(_map_integral(resolved, 16), rel=1e-4)


def test_radio_sources_with_a_zero_beam_paint_one_pixel_without_warnings(tmp_path) -> None:
    """A band with `fwhm` 0 (the default of `get_component_map`) gives the delta limit directly,
    without a division by zero in the Gaussian."""
    template = tmp_path / "radio.dat"
    _write_radio_source_table(template)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        point = _make_radio_sources(template).get_component_map(nside=16, fwhm_rad=0.0)
    narrow = _make_radio_sources(template).get_component_map(nside=16, fwhm_rad=np.deg2rad(1/60))

    np.testing.assert_array_equal(point, narrow)


def test_radio_sources_take_fwhm_in_radians_like_diffuse_components(tmp_path) -> None:
    """`CompList.get_sky_at_nu` passes the band's `fwhm_rad` to whatever components it holds, so
    both families have to read that argument the same way.

    `RadioSources` used to take arcmin here. Given radians it then painted every source through a
    beam ~3400x too narrow, collapsing each to a single pixel; the resulting ragged arrays crashed
    the numba projection kernel rather than quietly returning a wrong sky.
    """
    template = tmp_path / "radio.dat"
    _write_radio_source_table(template)
    one_degree = np.deg2rad(1.0)
    painted = _make_radio_sources(template).get_component_map(nside=64, fwhm_rad=one_degree)

    # A one-degree beam at nside 64 (55' pixels) must cover more than the source's own pixel.
    assert np.count_nonzero(painted) > 1
    # And it must be broader than what the old arcmin reading of the same number would give.
    as_arcmin = _make_radio_sources(template).get_component_map(nside=64,
                                                                fwhm_rad=np.deg2rad(1.0/60))
    assert np.count_nonzero(painted) > np.count_nonzero(as_arcmin)


def test_radio_sources_are_added_to_the_band_in_pixel_space(tmp_path) -> None:
    """`comp2band` paints the sources into `band.map`, not into the alms, so no flux is cut at the
    band lmax. `band2comp` is its transpose: <comp2band(x), y> = <x, band2comp(y)>."""
    from commander4.data_models.band import Band
    template = tmp_path / "radio.dat"
    _write_radio_source_table(template)
    source = _make_radio_sources(template, nu_ref=30.0)
    nside = 16
    band = Band(np.zeros((1, hp.Alm.getsize(3*nside - 1)), dtype=np.complex128), 30.0,
                np.deg2rad(2.0), nside)
    x = source.amps.copy()

    source.comp2band(band)
    painted = band.map.copy()

    assert np.all(band.alms == 0)
    # At nu_ref the SED is 1, and the beam integrates to 1: the map holds the full flux.
    assert _map_integral(painted, nside) == pytest.approx(x[0, 0]*source.mJysr_to_uKRJ, rel=1e-5)
    y = np.random.default_rng(0).normal(size=band.map.shape)
    band.map = y
    source.band2comp(band)
    transpose_y = source.amps.copy()
    assert np.sum(painted*y) == pytest.approx(np.sum(x*transpose_y), rel=1e-5)


def test_overlapping_radio_sources_add_up(tmp_path) -> None:
    """Two sources at the same position paint the same map as one source with their summed flux.

    A vectorized `map[all_pixels] += values` would silently keep only one of the repeated pixels.
    """
    pair, single = tmp_path / "pair.dat", tmp_path / "single.dat"
    with open(pair, "w") as handle:
        handle.write("# Glon(deg) Glat(deg) I(mJy) alpha_I\n0.0 0.0 2000.0 -0.7\n0.0 0.0 3000.0 -0.7\n")
    _write_radio_source_table(single, flux_mjy=5000.0)
    fwhm = np.deg2rad(5.0)
    pair_map = _make_radio_sources(pair).get_sky(nu=44.0, nside=16, fwhm_rad=fwhm)
    single_map = _make_radio_sources(single).get_sky(nu=44.0, nside=16, fwhm_rad=fwhm)
    np.testing.assert_allclose(pair_map, single_map, rtol=1e-6)


def _comp_list_at_amp_fwhm(amp_fwhm_rad):
    """Two components whose amplitudes carry `amp_fwhm_rad` of smoothing."""
    comp_list = _make_comp_list("I")   # intensity only: lmax here is 1, too low for spin-2 QU
    for comp in comp_list:
        comp.amp_fwhm_rad = amp_fwhm_rad
    return comp_list


def test_the_sky_is_only_as_sharp_as_the_coarsest_component() -> None:
    """One component with a beam makes the whole sky that coarse, or it mixes two resolutions."""
    comp_list = _comp_list_at_amp_fwhm(0.0)
    comp_list[0].amp_fwhm_rad = np.deg2rad(100.0/60)

    with pytest.raises(ValueError, match="cannot be realized any sharper"):
        comp_list.get_sky_at_nu(100.0, 2, "I", fwhm_rad=np.deg2rad(30.0/60))


def test_deconvolved_amplitudes_can_be_realized_at_any_beam() -> None:
    """The CG solver leaves `amp_fwhm_rad` zero, so nothing is out of reach."""
    comp_list = _comp_list_at_amp_fwhm(0.0)
    for fwhm in (0.0, np.deg2rad(1.0), None):
        assert comp_list.get_sky_at_nu(100.0, 2, "I", fwhm_rad=fwhm).shape == (1, 12*2**2)


def test_asking_for_a_sharper_sky_than_the_amplitudes_hold_is_refused() -> None:
    """Per-pixel amplitudes carry the common beam; a finer sky simply does not exist, and
    silently handing back a coarser one hides that from whoever asked."""
    comp_list = _comp_list_at_amp_fwhm(np.deg2rad(100.0/60))

    with pytest.raises(ValueError, match="cannot be realized any sharper"):
        comp_list.get_sky_at_nu(100.0, 2, "I", fwhm_rad=np.deg2rad(30.0/60))


def test_none_means_the_sharpest_these_amplitudes_can_give() -> None:
    """The way for a caller to say "best available" without naming a number."""
    comp_list = _comp_list_at_amp_fwhm(np.deg2rad(100.0/60))

    at_none = comp_list.get_sky_at_nu(100.0, 2, "I", fwhm_rad=None)
    at_amp = comp_list.get_sky_at_nu(100.0, 2, "I", fwhm_rad=np.deg2rad(100.0/60))

    assert np.array_equal(at_none, at_amp)


def test_the_amplitudes_own_beam_is_not_treated_as_too_sharp() -> None:
    """`amp_fwhm_rad` is set from a band's own fwhm, so equality here must not trip the check."""
    comp_list = _comp_list_at_amp_fwhm(np.deg2rad(100.0/60))

    sky = comp_list.get_sky_at_nu(100.0, 2, "I", fwhm_rad=np.deg2rad(100.0/60))
    assert sky.shape == (1, 12*2**2)
