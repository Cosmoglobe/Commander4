"""What the component-separation chain records beyond the amplitudes.

Commander3 spreads a sample's goodness-of-fit over `chisq_<postfix>.fits`, the per-band columns of
`fg_ind_mean_c<CCCC>.dat`, `sigma_l_<comp>_<postfix>.dat` and the `Dl_*` datasets in its chain. C4
keeps the equivalents in the one compsep chain file; these tests pin that each reaches it, and that
the numbers in it are self-consistent.
"""
import os

import h5py
import numpy as np
import pytest
from pixell.bunch import Bunch

from commander4.file_io import paths
from commander4.file_io.chain_writer import write_compsep_chain_to_file
from commander4.sky.comp_list import CompList


def _params(output_dir: str) -> Bunch:
    params = Bunch(output=Bunch(dir=output_dir, chains=Bunch(write=[1], maps_nside=2)))
    params.parameter_file_as_string = "parameters"
    return params


def _comp_list(lmax: int = 4) -> CompList:
    """One IQU CMB component with a power-law C(l) prior, at a small lmax."""
    cmb = Bunch(enabled=True, component_class="CMB",
                params=Bunch(lmax=lmax, polarization="IQU", shortname="cmb",
                             spatially_varying_MM=False, Cl_prior_amplitude=1e3,
                             Cl_prior_beta=-0.5, nu_ref=100.0))
    object.__setattr__(cmb, "_name", "CMB")
    params = Bunch(compsep=Bunch(bands=Bunch(), double_precision=True,
                                 ntask_compsep_QU=1))
    return CompList.init_from_params(Bunch({"cmb": cmb}), params)


def _write(tmp_path, diagnostics=None, band_frequencies=None, lmax=4) -> str:
    params = _params(str(tmp_path))
    paths.create_output_dirs(params.output)
    comp_list = _comp_list(lmax)
    for comp in comp_list.components:
        comp.allocate_empty_alms()
        rng = np.random.default_rng(0)
        comp.alms[:] = rng.normal(size=comp.alms.shape) + 1j*rng.normal(size=comp.alms.shape)
    write_compsep_chain_to_file(comp_list.joined(), params, chain=1, iter=1,
                                diagnostics=diagnostics, band_frequencies=band_frequencies)
    return os.path.join(paths.subdir(params, paths.CHAINS_COMPSEP), "chain01_iter0001.h5")


def test_sigma_l_matches_the_alms_written_beside_it(tmp_path) -> None:
    """C3's `sigma_l`: the realized power of exactly the alms in the same file."""
    import healpy as hp

    path = _write(tmp_path, lmax=4)

    with h5py.File(path) as f:
        alms = np.ascontiguousarray(f["comps/cmb/alms"][:], dtype=np.complex128)
        sigma_l = f["comps/cmb/sigma_l"][:]
        lmax = int(f["comps/cmb/lmax"][()])
    assert lmax == 4
    assert sigma_l.shape == (3, lmax + 1)   # One auto spectrum per stored alm row (T, E, B).
    expected = np.array([hp.alm2cl(alms[i], lmax=lmax) for i in range(3)])
    np.testing.assert_allclose(sigma_l, expected)


def test_the_cl_prior_parameters_are_recorded(tmp_path) -> None:
    """C3's `Dl_amp`/`Dl_beta`: the prior's model parameters, so a chain describes its own prior."""
    path = _write(tmp_path)

    with h5py.File(path) as f:
        prior = f["comps/cmb/Cl_prior"]
        assert f["comps/cmb/Cl_prior/Cl_prior_amplitude"][()] == pytest.approx(1e3)
        assert f["comps/cmb/Cl_prior/Cl_prior_beta"][()] == pytest.approx(-0.5)
        assert {"Cl_prior_FWHM", "Cl_prior_l_pivot", "Cl_prior_l_apod"} <= set(prior.keys())


def test_a_disabled_cl_prior_writes_nothing(tmp_path) -> None:
    """`Cl_prior_amplitude: null` is C3's `CL_TYPE none`, which writes no prior at all."""
    params = _params(str(tmp_path))
    paths.create_output_dirs(params.output)
    cmb = Bunch(enabled=True, component_class="CMB",
                params=Bunch(lmax=2, polarization="I", shortname="cmb",
                             spatially_varying_MM=False, Cl_prior_amplitude=None, nu_ref=100.0))
    object.__setattr__(cmb, "_name", "CMB")
    comp_list = CompList.init_from_params(
        Bunch({"cmb": cmb}),
        Bunch(compsep=Bunch(bands=Bunch(), double_precision=True, ntask_compsep_QU=1)))
    for comp in comp_list.components:
        comp.allocate_empty_alms()
    write_compsep_chain_to_file(comp_list, params, chain=1, iter=1)

    path = os.path.join(paths.subdir(params, paths.CHAINS_COMPSEP), "chain01_iter0001.h5")
    with h5py.File(path) as f:
        assert "Cl_prior_amplitude" not in f["comps/cmb/Cl_prior"]


def test_mixing_coefficients_are_written_per_band(tmp_path) -> None:
    """C3's `mixmat_<comp>_<band>.fits`. C4 indices are scalar, so each is a single number."""
    path = _write(tmp_path, band_frequencies={"Band30GHz": 30.0, "Band353GHz": 353.0})

    with h5py.File(path) as f:
        mixing = {name: float(f["comps/cmb/mixing"][name][()]) for name in f["comps/cmb/mixing"]}
    assert set(mixing) == {"Band30GHz", "Band353GHz"}
    # A CMB component in uK_RJ: the CMB-to-RJ factor falls steeply with frequency.
    assert mixing["Band30GHz"] > mixing["Band353GHz"] > 0.0


def test_the_diagnostics_tree_becomes_nested_groups(tmp_path) -> None:
    """The chi-squared, sampler stats and residuals land as groups, with `None` entries skipped."""
    diagnostics = {
        "chi2": {"total": 100.0, "ndof": 90, "reduced": 100.0/90, "z": 0.745,
                 "bands": {"Band30GHz_I": {"chi2": 60.0, "ndof": 50, "nu": 30.0},
                           "Band44GHz_I": {"chi2": 40.0, "ndof": 40, "nu": 44.0}},
                 "map": np.ones((3, 48), dtype=np.float32)},
        "mcmc": {"beta_group": {"numstep": 10, "n_accept": 4, "accept_rate": 0.4,
                                "params": {"sync": -3.1}}},
        "amplitude_groups": {"amps": {"n_iter": 3, "cg_residuals": np.array([1.0, 0.1, 0.01])}},
        "nothing_here": None,
    }

    path = _write(tmp_path, diagnostics=diagnostics)

    with h5py.File(path) as f:
        assert f["chi2/total"][()] == pytest.approx(100.0)
        assert f["chi2/bands/Band30GHz_I/chi2"][()] == pytest.approx(60.0)
        assert f["chi2/map"].shape == (3, 48)
        assert f["mcmc/beta_group/n_accept"][()] == 4
        assert f["mcmc/beta_group/params/sync"][()] == pytest.approx(-3.1)
        assert f["amplitude_groups/amps/cg_residuals"].shape == (3,)
        assert "nothing_here" not in f
        # The per-band terms must add up to the total the same file reports.
        bands = f["chi2/bands"]
        assert sum(bands[b]["chi2"][()] for b in bands) == pytest.approx(f["chi2/total"][()])
