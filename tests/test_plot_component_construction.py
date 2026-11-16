"""The chain plotter must reconstruct the same component views as the live pipeline."""

import healpy as hp
import numpy as np
from pixell.bunch import Bunch

from commander4.diagnostics import plotting
from commander4.standalone_tools.plot_chain import _build_component_list


def _cmb_params() -> Bunch:
    """A parameter file holding one IQU CMB component with lmax 11."""
    component = Bunch(
        enabled=True,
        component_class="CMB",
        params=Bunch(
            polarization="IQU",
            longname="CMB",
            shortname="cmb",
            lmax=11,
            spatially_varying_MM=False,
            Cl_prior_amplitude=None,
        ),
    )
    object.__setattr__(component, "_name", "CMB")
    return Bunch(components=Bunch(CMB=component), compsep=Bunch(double_precision=False))


def test_plotter_uses_live_component_lmax_and_polarization_views() -> None:
    components = _build_component_list(_cmb_params())

    assert [item.pol for item in components] == ["I", "QU"]
    assert [item.lmax for item in components] == [11, 11]
    assert [item.comp_name for item in components] == ["CMB", "CMB"]


def test_plotter_draws_a_diffuse_component_at_a_band_frequency() -> None:
    """At a band frequency the plotted map is the amplitude map times the SED."""
    cmb_I = _build_component_list(_cmb_params())[0]
    rng = np.random.default_rng(1)
    nalm = hp.Alm.getsize(cmb_I.lmax)
    cmb_I.amps = (rng.normal(size=(1, nalm)) + 1j*rng.normal(size=(1, nalm))).astype(np.complex64)

    at_100GHz = plotting._get_component_map(cmb_I, 100.0, 4, npol=1, ipol=0,
                                            smoothing_scale_radians=0.0)
    amplitude = plotting._get_component_map(cmb_I, np.nan, 4, npol=1, ipol=0,
                                            smoothing_scale_radians=0.0)
    np.testing.assert_allclose(at_100GHz, amplitude*cmb_I.get_sed(100.0), rtol=1e-6)
    assert cmb_I.get_sed(100.0) != 1.0
