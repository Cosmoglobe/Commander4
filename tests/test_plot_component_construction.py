"""The chain plotter must reconstruct the same component views as the live pipeline."""

from pixell.bunch import Bunch

from commander4.standalone_tools.plot_chain import _build_component_list


def test_plotter_uses_live_component_lmax_and_polarization_views() -> None:
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
    params = Bunch(
        components=Bunch(CMB=component),
        compsep=Bunch(double_precision=False),
    )

    components = _build_component_list(params)

    assert [item.eval_pol for item in components] == ["I", "QU"]
    assert [item.lmax for item in components] == [11, 11]
    assert [item.comp_name for item in components] == ["CMB", "CMB"]
