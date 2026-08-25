"""simgen's per-component truth maps: the input amplitude of each simulated sky component.

A truth map must be the component's *amplitude* -- unsmoothed, at the component's own reference
frequency, in the run's sky unit -- so that Commander4 can read it back as that component's
``init_from`` or ``amp_prior_mean_map``, and so that a recovered component map can be compared
against it. The defining property is therefore

    band_map(band) == beam(truth_map) * get_sed(band.freq),

with the beam and the SED being exactly the ones simgen applies when building the band map. The
tests below check that identity for each component class in a configuration where the two sides are
*exactly* comparable, then check that the written FITS file survives the round trip through
Commander4's own init-map reader.
"""
import os

import healpy as hp
import numpy as np
import pytest
from pixell.bunch import Bunch


NSIDE = 32
LMAX = 47          # well below 3*nside-1, so the map is comfortably band-limited


def _write_smooth_iqu_template(path: str, seed: int = 7) -> np.ndarray:
    """A band-limited random IQU map, used as a FITS component template (no PySM3 needed)."""
    ell = np.arange(LMAX + 1)
    cl = 1.0e4 / (ell + 10.0)**2
    np.random.seed(seed)
    m = np.asarray(hp.synfast([cl, cl, cl, np.zeros_like(cl)], NSIDE, lmax=LMAX, new=True),
                   dtype=np.float32)
    hp.write_map(path, m, overwrite=True, dtype=np.float32)
    return m


def _params(components: dict, bands: dict, nside: int = NSIDE) -> Bunch:
    from simgen.config import as_bunch_recursive
    return as_bunch_recursive({
        "general": {"nside": nside, "units": "uK_RJ", "float_precision": "single",
                    "seed": 11, "output_dir": "unused"},
        "components": components,
        "simulation": {"nscans": 1, "scan_duration_sec": 1,
                       "pointing": {"strategy": "raster"}, "noise": {"white": True}},
        "experiments": {"Exp": {"enabled": True, "bands": bands}},
    })


def _dust_cfg(template_path: str, nu_ref: float = 353.0) -> dict:
    return {"enabled": True, "component_class": "ThermalDust",
            "params": {"polarization": "IQU", "shortname": "dust", "lmax": LMAX,
                       "beta": 1.54, "T": 20.0, "nu_ref": nu_ref,
                       "template": {"source": "fits", "path": template_path}}}


def _cmb_cfg(nu_ref: float | None = None) -> dict:
    params = {"polarization": "IQU", "shortname": "cmb", "lmax": LMAX, "solar_dipole": False}
    if nu_ref is not None:
        params["nu_ref"] = nu_ref
    return {"enabled": True, "component_class": "CMB", "params": params}


def _band(freq: float, fwhm: float, nside: int = NSIDE) -> dict:
    return {"enabled": True, "freq": freq, "fwhm": fwhm, "fsamp": 10.0, "eval_nside": nside,
            "data_nside": nside, "sigma0": 1.0, "polarization": "IQU",
            "detectors": {"d0": {}}}


# --------------------------------------------------------------------- the defining identity

def test_diffuse_truth_map_is_the_band_map_amplitude(tmp_path):
    """beam(truth) * SED must reproduce band_map for a template-based foreground.

    The band's ``eval_nside`` equals ``general.nside``, so the ``ud_grade`` in both paths is the
    identity and the two sides are comparable to float32 rounding rather than to a tolerance.
    """
    from simgen.instrument import build_bands
    from simgen.sky import build_components

    template_path = str(tmp_path / "dust_template.fits")
    _write_smooth_iqu_template(template_path)
    params = _params({"ThermalDust": _dust_cfg(template_path)}, {"B217": _band(217.0, 40.0)})
    comp = build_components(params)[0]
    band = build_bands(params)[0]

    truth = comp.truth_map(NSIDE)
    assert truth.shape == (3, hp.nside2npix(NSIDE))
    # Exactly the operations band_map applies, but starting from the truth map.
    expected = hp.smoothing(truth, fwhm=band.fwhm_rad).astype(np.float32) * comp.c4.get_sed(217.0)

    np.testing.assert_allclose(comp.band_map(band), expected, rtol=1e-5, atol=1e-5)
    # The SED at 217 GHz is far from unity, so this is not a vacuous comparison.
    assert not np.allclose(comp.band_map(band), hp.smoothing(truth, fwhm=band.fwhm_rad))


def test_diffuse_truth_map_is_the_template_at_nu_ref(tmp_path):
    """The amplitude is the template itself: the SED is unity at nu_ref, and no beam is applied."""
    from simgen.sky import build_components

    template_path = str(tmp_path / "dust_template.fits")
    template = _write_smooth_iqu_template(template_path)
    params = _params({"ThermalDust": _dust_cfg(template_path)}, {"B353": _band(353.0, 40.0)})
    comp = build_components(params)[0]

    assert comp.c4.get_sed(353.0) == pytest.approx(1.0)
    np.testing.assert_allclose(comp.truth_map(NSIDE), template, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("nu_ref", [None, 70.0])
def test_cmb_truth_map_is_the_band_map_amplitude(nu_ref):
    """Same identity for the CMB, whose amplitude carries the uK_CMB -> uK_RJ conversion at nu_ref.

    Uses ``fwhm = 0``, where ``smoothalm`` is exactly the identity, so any difference between the
    two sides is a unit/reference-frequency error rather than a beam round-trip. Both the default
    reference (1 GHz, where the conversion is nearly unity and would hide a nu_ref mix-up) and an
    explicit one far from it are covered.
    """
    pytest.importorskip("camb")
    pytest.importorskip("pysm3")
    from simgen.instrument import build_bands
    from simgen.sky import _build_c4_component, build_components

    params = _params({"CMB": _cmb_cfg(nu_ref)}, {"B217": _band(217.0, 0.0)})
    comp = build_components(params)[0]
    band = build_bands(params)[0]

    # Scale with Commander4's own CMB SED, so this pins the simgen amplitude to the C4 convention
    # rather than to a re-derivation of it here.
    c4_sed = _build_c4_component(comp.comp_cfg, params.general).get_sed(217.0)
    assert abs(c4_sed - 1.0) > 0.1          # a real conversion, so the comparison is not vacuous
    expected = comp.truth_map(NSIDE) * c4_sed
    np.testing.assert_allclose(comp.band_map(band), expected, rtol=1e-5, atol=1e-5)


def test_cmb_truth_map_scales_with_nu_ref():
    """Changing nu_ref rescales the amplitude by exactly the thermodynamic-to-RJ ratio.

    The amplitude convention (uK_RJ referenced to nu_ref) is the one thing about the CMB truth map
    that cannot be read off the file, so it is worth pinning: a truth map written at one nu_ref and
    fed to a Commander4 CMB component declaring another would be silently mis-scaled.
    """
    pytest.importorskip("camb")
    import pysm3.units as u
    from simgen.sky import build_components

    a = build_components(_params({"CMB": _cmb_cfg(1.0)}, {"B100": _band(100.0, 0.0)}))[0]
    b = build_components(_params({"CMB": _cmb_cfg(100.0)}, {"B100": _band(100.0, 0.0)}))[0]

    def cmb_to_rj(nu):
        return (1.0 * u.uK_CMB).to(u.uK_RJ, equivalencies=u.cmb_equivalencies(nu * u.GHz)).value

    ratio = cmb_to_rj(100.0) / cmb_to_rj(1.0)
    assert ratio == pytest.approx(0.7, abs=0.15)   # a real, non-unity conversion at 100 GHz
    np.testing.assert_allclose(b.truth_map(NSIDE), a.truth_map(NSIDE) * ratio, rtol=1e-5)


def test_gridded_point_sources_truth_map_is_unsmoothed():
    """The synthetic point-source component's amplitude is the bare source grid."""
    from simgen.sky import build_components

    cfg = {"GriddedPointSources": {
        "enabled": True, "component_class": "GriddedPointSources",
        "params": {"polarization": "I", "shortname": "ps", "amplitude": 1.0e3,
                   "nlon": 4, "nlat": 3, "lon_range_deg": [0.0, 90.0],
                   "lat_range_deg": [-30.0, 30.0]}}}
    comp = build_components(_params(cfg, {"B100": _band(100.0, 30.0)}))[0]

    truth = comp.truth_map(NSIDE)
    hit = truth[0] > 0
    assert hit.sum() == 12                              # 4 x 3 distinct source pixels
    np.testing.assert_allclose(truth[0][hit], 1.0e3)    # unsmoothed: full amplitude in one pixel
    np.testing.assert_array_equal(truth[1:], 0.0)       # intensity only


# ------------------------------------------------------------------------- pipeline integration

def _pipeline_params(tmp_path, template_path, write_truth=None) -> tuple[str, str]:
    import yaml
    out_dir, debug_dir = str(tmp_path / "sim"), str(tmp_path / "debug")
    sim = {"nscans": 1, "scan_duration_sec": 4, "npsi": 256, "orbital_dipole": False,
           "compress": False, "debug_output_dir": debug_dir,
           "pointing": {"strategy": "raster", "patch_center_deg": [0.0, 0.0],
                        "patch_size_deg": [10.0, 10.0], "n_rows": 8, "samples_per_row": 16},
           "noise": {"white": True}}
    if write_truth is not None:
        sim["write_component_truth_maps"] = write_truth
    params = {
        "general": {"nside": NSIDE, "units": "uK_RJ", "float_precision": "single",
                    "seed": 3, "output_dir": out_dir},
        "components": {"ThermalDust": _dust_cfg(template_path)},
        "simulation": sim,
        "experiments": {"Exp": {"enabled": True, "bands": {"B353": _band(353.0, 40.0)}}},
    }
    path = str(tmp_path / "param.yml")
    with open(path, "w") as f:
        yaml.dump(params, f, sort_keys=False)
    return path, debug_dir


def test_pipeline_writes_truth_maps(tmp_path):
    """A full simgen run writes one truth FITS per component, tagged with its reference frame."""
    from astropy.io import fits
    from simgen import pipeline

    template_path = str(tmp_path / "dust_template.fits")
    template = _write_smooth_iqu_template(template_path)
    param_path, debug_dir = _pipeline_params(tmp_path, template_path)

    assert pipeline.run(param_path) == 0

    truth_path = os.path.join(debug_dir, "truth_ThermalDust.fits")
    written = np.atleast_2d(hp.read_map(truth_path, field=None))
    assert written.shape == (3, hp.nside2npix(NSIDE))
    np.testing.assert_allclose(written, template, rtol=1e-6, atol=1e-6)

    header = fits.getheader(truth_path, ext=1)
    assert header["NU_REF"] == pytest.approx(353.0)
    assert header["BUNIT"] == "uK_RJ"
    assert header["COMPCLS"] == "ThermalDust"


def test_truth_maps_can_be_switched_off(tmp_path):
    from simgen import pipeline

    template_path = str(tmp_path / "dust_template.fits")
    _write_smooth_iqu_template(template_path)
    param_path, debug_dir = _pipeline_params(tmp_path, template_path, write_truth=False)

    assert pipeline.run(param_path) == 0
    assert os.listdir(debug_dir)          # the other diagnostics are still written
    assert not os.path.exists(os.path.join(debug_dir, "truth_ThermalDust.fits"))


# ------------------------------------------------------- the point of the feature: init_from

def test_truth_map_is_readable_as_a_commander4_init_map(tmp_path):
    """The written file loads through Commander4's own init-map reader and round-trips.

    This is what the truth maps are for: `init_from` (start the chain at the truth, to isolate one
    sampler) and `amp_prior_mean_map` (the prior mean mu) both go through
    `_read_view_alms_from_fits`, which infers the polarization from the map's shape and converts
    from the component's declared `units` at its `nu_ref`. Declaring `units: uK_RJ` -- the unit
    simgen writes -- must make that conversion the identity.
    """
    pytest.importorskip("ducc0")
    from commander4.sky.comp_io import _load_component_alms
    from commander4.sky.diffuse_components import ThermalDust
    from simgen.sky import build_components, write_component_truth_maps

    template_path = str(tmp_path / "dust_template.fits")
    _write_smooth_iqu_template(template_path)
    params = _params({"ThermalDust": _dust_cfg(template_path)}, {"B353": _band(353.0, 40.0)})
    comp = build_components(params)[0]
    write_component_truth_maps(str(tmp_path), [comp], NSIDE, "uK_RJ")
    truth_path = str(tmp_path / "truth_ThermalDust.fits")

    global_params = Bunch(nside=NSIDE, float_precision="single",
                          MPI_config=Bunch(ntask_compsep_I=1, ntask_compsep_QU=1))
    comp_params = Bunch(polarization="IQU", shortname="dust", lmax=3 * NSIDE - 1,
                        spatially_varying_MM=False, Cl_prior_amplitude=None,
                        beta=1.54, T=20.0, nu_ref=353.0, units="uK_RJ")
    object.__setattr__(comp_params, "_name", "ThermalDust")
    c4_comp = ThermalDust(comp_params, global_params, allocate_empty_alms=True, eval_pol="I",
                          comp_name="ThermalDust")

    _load_component_alms(c4_comp, truth_path)

    recovered = hp.alm2map(c4_comp.alms[0].astype(np.complex128), NSIDE, lmax=c4_comp.lmax)
    truth_I = comp.truth_map(NSIDE)[0]
    assert np.std(recovered - truth_I) < 1e-3 * np.std(truth_I)
