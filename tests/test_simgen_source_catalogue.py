"""The point-source catalogue simgen writes, read back by Commander4's `RadioSources`.

simgen paints sources as pixel values in uK_RJ; `RadioSources` describes them as flux densities in
mJy. The catalogue is the bridge, and it is only useful if the same sources come back out, so what
is pinned here is that the total brightness survives the round trip.
"""
import os
import sys

import numpy as np
import pytest
from pixell.bunch import Bunch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sims"))

from simgen.sky import GriddedPointSources, write_source_catalogue
from commander4.sky.point_sources import RadioSources

NSIDE = 128
AMPLITUDE = 2000.0
NU_REF = 100.0


def _simgen_sources(beta=0.0):
    cfg = Bunch(_name="PointSources", component_class="GriddedPointSources", enabled=True,
                params=Bunch(polarization="I", shortname="ptsrc", amplitude=AMPLITUDE,
                             nlon=12, nlat=6, lat_range_deg=[-70.0, 70.0], beta=beta,
                             nu_ref=NU_REF))
    return GriddedPointSources(cfg, Bunch(nside=NSIDE, units="uK_RJ", float_precision="double"))


def _radio_sources(catalogue_path):
    cfg = Bunch(_name="RadioSources", polarization="I", longname="RadioSources",
                shortname="radsources", template_path=str(catalogue_path), nu_0=NU_REF,
                lmax="full", smoothing_scale=0, sample_alphas=False)
    return RadioSources(cfg, Bunch(nside=NSIDE, float_precision="double", lmax=3*NSIDE-1,
                                   polarization="I"))


@pytest.fixture
def catalogue(tmp_path):
    comp = _simgen_sources()
    write_source_catalogue(str(tmp_path), comp, NSIDE)
    return comp, tmp_path / "catalogue_PointSources.dat"


def test_the_catalogue_has_one_row_per_source(catalogue):
    comp, path = catalogue
    _, table = comp.catalogue_table(NSIDE)
    data_lines = [ln for ln in path.read_text().split("\n") if ln.strip()
                  and not ln.strip().startswith("#")]
    assert len(data_lines) == table.shape[0] == 72


def test_commander4_reads_the_catalogue_columns(catalogue):
    """The header names must be the ones `read_dat_to_bunch` looks for, or construction fails."""
    _, path = catalogue
    c4 = _radio_sources(path)
    assert c4.lonlat_arr.shape == (72, 2)
    assert c4.alpha_arr.shape == (72,)


def test_total_brightness_survives_the_round_trip(catalogue):
    """simgen's pixel amplitudes and the catalogue's mJy fluxes must describe the same sky.

    Only the integral is pinned: simgen smooths in harmonic space and is band-limited at lmax,
    while `RadioSources` paints the beam in real space, so the two differ slightly at the peaks.
    """
    comp, path = catalogue
    beam_rad = np.deg2rad(1.0)
    band = Bunch(freq=217.0, eval_nside=NSIDE, fwhm_rad=beam_rad, polarization="I")
    sim_map = comp.band_map(band)[0]
    # `get_sky` takes the beam in radians, the same unit `SkyModel` passes both component families.
    c4_map = _radio_sources(path).get_sky(band.freq, NSIDE, fwhm=beam_rad)[0]
    assert c4_map.sum() == pytest.approx(sim_map.sum(), rel=1e-3)


def test_a_flat_spectrum_stays_flat_across_bands(catalogue):
    """beta = 0 means the same amplitude in every band, which is `alpha_I = 2` on the C4 side."""
    _, path = catalogue
    c4 = _radio_sources(path)
    assert np.allclose(c4.alpha_arr, 2.0)
    assert np.allclose(c4.get_sed(30.0), 1.0)
    assert np.allclose(c4.get_sed(353.0), 1.0)


def test_a_sloped_spectrum_maps_to_alpha_minus_two():
    """`RadioSources.get_sed` uses `alpha - 2`, so simgen's beta must be written as beta + 2."""
    comp = _simgen_sources(beta=-0.7)
    _, table = comp.catalogue_table(NSIDE)
    assert np.allclose(table[:, 3], 1.3)


def test_flux_scales_with_the_pixel_area_it_was_painted_on(tmp_path):
    """A source is one pixel of the simulation grid, so the same amplitude at a finer nside is a
    proportionally smaller flux. The catalogue is tied to the nside it was written for."""
    comp = _simgen_sources()
    _, coarse = comp.catalogue_table(64)
    _, fine = comp.catalogue_table(128)
    assert fine[0, 2] == pytest.approx(coarse[0, 2]/4.0, rel=1e-6)
