"""Tests for the MCMC sampling-group chi-squared mask (C3's MCMC_SAMPLING_GROUP_CHISQ_MASK).

The mask restricts which pixels enter the accept/reject chi-squared of the sampler. It is not a
data mask: it never touches the noise model or the amplitude solve.
"""

import healpy as hp
import numpy as np
import pytest
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.compsep.processing import _read_chisq_masks
from commander4.compsep.spectral_sampling import chi2_of_chisq_bands, resolve_chisq_mask
from commander4.data_models.detector_map import DetectorMap
from commander4.sky.comp_list import CompList

NSIDE = 2
NPIX = 12*NSIDE**2


def _zero_sky() -> CompList:
    """A single CMB component with zero alms, so the realized sky is identically zero."""
    cmb = Bunch(enabled=True, component_class="CMB",
                params=Bunch(lmax=2, polarization="I", shortname="cmb",
                             spatially_varying_MM=False, Cl_prior_amplitude=None))
    comp_list = CompList.init_from_params(Bunch(components=Bunch(CMB=cmb),
                                                compsep=Bunch(double_precision=True)))
    for comp in comp_list:
        comp.amps[:] = 0.0
    return comp_list


def _chi2(map_sky, inv_var, keep=slice(None), chisq_bands=("Band100",)) -> float:
    """The sampler's chi-squared of one band map against a zero sky, on one rank."""
    band_map = DetectorMap(np.asarray(map_sky, dtype=np.float64),
                           np.asarray(inv_var, dtype=np.float64),
                           nu=100.0, fwhm=0.0, nside=NSIDE, double_precision=True, lmax=2)
    mpi_info = Bunch(compsep=Bunch(comm=MPI.COMM_SELF),
                     band=Bunch(name="Band100", identifier="Band100_I"))
    group = Bunch(chisq_bands=list(chisq_bands))
    return chi2_of_chisq_bands(mpi_info, group, band_map, _zero_sky(), keep)


class TestResolveChisqMask:
    def test_thresholds_at_half(self):
        raw = np.array([0.0, 0.4, 0.5, 0.6, 1.0] + [1.0]*(NPIX - 5))
        keep = resolve_chisq_mask(raw, NSIDE)
        # C3's convention: strictly greater than 0.5 is kept, so 0.5 itself is masked out.
        np.testing.assert_array_equal(keep[:5], [False, False, False, True, True])

    def test_udgrades_to_the_band_nside(self):
        """A mask given at a different nside is brought to the band's resolution."""
        coarse = np.ones(12)          # nside 1
        coarse[0] = 0.0
        keep = resolve_chisq_mask(coarse, NSIDE)
        assert keep.size == NPIX
        # The four nside-2 children of the masked nside-1 pixel are dropped, nothing else.
        assert keep.sum() == NPIX - 4


class TestMaskedChi2:
    def test_masked_chi2_equals_explicit_partial_sum(self):
        rng = np.random.default_rng(7)
        map_sky = rng.normal(0.0, 2.0, (1, NPIX))
        rms = rng.uniform(0.5, 2.0, (1, NPIX))
        keep = rng.random(NPIX) > 0.5

        expected = np.sum((map_sky[0, keep]/rms[0, keep])**2)
        assert _chi2(map_sky, 1.0/rms**2, keep=keep) == pytest.approx(expected)

    def test_zero_weight_pixels_contribute_zero_not_nan(self):
        """Unobserved pixels (map_inv_var = 0) must not poison the chi-squared."""
        map_sky = np.full((1, NPIX), 3.0)
        inv_var = np.ones((1, NPIX))
        inv_var[0, :6] = 0.0

        assert _chi2(map_sky, inv_var) == pytest.approx((NPIX - 6)*3.0**2)

    def test_a_band_outside_chisq_bands_contributes_nothing(self):
        map_sky = np.full((1, NPIX), 3.0)
        assert _chi2(map_sky, np.ones((1, NPIX)), chisq_bands=["OtherBand"]) == 0.0


class TestReadChisqMasks:
    def test_reads_only_groups_that_define_a_mask(self, tmp_path):
        mask = np.ones(NPIX)
        mask[:8] = 0.0
        path = tmp_path / "chisq_mask.fits"
        hp.write_map(str(path), mask, overwrite=True)

        groups = {
            "with_mask": Bunch(name="with_mask", chisq_mask=str(path)),
            "without_mask": Bunch(name="without_mask", chisq_mask=None),
            "empty_mask": Bunch(name="empty_mask", chisq_mask=""),
        }
        masks = _read_chisq_masks(MPI.COMM_SELF, groups)

        assert set(masks) == {"with_mask"}
        np.testing.assert_allclose(masks["with_mask"], mask)
