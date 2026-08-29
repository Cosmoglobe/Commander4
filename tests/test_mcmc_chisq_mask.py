"""Tests for the MCMC sampling-group chi-squared mask (C3's MCMC_SAMPLING_GROUP_CHISQ_MASK).

The mask restricts which pixels enter the MH accept/reject likelihood. It is not a data
mask: it never touches the noise model or the amplitude solve, only `local_loglike`.
"""

import healpy as hp
import numpy as np
import pytest
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.compsep.processing import MCMCSamplingGroupConfig, _read_chisq_masks
from commander4.data_models.detector_map import DetectorMap
from commander4.sky.comp_list import CompList
from commander4.compsep.mcmc import MCMCSamplingGroup, resolve_chisq_mask

NSIDE = 2
NPIX = 12*NSIDE**2


class _TrivialGroup(MCMCSamplingGroup):
    """Concrete MCMCSamplingGroup with no parameters, to exercise the base-class likelihood."""

    def has_parameters(self) -> bool:
        return False

    def capture_state(self):
        return {}

    def propose(self, current_state):
        return {}, True

    def apply_state(self, state) -> None:
        pass


def _zero_comp_list() -> CompList:
    """A single CMB component with zero alms, so the realized sky model is identically zero."""
    params = Bunch(compsep=Bunch(nside=NSIDE, double_precision=True,
                                 MPI_config=Bunch(ntask_compsep_I=1, ntask_compsep_QU=1)))
    cmb = Bunch(enabled=True, component_class="CMB",
                params=Bunch(lmax=2, polarization="I", shortname="cmb",
                             spatially_varying_MM=False, Cl_prior_amplitude=None))
    object.__setattr__(cmb, "_name", "CMB")
    comp_list = CompList.init_from_params(Bunch({"CMB": cmb}), params)
    for comp in comp_list:
        comp.alms[:] = 0.0
    return comp_list


def _make_group(map_sky, rms, chisq_mask=None) -> _TrivialGroup:
    det_map = DetectorMap(np.asarray(map_sky, dtype=np.float64),
                          np.asarray(rms, dtype=np.float64),
                          nu=100.0, fwhm=0.0, nside=NSIDE, double_precision=True, lmax=2)
    return _TrivialGroup(MPI.COMM_SELF, det_map, _zero_comp_list(), target_pol="I",
                         chisq_active=True, chisq_mask=chisq_mask)


class TestResolveChisqMask:
    def test_none_stays_none(self):
        assert resolve_chisq_mask(None, NSIDE) is None

    def test_all_sky_mask_collapses_to_none(self):
        """A mask that keeps everything takes the cheaper full-sky path."""
        assert resolve_chisq_mask(np.ones(NPIX), NSIDE) is None

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


class TestMaskedLikelihood:
    def test_masked_pixels_do_not_contribute(self):
        map_sky = np.zeros((1, NPIX))
        map_sky[0, :4] = 10.0                      # large residual, entirely inside the masked area
        keep = np.ones(NPIX, dtype=bool)
        keep[:4] = False

        masked = _make_group(map_sky, np.ones((1, NPIX)), chisq_mask=keep.astype(float))
        assert masked.local_loglike() == pytest.approx(0.0)

        unmasked = _make_group(map_sky, np.ones((1, NPIX)))
        assert unmasked.local_loglike() == pytest.approx(-0.5*4*10.0**2)

    def test_masked_likelihood_equals_explicit_partial_sum(self):
        rng = np.random.default_rng(7)
        map_sky = rng.normal(0.0, 2.0, (1, NPIX))
        rms = rng.uniform(0.5, 2.0, (1, NPIX))
        keep = rng.random(NPIX) > 0.5

        group = _make_group(map_sky, rms, chisq_mask=keep.astype(float))
        expected = -0.5*np.sum((map_sky[0, keep]/rms[0, keep])**2)
        assert group.local_loglike() == pytest.approx(expected)

    def test_zero_weight_pixels_contribute_zero_not_nan(self):
        """Unobserved pixels (inv_n = 0, rms = inf) must not poison the chi-squared."""
        map_sky = np.full((1, NPIX), 3.0)
        rms = np.ones((1, NPIX))
        rms[0, :6] = np.inf                        # inv_n_map = 0 there

        loglike = _make_group(map_sky, rms).local_loglike()
        assert np.isfinite(loglike)
        assert loglike == pytest.approx(-0.5*(NPIX - 6)*3.0**2)


class TestReadChisqMasks:
    def test_reads_only_groups_that_define_a_mask(self, tmp_path):
        mask = np.ones(NPIX)
        mask[:8] = 0.0
        path = tmp_path / "chisq_mask.fits"
        hp.write_map(str(path), mask, overwrite=True)

        groups = {
            "with_mask": MCMCSamplingGroupConfig(name="with_mask", chisq_mask=str(path)),
            "without_mask": MCMCSamplingGroupConfig(name="without_mask"),
            "empty_mask": MCMCSamplingGroupConfig(name="empty_mask", chisq_mask=""),
        }
        masks = _read_chisq_masks(MPI.COMM_SELF, groups)

        assert set(masks) == {"with_mask"}
        np.testing.assert_allclose(masks["with_mask"], mask)

    def test_no_groups_gives_empty_mapping(self):
        assert _read_chisq_masks(MPI.COMM_SELF, {}) == {}
