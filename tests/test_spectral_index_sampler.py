import importlib
import sys
import types

import numpy as np
import pytest
from mpi4py import MPI


class AttrBunch(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


class FakeCompList:
    def __init__(self, comp_list):
        self.comp_list = list(comp_list)

    def __iter__(self):
        for item in self.comp_list:
            yield item

    def __len__(self):
        return len(self.comp_list)

    def __getitem__(self, index):
        return self.comp_list[index]

    def split_for_eval_pol(self, target_pol):
        # The FakeSkyModel filters by `is_pol`, so returning the full list per polarization is
        # sufficient for the single-band, single-polarization likelihood exercised here.
        return self


class StubComponent:
    pass


class FakeDetectorMap:
    def __init__(self, map_sky, map_rms, nu, fwhm, nside, double_precision=True):
        self.map_sky = np.array(map_sky, dtype=np.float64, copy=True)
        self._map_rms = np.array(map_rms, dtype=np.float64, copy=True)
        self.nu = nu
        self.fwhm = fwhm
        self.fwhm_rad = fwhm  # MCMCSamplingGroup.local_loglike realizes the model at this resolution.
        self.nside = nside
        self.double_precision = double_precision

    @property
    def map_rms(self):
        return self._map_rms.copy()

    @property
    def pol(self):
        return self.map_sky.shape[0] == 2


class FakeSkyModel:
    def __init__(self, components):
        self._components = components

    def get_sky_at_nu(self, nu, nside, pols_required, fwhm=None):
        npix = 12 * nside**2
        if pols_required == "I":
            skymap = np.zeros((1, npix), dtype=np.float64)
            for component in self._components:
                if not component.is_pol:
                    skymap[0] += component.get_sky(nu, nside, fwhm)[0]
            return skymap
        if pols_required == "QU":
            skymap = np.zeros((2, npix), dtype=np.float64)
            for component in self._components:
                if component.is_pol:
                    skymap += component.get_sky(nu, nside, fwhm)
            return skymap
        raise ValueError("Unsupported polarization in test stub.")


pixell_module = types.ModuleType("pixell")
bunch_module = types.ModuleType("pixell.bunch")
bunch_module.Bunch = AttrBunch
pixell_module.bunch = bunch_module
sys.modules["pixell"] = pixell_module
sys.modules["pixell.bunch"] = bunch_module

component_module = types.ModuleType("commander4.sky_models.component")
component_module.Component = StubComponent
component_module.CompList = FakeCompList
sys.modules["commander4.sky_models.component"] = component_module

detector_map_module = types.ModuleType("commander4.data_models.detector_map")
detector_map_module.DetectorMap = FakeDetectorMap
sys.modules["commander4.data_models.detector_map"] = detector_map_module

sky_model_module = types.ModuleType("commander4.sky_models.sky_model")
sky_model_module.SkyModel = FakeSkyModel
sys.modules["commander4.sky_models.sky_model"] = sky_model_module

spectral_index_sampler = importlib.import_module("commander4.solvers.spectral_index_sampler")


class FakeComponent:
    def __init__(self, shortname, eval_pol, beta, amplitude, comp_params=None, sample=False,
                 proposal_sigma=None, bounds=None):
        self.eval_pol = eval_pol
        self.shortname = shortname
        self.longname = shortname
        if comp_params is None:
            comp_params = AttrBunch({"shortname": shortname, "beta": beta})
            if sample:
                comp_params["sample_spectral_index"] = True
                comp_params["spectral_index_proposal_sigma"] = proposal_sigma
            if bounds is not None:
                comp_params["spectral_index_bounds"] = bounds
        self.comp_params = comp_params
        self.beta = beta
        self.alms = self._make_alms(amplitude)

    @property
    def is_pol(self):
        return self.eval_pol == "QU"

    @property
    def npol(self):
        return 2 if self.is_pol else 1

    @staticmethod
    def _make_alms(amplitude, npol=1):
        return np.full((npol, 1), amplitude, dtype=np.float64)

    @property
    def alms(self):
        return self._data

    @alms.setter
    def alms(self, alms):
        self._data = np.array(alms, dtype=np.float64, copy=True)

    def get_sed(self, nu):
        return np.asarray(nu, dtype=np.float64)**self.beta

    def get_sky(self, nu, nside, fwhm=0.0):
        npix = 12 * nside**2
        amplitude = float(self.alms[0, 0])
        base = np.full((self.npol, npix), amplitude, dtype=np.float64)
        return base * self.get_sed(nu)


def make_detector_data(component, nu=2.0, nside=1, rms=1.0):
    map_sky = component.get_sky(nu, nside)
    map_rms = np.full_like(map_sky, rms, dtype=np.float64)
    return FakeDetectorMap(map_sky, map_rms, nu=nu, fwhm=0.0, nside=nside,
                           double_precision=True)


def make_iqu_components(beta=1.5, amplitude=2.0, bounds=(1.0, 2.0), proposal_sigma=0.1):
    shared_params = AttrBunch({
        "shortname": "dust",
        "beta": beta,
        "sample_spectral_index": True,
        "spectral_index_proposal_sigma": proposal_sigma,
        "spectral_index_bounds": list(bounds),
    })
    comp_i = FakeComponent("dust_I", "I", beta, amplitude, comp_params=shared_params)
    comp_qu = FakeComponent("dust_QU", "QU", beta, amplitude,
                            comp_params=shared_params)
    comp_qu.alms = FakeComponent._make_alms(amplitude, npol=2)
    return comp_i, comp_qu


class TestDiscoverSpectralIndexGroups:
    def test_groups_shared_iqu_component_once(self):
        comp_i, comp_qu = make_iqu_components()
        groups = spectral_index_sampler._discover_spectral_index_groups(
            FakeCompList([comp_i, comp_qu]), None
        )

        assert len(groups) == 1
        assert groups[0].name == "dust"
        assert groups[0].proposal_sigma == pytest.approx(0.1)
        assert groups[0].bounds == pytest.approx((1.0, 2.0))
        assert groups[0].prior is None
        assert groups[0].components == (comp_i, comp_qu)


def _make_group(detector_data, comp_list, target_pol="I"):
    """Build a single-rank SpectralIndexSamplingGroup over `comp_list` on COMM_SELF."""
    return spectral_index_sampler.SpectralIndexSamplingGroup(
        MPI.COMM_SELF, detector_data, comp_list, target_pol=target_pol,
        chisq_active=True, selected_comps=None)


class TestSampleSpectralIndicesMH:
    """The MH machinery now lives in ``MCMCSamplingGroup.run``; these drive a real
    ``SpectralIndexSamplingGroup`` on ``COMM_SELF`` and check the accept/reject side effects on the
    component ``beta`` (the source of truth ``apply_state`` writes) and the coupled amplitudes.

    The coupled amplitude re-solve is stubbed by the ``resolve_amplitudes`` callback ``run`` invokes;
    the likelihood is either monkeypatched per step (``local_loglike``) or, in the integration test,
    left as the real whitened-residual chi-squared."""

    def test_out_of_bounds_rejects_without_resolve(self, monkeypatch):
        comp_i, comp_qu = make_iqu_components()
        detector_data = make_detector_data(comp_i)
        group = _make_group(detector_data, FakeCompList([comp_i, comp_qu]))

        def fail_if_called():
            raise AssertionError("resolve_amplitudes must not run for out-of-bounds proposals")

        # Proposal 2.5 is outside the (1.0, 2.0) bounds -> rejected before the amplitude re-solve.
        monkeypatch.setattr(np.random, "normal", lambda loc, scale: 2.5)

        group.run(numstep=1, resolve_amplitudes=fail_if_called)

        assert comp_i.beta == pytest.approx(1.5)
        assert comp_qu.beta == pytest.approx(1.5)
        np.testing.assert_allclose(comp_i.alms, np.array([[2.0]]))

    def test_rejection_restores_parameters_and_alms(self, monkeypatch):
        comp_i, comp_qu = make_iqu_components()
        detector_data = make_detector_data(comp_i)
        group = _make_group(detector_data, FakeCompList([comp_i, comp_qu]))

        def resolve():  # The coupled amplitude re-solve mutates the shared component amplitudes.
            comp_i.alms = np.array([[9.0]])

        # loglike drops on the proposal and random() forces a reject; state must be fully reverted.
        loglikes = iter([0.0, -5.0])
        monkeypatch.setattr(group, "local_loglike", lambda: next(loglikes))
        monkeypatch.setattr(np.random, "normal", lambda loc, scale: 1.6)
        monkeypatch.setattr(np.random, "random", lambda: 0.5)

        group.run(numstep=1, resolve_amplitudes=resolve)

        assert comp_i.beta == pytest.approx(1.5)
        assert comp_qu.beta == pytest.approx(1.5)
        np.testing.assert_allclose(comp_i.alms, np.array([[2.0]]))

    def test_acceptance_keeps_proposed_parameters_and_alms(self, monkeypatch):
        comp_i, comp_qu = make_iqu_components()
        detector_data = make_detector_data(comp_i)
        group = _make_group(detector_data, FakeCompList([comp_i, comp_qu]))

        def resolve():
            comp_i.alms = np.array([[9.0]])

        # loglike rises on the proposal -> accepted; the proposed beta and re-solved alms are kept.
        loglikes = iter([0.0, 2.0])
        monkeypatch.setattr(group, "local_loglike", lambda: next(loglikes))
        monkeypatch.setattr(np.random, "normal", lambda loc, scale: 1.6)
        monkeypatch.setattr(np.random, "random", lambda: 0.5)

        group.run(numstep=1, resolve_amplitudes=resolve)

        assert comp_i.beta == pytest.approx(1.6)
        assert comp_qu.beta == pytest.approx(1.6)
        np.testing.assert_allclose(comp_i.alms, np.array([[9.0]]))

    def test_single_rank_integration_uses_real_likelihood(self, monkeypatch):
        comp_i, comp_qu = make_iqu_components(beta=1.0, amplitude=2.0)

        target_map = np.full((1, 12), 8.0, dtype=np.float64)
        detector_data = FakeDetectorMap(target_map, np.ones_like(target_map), nu=2.0,
                        fwhm=0.0, nside=1, double_precision=True)
        group = _make_group(detector_data, FakeCompList([comp_i, comp_qu]))

        def resolve():  # Fit the amplitude exactly to the proposed index so the residual vanishes.
            target_amplitude = detector_data.map_sky[0, 0] / (detector_data.nu**comp_i.beta)
            comp_i.alms = np.array([[target_amplitude]])

        monkeypatch.setattr(np.random, "normal", lambda loc, scale: 1.5)
        monkeypatch.setattr(np.random, "random", lambda: 0.5)

        group.run(numstep=1, resolve_amplitudes=resolve)

        # The proposed beta=1.5 gives a perfect fit (higher likelihood), so it is accepted.
        assert comp_i.beta == pytest.approx(1.5)
        assert comp_qu.beta == pytest.approx(1.5)
        np.testing.assert_allclose(comp_i.alms, np.array([[8.0 / (2.0**1.5)]]))