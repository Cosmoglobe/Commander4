"""Commander3-equivalence tests for polarized far-sidelobe convolution."""

import healpy as hp
import numpy as np
from mpi4py import MPI

import commander4.tod.sidelobe_deconvolve as sidelobe


class FakeHDF(dict):
    """Small dictionary-backed context manager with the h5py indexing interface."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeBandComm:
    """Single-rank communicator used by the projector construction test."""

    def Get_rank(self) -> int:
        return 0

    def bcast(self, value, root: int):
        return value


class FakeConvolverPlan:
    """Record the arrays passed to the ducc0 plan without doing an SHT."""

    def __init__(self, lmax: int, kmax: int, epsilon: float, nthreads: int):
        self.lmax = lmax
        self.kmax = kmax
        self.epsilon = epsilon
        self.nthreads = nthreads
        self.calls = []

    def Npsi(self) -> int:
        return 2*self.kmax + 1

    def Ntheta(self) -> int:
        return 1

    def Nphi(self) -> int:
        return 1

    def getPlane(self, slm, blm, mbeam: int, planes) -> None:
        self.calls.append((slm, blm, mbeam))
        planes[:] = blm[0].real

    def prepPsi(self, cube) -> None:
        pass


def test_construct_model_matches_commander3_polarized_limits(monkeypatch) -> None:
    file_lmax = 105
    file_mmax = 102
    file_data = {
        "27M/sllmax": np.array([file_lmax]),
        "27M/slmmax": np.array([file_mmax]),
    }
    for value, component in enumerate(("T", "E", "B"), start=1):
        beam = np.zeros((file_lmax + 1)**2)
        beam[0] = value
        file_data[f"27M/sl/{component}"] = beam

    fake_hdf = FakeHDF(file_data)
    plans = []

    def make_plan(**kwargs):
        plan = FakeConvolverPlan(**kwargs)
        plans.append(plan)
        return plan

    expected_slm = np.zeros((3, hp.Alm.getsize(100)), dtype=np.complex128)
    map2alm_calls = []

    def fake_map2alm(maps, **kwargs):
        map2alm_calls.append((maps, kwargs))
        return expected_slm

    monkeypatch.setattr(sidelobe.h5py, "File", lambda *args, **kwargs: fake_hdf)
    monkeypatch.setattr(sidelobe.totalconvolve, "ConvolverPlan", make_plan)
    monkeypatch.setattr(sidelobe.hp, "map2alm", fake_map2alm)

    projector = sidelobe.FarBeamProjector.__new__(sidelobe.FarBeamProjector)
    projector.instrument_file = "instrument.h5"
    projector.detnames = ["27M"]
    projector.nthreads = 2
    projector.config = sidelobe.FarBeamConfig(enabled=True, lmax=100, mmax=100)
    sky = np.zeros((3, hp.nside2npix(1)))
    # A real single-rank node communicator, so the cubes go through actual MPI shared memory.
    projector.construct_model(FakeBandComm(), MPI.COMM_SELF, sky)

    assert len(plans) == 1
    assert plans[0].lmax == 100
    assert plans[0].kmax == 100
    assert len(plans[0].calls) == 3*101
    for component, (slm, blm, mbeam) in enumerate(plans[0].calls[:3]):
        assert np.shares_memory(slm, expected_slm)
        assert np.array_equal(slm, expected_slm[component])
        assert blm.shape == (hp.Alm.getsize(100, 100),)
        assert blm[0] == 2.0*(component + 1)
        assert mbeam == 0
    assert map2alm_calls[0][0] is sky
    assert map2alm_calls[0][1] == {"lmax": 100, "iter": 0, "pol": True}
    assert np.all(projector.cubes[0][:201] == 12.0)

    # The cubes live in an MPI window that nothing releases on garbage collection.
    projector.free()
    assert projector.cubes == []


def test_projection_evaluates_every_sample_at_the_ducc_angle() -> None:
    class InterpolationPlan:
        def __init__(self):
            self.calls = []

        def interpol(self, cube, spin, deriv, theta, phi, psi, res) -> None:
            self.calls.append(psi.copy())
            res[:] = theta + phi + psi

    projector = sidelobe.FarBeamProjector.__new__(sidelobe.FarBeamProjector)
    projector.nside = 1
    projector.polangs = np.array([0.3])
    projector.plan = InterpolationPlan()
    projector.cubes = [np.zeros(1)]

    pix = np.arange(13) % hp.nside2npix(1)
    psi = np.linspace(0.0, 0.6, pix.size)
    result = projector.get_projection(pix, psi, 0)

    assert result.shape == pix.shape
    assert len(projector.plan.calls) == 1
    assert projector.plan.calls[0].size == pix.size
    expected_psi = np.mod(psi - 0.3, 2*np.pi)
    assert np.array_equal(projector.plan.calls[0], expected_psi)
