"""Commander3-equivalence tests for far-sidelobe convolution."""

from types import SimpleNamespace

import healpy as hp
import numpy as np
import pytest
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


def _fake_instrument_file(components: tuple[str, ...]) -> FakeHDF:
    """Detector 27M with sidelobe limits (105, 102) and, per stored component, a beam whose
    monopole is 1 (T), 2 (E) or 3 (B)."""
    file_lmax = 105
    file_data = {"27M/sllmax": np.array([file_lmax]), "27M/slmmax": np.array([102])}
    for component in components:
        beam = np.zeros((file_lmax + 1)**2)
        beam[0] = ("T", "E", "B").index(component) + 1
        file_data[f"27M/sl/{component}"] = beam
    return FakeHDF(file_data)


def _bare_projector(pols: str = "IQU") -> sidelobe.FarBeamProjector:
    projector = sidelobe.FarBeamProjector.__new__(sidelobe.FarBeamProjector)
    projector.pols = pols
    projector.instrument_file = "instrument.h5"
    projector.detnames = ["27M"]
    projector.nthreads = 2
    projector.far_beam_deconvolution_cfg = sidelobe.FarBeamConfig(enabled=True, lmax=100, mmax=100)
    return projector


@pytest.mark.parametrize("pols, stored, used", [
    ("IQU", ("T", "E", "B"), ("T", "E", "B")),
    ("IQU", ("T",), ("T",)),  # a missing E or B beam is skipped
    ("I", ("T", "E", "B"), ("T",)),  # an intensity band has no E or B sky
])
def test_construct_model_matches_commander3_limits(monkeypatch, pols, stored, used) -> None:
    """Every used T/E/B beam is paired with its own sky component."""
    fake_hdf = _fake_instrument_file(stored)
    plans = []

    def make_plan(**kwargs):
        plan = FakeConvolverPlan(**kwargs)
        plans.append(plan)
        return plan

    expected_slm = np.zeros((3, hp.Alm.getsize(100)), dtype=np.complex128)
    map2alm_calls = []

    def fake_map2alm(maps, **kwargs):
        map2alm_calls.append((maps, kwargs))
        # Like healpy: T/E/B alms for a polarized transform, one flat alm array otherwise.
        return expected_slm if kwargs["pol"] else expected_slm[0]

    monkeypatch.setattr(sidelobe.h5py, "File", lambda *args, **kwargs: fake_hdf)
    monkeypatch.setattr(sidelobe.totalconvolve, "ConvolverPlan", make_plan)
    monkeypatch.setattr(sidelobe.hp, "map2alm", fake_map2alm)

    projector = _bare_projector(pols)
    sky = np.zeros((len(pols), hp.nside2npix(1)))
    # A real single-rank node communicator, so the cubes go through actual MPI shared memory.
    projector.construct_model(FakeBandComm(), MPI.COMM_SELF, sky)

    assert len(plans) == 1
    assert plans[0].lmax == 100
    assert plans[0].kmax == 100
    assert len(plans[0].calls) == len(used)*101
    for component, (slm, blm, mbeam) in zip(used, plans[0].calls):
        isky = ("T", "E", "B").index(component)
        assert np.shares_memory(slm, expected_slm)
        assert np.array_equal(slm, expected_slm[isky])
        assert blm.shape == (hp.Alm.getsize(100, 100),)
        assert blm[0] == 2.0*(isky + 1)
        assert mbeam == 0
    assert map2alm_calls[0][0] is sky
    assert map2alm_calls[0][1] == {"lmax": 100, "iter": 0, "pol": pols == "IQU"}
    # The fake getPlane writes the beam monopole (times beam_norm = 2), summed over components.
    expected_cube = 2.0*sum(("T", "E", "B").index(c) + 1 for c in used)
    assert np.all(projector.cubes[0][:201] == expected_cube)

    # The cubes live in an MPI window that nothing releases on garbage collection.
    projector.free()
    assert projector.cubes == []


def test_init_rejects_a_band_without_intensity_and_a_detector_without_polang(monkeypatch) -> None:
    monkeypatch.setattr(sidelobe.FarBeamProjector, "construct_model", lambda *args: None)
    det = SimpleNamespace(name="27M", det_idx_fullband=0, polang=None)
    band = SimpleNamespace(pols="IQU", nside=1, instrument_filepath="instrument.h5", ndet=1,
                           iter_detector_scans=lambda: [(0, det)])
    tod_samples = SimpleNamespace(det_names=["27M"])
    cfg = sidelobe.FarBeamConfig(enabled=True)
    with pytest.raises(ValueError, match="27M has no polang"):
        sidelobe.FarBeamProjector(None, None, band, tod_samples, None, cfg)
    band.pols = "QU"
    with pytest.raises(ValueError, match="needs an I or I/Q/U band"):
        sidelobe.FarBeamProjector(None, None, band, tod_samples, None, cfg)


def test_construct_model_requires_the_T_beam(monkeypatch) -> None:
    fake_hdf = _fake_instrument_file(("E", "B"))
    monkeypatch.setattr(sidelobe.h5py, "File", lambda *args, **kwargs: fake_hdf)
    with pytest.raises(ValueError, match="No sidelobe T beam for 27M"):
        _bare_projector().construct_model(FakeBandComm(), MPI.COMM_SELF, None)


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
