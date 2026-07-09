import numpy as np
import ducc0
import healpy as hp
from mpi4py import MPI
from numpy.testing import assert_allclose

from commander4.data_models.detector_map import DetectorMap
from commander4.solvers.preconditioners import JointPreconditioner
from commander4.utils.math_operations import alm_dot_product


def _spin2_cross_channel_leakage(q_rms: float, u_rms: float, nside: int, lmax: int) -> float:
    map_sky = np.zeros((2, 12 * nside**2), dtype=np.float64)
    map_rms = np.vstack([
        np.full(12 * nside**2, q_rms, dtype=np.float64),
        np.full(12 * nside**2, u_rms, dtype=np.float64),
    ])
    detector_map = DetectorMap(
        map_sky,
        map_rms,
        nu=100.0,
        fwhm=0.0,
        nside=nside,
        double_precision=True,
        lmax=lmax,
    )

    alm = np.zeros((2, hp.Alm.getsize(lmax)), dtype=np.complex128)
    alm[0, hp.Alm.getidx(lmax, min(10, lmax), min(2, lmax))] = 1.0
    transformed = detector_map.apply_inv_N_alm(alm, nthreads=1, inplace=False)

    main_norm = np.sqrt(alm_dot_product(transformed[0], transformed[0], lmax))
    cross_norm = np.sqrt(alm_dot_product(transformed[1], transformed[1], lmax))
    return float(cross_norm / main_norm)


class _DummyDetectorMap:
    def __init__(self, inv_n_map: np.ndarray, lmax: int):
        self.inv_n_map = inv_n_map
        self.lmax = lmax


class _DummyBand:
    def __init__(self, nu: float, fwhm: float):
        self.nu = nu
        self.fwhm = fwhm


class _DummyCompSep:
    def __init__(self, inv_n_map: np.ndarray, lmax: int):
        self.CompSep_comm = MPI.COMM_SELF
        self.det_map = _DummyDetectorMap(inv_n_map, lmax)
        self.my_band = _DummyBand(nu=100.0, fwhm=0.0)


class _DummyDiffuseComp:
    def __init__(self, lmax: int, npol: int):
        self.lmax = lmax
        self.npol = npol
        self._alms = np.zeros((npol, hp.Alm.getsize(lmax)), dtype=np.complex128)

    @property
    def alms(self) -> np.ndarray:
        return self._alms

    @alms.setter
    def alms(self, value: np.ndarray) -> None:
        self._alms = value

    @property
    def P_Cl_prior_inv(self) -> np.ndarray:
        return np.ones(self.lmax + 1, dtype=np.float64)

    def get_sed(self, nu: float) -> float:
        return 1.0


class _DummyCompList:
    def __init__(self, comps):
        self._comps = comps

    def __len__(self):
        return len(self._comps)

    def __getitem__(self, index):
        return self._comps[index]

    def __iter__(self):
        return iter(self._comps)


def test_spin2_standard_mode_matches_grad_only_when_curl_is_zero():
    nside = 16
    lmax = 12
    alm = np.zeros((2, hp.Alm.getsize(lmax)), dtype=np.complex128)
    alm[0, hp.Alm.getidx(lmax, 6, 2)] = 1.0 + 0.5j
    geom = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()

    standard_map = ducc0.sht.synthesis(alm=alm, lmax=lmax, spin=2, nthreads=1, **geom)
    grad_only_map = ducc0.sht.synthesis(
        alm=alm[:1],
        lmax=lmax,
        spin=2,
        nthreads=1,
        mode="GRAD_ONLY",
        **geom,
    )

    assert_allclose(standard_map, grad_only_map, rtol=1e-12, atol=1e-12)


def test_spin2_equal_qu_weights_have_small_cross_channel_leakage():
    leakage = _spin2_cross_channel_leakage(q_rms=1.0, u_rms=1.0, nside=32, lmax=32)
    assert leakage < 1.0e-2


def test_spin2_unequal_qu_weights_induce_large_cross_channel_leakage():
    leakage = _spin2_cross_channel_leakage(q_rms=1.0, u_rms=2.0, nside=32, lmax=32)
    assert leakage > 1.0e-1


def test_spin2_equal_weight_leakage_drops_when_nside_increases():
    coarse = _spin2_cross_channel_leakage(q_rms=1.0, u_rms=1.0, nside=32, lmax=32)
    fine = _spin2_cross_channel_leakage(q_rms=1.0, u_rms=1.0, nside=64, lmax=32)
    assert fine < coarse


def test_spin2_equal_weight_leakage_grows_when_lmax_increases():
    low_lmax = _spin2_cross_channel_leakage(q_rms=1.0, u_rms=1.0, nside=32, lmax=16)
    high_lmax = _spin2_cross_channel_leakage(q_rms=1.0, u_rms=1.0, nside=32, lmax=64)
    assert high_lmax > low_lmax


def test_spin2_unequal_weight_leakage_is_not_fixed_by_higher_nside():
    coarse = _spin2_cross_channel_leakage(q_rms=1.0, u_rms=2.0, nside=32, lmax=32)
    fine = _spin2_cross_channel_leakage(q_rms=1.0, u_rms=2.0, nside=64, lmax=32)
    assert fine > 1.0e-1
    assert coarse > 1.0e-1


def test_joint_preconditioner_uses_trace_weight_for_spin2_channels():
    nside = 16
    npix = 12 * nside**2
    inv_n_map = np.vstack([
        np.full(npix, 1.0, dtype=np.float64),
        np.full(npix, 0.25, dtype=np.float64),
    ])
    compsep = _DummyCompSep(inv_n_map=inv_n_map, lmax=16)
    comp_list = _DummyCompList([_DummyDiffuseComp(lmax=16, npol=2)])

    precond = JointPreconditioner(compsep, comp_list)
    _, _, _, block_inv = precond.ell_block_data[4]

    assert_allclose(block_inv, np.eye(2) * block_inv[0, 0], rtol=1e-12, atol=1e-12)