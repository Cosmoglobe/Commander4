import numpy as np
import healpy as hp
import pytest
from mpi4py import MPI
from numpy.testing import assert_allclose

from commander4.data_models.detector_map import DetectorMap
from commander4.compsep import preconditioners
from commander4.compsep.preconditioners import JointPreconditioner
from commander4.math_utils.alm import alm_dot_product


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


def test_spin2_equal_qu_weights_have_small_cross_channel_leakage():
    leakage = _spin2_cross_channel_leakage(q_rms=1.0, u_rms=1.0, nside=32, lmax=32)
    assert leakage < 1.0e-2


def test_spin2_unequal_qu_weights_induce_large_cross_channel_leakage():
    leakage = _spin2_cross_channel_leakage(q_rms=1.0, u_rms=2.0, nside=32, lmax=32)
    assert leakage > 1.0e-1


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

class _SEDDiffuseComp(_DummyDiffuseComp):
    """A diffuse component with a settable SED, so the mixing matrix is not degenerate."""

    def __init__(self, lmax: int, npol: int, sed: float):
        super().__init__(lmax=lmax, npol=npol)
        self.sed = sed

    def get_sed(self, nu: float) -> float:
        return self.sed


def _single_band_precond_block(preconditioner_name: str, ell: int):
    """Build one preconditioner over a single 60' band with two differently-mixing components."""
    lmax, nside = 8, 8
    inv_n_map = np.full((1, 12*nside**2), 2.0, dtype=np.float64)
    compsep = _DummyCompSep(inv_n_map=inv_n_map, lmax=lmax)
    compsep.my_band.fwhm = 60.0   # arcmin
    comp_list = _DummyCompList([_SEDDiffuseComp(lmax=lmax, npol=1, sed=sed) for sed in (1.0, 3.0)])
    precond = getattr(preconditioners, preconditioner_name)(compsep, comp_list)
    return precond, precond.ell_block_data[ell][3], comp_list


def test_preconditioner_variants_build_the_block_their_name_claims():
    """`BeamOnly` and `MixingMatrix` are `Joint` with the other factors set to unity.

    Each is checked against a hand-built reference at one multipole, so switching a factor off
    cannot silently keep it. Dropping the mixing must give the identity in component space: setting
    every mixing coefficient to one instead would declare the components perfectly degenerate and
    make the block singular.
    """
    ell, lmax = 4, 8
    npix = 12*8**2
    noise_weight = 2.0 * npix/(4*np.pi)
    beam_squared = hp.gauss_beam(np.deg2rad(60.0/60), lmax=lmax)[ell]**2
    mixing = np.array([1.0, 3.0])
    identity = np.eye(2)
    # The dummy component's P_Cl_prior_inv is one, so the prior term is the identity.
    references = {
        "JointPreconditioner": identity + beam_squared*noise_weight*np.outer(mixing, mixing),
        "BeamOnlyPreconditioner": identity + beam_squared*identity,
        "MixingMatrixPreconditioner": identity + np.outer(mixing, mixing),
    }
    for name, expected in references.items():
        _, block_inv, _ = _single_band_precond_block(name, ell)
        assert_allclose(block_inv, np.linalg.inv(expected), rtol=1e-10, atol=1e-12,
                        err_msg=f"{name} does not build the block its name claims")
        # A preconditioner must stay positive definite to be usable by CG at all.
        assert np.all(np.linalg.eigvalsh(block_inv) > 0.0), f"{name} block is not positive definite"


@pytest.mark.parametrize(
    "preconditioner_name",
    ["JointPreconditioner", "BeamOnlyPreconditioner", "MixingMatrixPreconditioner"],
)
def test_preconditioner_variants_apply_to_a_complist(preconditioner_name: str):
    """Every selectable preconditioner takes and returns a CompList, not a bare alm array."""
    precond, _, comp_list = _single_band_precond_block(preconditioner_name, ell=4)
    comp_list[0].alms[:] = 1.0
    result = precond(comp_list)
    assert len(result) == len(comp_list)
    assert result[0].alms.shape == comp_list[0].alms.shape
    # The preconditioner must actually change the amplitudes, not silently pass them through.
    assert not np.allclose(result[0].alms, comp_list[0].alms)
