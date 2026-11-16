import numpy as np
import healpy as hp
import pytest
from mpi4py import MPI
from numpy.testing import assert_allclose

from commander4.data_models.detector_map import DetectorMap
from commander4.compsep import preconditioners
from commander4.compsep.preconditioners import JointPreconditioner
from commander4.sky.comp_list import CompList
from commander4.sky.diffuse_components import DiffuseComponent
from commander4.math_utils.alm import alm_dot_product


def _spin2_cross_channel_leakage(q_rms: float, u_rms: float, nside: int, lmax: int) -> float:
    map_sky = np.zeros((2, 12 * nside**2), dtype=np.float64)
    map_inv_var = np.vstack([
        np.full(12 * nside**2, 1.0/q_rms**2, dtype=np.float64),
        np.full(12 * nside**2, 1.0/u_rms**2, dtype=np.float64),
    ])
    detector_map = DetectorMap(
        map_sky,
        map_inv_var,
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
    def __init__(self, map_inv_var: np.ndarray, lmax: int):
        self.map_inv_var = map_inv_var
        self.lmax = lmax


class _DummyBand:
    def __init__(self, nu: float, fwhm: float):
        self.nu = nu
        self.fwhm = fwhm


class _DummyCompSep:
    def __init__(self, map_inv_var: np.ndarray, lmax: int):
        self.CompSep_comm = MPI.COMM_SELF
        self.band_map = _DummyDetectorMap(map_inv_var, lmax)
        self.my_band = _DummyBand(nu=100.0, fwhm=0.0)


class _DummyDiffuseComp(DiffuseComponent):
    """Only what the preconditioners read. `DiffuseComponent.__init__` is skipped, because it needs
    a parameter file; the class is inherited so that the preconditioners treat it as diffuse."""
    # Any value but None: the component has a prior. Its C_l is one (see `P_Cl_prior_inv`).
    Cl_prior_amplitude = 1.0

    def __init__(self, lmax: int, npol: int):
        self.lmax = lmax
        self.pol = "I" if npol == 1 else "QU"
        self.amps = np.zeros((npol, hp.Alm.getsize(lmax)), dtype=np.complex128)

    @property
    def P_Cl_prior(self) -> np.ndarray:
        return np.ones(self.lmax + 1, dtype=np.float64)

    @property
    def P_Cl_prior_inv(self) -> np.ndarray:
        return np.ones(self.lmax + 1, dtype=np.float64)

    def get_sed(self, nu: float) -> float:
        return 1.0


class _NonDiffuseComp:
    """Stands in for a point-source component, which the preconditioners leave unchanged."""

    def __init__(self):
        self.amps = np.full((1, 3), 5.0)


def test_diagonal_preconditioner_gives_each_component_its_own_diagonal():
    # A non-diffuse component placed first must not shift the diagonals of the ones after it.
    lmax, npix = 4, 12*4**2
    compsep = _DummyCompSep(map_inv_var=np.full((1, npix), 2.0), lmax=lmax)
    comp_list = CompList([_NonDiffuseComp(), _DummyDiffuseComp(lmax=lmax, npol=1)])
    comp_list[1].amps[:] = 1.0

    result = preconditioners.DiagonalJointPreconditioner(compsep, comp_list)(comp_list)

    # Unit SED, no beam, C_l = 1: the diagonal is 1 (prior) + the mean weight per steradian.
    assert_allclose(result[1].amps, 1.0/(1.0 + 2.0*npix/(4*np.pi)), rtol=1e-12)
    assert_allclose(result[0].amps, 5.0)


def test_spin2_equal_qu_weights_have_small_cross_channel_leakage():
    leakage = _spin2_cross_channel_leakage(q_rms=1.0, u_rms=1.0, nside=32, lmax=32)
    assert leakage < 1.0e-2


def test_spin2_unequal_qu_weights_induce_large_cross_channel_leakage():
    leakage = _spin2_cross_channel_leakage(q_rms=1.0, u_rms=2.0, nside=32, lmax=32)
    assert leakage > 1.0e-1


def test_joint_preconditioner_uses_trace_weight_for_spin2_channels():
    nside = 16
    npix = 12 * nside**2
    map_inv_var = np.vstack([
        np.full(npix, 1.0, dtype=np.float64),
        np.full(npix, 0.25, dtype=np.float64),
    ])
    compsep = _DummyCompSep(map_inv_var=map_inv_var, lmax=16)
    comp_list = CompList([_DummyDiffuseComp(lmax=16, npol=2)])

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
    map_inv_var = np.full((1, 12*nside**2), 2.0, dtype=np.float64)
    compsep = _DummyCompSep(map_inv_var=map_inv_var, lmax=lmax)
    compsep.my_band.fwhm = 60.0   # arcmin
    comp_list = CompList([_SEDDiffuseComp(lmax=lmax, npol=1, sed=sed) for sed in (1.0, 3.0)])
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


def test_joint_preconditioner_has_no_prior_term_for_a_component_without_a_prior():
    """C3's `CL_TYPE = none`: such a component gets no identity on its diagonal. Above the band
    lmax nothing constrains it, and the preconditioner must leave it untouched (zero) there."""
    comp_lmax, band_lmax, nside = 10, 8, 8
    compsep = _DummyCompSep(map_inv_var=np.full((1, 12*nside**2), 2.0), lmax=band_lmax)
    compsep.my_band.fwhm = 60.0   # arcmin
    comps = [_SEDDiffuseComp(lmax=comp_lmax, npol=1, sed=sed) for sed in (1.0, 3.0)]
    comps[0].Cl_prior_amplitude = None
    precond = JointPreconditioner(compsep, CompList(comps))

    noise_weight = 2.0 * 12*nside**2/(4*np.pi)
    beam_squared = hp.gauss_beam(np.deg2rad(60.0/60), lmax=band_lmax)[4]**2
    mixing = np.array([1.0, 3.0])
    expected = np.diag([0.0, 1.0]) + beam_squared*noise_weight*np.outer(mixing, mixing)
    assert_allclose(precond.ell_block_data[4][3], np.linalg.inv(expected), rtol=1e-10)
    assert_allclose(precond.ell_block_data[9][3], np.diag([0.0, 1.0]), atol=1e-12)


@pytest.mark.parametrize(
    "preconditioner_name",
    ["JointPreconditioner", "BeamOnlyPreconditioner", "MixingMatrixPreconditioner"],
)
def test_preconditioner_variants_apply_to_a_complist(preconditioner_name: str):
    """Every selectable preconditioner takes and returns a CompList, not a bare alm array."""
    precond, _, comp_list = _single_band_precond_block(preconditioner_name, ell=4)
    comp_list[0].amps[:] = 1.0
    result = precond(comp_list)
    assert len(result) == len(comp_list)
    assert result[0].amps.shape == comp_list[0].amps.shape
    # The preconditioner must actually change the amplitudes, not silently pass them through.
    assert not np.allclose(result[0].amps, comp_list[0].amps)
