import numpy as np
import healpy as hp
import pytest
from mpi4py import MPI
from numpy.testing import assert_allclose

from commander4.data_models.band_map import BandMap
from commander4.compsep import preconditioners
from commander4.compsep.preconditioners import JointPreconditioner
from commander4.sky.comp_list import CompList
from commander4.sky.diffuse_components import DiffuseComponent


def _band_map(map_inv_var: np.ndarray, lmax: int, fwhm_arcmin: float = 0.0) -> BandMap:
    """A 100 GHz band with the given inverse-noise map; its sky map, which the preconditioners do
    not read, is zero."""
    nside = hp.npix2nside(map_inv_var.shape[-1])
    return BandMap(np.zeros_like(map_inv_var), map_inv_var, nu=100.0, fwhm_arcmin=fwhm_arcmin,
                   nside=nside, lmax=lmax)


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
    band_map = _band_map(np.full((1, npix), 2.0), lmax=lmax)
    comp_list = CompList([_NonDiffuseComp(), _DummyDiffuseComp(lmax=lmax, npol=1)])
    comp_list[1].amps[:] = 1.0

    precond = preconditioners.DiagonalJointPreconditioner(MPI.COMM_SELF, band_map, comp_list)
    result = precond(comp_list)

    # Unit SED, no beam, C_l = 1: the diagonal is 1 (prior) + the mean weight per steradian.
    assert_allclose(result[1].amps, 1.0/(1.0 + 2.0*npix/(4*np.pi)), rtol=1e-12)
    assert_allclose(result[0].amps, 5.0)


def test_joint_preconditioner_has_only_the_prior_above_the_band_lmax():
    # The band sees the component only up to its own lmax, as in `comp2band`.
    comp_lmax, band_lmax, npix = 6, 3, 12*4**2
    band_map = _band_map(np.full((1, npix), 2.0), lmax=band_lmax)
    comp_list = CompList([_DummyDiffuseComp(lmax=comp_lmax, npol=1)])
    comp_list[0].amps[:] = 1.0

    result = JointPreconditioner(MPI.COMM_SELF, band_map, comp_list)(comp_list)

    ell = hp.Alm.getlm(comp_lmax)[0]
    assert_allclose(result[0].amps[0, ell <= band_lmax], 1.0/(1.0 + 2.0*npix/(4*np.pi)), rtol=1e-12)
    assert_allclose(result[0].amps[0, ell > band_lmax], 1.0, rtol=1e-12)


def test_joint_preconditioner_uses_trace_weight_for_spin2_channels():
    nside = 16
    npix = 12 * nside**2
    map_inv_var = np.vstack([
        np.full(npix, 1.0, dtype=np.float64),
        np.full(npix, 0.25, dtype=np.float64),
    ])
    comp_list = CompList([_DummyDiffuseComp(lmax=16, npol=2)])

    precond = JointPreconditioner(MPI.COMM_SELF, _band_map(map_inv_var, lmax=16), comp_list)
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
    band_map = _band_map(map_inv_var, lmax=lmax, fwhm_arcmin=60.0)
    comp_list = CompList([_SEDDiffuseComp(lmax=lmax, npol=1, sed=sed) for sed in (1.0, 3.0)])
    precond = getattr(preconditioners, preconditioner_name)(MPI.COMM_SELF, band_map, comp_list)
    return precond, precond.ell_block_data[ell][3], comp_list


def test_preconditioner_variants_build_the_block_their_name_claims():
    """`BeamOnly` and `MixingMatrix` are `Joint` with the other factors set to unity, and
    `DiagonalJoint` is `Joint` without the coupling between components.

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
        "DiagonalJointPreconditioner": identity + beam_squared*noise_weight*np.diag(mixing**2),
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
    band_map = _band_map(np.full((1, 12*nside**2), 2.0), lmax=band_lmax, fwhm_arcmin=60.0)
    comps = [_SEDDiffuseComp(lmax=comp_lmax, npol=1, sed=sed) for sed in (1.0, 3.0)]
    comps[0].Cl_prior_amplitude = None
    precond = JointPreconditioner(MPI.COMM_SELF, band_map, CompList(comps))

    noise_weight = 2.0 * 12*nside**2/(4*np.pi)
    beam_squared = hp.gauss_beam(np.deg2rad(60.0/60), lmax=band_lmax)[4]**2
    mixing = np.array([1.0, 3.0])
    expected = np.diag([0.0, 1.0]) + beam_squared*noise_weight*np.outer(mixing, mixing)
    assert_allclose(precond.ell_block_data[4][3], np.linalg.inv(expected), rtol=1e-10)
    assert_allclose(precond.ell_block_data[9][3], np.diag([0.0, 1.0]), atol=1e-12)


@pytest.mark.parametrize(
    "preconditioner_name",
    ["JointPreconditioner", "BeamOnlyPreconditioner", "MixingMatrixPreconditioner",
     "DiagonalJointPreconditioner"],
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
