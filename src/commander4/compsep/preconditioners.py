"""Preconditioners M ~ H^-1 for the component-separation CG solve H x = b.

The choice of preconditioner is a class-name as a string in the parameter file.
`JointPreconditioner` is the "proper" one, but the file also contains four "worse" ones,
which can be used for testing.
"""
import numpy as np
import healpy as hp
from mpi4py import MPI

from commander4.data_models.detector_map import DetectorMap
from commander4.sky.comp_list import CompList
from commander4.sky.diffuse_components import DiffuseComponent


class NoPreconditioner:
    """No preconditioning: returns a copy of the input."""
    def __init__(self, solver_comm: MPI.Comm, band_map: DetectorMap, comp_list: CompList):
        pass

    def __call__(self, comp_list: CompList):
        return comp_list.copy()


class JointPreconditioner:
    """Block preconditioner for the component-separation CG system.

    The CG solves H x = b, with H = I + S^(1/2) A^T N^(-1) A S^(1/2), where A holds the mixing
    matrix, the spherical-harmonic transforms and the beam of each band. This preconditioner
    replaces each band's N^(-1) by one constant weight. H is then the same for every m at a given
    ell, and splits into one small dense block per ell, which couples the components (and, for QU,
    the two polarization rows). The inverse of that block is the preconditioner:

        H_ell ~= D_ell (kron(I_pol, D_ell^(-2)) + K_ell) D_ell,
        K_ell  = sum_b B_b(ell)^2 kron(W_b, m_b m_b^T),

    with D_ell the square roots of the components' C_l priors, B_b the beam of band b, W_b its
    polarization weight block and m_b its mixing vector. A band contributes only up to its own
    lmax. It is close to the exact inverse when the noise is uniform over the sky.

    Point sources are skipped, i.e. left unchanged. Only the master rank builds and applies the
    preconditioner; the other ranks return their input.
    """

    # The parts of the left-hand side that are kept; the subclasses below switch some off. A part
    # switched off is replaced by unity: B_b(ell) = 1, W_b = I_pol, or m_b m_b^T -> I_comp. The
    # mixing must become the identity, not all ones: ones would make every component perfectly
    # degenerate with every other, and the block singular.
    include_beam = True
    include_noise = True
    include_mixing = True
    # If False, each component keeps only its own diagonal entry of the component block, as if the
    # components were independent of each other.
    couple_components = True

    def __init__(self, solver_comm: MPI.Comm, band_map: DetectorMap, comp_list: CompList):
        self.is_master = solver_comm.Get_rank() == 0
        self.ell_block_data = []
        self.npol = 0

        # Every rank holds one band. The master gathers them all and builds the blocks.
        all_fwhm_rad = solver_comm.gather(band_map.fwhm_rad, root=0)
        all_map_inv_var = solver_comm.gather(band_map.map_inv_var, root=0)
        all_freqs = solver_comm.gather(band_map.nu, root=0)
        all_band_lmax = solver_comm.gather(band_map.lmax, root=0)
        if not self.is_master:
            return

        nband = len(all_fwhm_rad)
        diffuse_comp_indices = [
            icomp for icomp, comp in enumerate(comp_list) if isinstance(comp, DiffuseComponent)
        ]
        if not diffuse_comp_indices:
            return

        diffuse_comps = [comp_list[icomp] for icomp in diffuse_comp_indices]
        self.npol = diffuse_comps[0].npol
        diffuse_lmax = np.array([comp.lmax for comp in diffuse_comps], dtype=np.int64)
        max_lmax = int(np.max(diffuse_lmax))

        # Each band's weight block W_b. For a constant inverse variance w per pixel, Y^T N^-1 Y is
        # w*npix/(4*pi) times the identity. The two rows of a spin-2 alm are E/B-like channels, not
        # Q and U, so separate Q and U weights would mean nothing there. They get one weight, the
        # mean of the two: the trace part of the Stokes weight matrix, which does not change under
        # rotation.
        band_pol_matrices = []
        for map_inv_var in all_map_inv_var:
            if not self.include_noise:
                band_pol_matrices.append(np.eye(self.npol, dtype=np.float64))
                continue
            npix = map_inv_var.shape[-1]
            stokes_weights = np.mean(map_inv_var, axis=-1).astype(np.float64, copy=False)
            stokes_weights *= npix/(4*np.pi)
            if self.npol == 1:
                band_pol_matrices.append(np.array([[stokes_weights[0]]], dtype=np.float64))
            else:
                pol_weight = np.mean(stokes_weights)
                band_pol_matrices.append(np.eye(self.npol, dtype=np.float64) * pol_weight)

        # The SED is the same over the whole sky, so a band has one mixing coefficient per
        # component.
        mixing_matrix = np.empty((nband, len(diffuse_comps)), dtype=np.float64)
        for iband, band_freq in enumerate(all_freqs):
            for jcomp, comp in enumerate(diffuse_comps):
                mixing_matrix[iband, jcomp] = comp.get_sed(np.float64(band_freq))

        beam_windows_squared = [
            hp.gauss_beam(fwhm_rad, lmax=band_lmax)**2 if self.include_beam
            else np.ones(band_lmax + 1, dtype=np.float64)
            for fwhm_rad, band_lmax in zip(all_fwhm_rad, all_band_lmax)
        ]
        prior_inv = [
            comp.P_Cl_prior_inv.astype(np.float64, copy=False)
            for comp in diffuse_comps
        ]
        # A component without a prior has no prior term (the D_ell^(-2) below); its D_ell is 1.
        has_prior = np.array([comp.Cl_prior_amplitude is not None for comp in diffuse_comps])

        self.ell_block_data = [None] * (max_lmax + 1)
        for ell in range(max_lmax + 1):
            # The block holds only the components whose lmax reaches this ell.
            active_local = np.flatnonzero(diffuse_lmax >= ell)
            if active_local.size == 0:
                continue

            active_global = [diffuse_comp_indices[iloc] for iloc in active_local]
            # The alm indices of all m at this ell, which share the block, for `__call__`.
            alm_indices = [
                np.array([hp.Alm.getidx(comp_list[icomp].lmax, ell, m) for m in range(ell + 1)],
                         dtype=np.int64)
                for icomp in active_global
            ]

            active_prior_inv = np.array([prior_inv[iloc][ell] for iloc in active_local],
                                        dtype=np.float64)
            active_prior_inv_sqrt = np.sqrt(active_prior_inv)
            block_prior_inv_sqrt = np.tile(active_prior_inv_sqrt, self.npol)

            # The block is built as D_ell^(-1) H_ell D_ell^(-1) = D_ell^(-2) + K_ell instead of
            # H_ell = I + D_ell K_ell D_ell. It stays well-conditioned when the C_l prior (D_ell)
            # is huge.
            system_block = np.kron(np.eye(self.npol, dtype=np.float64),
                                   np.diag(active_prior_inv * has_prior[active_local]))
            for iband in range(nband):
                if ell > all_band_lmax[iband]:
                    continue
                pol_block = band_pol_matrices[iband]
                if not np.any(pol_block):
                    continue
                # The band's term is (polarization block) x (component block), times the beam at
                # this ell. The mixing outer product m_b m_b^T is what couples the components.
                if self.include_mixing:
                    scaled_mixing = mixing_matrix[iband, active_local]
                    comp_block = np.outer(scaled_mixing, scaled_mixing)
                else:
                    comp_block = np.eye(active_local.size, dtype=np.float64)
                if not self.couple_components:
                    comp_block = np.diag(np.diag(comp_block))
                system_block += beam_windows_squared[iband][ell] * np.kron(pol_block, comp_block)
            # Remove roundoff asymmetry, and floor tiny eigenvalues so that a nearly singular block
            # still has an inverse.
            system_block = 0.5 * (system_block + system_block.T)
            eigvals, eigvecs = np.linalg.eigh(system_block)
            eig_floor = np.finfo(np.float64).eps * max(1.0, eigvals[-1]) * system_block.shape[0]
            eigvals = np.clip(eigvals, eig_floor, None)
            block_inv = (eigvecs / eigvals) @ eigvecs.T
            # A component without a prior that no band sees at this ell has an all-zero row. It is
            # set to 0, not scaled by the inverse of the eigenvalue floor.
            unconstrained = np.diag(system_block) == 0
            block_inv[unconstrained, :] = 0.0
            block_inv[:, unconstrained] = 0.0

            self.ell_block_data[ell] = (
                active_global,
                alm_indices,
                block_prior_inv_sqrt,
                block_inv,
            )

    def __call__(self, comp_list: CompList) -> CompList:
        """Apply the preconditioner on the master rank; the other ranks return their input.

        For each ell, the coefficients of all components and polarizations form a matrix of shape
        ``(npol*ncomp, ell + 1)``, and one matrix product applies the block to every m.
        """
        if not self.is_master:
            return comp_list

        comp_list_out = comp_list.copy()
        for ell_data in self.ell_block_data:
            if ell_data is None:
                continue
            active_global, alm_indices, block_prior_inv_sqrt, block_inv = ell_data
            nmodes = alm_indices[0].size
            coeff_dtype = comp_list_out[active_global[0]].amps.dtype

            # Stack all active component coefficients for this ell into one dense array.
            coeffs = np.empty((self.npol * len(active_global), nmodes), dtype=coeff_dtype)
            for ipol in range(self.npol):
                offset = ipol * len(active_global)
                for iactive, icomp in enumerate(active_global):
                    coeffs[offset + iactive] = comp_list_out[icomp].amps[ipol,
                                                                         alm_indices[iactive]]

            # D_ell^(-1) block_inv D_ell^(-1), undoing the scaling the block was built with.
            coeffs = block_prior_inv_sqrt[:, None] * coeffs
            coeffs = block_inv @ coeffs
            coeffs = block_prior_inv_sqrt[:, None] * coeffs

            # Scatter the updated coefficients back to the component objects.
            for ipol in range(self.npol):
                offset = ipol * len(active_global)
                for iactive, icomp in enumerate(active_global):
                    comp_list_out[icomp].amps[ipol, alm_indices[iactive]] = coeffs[offset + iactive]

        return comp_list_out


class BeamOnlyPreconditioner(JointPreconditioner):
    """`JointPreconditioner` with the beam alone: H ~ 1 + S^(1/2) B^T B S^(1/2)."""
    include_beam = True
    include_noise = False
    include_mixing = False
    couple_components = True


class MixingMatrixPreconditioner(JointPreconditioner):
    """`JointPreconditioner` with the mixing matrix alone: H ~ 1 + S^(1/2) M^T M S^(1/2).

    It keeps the coupling between components, which is what makes the system ill-conditioned. Its
    only ell dependence is the C_l prior.
    """

    include_beam = False
    include_noise = False
    include_mixing = True
    couple_components = True


class DiagonalJointPreconditioner(JointPreconditioner):
    """`JointPreconditioner` without the coupling between components.

    Each component gets only its own term sum_b B_b(l)^2 w_b m_bc^2. The difference from
    `JointPreconditioner` shows how much of its benefit comes from the coupling.
    """
    include_beam = True
    include_noise = True
    include_mixing = True
    couple_components = False


# `NoiseOnlyPreconditioner`: the diagonal of Y^T N^-1 Y, with its ell dependence from Wigner 3j
# symbols. A starting point that does not run: it never reproduced the diagonal in testing, needs
# `py3nj` (not a dependency), and is written for the solver before `CompList` (one component per
# rank, `CompSepSolver` attributes that no longer exist, plain alm arrays). Its ell_3 = 0 term is
# the constant weight `JointPreconditioner` uses, so with constant noise a noise-only variant is
# just `JointPreconditioner` with `include_noise` alone; this version adds the ell dependence.
#
# class NoiseOnlyPreconditioner:
#     def __init__(self, solver: CompSepSolver):
#         import py3nj
#         self.solver = solver
#         mycomp = solver.solver_comm.Get_rank()
#
#         # Since the noise-map has no component-dependence (while the A-matrix does), we simply
#         # have the same weights per component, and use the average of the band-weights.
#         w = solver.map_inv_var
#         w_alm = None
#         # The different components have different lmax, so we loop over each.
#         for icomp in range(solver.ncomp):
#             lmax = solver.lmax_per_comp[icomp]
#             # Create alms at the specific lmax used by this component.
#             temp_w_alm = hp.map2alm(w, lmax=lmax)
#             if mycomp == icomp:
#                 # Reduce to the rank holding this component.
#                 w_alm = solver.solver_comm.reduce(temp_w_alm, op=MPI.SUM, root=icomp)
#                 w_alm /= solver.solver_comm.Get_size()
#             else:
#                 # Reduce to the rank holding this component.
#                 solver.solver_comm.reduce(temp_w_alm, op=MPI.SUM, root=icomp)
#
#         if mycomp >= solver.ncomp:
#             return
#
#         self.my_comp_lmax = solver.my_comp_lmax
#         # Not the same as the real-valued alms.
#         my_alm_len_complex = ((self.my_comp_lmax+1)*(self.my_comp_lmax+2))//2
#         self.YTNY = np.zeros(my_alm_len_complex, dtype=np.complex128)
#         w_alm_only_m0 = np.zeros(self.my_comp_lmax + 1, dtype=np.complex128)
#         for l in range(self.my_comp_lmax + 1):
#             idx = hp.Alm.getidx(self.my_comp_lmax, l, 0)
#             w_alm_only_m0[l] = w_alm[idx]
#
#         inv_sqrt_4pi = 1.0/np.sqrt(4*np.pi)
#         for l in range(self.my_comp_lmax + 1):
#             l3_max = min(self.my_comp_lmax, 2 * l)
#             for m in range(0, l + 1):
#                 l3_arr = np.arange(0, l3_max + 1)
#                 l_arr = np.full_like(l3_arr, l)
#                 m_arr = np.full_like(l3_arr, m)
#
#                 value = (-1)**m*py3nj.wigner3j(2*l_arr, 2*l_arr, 2*l3_arr, 2*m_arr, -2*m_arr, 0)*\
#                     py3nj.wigner3j(2*l_arr, 2*l_arr, 2*l3_arr, 0, 0, 0) * w_alm_only_m0[l3_arr]*\
#                     np.sqrt((2*l_arr + 1)**2*(2*l3_arr + 1))*inv_sqrt_4pi
#                 idx = hp.Alm.getidx(self.my_comp_lmax, l, m)
#                 self.YTNY[idx] += np.sum(value)
#
#     def __call__(self, a_array: NDArray):
#         solver = self.solver
#         mycomp = solver.solver_comm.Get_rank()
#
#         if mycomp >= solver.ncomp:  # nothing to do
#             return a_array
#         # Convert from real to complex alms, apply the Y^T N^-1 Y matrix, and then convert back.
#         a_array_out = alm_real2complex(a_array, self.my_comp_lmax)
#         a_array_out /= self.YTNY
#         a_array_out = alm_complex2real(a_array_out, self.my_comp_lmax)
#         return a_array_out
