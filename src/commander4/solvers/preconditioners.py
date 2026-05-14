# Solves NameError arising if performing early evaluation of type hints.
# Needed together with below if-test, since we have a cirular import.
from __future__ import annotations

import numpy as np
import ctypes as ct
import healpy as hp
from mpi4py import MPI
from numpy.typing import NDArray
from pixell import curvedsky
from copy import deepcopy
import logging
from commander4.utils.math_operations import alm_real2complex, alm_complex2real, inplace_arr_prod
from commander4.utils.ctypes_lib import load_cmdr4_ctypes_lib

import typing
# Only import when performing type checking, avoiding circular import during normal runtime.
if typing.TYPE_CHECKING:
    from commander4.solvers.CG_compsep_solver import CompSepSolver
    from commander4.sky_models.component import Component, CompList
    from commander4.utils.mapmaker import WeightsMapmakerIQU

logger = logging.getLogger(__name__)


class NoPreconditioner:
    """ Preconditioner for the case where no preconditioner is used.
        Returns the input array unchanged.
    """
    def __init__(self, compsep: CompSepSolver, complist: CompList):
        pass

    def __call__(self, complist: CompList):
        return deepcopy(complist)



class BeamOnlyPreconditioner:
    """ Preconditioner for the beam-smoothing only case: A = B^TB.
        Calculates the A^-1 operator for this case, which is exact, as B is diagonal in alm space.
    """
    def __init__(self, compsep: CompSepSolver, single_fwhm_value=None):
        """
        Args:
            compsep (CompSepSolver): The CompSepSolver object from which this class is initialized.
            single_fwhm_value (float): If provided, use this fwhm instead of the "correct"
            sum of all beams.
        """
        self.compsep = compsep
        compsep = self.compsep
        mycomp = compsep.CompSep_comm.Get_rank()
        all_fwhm = np.array(compsep.CompSep_comm.allgather(self.compsep.my_band_fwhm_rad))

        if mycomp >= compsep.ncomp:  # nothing to do
            return
        
        lmax = compsep.lmax_per_comp[mycomp]
        self.beam_window_squared_sum = np.zeros(lmax + 1)

        for fwhm in all_fwhm:
            # Create beam window function. Square the beam window since it appears twice in the system matrix
            beam_window_squared = hp.gauss_beam(fwhm, lmax=lmax)**2
                
            # Add regularization to avoid division by very small values
            min_beam = 1e-10
            beam_window_squared = np.maximum(beam_window_squared, min_beam)

            # Add up the individual contributions to the beam from each frequency.                
            self.beam_window_squared_sum += beam_window_squared


    def __call__(self, a_array: NDArray):
        # Apply inverse squared beam (divide by beam window squared)
        a_array_out = alm_real2complex(a_array, self.compsep.my_comp_lmax)
        a_array_out = hp.almxfl(a_array_out, 1.0/self.beam_window_squared_sum, inplace=True)
        a_array_out = alm_complex2real(a_array_out, self.compsep.my_comp_lmax)
        return a_array_out



class NoiseOnlyPreconditioner:
    """ Preconditioner accounting only for the diagonal of the noise covariance
        matrix: A = Y^T N^-1 Y. Calculates the A^-1 operator for this case, which is only the
        l- m-diagonal of A. NB: I don't think this preconditioner is correct, I'm unable to get it
        to reproduce the diagonal when testing.
    """
    def __init__(self, compsep: CompSepSolver):
        """
        Arguments:
            compsep (CompSepSolver): The CompSepSolver object from which this class is initialized.
        """
        import py3nj
        self.compsep = compsep
        mycomp = compsep.CompSep_comm.Get_rank()

        # Since the noise-map has no component-dependence (while the A-matrix does), we simply
        # have the same weights per component, and use the average of the band-weights.
        w = compsep.map_inv_var
        w_alm = None
        # The different components have different lmax, so we loop over each.
        for icomp in range(compsep.ncomp):
            lmax = compsep.lmax_per_comp[icomp]
            # Create alms at the specific lmax used by this component.
            temp_w_alm = hp.map2alm(w, lmax=lmax)
            if mycomp == icomp:
                # Reduce to the rank holding this component.
                w_alm = compsep.CompSep_comm.reduce(temp_w_alm, op=MPI.SUM, root=icomp)
                w_alm /= compsep.CompSep_comm.Get_size()
            else:
                # Reduce to the rank holding this component.
                compsep.CompSep_comm.reduce(temp_w_alm, op=MPI.SUM, root=icomp)

        if mycomp >= compsep.ncomp:
            return

        self.my_comp_lmax = compsep.my_comp_lmax
        # Not the same as the real-valued alms.
        my_alm_len_complex = ((self.my_comp_lmax+1)*(self.my_comp_lmax+2))//2
        self.YTNY = np.zeros(my_alm_len_complex, dtype=np.complex128)
        w_alm_only_m0 = np.zeros(self.my_comp_lmax + 1, dtype=np.complex128)
        for l in range(self.my_comp_lmax + 1):
            idx = hp.Alm.getidx(self.my_comp_lmax, l, 0)
            w_alm_only_m0[l] = w_alm[idx]

        inv_sqrt_4pi = 1.0/np.sqrt(4*np.pi)
        for l in range(self.my_comp_lmax + 1):
            l3_max = min(self.my_comp_lmax, 2 * l)
            for m in range(0, l + 1):
                l3_arr = np.arange(0, l3_max + 1)
                l_arr = np.full_like(l3_arr, l)
                m_arr = np.full_like(l3_arr, m)

                value = (-1)**m*py3nj.wigner3j(2*l_arr, 2*l_arr, 2*l3_arr, 2*m_arr, -2*m_arr, 0) * \
                    py3nj.wigner3j(2*l_arr, 2*l_arr, 2*l3_arr, 0, 0, 0) * w_alm_only_m0[l3_arr] * \
                    np.sqrt((2*l_arr + 1)**2*(2*l3_arr + 1))*inv_sqrt_4pi
                idx = hp.Alm.getidx(self.my_comp_lmax, l, m)
                self.YTNY[idx] += np.sum(value)
        # alm_plotter(self.YTNY[icomp], lmax, filename=f"YTNY_{icomp}.png")


    def __call__(self, a_array: NDArray):
        compsep = self.compsep
        mycomp = compsep.CompSep_comm.Get_rank()

        if mycomp >= compsep.ncomp:  # nothing to do
            return a_array
        # Convert from real to complex alms, apply the Y^T N^-1 Y matrix, and then convert back.
        a_array_out = alm_real2complex(a_array, self.my_comp_lmax)
        a_array_out /= self.YTNY
        a_array_out = alm_complex2real(a_array_out, self.my_comp_lmax)
        return a_array_out


class MixingMatrixPreconditioner:
    """ Preconditioner accounting only for the mixing matrix.
        Calculates the A^-1 operator for this case, which, since it's both pixel-independent and
        l-m-independent, is only a small matrix depending on frequency and components.
        This small matrix can be inverted directly.
    """
    def __init__(self, compsep: CompSepSolver):
        self.compsep = compsep
        M = np.empty((compsep.nband, compsep.ncomp), dtype=np.float64)
        for icomp in range(compsep.ncomp):
            comp = compsep.comp_list[icomp]
            M[:,icomp] = comp.get_sed(compsep.freqs)
        MT_M = np.matmul(M.T, M)
        self.MT_M_inv = np.linalg.inv(MT_M)
        self.my_comp = compsep.CompSep_comm.Get_rank()
        self.is_holding_comp = self.my_comp < compsep.ncomp
        self.full_size = np.sum(compsep.alm_len_percomp)
        if self.is_holding_comp:
            self.my_size = compsep.alm_len_percomp[self.my_comp]
            color = 0
        else:
            self.my_size = 0
            color = MPI.UNDEFINED
        self.CompSep_subcomm = self.compsep.CompSep_comm.Split(color, key=self.my_comp)
        

    def __call__(self, a_array: NDArray):
        if self.is_holding_comp:
            a_array = alm_real2complex(a_array, self.compsep.my_comp_lmax)
            a_map = np.empty((self.compsep.npix,), dtype=np.float64)
            curvedsky.alm2map_healpix(a_array, a_map, spin=0,
                                      nthread=self.compsep.params.nthreads_compsep)
            a_map_all = self.CompSep_subcomm.allgather(a_map)
            a_map_all = np.array(a_map_all)
            a_map_all = np.matmul(self.MT_M_inv, a_map_all)
            a_map_me = a_map_all[self.my_comp]
            curvedsky.map2alm_healpix(a_map_me, a_array, niter=3, spin=0,
                                      nthread=self.compsep.params.nthreads_compsep)
            a_array = alm_complex2real(a_array, self.compsep.my_comp_lmax)
        return a_array



class DiagonalJointPreconditioner:
    """ Preconditioner taking beam, noise, and mixing matrices into account, but all only partially.
        only the component-l-m-diagonal of A is calculated, and for the noise covariance we assume
        constant rms across the map (but not across components).
        #TODO: 1. Get Wigner 3j stuff to work to get full N-diagonal. 2. Get the full mixing matrix
        # stuff implemented.
    """
    def __init__(self, compsep: CompSepSolver, comp_list:CompList):
        self.compsep = compsep
        self.is_master = compsep.CompSep_comm.Get_rank() == 0

        # Gather all necessary per-band information from all ranks (all ranks hold a band).
        # TODO: Currently the master rank temporarily has to hold ALL inverse-variance bands.
        # It would be easy to re-write this so they send them one-and-one as we make the diagonal.
        all_fwhm_rad = compsep.CompSep_comm.gather(np.deg2rad(compsep.my_band.fwhm/60), root=0)
        all_map_inv_var = compsep.CompSep_comm.gather(compsep.det_map.inv_n_map, root=0)
        all_freqs = compsep.CompSep_comm.gather(compsep.my_band.nu, root=0)

        # We can now get rid of the ranks that do not hold components.
        if not self.is_master:
            return

        nband = len(all_fwhm_rad)
        ncomp = len(comp_list)
        self.A_diag_inv_list = []
        for icomp in range(ncomp):
            if not hasattr(comp_list[icomp], "alms"): #FIXME: for now workaround to exclude point sources
                continue
            # Construct the full mixing matrix M on all ranks
            M = np.empty((nband, ncomp), dtype=np.float64)
            for jcomp in range(ncomp):
                comp = comp_list[jcomp]
                if not hasattr(comp, "alms"): #FIXME: for now workaround to exclude point sources
                    continue
                M[:, jcomp] = comp.get_sed(np.array(all_freqs, dtype=np.float64))

            # This is our estimate of the inverse of A, which serves as a preconditioner for A.
            A_diag = np.zeros(comp_list[icomp].alm_len_complex, dtype=np.complex128)
            # Loop over all frequency bands to build the diagonal term
            for iband in range(nband):
                M_fc = M[iband, icomp]

                # Calculate beam operator for this frequency band
                beam_window_squared = hp.gauss_beam(all_fwhm_rad[iband],
                                                    lmax=comp_list[icomp].lmax)**2
                beam_op_complex = hp.almxfl(np.ones(comp_list[icomp].alm_len_complex,
                                                    dtype=np.complex128), beam_window_squared)

                mean_weights = np.mean(all_map_inv_var[iband])
                npix = all_map_inv_var[iband].shape[-1]
                mean_weights *= npix/(4*np.pi)

                # Add the weighted contribution of this frequency band to the total
                A_diag += M_fc**2 * beam_op_complex * mean_weights

            hp.almxfl(A_diag, comp_list[icomp].P_smoothing_prior, inplace=True)
            # +1 because of the re-writing of the LHS equation with a S^{1/2} scaling.
            A_diag += 1
            # Regularize the final operator to avoid division by zero
            self.A_diag_inv_list.append(1.0/A_diag)

    def __call__(self, a_complist: CompList) -> CompList:
        if not self.is_master:
            return a_complist

        # Need to parse the list to make copy, as the list.copy() is a shallow copy.
        a_complist_out = deepcopy(a_complist)
        for icomp in range(len(a_complist)):
            #comp_lmax = self.compsep.lmax_per_comp[icomp]
            if hasattr(a_complist[icomp], "alms"):
                # Apply the already calculated diagonal preconditioner.
                a_complist_out[icomp].alms *= self.A_diag_inv_list[icomp]
            else:
                pass
 
        return a_complist_out


class JointPreconditioner:
    """Block preconditioner for the component-separation CG system.

    The exact solver applies the left-hand side

        I + S^(1/2) A^T N^(-1) A S^(1/2),

    where ``A`` collects the mixing matrix, spherical-harmonic synthesis/analysis, and beam
    convolution for all bands. This operator is expensive because it is applied through map-space
    transforms and MPI communication, and because in the discrete polarized case it is not exactly
    diagonal in harmonic space.

    This preconditioner keeps the parts of the operator that are cheap and dominant when the test
    problem is close to isotropic:

    - the full component-component coupling for each multipole ``ell``
    - the beam suppression per band and per ``ell``
    - the smoothing-prior scaling that comes from the ``S^(1/2)`` reparameterization
    - a small polarization-space block for each ``ell``

    The approximation drops the remaining alm off-diagonal structure coming from the discrete
    spin-weighted transforms. In particular, it assumes that each ``ell`` can be treated
    independently, and that all ``m`` values at fixed ``ell`` share the same small dense block.

    For intensity this reduces to a per-``ell`` component block. For polarization we work in the
    two spin-2 harmonic channels used by ``ducc0.sht``. These are grad/curl-like channels, not
    literal Q/U harmonic coefficients, so it is not sensible to feed distinct Q and U weights
    directly onto the two rows of the harmonic vector. Instead, the isotropic polarized
    approximation retains only the trace part of the Stokes-space weight matrix,

        W_pol -> 0.5 * trace(W_pol) * I,

    which is the rotationally invariant part seen by the spin-2 harmonic channels.

    The resulting preconditioner block for a fixed ``ell`` is built as

        M_ell^(-1) ~= kron(I_pol, D_ell^(-2))
                     + sum_b B_b(ell)^2 * kron(W_b, m_b m_b^T),

    where ``D_ell`` is the diagonal matrix of prior square-roots for the active components,
    ``W_b`` is the small polarization-space weight block for band ``b``, and ``m_b`` is the band
    mixing vector evaluated at that band's frequency. The final inverse is applied in the
    similarity-transformed basis ``D_ell^(-1) A_ell D_ell^(-1)`` to avoid numerical problems when
    the smoothing prior amplitude is very large.

    Notes
    -----
    - Only the master rank constructs and applies this preconditioner. Worker ranks return their
      input unchanged.
    - Components without harmonic alms, such as current point-source representations, are skipped.
    - Bands only contribute up to their own ``lmax``.
    - This is a preconditioner for the isotropic part of the operator, not an exact inverse of the
      full discrete polarized system.
    """
    def __init__(self, compsep: CompSepSolver, comp_list:CompList):
        self.compsep = compsep
        self.is_master = compsep.CompSep_comm.Get_rank() == 0
        self.ell_block_data = []
        self.npol = 0

        # Gather the band-local ingredients needed to build the isotropic approximation. Every rank
        # holds one band, but only the master rank actually assembles the small dense blocks.
        all_fwhm_rad = compsep.CompSep_comm.gather(np.deg2rad(compsep.my_band.fwhm/60), root=0)
        all_map_inv_var = compsep.CompSep_comm.gather(compsep.det_map.inv_n_map, root=0)
        all_freqs = compsep.CompSep_comm.gather(compsep.my_band.nu, root=0)
        all_band_lmax = compsep.CompSep_comm.gather(compsep.det_map.lmax, root=0)

        # Worker ranks participate in the gather above, but the actual block construction is done
        # only on the master rank.
        if not self.is_master:
            return

        nband = len(all_fwhm_rad)
        # Restrict the preconditioner to components represented by diffuse harmonic alms.
        diffuse_comp_indices = [
            icomp for icomp, comp in enumerate(comp_list) if hasattr(comp, "alms")
        ]
        if not diffuse_comp_indices:
            return

        diffuse_comps = [comp_list[icomp] for icomp in diffuse_comp_indices]
        self.npol = diffuse_comps[0].npol
        diffuse_lmax = np.array([comp.lmax for comp in diffuse_comps], dtype=np.int64)
        max_lmax = int(np.max(diffuse_lmax))

        # Convert each band's inverse-noise map to the corresponding isotropic weight block. The
        # factor npix/(4*pi) is the continuum normalization of a constant inverse-noise map in the
        # harmonic inner product. For polarization we keep only the trace part of the Stokes-space
        # weight matrix so that the approximation acts sensibly on the two spin-2 harmonic channels.
        band_pol_matrices = []
        for inv_n_map in all_map_inv_var:
            npix = inv_n_map.shape[-1]
            stokes_weights = np.mean(inv_n_map, axis=-1).astype(np.float64, copy=False)
            stokes_weights *= npix/(4*np.pi)
            if self.npol == 1:
                band_pol_matrices.append(np.array([[stokes_weights[0]]], dtype=np.float64))
            else:
                # The two rows of a spin-2 alm object are grad/curl-like harmonic channels, not Q/U.
                # Using separate Q and U weights here would therefore impose an unphysical harmonic
                # weighting. The trace part is the rotationally invariant piece of the constant
                # polarized weight matrix.
                pol_weight = np.mean(stokes_weights)
                band_pol_matrices.append(np.eye(self.npol, dtype=np.float64) * pol_weight)

        # Mixing is assumed to be spatially constant, so each band contributes only a single
        # frequency-dependent mixing vector.
        mixing_matrix = np.empty((nband, len(diffuse_comps)), dtype=np.float64)
        for iband, band_freq in enumerate(all_freqs):
            for jcomp, comp in enumerate(diffuse_comps):
                mixing_matrix[iband, jcomp] = comp.get_sed(np.float64(band_freq))

        # Precompute the per-band beam transfer functions and the diagonal prior factors.
        beam_windows_squared = [
            hp.gauss_beam(fwhm_rad, lmax=band_lmax)**2
            for fwhm_rad, band_lmax in zip(all_fwhm_rad, all_band_lmax)
        ]
        prior_inv = [
            comp.P_smoothing_prior_inv.astype(np.float64, copy=False)
            for comp in diffuse_comps
        ]

        self.ell_block_data = [None] * (max_lmax + 1)
        for ell in range(max_lmax + 1):
            # Only components defined up to the current ell participate in this block.
            active_local = np.flatnonzero(diffuse_lmax >= ell)
            if active_local.size == 0:
                continue

            active_global = [diffuse_comp_indices[iloc] for iloc in active_local]
            # All m-modes at fixed ell share the same dense block. We store the alm indices once so
            # that application only becomes gather -> dense matmul -> scatter.
            alm_indices = [
                np.array([hp.Alm.getidx(comp_list[icomp].lmax, ell, m) for m in range(ell + 1)],
                         dtype=np.int64)
                for icomp in active_global
            ]

            active_prior_inv = np.array([prior_inv[iloc][ell] for iloc in active_local],
                                        dtype=np.float64)
            active_prior_inv_sqrt = np.sqrt(active_prior_inv)
            block_prior_inv_sqrt = np.tile(active_prior_inv_sqrt, self.npol)

            # Build the dense block in the similarity-transformed basis
            #
            #     D_ell^(-1) A_ell D_ell^(-1) = D_ell^(-2) + K_ell,
            #
            # rather than inverting I + D_ell K_ell D_ell directly. This keeps the block numerically
            # stable when the smoothing prior amplitude is large and D_ell contains huge values.
            system_block = np.kron(np.eye(self.npol, dtype=np.float64), np.diag(active_prior_inv))
            for iband in range(nband):
                # A band contributes only while it has support at this ell.
                if ell > all_band_lmax[iband]:
                    continue
                pol_block = band_pol_matrices[iband]
                if not np.any(pol_block):
                    continue
                # The isotropic band contribution factorizes into a polarization block and a
                # component-mixing outer product, scaled by the beam window at this ell.
                scaled_mixing = mixing_matrix[iband, active_local]
                system_block += beam_windows_squared[iband][ell] * np.kron(
                    pol_block,
                    np.outer(scaled_mixing, scaled_mixing),
                )
            # Enforce exact symmetry before diagonalization to suppress tiny roundoff asymmetries.
            system_block = 0.5 * (system_block + system_block.T)
            eigvals, eigvecs = np.linalg.eigh(system_block)
            # Floor tiny eigenvalues so that the preconditioner remains well-defined even when the
            # block is numerically close to singular.
            eig_floor = np.finfo(np.float64).eps * max(1.0, eigvals[-1]) * system_block.shape[0]
            eigvals = np.clip(eigvals, eig_floor, None)
            block_inv = (eigvecs / eigvals) @ eigvecs.T

            self.ell_block_data[ell] = (
                active_global,
                alm_indices,
                block_prior_inv_sqrt,
                block_inv,
            )

    def __call__(self, a_complist: CompList) -> CompList:
        """Apply the preconditioner to a component list on the master rank.

        For each ell we gather all active component coefficients for all polarization channels into
        one dense matrix of shape ``(npol * nactive_comp, ell + 1)``. The same precomputed inverse
        block is then applied to every m at this ell in one batched matrix multiplication, and the
        result is scattered back into the component alms.
        """
        if not self.is_master:
            return a_complist

        # Need to parse the list to make copy, as the list.copy() is a shallow copy.
        a_complist_out = deepcopy(a_complist)
        for ell_data in self.ell_block_data:
            if ell_data is None:
                continue
            active_global, alm_indices, block_prior_inv_sqrt, block_inv = ell_data
            nmodes = alm_indices[0].size
            coeff_dtype = a_complist_out[active_global[0]].alms.dtype

            # Stack all active component coefficients for this ell into one dense array.
            coeffs = np.empty((self.npol * len(active_global), nmodes), dtype=coeff_dtype)
            for ipol in range(self.npol):
                offset = ipol * len(active_global)
                for iactive, icomp in enumerate(active_global):
                    coeffs[offset + iactive] = a_complist_out[icomp].alms[
                        ipol, alm_indices[iactive]
                    ]

            # Apply D_ell^(-1) A_ell^(-1) D_ell^(-1) in the same similarity-transformed basis used
            # during construction.
            coeffs = block_prior_inv_sqrt[:, None] * coeffs
            coeffs = block_inv @ coeffs
            coeffs = block_prior_inv_sqrt[:, None] * coeffs

            # Scatter the updated coefficients back to the component objects.
            for ipol in range(self.npol):
                offset = ipol * len(active_global)
                for iactive, icomp in enumerate(active_global):
                    a_complist_out[icomp].alms[ipol, alm_indices[iactive]] = coeffs[
                        offset + iactive
                    ]
 
        return a_complist_out


class InvNPreconditionerIQU:
    """ Standard diagonal preconditioner for CG mapmaker. It builds an estimate of the diagonal of the A matrix
        by estimating the RMS of the i-th pixel as sigma_0/n_hit_i.
    """

    def __init__(self, inv_N_IQU:NDArray, double_prec=True):
        """
        Initialize preconditioner starting from the RMS.
        """
        assert inv_N_IQU.ndim == 2, "InvN_IQU must be 2-dimensional array."
        assert inv_N_IQU.shape[0] == 3, "InvN_IQU must be must have shape (3,npix)."
        self.npix = inv_N_IQU.shape[1]
        self.inv_N_IQU = inv_N_IQU

    def __call__(self, map: NDArray) -> NDArray:
        assert map.shape[1] == self.npix, "map should have same npix as the preconditioner"
        assert map.shape[0] == 3, "map should have 3 polarization components: I, Q and U."
        map_out = np.copy(map)
        inplace_arr_prod(map_out, self.inv_N_IQU)
        return map_out


class InvNPreconditionerI:
    """ Standard diagonal preconditioner for temperature-only CG mapmaker. It builds an estimate of 
    the diagonal of the A matrix by estimating the RMS of the i-th pixel as sigma_0/n_hit_i.
    """

    def __init__(self, inv_N_map:NDArray):
        """
        Initialize preconditioner starting from the mapmaker object it will be used in.
        It precomputes the hitmap
        """
        self.inv_N_map = inv_N_map.reshape((1,-1)) if inv_N_map.ndim == 1 else inv_N_map
        self.npix = self.inv_N_map.shape[1]

    def __call__(self, map: NDArray) -> NDArray:
        assert map.shape[1] == self.npix
        map_out = np.copy(map)
        logger.debug(f"## Preconditioner called. map shape: {map.shape}, inv N shape: {self.inv_N_map.shape}")
        # this allows it to be applied to IQU maps as well
        map_out = map_out.reshape((1,-1)) if map_out.ndim == 1 else map_out
        if map_out.shape[0] == 1:
            inplace_arr_prod(map_out, self.inv_N_map)
        else:
            for i in range(map_out.shape[0]):
                inplace_arr_prod(map_out[i,:], self.inv_N_map)
        return map_out







