from __future__ import annotations  # Solves NameError arising if performing early evaluation of type hints. Needed together with below if-test, since we have a cirular import.
import numpy as np
import healpy as hp
from mpi4py import MPI
import typing
from numpy.typing import NDArray
from pixell import curvedsky
from copy import deepcopy
from src.python.utils.math_operations import alm_real2complex, alm_complex2real

if typing.TYPE_CHECKING:  # Only import when performing type checking, avoiding circular import during normal runtime.
    from src.python.solvers.comp_sep_solvers import CompSepSolver


class NoPreconditioner:
    """ Preconditioner for the case where no preconditioner is used.
        Returns the input array unchanged.
    """
    def __init__(self, compsep: CompSepSolver):
        """
        Arguments:
            compsep (CompSepSolver): The CompSepSolver object from which this class is initialized.
        """
        self.compsep = compsep


    def __call__(self, a_array: NDArray):
        return deepcopy(a_array)



class BeamOnlyPreconditioner:
    """ Preconditioner for the beam-smoothing only case: A = B^TB.
        Calculates the A^-1 operator for this case, which is exact, as B is diagonal in alm space.
    """
    def __init__(self, compsep: CompSepSolver, single_fwhm_value=None):
        """
        Arguments:
            compsep (CompSepSolver): The CompSepSolver object from which this class is initialized.
            single_fwhm_value (float): If provided, use this fwhm instead of the "correct" sum of all beams.
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
    """ Preconditioner accounting only for the diagonal of the noise covariance matrix: A = Y^T N^-1 Y.
        Calculates the A^-1 operator for this case, which is only the l- m-diagonal of A.
        NB: I don't think this preconditioner is correct, I'm unable to get it to reproduce the diagonal when testing.
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
        for icomp in range(compsep.ncomp): # The different components have different lmax, so we loop over each.
            lmax = compsep.lmax_per_comp[icomp]
            temp_w_alm = hp.map2alm(w, lmax=lmax)  # Create alms at the specific lmax used by this component.
            if mycomp == icomp:
                w_alm = compsep.CompSep_comm.reduce(temp_w_alm, op=MPI.SUM, root=icomp)  # Reduce to the rank holding this component.
                w_alm /= compsep.CompSep_comm.Get_size()
            else:
                compsep.CompSep_comm.reduce(temp_w_alm, op=MPI.SUM, root=icomp)  # Reduce to the rank holding this component.

        if mycomp >= compsep.ncomp:
            return

        self.my_comp_lmax = compsep.my_comp_lmax
        my_alm_len_complex = ((self.my_comp_lmax+1)*(self.my_comp_lmax+2))//2  # Not the same as the real-valued alms.
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
        Calculates the A^-1 operator for this case, which, since it's both pixel-independent and l-m-independent,
        is only a small matrix depending on frequency and components. This small matrix can be inverted directly.
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
            curvedsky.alm2map_healpix(a_array, a_map, spin=0, nthread=self.compsep.params.nthreads_compsep)
            a_map_all = self.CompSep_subcomm.allgather(a_map)
            a_map_all = np.array(a_map_all)
            a_map_all = np.matmul(self.MT_M_inv, a_map_all)
            a_map_me = a_map_all[self.my_comp]
            curvedsky.map2alm_healpix(a_map_me, a_array, niter=3, spin=0, nthread=self.compsep.params.nthreads_compsep)
            a_array = alm_complex2real(a_array, self.compsep.my_comp_lmax)
        return a_array



class JointPreconditioner:
    """ Preconditioner taking beam, noise, and mixing matrices into account, but all only partially.
        only the component-l-m-diagonal of A is calculated, and for the noise covariance we assume
        constant rms across the map (but not across components).
        #TODO: 1. Get Wigner 3j stuff to work to get full N-diagonal. 2. Get the full mixing matrix stuff implemented.
    """
    def __init__(self, compsep: CompSepSolver):
        self.compsep = compsep
        # self.my_comp = compsep.CompSep_comm.Get_rank()
        self.is_master = compsep.CompSep_comm.Get_rank() == 0
        # self.is_holding_comp = self.my_comp < compsep.ncomp

        # lmax = compsep.my_comp_lmax
        # self.my_comp_lmax = lmax
        # my_alm_len_complex = hp.Alm.getsize(lmax)
        
        # Gather all necessary per-band information from all ranks (all ranks hold a band).
        # NB, this solution is not ideal, as all ranks now hold all rms maps, substantially increasing memory footprint.
        all_fwhm_rad = compsep.CompSep_comm.allgather(compsep.my_band_fwhm_rad)
        all_map_inv_var = compsep.CompSep_comm.allgather(compsep.map_inv_var)
        nband = len(all_fwhm_rad)

        # We can now get rid of the ranks that do not hold components.
        if not self.is_master:
            return

        self.A_diag_inv_list = []
        for icomp in range(compsep.ncomp):
            # Construct the full mixing matrix M on all ranks
            M = np.empty((nband, compsep.ncomp), dtype=np.float64)
            for jcomp in range(compsep.ncomp):
                comp = compsep.comp_list[jcomp]
                M[:, jcomp] = comp.get_sed(compsep.freqs)

            # This is our estimate of the inverse of A, which serves as a preconditioner for A.
            A_diag = np.zeros(compsep.alm_len_percomp_complex[icomp], dtype=np.complex128)
            # Loop over all frequency bands to build the diagonal term
            for iband in range(nband):
                M_fc = M[iband, icomp]

                # Calculate beam operator for this frequency band
                beam_window_squared = hp.gauss_beam(all_fwhm_rad[iband], lmax=compsep.lmax_per_comp[icomp])**2
                beam_op_complex = hp.almxfl(np.ones(compsep.alm_len_percomp_complex[icomp], dtype=np.complex128), beam_window_squared)

                mean_weights = np.mean(all_map_inv_var[iband])

                # Add the weighted contribution of this frequency band to the total
                A_diag += M_fc**2 * beam_op_complex * mean_weights

            hp.almxfl(A_diag, compsep.per_comp_P_smooth[icomp], inplace=True)
            # +1 because of the re-writing of the LHS equation with a S^{1/2} scaling.
            A_diag += 1
            # Regularize the final operator to avoid division by zero
            min_val = 1e-30
            A_diag[np.abs(A_diag) < min_val] = min_val
            self.A_diag_inv_list.append(1.0/A_diag)

    def __call__(self, a_array: list[NDArray]) -> list[NDArray]:
        if not self.is_master:
            return a_array

        # Need to parse the list to make copy, as the list.copy() is a shallow copy.
        a_array_out = [a.copy() for a in a_array]
        for icomp in range(self.compsep.ncomp):
            comp_lmax = self.compsep.lmax_per_comp[icomp]

            if self.compsep.params.CG_real_alm_mode:
                a_array_out[icomp] = alm_real2complex(a_array_out[icomp], comp_lmax)
            
            # Apply the already calculated diagonal preconditioner.
            a_array_out[icomp] *= self.A_diag_inv_list[icomp]

            if self.compsep.params.CG_real_alm_mode:
                a_array_out[icomp] = alm_complex2real(a_array_out[icomp], comp_lmax)
        
        return a_array_out