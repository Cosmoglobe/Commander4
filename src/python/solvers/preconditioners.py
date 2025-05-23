from __future__ import annotations  # Solves NameError arising if performing early evaluation of type hints. Needed together with below if-test, since we have a cirular import.
import numpy as np
import healpy as hp
from mpi4py import MPI
import typing
from numpy.typing import NDArray

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
        return a_array



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


    def __call__(self, a_array: NDArray):
        compsep = self.compsep
        mycomp = compsep.CompSep_comm.Get_rank()

        if mycomp >= compsep.ncomp:  # nothing to do
            return a_array
        a = a_array.copy()  # I'm not sure if the copy is needed
        
        lmax = compsep.lmax_per_comp[mycomp]
        beam_window_squared_sum = np.zeros(lmax + 1)

        for fwhm in compsep.fwhm_rad_allbands:
            # Create beam window function. Square the beam window since it appears twice in the system matrix
            beam_window_squared = hp.gauss_beam(fwhm, lmax=lmax)**2
                
            # Add regularization to avoid division by very small values
            min_beam = 1e-10
            beam_window_squared = np.maximum(beam_window_squared, min_beam)

            # Add up the individual contributions to the beam from each frequency.                
            beam_window_squared_sum += beam_window_squared
        # Apply inverse squared beam (divide by beam window squared)
        a = hp.almxfl(a, 1.0/beam_window_squared_sum)

        return a



class NoiseOnlyPreconditioner:
    """ Preconditioner accounting only for the diagonal of the noise covariance matrix: A = Y^T N^-1 Y.
        Calculates the A^-1 operator for this case, which is only the l- m-diagonal of A.
    """
    def __init__(self, compsep: CompSepSolver):
        """
        Arguments:
            compsep (CompSepSolver): The CompSepSolver object from which this class is initialized.
        """
        import py3nj
        self.compsep = compsep
        mycomp = compsep.CompSep_comm.Get_rank()

        w = 1.0/compsep.map_rms**2
        w_alm = hp.map2alm(w, lmax=compsep.lmax)
        # MR FIXME: the next two lines look like an expensive no-op to me ...
        # or is compsep.map_rms different on different tasks?
        w_alm = compsep.CompSep_comm.allreduce(w_alm, op=MPI.SUM)
        w_alm /= compsep.CompSep_comm.Get_size()

        if mycomp >= compsep.ncomp:
            return

        self.YTNY = np.zeros(compsep.alm_len_percomp[mycomp], dtype=np.complex128)
        lmax = compsep.lmax_per_comp[mycomp]
        w_alm_only_m0 = np.zeros(lmax + 1, dtype=np.complex128)
        for l in range(lmax + 1):
            idx = hp.Alm.getidx(lmax, l, 0)
            w_alm_only_m0[l] = w_alm[idx]

        inv_sqrt_4pi = 1.0/np.sqrt(4*np.pi)
        for l in range(lmax + 1):
            l3_max = min(lmax, 2 * l)
            for m in range(0, l + 1):
                l3_arr = np.arange(0, l3_max + 1)
                l_arr = np.full_like(l3_arr, l)
                m_arr = np.full_like(l3_arr, m)

                value = (-1)**m*py3nj.wigner3j(2*l_arr, 2*l_arr, 2*l3_arr, 2*m_arr, -2*m_arr, 0) * \
                    py3nj.wigner3j(2*l_arr, 2*l_arr, 2*l3_arr, 0, 0, 0) * w_alm_only_m0[l3_arr] * \
                    np.sqrt((2*l_arr + 1)**2*(2*l3_arr + 1))*inv_sqrt_4pi
                idx = hp.Alm.getidx(lmax, l, m)
                self.YTNY[idx] += np.sum(value)
        # alm_plotter(self.YTNY[icomp], lmax, filename=f"YTNY_{icomp}.png")


    def __call__(self, a_array: NDArray):
        compsep = self.compsep
        mycomp = compsep.CompSep_comm.Get_rank()

        if mycomp >= compsep.ncomp:  # nothing to do
            return a_array

        return a_array / self.YTNY
