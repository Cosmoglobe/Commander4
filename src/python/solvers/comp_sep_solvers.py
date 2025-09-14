import numpy as np
import healpy as hp
import time
from pixell import utils, curvedsky
from pixell.bunch import Bunch
import logging
from mpi4py import MPI
from numpy.typing import NDArray
from typing import Callable

from src.python.output.log import logassert
from src.python.data_models.detector_map import DetectorMap
from src.python.sky_models.component import DiffuseComponent
from src.python.utils.math_operations import alm_to_map, alm_to_map_adjoint, alm_real2complex,\
    alm_complex2real, gaussian_random_alm
from src.python.solvers.dense_matrix_math import DenseMatrix
import src.python.solvers.preconditioners as preconditioners


def amplitude_sampling_per_pix(proc_comm: MPI.Comm, detector_data: DetectorMap,
                               comp_list: list[DiffuseComponent], params: Bunch
                               )-> list[DiffuseComponent]:
    """A (quite inefficient) pixel-by-pixel solver for the component separation problem. This only
       works if assuming there is no beam, or all maps are smoothed to the same resolution.
    """
    logger = logging.getLogger(__name__)
    if proc_comm.Get_rank() == 0:
        logger.info("Starting pixel-by-pixel component separation.")
    map_sky = detector_data.map_sky.copy()
    band_freq = detector_data.nu
    map_rms = detector_data.map_rms
    if params.smooth_to_common_res:
        fwhm = detector_data.fwhm
        all_fwhm = proc_comm.allgather(fwhm)
        max_fwhm = np.max(all_fwhm)
        my_smoothing_fwhm = np.sqrt(max_fwhm**2 - fwhm**2)
        logger.info(f"{detector_data.nu} GHz map with FWHM = {fwhm:.1f} arcmin will be smoothed by "
                    f"{my_smoothing_fwhm:.1f} arcmin to reach {max_fwhm:.1f} arcmin.")
        if params.smooth_to_common_res:
            if map_sky[1] is None:  # No polarization
                map_sky[0] = hp.smoothing(map_sky[0], np.deg2rad(my_smoothing_fwhm/60.0))
            elif map_sky[0] is None:
                map_sky[1] = hp.smoothing(map_sky[1], np.deg2rad(my_smoothing_fwhm/60.0))
                map_sky[2] = hp.smoothing(map_sky[2], np.deg2rad(my_smoothing_fwhm/60.0))
            else:
                map_sky = hp.smoothing(map_sky, np.deg2rad(my_smoothing_fwhm/60.0), pol=True)

    ncomp_full = len(comp_list)
    all_freq = proc_comm.gather(band_freq, root=0)
    all_map_sky = proc_comm.gather(map_sky, root=0)
    all_map_rms = proc_comm.gather(map_rms, root=0)
    nside = detector_data.nside
    npix = 12*nside**2
    comp_maps = [None, None, None]
    if proc_comm.Get_rank() == 0:
        for ipol in range(3):
            t0 = time.time()
            ncomp = len(comp_list)
            if ipol > 0:
                ncomp = len([comp for comp in comp_list if comp.polarized])
            freqs = []
            maps_sky = []
            maps_rms = []
            for iband in range(len(all_freq)):
                if all_map_sky[iband][ipol] is not None:
                    freqs.append(all_freq[iband])
                    maps_sky.append(all_map_sky[iband][ipol])
                    maps_rms.append(all_map_rms[iband][ipol])
            freqs = np.array(freqs)
            maps_sky = np.array(maps_sky)
            maps_rms = np.array(maps_rms)
            nband = len(freqs)
            comp_maps[ipol] = np.zeros((ncomp, npix))
            M = np.empty((nband, ncomp))
            idx = 0
            for i in range(ncomp):
                if ipol == 0 or comp_list[i].polarized:
                    M[:,idx] = comp_list[i].get_sed(freqs)
                    idx += 1

            rand = np.random.randn(npix,nband)
            for i in range(npix):
                xmap = 1/maps_rms[:,i]
                x = M.T.dot((xmap**2*maps_sky[:,i]))
                x += M.T.dot(rand[i]*xmap)
                A = (M.T.dot(np.diag(xmap**2)).dot(M))
                n_failures = 0
                try:
                    comp_maps[ipol][:,i] = np.linalg.solve(A, x)
                except np.linalg.LinAlgError:
                    comp_maps[ipol][:,i] = 0
                    n_failures += 1
            if n_failures > 0:
                logger.warning(f"Pixel-by-pixel component separation failed for {n_failures}"
                               f"out of {npix} pixels for polarization {ipol+1}/3.")
            logger.info(f"Finished pixel-by-pixel component separation in {time.time()-t0:.2f}s "
                        f"for polarization {ipol+1} of 3.")

            # import cmdr4_support
            # t0 = time()
            # comp_maps2 = cmdr4_support.utils.amplitude_sampling_per_pix_helper(map_sky, map_rms, M, rand, nnthreads=1)
            # logger.info(f"Time for native solution: {time()-t0}s.")
            # import ducc0
            # logger.info(f"L2 error between solutions: {ducc0.misc.l2error(comp_maps, comp_maps2)}.")

    comp_maps = proc_comm.bcast(comp_maps, root=0)
    for icomp in range(ncomp_full):
        alm_len = ((comp_list[icomp].lmax+1)*(comp_list[icomp].lmax+2))//2
        comp_alms = np.zeros((1,alm_len), dtype=np.complex128)
        comp_list[icomp].component_alms_intensity = curvedsky.map2alm_healpix(comp_maps[0][icomp], comp_alms, niter=1, spin=0)
        if comp_list[icomp].polarized:
            alm_len = ((comp_list[icomp].lmax+1)*(comp_list[icomp].lmax+2))//2
            comp_alms = np.zeros((2,alm_len), dtype=np.complex128)
            pol_alms = curvedsky.map2alm_healpix(np.array([comp_maps[1][icomp], comp_maps[2][icomp]]), comp_alms, niter=1, spin=2)
            comp_list[icomp].component_alms_polarization = pol_alms
    return comp_list



def _project_alms(alms_in, lmax_in, lmax_out):
    """
    Projects alms from one lmax resolution to another, handling truncation or zero-padding.
    Importantly, this function is the adjoint of itself.
    """
    if lmax_in == lmax_out:
        return alms_in.copy()

    alms_out = np.zeros_like(alms_in, shape=(alms_in.shape[0], hp.Alm.getsize(lmax_out)))
    
    # Determine the number of modes to copy
    l_copy = min(lmax_in, lmax_out)
    m_copy = min(lmax_in, lmax_out)
    
    # Copy alm data up to the minimum lmax
    for m in range(m_copy + 1):
        idx_in = hp.Alm.getidx(lmax_in, np.arange(m, l_copy + 1), m)
        idx_out = hp.Alm.getidx(lmax_out, np.arange(m, l_copy + 1), m)
        alms_out[:, idx_out] = alms_in[:, idx_in]
        
    return alms_out


class CompSepSolver:
    """ Class for performing global component separation using the preconditioned conjugate gradient
        method. After initializing the class, the solve() method should be called to perform the
        component separation. Note that the solve() method will in-place update (as well as return)
        the 'comp_list' argument passed to the constructor.

        The component separation problem is an Ax = b equation on the form
        (S^-1 + Y^T M^T Y^-1^T N^-1 B M Y) a
            = Y^T M^T Y^-1^T B^T d + Y^T M^T Y^-1^T B^T N^-{1/2} \eta_1 + S^-1 mu + S^{-1/2} eta_2,
        where B is the beam smoothing, M is the mixing matrix, N is the noise covariance matrix,
        Y is alm->map spherical harmonic synthesis, d is the observed frequency maps (as alms),
        a is the component maps we want to solve for (as alms), and z1 and z2 are random numbers
        drawn from N(0,1). For better numerical stability, we actually solve the equivalent equation
        (1 + S^{1/2} Y^T M^T Y^-1^T B^T N^-1 B Y^-1 M Y S^{1/2})[S^{-1/2} a]
            = S^{1/2} Y^T M^T {Y^{-1}}^T B^T N^{-1} d
            + S^{1/2}A^TN^{-1/2} eta_1 + S^{-1/2} mu + eta_2.
    """
    def __init__(self, comp_list: list[DiffuseComponent], map_sky: NDArray, map_rms: NDArray, freq: float, fwhm: float, params: Bunch, CompSep_comm: MPI.Comm, pol: bool):
        self.logger = logging.getLogger(__name__)
        self.CompSep_comm = CompSep_comm
        self.params = params
        self.map_sky = map_sky
        self.map_rms = map_rms
        self.freqs = np.array(CompSep_comm.allgather(freq))
        self.my_band_npix = map_rms.shape[-1]
        self.nband = len(self.freqs)
        self.my_rank = CompSep_comm.Get_rank()
        self.my_band_nside = hp.npix2nside(self.my_band_npix)
        self.per_band_nside = np.array(self.CompSep_comm.allgather(self.my_band_nside))
        self.per_band_npix = 12*self.per_band_nside**2
        self.my_band_lmax = int(2.5*self.my_band_nside)  # Slightly higher than 2*NSIDE to avoid accumulation of numeric junk.
        self.my_band_alm_len = ((self.my_band_lmax+1)*(self.my_band_lmax+2))//2
        self.per_band_lmax = np.array(self.CompSep_comm.allgather(self.my_band_lmax))
        self.per_band_alm_len = ((self.per_band_lmax+1)*(self.per_band_lmax+2))//2
        self.comp_list = comp_list
        self.pol = pol
        if pol:
            self.comps_SED = np.array([comp.get_sed(self.freqs) for comp in comp_list if comp.polarized])
            self.lmax_per_comp = np.array([comp.lmax for comp in comp_list if comp.polarized])
            self.per_comp_P_smooth = [comp.P_smoothing_prior for comp in comp_list if comp.polarized]
            self.per_comp_P_smooth_inv = [comp.P_smoothing_prior_inv for comp in comp_list if comp.polarized]
            self.per_comp_spatial_MM = np.array([comp.spatially_varying_MM for comp in comp_list if comp.polarized])
        else:  # We currently assume that all provided components are to be included in intensity.
            self.comps_SED = np.array([comp.get_sed(self.freqs) for comp in comp_list])
            self.lmax_per_comp = np.array([comp.lmax for comp in comp_list])
            self.per_comp_P_smooth = [comp.P_smoothing_prior for comp in comp_list]
            self.per_comp_P_smooth_inv = [comp.P_smoothing_prior_inv for comp in comp_list]
            self.per_comp_spatial_MM = np.array([comp.spatially_varying_MM for comp in comp_list])
        self.ncomp = len(self.comps_SED)
        self.is_holding_comp = self.my_rank < self.ncomp
        self.alm_len_percomp_complex = np.array([((lmax+1)*(lmax+2))//2 for lmax in self.lmax_per_comp])
        self.alm_len_percomp = np.array([(lmax+1)**2 for lmax in self.lmax_per_comp])
        self.my_band_fwhm_rad = np.deg2rad(fwhm/60.0)
        self.npol = 2 if pol else 1
        self.spin = 2 if pol else 0
        self.alm_start_idx_per_comp = [0]
        for icomp in range(self.ncomp):
            self.alm_start_idx_per_comp.append(self.alm_start_idx_per_comp[-1] + self.alm_len_percomp[icomp])
        self.alm_start_idx_per_comp_complex = [0]
        for icomp in range(self.ncomp):
            self.alm_start_idx_per_comp_complex.append(self.alm_start_idx_per_comp_complex[-1] + self.alm_len_percomp_complex[icomp])


    def _calc_dot(self, a: NDArray, b: NDArray):
        """ Calculates the dot product of two sets of real a_lms which are distributed across the
            self.ncomp first ranks of the self.CompSep_comm communicator.
        """
        res = 0.
        if self.is_holding_comp:
            res = np.dot(a.flatten(), b.flatten())
        res = self.CompSep_comm.allreduce(res, op=MPI.SUM)
        return res


    def _calc_dot_complex(self, a: NDArray, b: NDArray):
        """ Calculates the dot product of two sets of complex a_lms which are distributed across the
            self.ncomp first ranks of the self.CompSep_comm communicator.
            NB: This function is currently NOT USED, as we transitioned to using real alms in the CG.
        """
        res = 0.
        if self.is_holding_comp:
            lmax = self.my_comp_lmax
            res = np.dot(a[0:lmax+1].real, b[0:lmax+1].real)
            res += 2 * np.dot(a[lmax+1:].real, b[lmax+1:].real)
            res += 2 * np.dot(a[lmax+1:].imag, b[lmax+1:].imag)
        res = self.CompSep_comm.allreduce(res, op=MPI.SUM)
        return res


    def apply_A(self, a_in: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
        mythreads = self.params.nthreads_compsep

        band_alms = np.zeros((self.npol, self.my_band_alm_len), dtype=np.complex128)
        if (self.per_comp_spatial_MM).any():
            band_map = np.zeros((self.npol, self.my_band_npix))
            for icomp in range(self.ncomp):
                if self.per_comp_spatial_MM[icomp]:  # If this component has a MM that is pixel-depnedent.
                    alm_in_band_space = _project_alms(a_in[icomp], self.lmax_per_comp[icomp], self.my_band_lmax)
                    # Y a
                    comp_map = alm_to_map(alm_in_band_space, self.my_band_nside, self.my_band_lmax, spin=self.spin, nthreads=mythreads)
                    # M Y a
                    comp_map *= self.comps_SED[icomp, self.my_rank]
                    band_map += comp_map
            # Y^-1 M Y a
            curvedsky.map2alm_healpix(band_map, band_alms, niter=1, spin=self.spin, nthread=mythreads)

        for icomp in range(self.ncomp):
            if not self.per_comp_spatial_MM[icomp]:
                alm_in_band_space = _project_alms(a_in[icomp], self.lmax_per_comp[icomp], self.my_band_lmax)
                alm_in_band_space *= self.comps_SED[icomp, self.my_rank]
                band_alms += alm_in_band_space

        # B Y^-1 M Y a
        hp.smoothalm(band_alms, self.my_band_fwhm_rad, inplace=True)

        return band_alms



    def apply_A_adjoint(self, a_in: NDArray) -> NDArray:
        mythreads = self.params.nthreads_compsep
        
        # B^T a
        hp.smoothalm(a_in, self.my_band_fwhm_rad, inplace=True)
        a_final = [np.zeros((self.npol, self.alm_len_percomp_complex[icomp]), dtype=np.complex128) for icomp in range(self.ncomp)]

        if (self.per_comp_spatial_MM).any():
            band_map = np.zeros((self.npol, self.my_band_npix))
            curvedsky.map2alm_healpix(band_map, a_in, niter=1, adjoint=True, spin=self.spin, nthread=mythreads)
            for icomp in range(self.ncomp):
                if self.per_comp_spatial_MM[icomp]:
                    local_comp_lmax = self.lmax_per_comp[icomp]

                    tmp_map = band_map*self.comps_SED[icomp,self.my_rank]
                    tmp_alm = alm_to_map_adjoint(tmp_map, self.my_band_nside, self.my_band_lmax, spin=self.spin, nthreads=mythreads)

                    summed_alms_for_comp = _project_alms(tmp_alm, self.my_band_lmax, local_comp_lmax)
                    a_final[icomp][:] = summed_alms_for_comp
        for icomp in range(self.ncomp):
            if not self.per_comp_spatial_MM[icomp]:
                local_comp_lmax = self.lmax_per_comp[icomp]
                tmp_alm = a_in*self.comps_SED[icomp,self.my_rank]
                summed_alms_for_comp = _project_alms(tmp_alm, self.my_band_lmax, local_comp_lmax)
                a_final[icomp][:] = summed_alms_for_comp

        return a_final



    def apply_N_inv(self, a):
        mythreads = self.params.nthreads_compsep

        # Y a
        a = alm_to_map(a, self.my_band_nside, self.my_band_lmax, spin=self.spin, nthreads=mythreads)

        # N^-1 Y a
        a /= self.map_rms**2

        # Y^T N^-1 Y a
        a = alm_to_map_adjoint(a, self.my_band_nside, self.my_band_lmax, spin=self.spin, nthreads=mythreads)

        return a


    def apply_LHS_matrix(self, a_in: NDArray) -> NDArray:
        """ Applies the A matrix to inputed component alms a, where A represents the entire LHS of
            the Ax=b system for global component separation. The full A matrix is:
            (1 + S^{1/2} Y^T M^T Y^-1^T B^T N^-1 B Y^-1 M Y S^{1/2}).
            This function should be called by all ranks holding a frequency map, even if they do
            not hold a compoenent, as they are still needed to compute the LHS operation.
            Args:
                a_in (np.array): The a_lm of the component residing on this MPI rank.
                                 Should have shape (npol, nalm). Should be a zero-sized array
                                 for MPI ranks not holding a component.)
            Returns:
                Aa (np.array): The result of applying A to the input alms. Will return a zero-sized
                               array if this MPI rank does not hold a component.                               
        """
        logger = logging.getLogger(__name__)

        # logassert(a_in.dtype == np.float64, f"Provided component array is of type {a_in.dtype} and not np.float64. This operator takes and returns real alms (and converts to and from complex interally).", logger)
        # logassert(a_in.shape[-1] == self.my_comp_alm_len, f"Provided component array is of length {a_in.shape[-1]}, not {self.my_comp_alm_len}.", logger)
        myrank = self.CompSep_comm.Get_rank()
        mythreads = self.params.nthreads_compsep

        # if mycomp < self.ncomp:  # this task actually holds a component
        #     a = alm_real2complex(a_in, self.my_comp_lmax) # Convert the real input alms to complex alms.

        #     # S^{1/2} a
        #     for ipol in range(a.shape[0]):
        #         hp.almxfl(a[ipol], np.sqrt(self.my_comp_P_smooth), inplace=True)
        # else:
        #     a = None

        # Create empty a array for all ranks.
        a = [np.zeros((self.npol, self.alm_len_percomp_complex[icomp]), dtype=np.complex128) for icomp in range(self.ncomp)]
        if myrank == 0:  # this task actually holds a component
            for icomp in range(self.ncomp):
                start_idx = self.alm_start_idx_per_comp[icomp]
                stop_idx = self.alm_start_idx_per_comp[icomp+1]
                a_local_comp = a_in[:,start_idx:stop_idx]
                a[icomp] = alm_real2complex(a_local_comp, self.lmax_per_comp[icomp]) # Convert the real input alms to complex alms.
                for ipol in range(self.npol):
                    hp.almxfl(a[icomp][ipol], np.sqrt(self.per_comp_P_smooth[icomp]), inplace=True)


        # Spread initial a to all ranks from master.
        for icomp in range(self.ncomp):
            self.CompSep_comm.Bcast(a[icomp], root=0)

        # B Y^-1 M Y S^{1/2} a
        a = self.apply_A(a)
        # print(f"{self.CompSep_comm.Get_rank()} <<")
        # Y^T N^-1 Y B Y^-1 M Y S^{1/2} a
        a = self.apply_N_inv(a)
        # print(f"{self.CompSep_comm.Get_rank()} >>")

        # Y^T M^T Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y S^{1/2} a
        a = self.apply_A_adjoint(a)

        # if mycomp < self.ncomp:
        for icomp in range(self.ncomp):
            # Accumulate solution on master
            send, recv = (MPI.IN_PLACE, a[icomp]) if myrank == 0 else (a[icomp], None)
            self.CompSep_comm.Reduce(send, recv, op=MPI.SUM, root=0)

            # S^{1/2} Y^T M^T Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y S^{1/2} a
            if myrank == 0:
                for ipol in range(self.npol):
                    hp.almxfl(a[icomp][ipol], np.sqrt(self.per_comp_P_smooth[icomp]), inplace=True)                


        if myrank == 0:
            for icomp in range(self.ncomp):
                a[icomp] = alm_complex2real(a[icomp], self.lmax_per_comp[icomp]) # Convert complex alm back to real before returning.
                a[icomp] += a_in[:,self.alm_start_idx_per_comp[icomp]:self.alm_start_idx_per_comp[icomp+1]]
            a = np.concatenate(a, axis=-1)
        else:
            a = np.zeros((0,), dtype=np.float64)  # zero-sized array

        # Adds input vector to output, since (1 + S^{1/2}...)a = a + (S^{1/2}...)a
        # return a_in + a
        return a


    def calc_RHS_mean(self) -> NDArray:
        """ Caculates the right-hand-side b-vector of the Ax=b CompSep equation for the Wiener filtered (or mean-field) solution.
            If used alone on the right-hand-side, gives the deterministic maximum likelihood map-space solution, but a biased PS solution.
        """
        myrank = self.CompSep_comm.Get_rank()
        mythreads = self.params.nthreads_compsep

        # N^-1 d
        b = self.map_sky/self.map_rms**2

        # b_out = [np.zeros((self.npol, self.alm_len_percomp_complex[icomp]), dtype=np.complex128) for icomp in range(self.ncomp)]
        # # Y^T N^-1 d
        # for icomp in range(self.ncomp):
        #     b_out[icomp] = alm_to_map_adjoint(b, self.my_band_nside, self.lmax_per_comp[icomp], spin=self.spin, nthreads=mythreads)

        b = alm_to_map_adjoint(b, self.my_band_nside, self.my_band_lmax, spin=self.spin, nthreads=mythreads)

        # (Y^T M^T Y^-1^T B^T) Y^T N^-1 d
        b = self.apply_A_adjoint(b)

        # S^{1/2} Y^T M^T Y^-1^T B^T Y^T N^-1 d
        # if self.is_holding_comp:  # This task actually holds a component
        #     for ipol in range(b.shape[0]):
        #         hp.almxfl(b[ipol], np.sqrt(self.my_comp_P_smooth), inplace=True)               
        #     b = alm_complex2real(b, self.my_comp_lmax)
        # else:
        #     b = np.zeros((0,), dtype=np.float64)
        # if self.CompSep_comm.Get_rank() == 0:
        #     for icomp in range(self.ncomp):
        #         a[icomp] = alm_complex2real(a[icomp], self.lmax_per_comp[icomp]) # Convert complex alm back to real before returning.
        #         a[icomp] += a_in[icomp]
        # else:
        #     a = np.zeros((0,), dtype=np.float64)  # zero-sized array

        # for icomp in range(self.ncomp):
        #     for ipol in range(self.npol):
        #         hp.almxfl(b[icomp][ipol], np.sqrt(self.per_comp_P_smooth[icomp]), inplace=True)

        for icomp in range(self.ncomp):
            # Accumulate solution on master
            send, recv = (MPI.IN_PLACE, b[icomp]) if myrank == 0 else (b[icomp], None)
            self.CompSep_comm.Reduce(send, recv, op=MPI.SUM, root=0)

            # S^{1/2} Y^T M^T Y^-1^T B^T Y^T N^-1 d
            if myrank == 0:
                for ipol in range(self.npol):
                    hp.almxfl(b[icomp][ipol], np.sqrt(self.per_comp_P_smooth[icomp]), inplace=True)                

        if myrank == 0:
            for icomp in range(self.ncomp):
                b[icomp] = alm_complex2real(b[icomp], self.lmax_per_comp[icomp]) # Convert complex alm back to real before returning.
            b = np.concatenate(b, axis=-1)
        else:
            b = np.zeros((0,), dtype=np.float64)  # zero-sized array

        return b


    def calc_RHS_fluct(self) -> NDArray:
        """ Calculates the right-hand-side fluctuation vector. Provides unbiased realizations (of foregrounds or the CMB) if added
            together with the right-hand-side of the Wiener filtered solution : Ax = b_mean + b_fluct.
        """
        mythreads = self.params.nthreads_compsep

        # eta_1
        b = np.random.normal(0.0, 1.0, self.map_rms.shape)

        # N^-1/2 eta_1
        b /= self.map_rms

        # Y^T N^-1 eta_1
        b = alm_to_map_adjoint(b, self.my_band_nside, self.global_lmax, spin=self.spin, nthreads=mythreads)

        # (Y^T M^T Y^-1^T B^T) Y^T N^-1 eta_1
        b = self.apply_A_adjoint(b)

        # S^{1/2} Y^T M^T Y^-1^T B^T Y^T N^-1 eta_1
        if self.is_holding_comp:  # This task actually holds a component
            for ipol in range(b.shape[0]):
                hp.almxfl(b[ipol], np.sqrt(self.my_comp_P_smooth), inplace=True)               
            b = alm_complex2real(b, self.my_comp_lmax)
        else:
            b = np.zeros((0,), dtype=np.float64)

        return b


    def calc_RHS_prior_mean(self) -> NDArray:
        if self.is_holding_comp:  # This task actually holds a component
            mu = np.zeros((self.npol, self.my_comp_alm_len_complex), dtype=np.complex128)
            for ipol in range(mu.shape[0]):
                hp.almxfl(mu[ipol], self.my_comp_P_smooth_inv, inplace=True)
            mu = alm_complex2real(mu, self.my_comp_lmax)
        else:
            mu = np.zeros((0,), dtype=np.float64)
        return mu


    def calc_RHS_prior_fluct(self) -> NDArray:
        if self.is_holding_comp:  # This task actually holds a component
            eta2 = np.zeros((self.npol, self.my_comp_alm_len_complex), dtype=np.complex128)
            for ipol in range(eta2.shape[0]):
                eta2[ipol] = gaussian_random_alm(self.my_comp_lmax, self.my_comp_lmax, self.spin, 1)
                hp.almxfl(eta2[ipol], np.sqrt(self.my_comp_P_smooth_inv), inplace=True)
            eta2 = alm_complex2real(eta2, self.my_comp_lmax)
        else:
            eta2 = np.zeros((0,), dtype=np.float64)
        return eta2



    def solve_CG(self, LHS: Callable, RHS: NDArray, x0: NDArray, M = None, x_true = None) -> NDArray:
        """ Solves the equation Ax=b for x given A (LHS) and b (RHS) using CG from the pixell package.
            Assumes that both x and b are in alm space.

            Args:
                LHS (callable): A function/callable taking x as argument and returning Ax.
                RHS (np.array): A Numpy array representing b, in alm space.
                x0 (np.array): Initial guess for x.
                M (callable): Preconditioner for the CG solver. A function/callable which approximates A^-1 (optional).
                x_true (np.array): True solution for x, in order to print the true error (optional, used for testing).
            Returns:
                m_bestfit (np.array): The resulting best-fit solution, in alm space, for the component held by this rank.
                                      A zero-sized array for ranks holding no component.
        """
        max_iter = self.params.CG_max_iter_pol if self.pol else self.params.CG_max_iter

        logger = logging.getLogger(__name__)
        checkpoint_interval = 5
        master = self.CompSep_comm.Get_rank() == 0
        mycomp = self.CompSep_comm.Get_rank()

        # mydot = lambda a,b: self._calc_dot(a,b)
        # mydot = lambda a,b: np.sum([np.dot(a[icomp],b[icomp]) for icomp in range(self.ncomp)]) if len(a) > 0 else 0
        mydot = lambda a,b: np.dot(a.flatten(),b.flatten())
        if M is None:
            CG_solver = utils.CG(LHS, RHS, dot=mydot, x0=x0)
        else:
            CG_solver = utils.CG(LHS, RHS, dot=mydot, x0=x0, M=M)
        self.CG_residuals = np.zeros((max_iter))
        if x_true is not None:
            # self.x_true_allcomps = self.CompSep_comm.allgather()
            self.xtrue_A_xtrue = x_true.dot(LHS(x_true))  # The normalization factor for the true error (contribution from my rank).
            self.xtrue_A_xtrue = self.CompSep_comm.allreduce(self.xtrue_A_xtrue, MPI.SUM)  # Dot product is linear, so we can just sum the contributions.
        if master:
            logger.info("CG starting up!")
        iter = 0
        t0 = time.time()
        stop_CG = False
        while not stop_CG:
            CG_solver.step()
            self.CG_residuals[iter] = CG_solver.err
            iter += 1
            if iter%checkpoint_interval == 0:
                if master:
                    logger.info(f"CG iter {iter:3d} - Residual {np.mean(self.CG_residuals[iter-checkpoint_interval:iter]):.3e} ({(time.time() - t0)/checkpoint_interval:.1f}s/iter)")
                    t0 = time.time()
                if mycomp < self.ncomp:
                    if x_true is not None:
                        CG_errors_true = np.linalg.norm(CG_solver.x-x_true)/np.linalg.norm(x_true)
                        CG_Anorm_error = (CG_solver.x-x_true).dot(LHS(CG_solver.x-x_true))
                        CG_Anorm_error = self.CompSep_holdingcomp_comm.allreduce(CG_Anorm_error, MPI.SUM)/self.xtrue_A_xtrue  # Collect error contributions from all ranks.
                        time.sleep(0.01*mycomp)  # Getting the prints in the same order every time.
                        if master:
                            logger.info(f"CG iter {iter:3d} - True A-norm error: {CG_Anorm_error:.3e}")  # A-norm error is only defined for the full vector.
                        logger.info(f"CG iter {iter:3d} - {self.comp_list[mycomp].longname} - True L2 error: {CG_errors_true:.3e}")  # We can print the individual component L2 errors.
                else:
                    if x_true is not None:
                        LHS(np.zeros((0,), dtype=np.float64))  # Matching LHS call for the calculation of LHS(CG_solver.x-x_true).
            if iter >= max_iter:
                if master:
                    logger.warning(f"Maximum number of iterations ({max_iter}) reached in CG.")
                stop_CG = True
            if CG_solver.err < self.params.CG_err_tol:
                stop_CG = True
            stop_CG = self.CompSep_comm.bcast(stop_CG, root=0)
        self.CG_residuals = self.CG_residuals[:iter]
        if master:
            logger.info(f"CG finished after {iter} iterations with a residual of {CG_solver.err:.3e} (err tol = {self.params.CG_err_tol})")
            s_bestfit_compact = CG_solver.x
            s_bestfit_list = []
            for icomp in range(self.ncomp):
                local_comp = s_bestfit_compact[:,self.alm_start_idx_per_comp[icomp]:self.alm_start_idx_per_comp[icomp+1]]
                local_comp = alm_real2complex(local_comp, self.lmax_per_comp[icomp])  # CG search uses real-valued alms, convert to complex, which is used outside CG.
                for ipol in range(self.npol):
                    hp.almxfl(local_comp[ipol], np.sqrt(self.per_comp_P_smooth[icomp]), inplace=True)
                s_bestfit_list.append(local_comp)
        else:
            s_bestfit_list = [] # np.zeros((0,), dtype=np.complex128)
        return s_bestfit_list



    def solve(self, seed=None) -> list[DiffuseComponent]:
        mycomp = self.CompSep_comm.Get_rank()

        RHS1 = self.calc_RHS_mean()
        # RHS2 = self.calc_RHS_fluct()
        # RHS3 = self.calc_RHS_prior_mean()
        # RHS4 = self.calc_RHS_prior_fluct()
        RHS = RHS1 # + RHS2 + RHS3 + RHS4
        # self.logger.info(f"Mean amplitude of each RHS component: {np.mean(np.abs(RHS1)):.2e}"
        #                  f"{np.mean(np.abs(RHS2)):.2e} {np.mean(np.abs(RHS3)):.2e} {np.mean(np.abs(RHS4)):.2e}")

        # Initialize the precondidioner class, which is in the module "solvers.preconditioners", and has a name specified by self.params.compsep.preconditioner.
        precond = getattr(preconditioners, self.params.compsep.preconditioner)(self)

        if self.params.compsep.dense_matrix_debug_mode:  # For testing preconditioner with a true solution as reference, first solving for exact solution with dense matrix math.
            M_A_matrix = lambda a : self.apply_LHS_matrix(precond(a))

            # Testing the initial LHS (A) matrix
            dense_matrix = DenseMatrix(self.CompSep_comm, self.apply_LHS_matrix, self.alm_len_percomp, matrix_name="A")
            x_true = dense_matrix.solve_by_inversion(RHS)
            dense_matrix.test_matrix_hermitian()
            dense_matrix.print_sing_vals()
            dense_matrix.test_matrix_eigenvalues()
            dense_matrix.print_matrix_diag()

            # Testing the preconditioning matrix (M) alone
            dense_matrix = DenseMatrix(self.CompSep_comm, precond, self.alm_len_percomp, matrix_name="M")
            dense_matrix.test_matrix_hermitian()  # Preconditioner matrix needs to be Hermitian.
            dense_matrix.test_matrix_eigenvalues()  # and to have positive and real eigenvalues

            # Testing the combined preconditioned system (MA)  (this combined matrix will generally not be Hermitian positive-definite).
            dense_matrix = DenseMatrix(self.CompSep_comm, M_A_matrix, self.alm_len_percomp, matrix_name="MA")
            dense_matrix.print_sing_vals()  # Check how much singular values (condition number) of preconditioned system improved.
            dense_matrix.print_matrix_diag()


        if seed is not None:
            np.random.seed(seed)
        # if mycomp < self.ncomp:
        #     x0 = np.zeros((self.npol, self.alm_len_percomp[mycomp],), dtype=np.float64)
            # np.random.seed(42)
            # x0[0] = np.random.normal(0, 1, (self.alm_len_percomp[mycomp],))
        if self.my_rank == 0:
            # x0 = [np.zeros((self.npol, self.alm_len_percomp[icomp]), dtype=np.float64) for icomp in range(self.ncomp)]
            tot_alm_len = np.sum(self.alm_len_percomp)
            x0 = np.zeros((self.npol, tot_alm_len), dtype=np.float64)
        else:
            # x0 = [] # np.zeros((0,), dtype=np.float64)
            x0 = np.zeros((0,), dtype=np.float64)
        sol_list = self.solve_CG(self.apply_LHS_matrix, RHS, x0, M=precond, x_true=x_true if self.params.compsep.dense_matrix_debug_mode else None)
        sol_list = self.CompSep_comm.bcast(sol_list, root=0)
        for icomp in range(self.ncomp):
            if self.spin == 0:
                self.comp_list[icomp].component_alms_intensity = sol_list[icomp]
            elif self.spin == 2:
                self.comp_list[icomp].component_alms_polarization = sol_list[icomp]

        return self.comp_list
