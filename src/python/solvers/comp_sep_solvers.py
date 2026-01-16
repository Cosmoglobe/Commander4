import numpy as np
import healpy as hp
import time
from pixell import utils, curvedsky
from pixell.bunch import Bunch
import logging
from mpi4py import MPI
from numpy.typing import NDArray
from typing import Callable
from copy import deepcopy

from src.python.output.log import logassert
from src.python.data_models.detector_map import DetectorMap
from src.python.sky_models.component import Component, DiffuseComponent
from src.python.utils.math_operations import alm_to_map, alm_to_map_adjoint, alm_real2complex,\
    alm_complex2real, gaussian_random_alm, project_alms, almxfl, inplace_add_scaled_vec,\
    inplace_arr_prod, inplace_scale, almlist_dot_complex, almlist_dot_real, map_to_alm,\
    map_to_alm_adjoint, complist_dot
from src.python.solvers.dense_matrix_math import DenseMatrix
from src.python.solvers.CG_driver import distributed_CG
import src.python.solvers.preconditioners as preconditioners
from src.python.data_models.band import Band

MPI_LIMIT_32BIT = 2**31 - 1

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
    if comp_maps[0][0].dtype == np.float64:
        complex_dtype = np.complex128
    else:
        complex_dtype = np.complex64
    for icomp in range(ncomp_full):
        alm_len = ((comp_list[icomp].lmax+1)*(comp_list[icomp].lmax+2))//2
        comp_alms = np.zeros((1,alm_len), dtype=complex_dtype)
        comp_list[icomp].component_alms_intensity = curvedsky.map2alm_healpix(comp_maps[0][icomp], comp_alms, niter=3, spin=0)
        if comp_list[icomp].polarized:
            alm_len = ((comp_list[icomp].lmax+1)*(comp_list[icomp].lmax+2))//2
            comp_alms = np.zeros((2,alm_len), dtype=complex_dtype)
            pol_alms = curvedsky.map2alm_healpix(np.array([comp_maps[1][icomp], comp_maps[2][icomp]]), comp_alms, niter=3, spin=2)
            comp_list[icomp].component_alms_polarization = pol_alms
    return comp_list


class CompSepSolver:
    """ Class for performing global component separation using the preconditioned conjugate gradient
        method. After initializing the class, the solve() method should be called to perform the
        component separation. Note that the solve() method will in-place update (as well as return)
        the 'comp_list' argument passed to the constructor.

        The component separation problem is an Ax = b equation on the form
        (S^-1 + Y^T M^T Y^-1^T N^-1 B M Y) a
            = Y^T M^T Y^-1^T B^T d + Y^T M^T Y^-1^T B^T N^-{1/2} eta_1 + S^-1 mu + S^{-1/2} eta_2,
        where B is the beam smoothing, M is the mixing matrix, N is the noise covariance matrix,
        Y is alm->map spherical harmonic synthesis, d is the observed frequency maps (as alms),
        a is the component maps we want to solve for (as alms), and z1 and z2 are random numbers
        drawn from N(0,1). For better numerical stability, we actually solve the equivalent equation
        (1 + S^{1/2} Y^T M^T Y^-1^T B^T N^-1 B Y^-1 M Y S^{1/2})[S^{-1/2} a]
            = S^{1/2} Y^T M^T {Y^{-1}}^T B^T N^{-1} d
            + S^{1/2}A^TN^{-1/2} eta_1 + S^{-1/2} mu + eta_2.
    """
    def __init__(self, det_map: DetectorMap,
                 params: Bunch, CompSep_comm: MPI.Comm):
        
        self.logger = logging.getLogger(__name__)
        self.CompSep_comm = CompSep_comm
        self.my_rank = CompSep_comm.Get_rank()
        self.det_map = det_map
        self.params = params
        #self.comp_list = comp_list
        #self.ncomp = len(comp_list)
        self.my_band = Band.init_from_detector(det_map = det_map, double_precision = params.CG_float_precision == "double")

        if params.CG_float_precision == "single":
            self.float_dtype = np.float32
            self.complex_dtype = np.complex64
        else:
            self.float_dtype = np.float64
            self.complex_dtype = np.complex128

        ### This info should stay within DetectorMap
        # self.map_sky = map_sky.astype(self.float_dtype)  # The sky map that my band holds.
        # self.map_inv_var = (1.0/map_rms**2).astype(self.float_dtype)  # The rms map of my band (TODO: this should eventually be abstracted away to a general N^-1 procedure).
        # self.my_band_npix = map_rms.shape[-1]
        # self.my_band_nside = hp.npix2nside(self.my_band_npix)

        #This is probably not used (?)
        # self.per_band_lmax = np.array(self.CompSep_comm.allgather(self.det_map.lmax))
        # self.per_band_nside = np.array(self.CompSep_comm.allgather(self.my_band_nside))
        # self.per_band_npix = 12*self.per_band_nside**2
        # self.nband = len(self.freqs)

        #THIS INFO SHOULD STAY IN THE ComponentClass
        #self.freqs = np.array(CompSep_comm.allgather(freq))
        
        # if pol:  #TODO: Polarized vs non-polarized should be handled more elegantly.
        #     #self.comps_SED = np.array([comp.get_sed(self.freqs) for comp in comp_list if comp.polarized]).astype(self.float_dtype)
        #     #self.lmax_per_comp = np.array([comp.lmax for comp in comp_list if comp.polarized])
        #     # self.per_comp_P_smooth = [comp.P_smoothing_prior.astype(self.float_dtype) for comp in comp_list if comp.polarized]
        #     # self.per_comp_P_smooth_sqrt = [np.sqrt(comp.P_smoothing_prior.astype(self.float_dtype)) for comp in comp_list if comp.polarized]
        #     #  self.per_comp_P_smooth_inv = [comp.P_smoothing_prior_inv.astype(self.float_dtype) for comp in comp_list if comp.polarized]
        #     # self.per_comp_P_smooth_inv_sqrt = [np.sqrt(comp.P_smoothing_prior_inv).astype(self.float_dtype) for comp in comp_list if comp.polarized]
        #     #  self.per_comp_spatial_MM = np.array([comp.spatially_varying_MM for comp in comp_list if comp.polarized])
        # else:  # We currently assume that all provided components are to be included in intensity.
        #     #self.comps_SED = np.array([comp.get_sed(self.freqs) for comp in comp_list]).astype(self.float_dtype)
        #     #self.lmax_per_comp = np.array([comp.lmax for comp in comp_list])
        #     # self.per_comp_P_smooth = [comp.P_smoothing_prior.astype(self.float_dtype) for comp in comp_list]
        #     # self.per_comp_P_smooth_sqrt = [np.sqrt(comp.P_smoothing_prior.astype(self.float_dtype)) for comp in comp_list]
        #     #  self.per_comp_P_smooth_inv = [comp.P_smoothing_prior_inv.astype(self.float_dtype) for comp in comp_list]
        #     # self.per_comp_P_smooth_inv_sqrt = [np.sqrt(comp.P_smoothing_prior_inv).astype(self.float_dtype) for comp in comp_list]
        #     # self.per_comp_spatial_MM = np.array([comp.spatially_varying_MM for comp in comp_list])
        
        # Real and complex alm representations have different length, so we keep track of both:
        self.alm_len_percomp_complex = np.array([((lmax+1)*(lmax+2))//2 for lmax in self.lmax_per_comp])
        self.alm_len_percomp_real = np.array([(lmax+1)**2 for lmax in self.lmax_per_comp])

        # Defining some convenience parameters, so we can e.g. check the lengths for the alm arrays
        # without having to if-test what type they are depending on what mode we are in.
        # if params.CG_real_alm_mode:
        #     self.alm_dtype = self.float_dtype
        # else:
        self.alm_dtype = self.complex_dtype

        # For simplicity all array will have shapes (1, ...) for non-polarization (and then (2, ...) for polarization).
        self.npol = det_map.npol
        self.spin = 2 if self.npol else 0

        # num-threads is either an int, or a list of one value per thread.
        if isinstance(self.params.nthreads_compsep, int):
            self.nthreads = self.params.nthreads_compsep
        else:
            self.nthreads = self.params.nthreads_compsep[self.CompSep_comm.Get_rank()]


    # def apply_A_old(self, a_in: list[NDArray[np.complexfloating]]) -> NDArray[np.complexfloating]:
    #     #a_in: alms of the different components
    #     mythreads = self.nthreads
    #     alm_len_real = ((self.det_map.lmax+1)*(self.det_map.lmax+2))//2
    #     band_alms = np.zeros((self.npol, alm_len_real), dtype=self.complex_dtype)
    #     if (self.per_comp_spatial_MM).any():
    #         band_map = np.zeros((self.npol, self.my_band_npix), dtype=self.float_dtype)
    #         for icomp in range(self.ncomp):
    #             if self.per_comp_spatial_MM[icomp]:  # If this component has a MM that is pixel-depnedent.
    #                 alm_in_band_space = project_alms(a_in[icomp], self.det_map.lmax)
    #                 # Y a
    #                 comp_map = alm_to_map(alm_in_band_space, self.my_band_nside, self.det_map.lmax, spin=self.spin, nthreads=mythreads)
    #                 # M Y a
    #                 for ipol in range(self.npol):
    #                     inplace_add_scaled_vec(band_map[ipol], comp_map[ipol], self.comps_SED[icomp, self.my_rank])
    #         # Y^-1 M Y a
    #         # curvedsky.map2alm_healpix(band_map, band_alms, niter=0, spin=self.spin, nthread=mythreads)
    #         band_alms = map_to_alm(band_map, self.my_band_nside, self.det_map.lmax, spin=self.spin, out=band_alms, nthreads=mythreads)

    #     for icomp in range(self.ncomp):
    #         if not self.per_comp_spatial_MM[icomp]:
    #             alm_in_band_space = project_alms(a_in[icomp], self.det_map.lmax)
    #             for ipol in range(self.npol):
    #                 inplace_add_scaled_vec(band_alms[ipol], alm_in_band_space[ipol], self.comps_SED[icomp, self.my_rank])
    #     # B Y^-1 M Y a
    #     self.det_map.apply_B(band_alms)

    #     return band_alms
    

    def project_all_comps_to_band(self, comp_list_in: list[Component], band_out:Band, inplace=True) -> NDArray[np.complexfloating]:
        """
        Projects all the components in comp_list_in to the band_out object, and updates it in-place.

        In Commander4 notation, applies A matrix, from comp list to band alms.
        """

        for comp in comp_list_in:
            comp.project_comp_to_band(band_out, nthreads=self.nthreads)
        # B Y^-1 M Y a
        alm_out = self.det_map.apply_B(band_out.alms)
        return alm_out


    def eval_all_comps_from_band(self, band_in:Band, comp_list_out:list[Component], inplace=True) -> list[Component]:
        """
        Evaluates the band_in's contribution to all the comp_list_out objects, and stores them in-place by default.

        In Commander4 notation, applies A_adj matrix, from band alms to comp list.
        """

        # B^T a
        self.det_map.apply_B(band_in.alms) #TODO: make band an argument, rather than inplicitly taking the one from self.
        
        # Y^T M^T Y^-1^T B^T a
        for comp in comp_list_out:
            comp.eval_comp_from_band(band_in, nthreads=self.nthreads)
        return comp_list_out

    # def apply_A_adjoint_old(self, a_in: NDArray[np.complexfloating]) -> list[NDArray[np.complexfloating]]:
    #     mythreads = self.nthreads
    #     a_final = [np.zeros((self.npol, self.alm_len_percomp_complex[icomp]), dtype=self.complex_dtype) for icomp in range(self.ncomp)]

    #     # B^T a
    #     self.det_map.apply_B(a_in)

    #     if (self.per_comp_spatial_MM).any():
    #         band_map = np.zeros((self.npol, self.my_band_npix), dtype=self.float_dtype)
    #         # Y^-1^T B^T a
    #         # curvedsky.map2alm_healpix(band_map, a_in, niter=0, adjoint=True, spin=self.spin, nthread=mythreads)
    #         band_map = map_to_alm_adjoint(a_in, self.my_band_nside, self.det_map.lmax, spin=self.spin, out=band_map, nthreads=mythreads)
    #         for icomp in range(self.ncomp):
    #             if self.per_comp_spatial_MM[icomp]:
    #                 local_comp_lmax = self.lmax_per_comp[icomp]

    #                 # M^T Y^-1 B^T a
    #                 tmp_map = band_map.copy()
    #                 for ipol in range(self.npol):
    #                     inplace_scale(tmp_map[ipol], self.comps_SED[icomp,self.my_rank])

    #                 # Y^T M^T Y^-1^T B^T a
    #                 tmp_alm = alm_to_map_adjoint(tmp_map, self.my_band_nside, self.det_map.lmax, spin=self.spin, nthreads=mythreads)

    #                 # Project alm from band to component lmax.
    #                 summed_alms_for_comp = project_alms(tmp_alm, local_comp_lmax)
    #                 a_final[icomp][:] = summed_alms_for_comp

    #     # For the components that don't have spatially dependent mixing matrix, we do it all in alm-space:
    #     for icomp in range(self.ncomp):
    #         if not self.per_comp_spatial_MM[icomp]:
    #             local_comp_lmax = self.lmax_per_comp[icomp]
    #             tmp_alm = a_in.copy()
    #             for ipol in range(self.npol):
    #                 inplace_scale(tmp_alm[ipol], self.comps_SED[icomp,self.my_rank])
    #             a_final[icomp][:] = project_alms(tmp_alm, local_comp_lmax)

    #     return a_final



    # def apply_N_inv(self, a):
    #     mythreads = self.nthreads

    #     # Y a
    #     a = alm_to_map(a, self.my_band_nside, self.det_map.lmax, spin=self.spin, nthreads=mythreads)

    #     # N^-1 Y a
    #     for ipol in range(self.npol):
    #         inplace_arr_prod(a[ipol], self.map_inv_var[ipol])

    #     # Y^T N^-1 Y a
    #     a = alm_to_map_adjoint(a, self.my_band_nside, self.my_band_lmax, spin=self.spin, nthreads=mythreads)

    #     return a


    def apply_LHS_matrix(self, comp_list_in: list[Component]) -> list[Component]:
        #This a_in should become a list of Component objects instead. 
        """ Applies the A matrix to inputed component alms a, where A represents the entire LHS of
            the Ax=b system for global component separation. The full A matrix is:
            (1 + S^{1/2} Y^T M^T Y^-1^T B^T N^-1 B Y^-1 M Y S^{1/2}).
            This function should be called by all ranks holding a frequency map, even if they do
            not hold a compoenent, as they are still needed to compute the LHS operation.
            Args:
                a_in (list[Component]): List of Components contributing to the local band
            Returns:
                Aa (np.array): The result of applying A to the input alms. Will return a zero-sized
                               array if this MPI rank does not hold a component.                               
        """
        logger = logging.getLogger(__name__)
        myrank = self.CompSep_comm.Get_rank()

        if myrank == 0:  # this task actually holds a component
            comp_list = deepcopy(comp_list_in)
            for comp in comp_list:
                # S^{1/2} a
                comp.apply_smoothing_prior_sqrt()

        # Spread initial a to all ranks from master.
        # NB: For some stupid reason the non-blocking mpi4py calls do not have 64-bit length support
        # and are therefore limited to <2GB arrays... We have to fallback to blocking communication
        # for >2GB arrays. In the future we should probably implement chunking instead.

            # here I think the non blocking is useless.
        for comp in comp_list:
            comp.bcast_data_blocking(self.CompSep_comm)
        # requests = []
        # for comp in comp_list:
        #     if comp_list[icomp].nbytes > MPI_LIMIT_32BIT:
        #         print(f"Fallback to blocking comm (array size = {comp.nbytes:.2e}B)")
        #         comp.bcast_data_blocking(self.CompSep_comm)
        #     else:
        #         req = comp.bcast_data_non_blocking(self.CompSep_comm)
        #         requests.append(req)
        # for request in requests:
        #     MPI.Request.Wait(request)

        # B Y^-1 M Y S^{1/2} a
        self.project_all_comps_to_band(comp_list, self.my_band)

        # Y^T N^-1 Y B Y^-1 M Y S^{1/2} a
        self.det_map.apply_inv_N_alm(self.my_band.alms, nthreads=self.nthreads)

        # Y^T M^T Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y S^{1/2} a
        self.eval_all_comps_from_band(self.my_band, comp_list)

        # Accumulate solution on master
        biggest_size_bytes = np.max([_array.nbytes for _array in comp_list._data])
        use_blocking = biggest_size_bytes > MPI_LIMIT_32BIT
        if use_blocking:
            print(f"Fallback to blocking comm (array size = {biggest_size_bytes:.2e}B)")
            for comp in comp_list:
                comp.accum_data_blocking(self.CompSep_comm)

        else:
            requests = []
            for comp in comp_list:
                req = comp.accum_data_non_blocking(self.CompSep_comm)
                requests.append(req)

        if myrank == 0:
            for icomp in range(len(comp_list)):
                # Since we used non-blocking reduce, master rank can start working on components
                # as they are received instead of waiting for all to be received.
                if not use_blocking:
                    MPI.Request.Wait(requests[icomp])  # Wait until all data for component icomp has been received.
                # S^{1/2} Y^T M^T Y^-1^T B^T Y^T N^-1 Y B Y^-1 M Y S^{1/2} a
                comp_list[icomp].apply_smoothing_prior_sqrt()
                # Adds input vector to output, since (1 + S^{1/2}...)a = a + (S^{1/2}...)a
                comp_list[icomp] += comp_list_in[icomp]
        else: # Worker ranks just wait for all their sends to complete.
            if not use_blocking:
                for icomp in range(self.ncomp):
                    MPI.Request.Wait(requests[icomp])
            comp_list = []

        return comp_list


    def calc_RHS_mean(self, comp_list: list[Component]) -> list[Component]:
        """ Caculates the right-hand-side b-vector of the Ax=b CompSep equation for the Wiener filtered (or mean-field) solution.
            If used alone on the right-hand-side, gives the deterministic maximum likelihood map-space solution, but a biased PS solution.
        """
        myrank = self.CompSep_comm.Get_rank()
        mythreads = self.nthreads

        # N^-1 d
        b_map = self.det_map.apply_inv_N_map(self.det_map.map_sky)

        # # Y^T N^-1 d
        b_alm = alm_to_map_adjoint(b_map, self.my_band.nside, self.my_band.lmax, spin=self.spin, nthreads=mythreads)
        b_band = Band.init_from_detector(det_map = self.det_map, double_precision = self.params.CG_float_precision == "double")
        b_band.alms = b_alm

        # (Y^T M^T Y^-1^T B^T) Y^T N^-1 d
        self.eval_all_comps_from_band(b_band, comp_list)

        # Accumulate solution on master
        for comp in comp_list:
            comp.accum_data_blocking(self.CompSep_comm)
            if myrank == 0:
                # S^{1/2} Y^T M^T Y^-1^T B^T Y^T N^-1 d
                comp.apply_smoothing_prior_sqrt()

        if myrank == 0:
            for comp in comp_list:
                self.logger.info(f"RHS1 comp-{comp.shortname}: {np.mean(np.abs(comp._data)):.2e}")
        else:
            b = []

        return b


    def calc_RHS_fluct(self, comp_list: list[Component]) -> list[Component]:
        """ Calculates the right-hand-side fluctuation vector. Provides unbiased realizations (of foregrounds or the CMB) if added
            together with the right-hand-side of the Wiener filtered solution : Ax = b_mean + b_fluct.
        """
        myrank = self.CompSep_comm.Get_rank()
        mythreads = self.nthreads

        # eta_1
        b_map = np.random.normal(0.0, 1.0, self.det_map.inv_var_map.shape)

        # N^-1/2 eta_1
        b_map *= np.sqrt(self.det_map.inv_var_map)

        # Y^T N^-1 eta_1
        b_alm = alm_to_map_adjoint(b_map, self.my_band.nside, self.my_band.lmax, spin=self.spin, nthreads=mythreads)
        b_band = Band.init_from_detector(det_map = self.det_map, double_precision = self.params.CG_float_precision == "double")
        b_band.alms = b_alm

        # (Y^T M^T Y^-1^T B^T) Y^T N^-1 eta_1
        self.eval_all_comps_from_band(b_band, comp_list)

        # Accumulate solution on master
        for comp in comp_list:
            comp.accum_data_blocking(self.CompSep_comm)
            if myrank == 0:
                # S^{1/2} Y^T M^T Y^-1^T B^T Y^T N^-1 eta_1
                comp.apply_smoothing_prior_sqrt()
        
        if myrank == 0:
            for comp in comp_list:
                self.logger.info(f"RHS1 comp-{comp.shortname}: {np.mean(np.abs(comp._data)):.2e}")
        else:
            b = []

        return b


    def calc_RHS_prior_mean(self, comp_list: list[Component]) -> list[Component]:
        
        #FIXME: how will this be for point sources?
        
        myrank = self.CompSep_comm.Get_rank()
        if myrank == 0:
            # Currently this will always return 0, since we have not yet implemented support for a spatial prior,
            # but when we do it will go here.
            mu_s = []
            for comp in comp_list:
                mu = np.zeros((self.npol, comp.alm_len_complex), dtype=self.complex_dtype)
                for ipol in range(self.npol):
                    almxfl(mu[ipol], comp.P_smoothing_prior.astype(self.float_dtype), inplace=True)
                self.logger.info(f"RHS3 comp-{comp.longname}: {np.mean(np.abs(mu)):.2e}")
                mu_s.append(mu)
        else:
            mu_s = []
        return mu_s


    def calc_RHS_prior_fluct(self, comp_list: list[Component]) -> list[Component]:
        
        #FIXME: how will this be for point sources?

        myrank = self.CompSep_comm.Get_rank()
        if myrank == 0:
            eta2_s = []
            for comp in comp_list:
                eta2 = np.zeros((self.npol, comp.alm_len_complex), dtype=self.complex_dtype)
                for ipol in range(self.npol):
                    eta2[ipol] = gaussian_random_alm(comp.lmax, comp.lmax, self.spin, 1)
                self.logger.info(f"RHS4 comp-{comp.longname}: {np.mean(np.abs(eta2)):.2e}")
                eta2_s.append(eta2)
        else:
            eta2_s = []
        return eta2_s


#TODO: overload of math functions such as dot products specifically tailored for some components such as point sources

    def solve_CG(self, LHS: Callable, RHS: list[Component], x0 = None, M = None,
                 x_true = None) -> list[Component]:
        """ Solves the equation Ax=b for x given A (LHS) and b (RHS) using CG from the pixell package.
            Assumes that both x and b are in alm space.

            Args:
                LHS (callable): A function/callable taking x as argument and returning Ax.
                RHS (list): A list of alm-vectors representing the right-hand-side of the equation.
                x0 (list): Initial guess for x, as list of alm-vectors for each component (optional).
                M (callable): Preconditioner for the CG solver. A callable which approximates A^-1 (optional).
                x_true (np.array): True solution for x, in order to print the true error (optional).
            Returns:
                m_bestfit (list): The resulting best-fit solution, for the component held by this
                                  rank, as a list of alm-vectors for each component.
        """
        max_iter = self.params.CG_max_iter_pol if self.det_map.pol else self.params.CG_max_iter

        logger = logging.getLogger(__name__)
        checkpoint_interval = 5
        master = self.CompSep_comm.Get_rank() == 0
        mycomp = self.CompSep_comm.Get_rank()

        mydot = complist_dot
        
        if M is None:
            CG_solver = distributed_CG(LHS, RHS, master, dot=mydot, x0=x0, destroy_b=True)
        else:
            CG_solver = distributed_CG(LHS, RHS, master, dot=mydot, x0=x0, M=M, destroy_b=True)
        self.CG_residuals = np.zeros((max_iter))
        if x_true is not None:
            if master:
                x_true_list = [x_true[i:j].reshape((1,-1)) for i,j in zip(self.alm_start_idx_per_comp[:-1], self.alm_start_idx_per_comp[1:])]
                self.xtrue_A_xtrue = x_true.dot(np.concatenate(LHS(x_true_list), axis=-1).flatten())
            else:
                LHS([])
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
                    logger.info(f"CG iter {iter:3d} - Residual {np.mean(self.CG_residuals[iter-checkpoint_interval:iter]):.6e} ({(time.time() - t0)/checkpoint_interval:.2f}s/iter)")
                    t0 = time.time()
                    if x_true is not None:
                        x_vec = np.concatenate(CG_solver.x, axis=-1)
                        CG_errors_true = np.linalg.norm(x_vec - x_true)/np.linalg.norm(x_true)
                        A_residual = LHS([x - y for x,y in zip(CG_solver.x, x_true_list)])
                        A_residual = np.concatenate(A_residual, axis=-1)
                        CG_Anorm_error = ((x_vec - x_true).flatten()).dot(A_residual.flatten())
                        logger.info(f"CG iter {iter:3d} - True A-norm error: {CG_Anorm_error:.3e}")  # A-norm error is only defined for the full vector.
                        logger.info(f"CG iter {iter:3d} - {self.comp_list[mycomp].longname} - True L2 error: {CG_errors_true:.3e}")  # We can print the individual component L2 errors.
                else:
                    if x_true is not None:
                        LHS([])  # Matching LHS call for the calculation of LHS(CG_solver.x-x_true).
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
        if master:
            s_bestfit_list = CG_solver.x
            for icomp in range(self.ncomp):
                # if self.params.CG_real_alm_mode:
                #     s_bestfit_list[icomp] = alm_real2complex(s_bestfit_list[icomp],
                #                                              self.lmax_per_comp[icomp])
                for ipol in range(self.npol):
                    almxfl(s_bestfit_list[icomp][ipol], self.per_comp_P_smooth_sqrt[icomp], inplace=True)
        else:
            s_bestfit_list = [np.zeros((self.npol, self.alm_len_percomp_complex[icomp]),
                                       dtype=self.alm_dtype) for icomp in range(self.ncomp)]
        for icomp in range(self.ncomp):
            self.CompSep_comm.Bcast(s_bestfit_list[icomp], root=0)
        
        return s_bestfit_list



    def solve(self, seed=None) -> list[DiffuseComponent]:
        if seed is not None:
            np.random.seed(seed)
        RHS1 = self.calc_RHS_mean()
        # RHS2 = self.calc_RHS_fluct()
        # RHS3 = self.calc_RHS_prior_mean()
        # RHS4 = self.calc_RHS_prior_fluct()
        # RHS = [_R1 + _R2 + _R3 + _R4 for _R1, _R2, _R3, _R4 in zip(RHS1, RHS2, RHS3, RHS4)]
        RHS = RHS1
        del(self.map_sky)

        # Initialize the precondidioner class, which is in the module "solvers.preconditioners", and has a name specified by self.params.compsep.preconditioner.
        precond = getattr(preconditioners, self.params.compsep.preconditioner)(self)

        if self.params.compsep.dense_matrix_debug_mode:  # For testing preconditioner with a true solution as reference, first solving for exact solution with dense matrix math.
            M_A_matrix = lambda a : self.apply_LHS_matrix(precond(a))

            # Testing the initial LHS (A) matrix
            dense_matrix = DenseMatrix(self, self.apply_LHS_matrix, "A")
            dense_matrix.test_matrix_hermitian()
            dense_matrix.print_sing_vals()
            dense_matrix.test_matrix_eigenvalues()
            dense_matrix.print_matrix_diag()

            # Testing the preconditioning matrix (M) alone
            dense_matrix = DenseMatrix(self, precond, "M")
            dense_matrix.test_matrix_hermitian()  # Preconditioner matrix needs to be Hermitian.
            dense_matrix.test_matrix_eigenvalues()  # and to have positive and real eigenvalues

            # Testing the combined preconditioned system (MA)  (this combined matrix will generally not be Hermitian positive-definite).
            dense_matrix = DenseMatrix(self, M_A_matrix, "MA")
            x_true = dense_matrix.solve_by_inversion(precond(RHS))
            dense_matrix.print_sing_vals()  # Check how much singular values (condition number) of preconditioned system improved.
            dense_matrix.print_matrix_diag()


        sol_list = self.solve_CG(self.apply_LHS_matrix, RHS, M=precond, x_true=x_true if self.params.compsep.dense_matrix_debug_mode else None)
        for icomp in range(self.ncomp):
            if self.spin == 0:
                self.comp_list[icomp].component_alms_intensity = sol_list[icomp]
            elif self.spin == 2:
                self.comp_list[icomp].component_alms_polarization = sol_list[icomp]

        return self.comp_list
