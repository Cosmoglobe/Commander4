"""Point-source components: amplitudes living on discrete sky positions rather than in alms.

A point-source component stores one amplitude per source and paints it onto the sky through the
band beam, so its map/alm operations are unrelated to the diffuse machinery. The two numba kernels
at the top do that painting and its adjoint.
"""
import healpy as hp
import numpy as np
import pysm3.units as pysm3u
from numba import njit, prange
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.data_models.band import Band
from commander4.sky.component import Component
from commander4.sky.beams import gauss_beam, get_gauss_beam_radius
from commander4.math_utils.sht import map_to_alm, map_to_alm_adjoint


# Painting source amplitudes onto a map, and its adjoint. Numba because both loop over every
# source's beam disc; `prange` parallelizes the projection across sources.
@njit(fastmath=True, parallel=True)
def _numba_proj2map(map, pix_disc_idx_list, beam_disc_val_list, amps, sed_s=None):
    for src_i in prange(len(pix_disc_idx_list)):
        map[pix_disc_idx_list[src_i]] += beam_disc_val_list[src_i] * amps[src_i]\
            * (sed_s[src_i] if sed_s is not None else 1)
    return map

@njit(fastmath=True, parallel=True)
def _numba_eval_from_map(map, pix_disc_idx_list, beam_disc_val_list, amps, sed_s=None):
    for src_i in range(len(pix_disc_idx_list)):
            amps[src_i] = np.sum(map[pix_disc_idx_list[src_i]] * beam_disc_val_list[src_i])\
                * (sed_s[src_i] if sed_s is not None else 1)
    return amps


# NON DIFFUSE COMPONENTS

class PointSourcesComponent(Component):
    default_shortname = "pscomp"
    legal_pols: tuple[str, ...] = ("I",)

    def __init__(self, comp_params: Bunch, global_params: Bunch, *,
                 shortname: str | None = None, comp_name: str | None = None,
                 eval_pol: str | None = None, allocate_empty_alms: bool = False):
        super().__init__(
            comp_params,
            global_params,
            shortname=shortname,
            comp_name=comp_name,
            eval_pol="I" if eval_pol is None else eval_pol,
            allocate_empty_alms=allocate_empty_alms,
        )
        self.defined_pol = "I"
        self.eval_pol = "I"

    @property
    def is_pol(self) -> bool:
        return False
    
    @property
    def npol(self) -> int:
        return 1

class RadioSources(PointSourcesComponent):
    default_shortname = "radsources"
    # `alpha_arr` is deliberately absent from the sed parmaeter names: it is one spectral index per
    # source, read from `template_path` alongside the source positions and amplitudes, so it belongs
    # with the template rather than in every iteration of the chain.
    sed_param_names = ("nu_ref",)

    def __init__(self, comp_params: Bunch, global_params: Bunch, *,
                 shortname: str | None = None, comp_name: str | None = None,
                 eval_pol: str | None = None, allocate_empty_alms: bool = False):
        super().__init__(
            comp_params,
            global_params,
            shortname=shortname,
            comp_name=comp_name,
            eval_pol=eval_pol,
            allocate_empty_alms=allocate_empty_alms,
        )
        #reference frequency
        self.nu_ref = comp_params.nu_0
        #tabulated data
        ps_bunch = self.read_dat_to_bunch(comp_params.template_path)
        #per-source amplitudes
        self._data = np.array(ps_bunch['I(mJy)'], dtype=np.float32).reshape((1,-1))
        #per-source spectral indexes
        self.alpha_arr = np.array(ps_bunch['alpha_I'], dtype=np.float32)
        self.lonlat_arr = np.array((ps_bunch['Glon(deg)'], ps_bunch['Glat(deg)']),
                                   dtype=np.float32).T #per-source list of coordinates
        del ps_bunch
        if self.alpha_arr.shape[0] != self.lonlat_arr.shape[0]\
        or self.alpha_arr.shape[0] != self._data.shape[1]:
            raise RuntimeError("Point Source tabulated data must be uniform in length.")

        self.pix_disc_idx_list = None   #per-source list of indexes of the pixel forming the disc
        self.beam_disc_val_list = None    #per-source list of beam values, for each pix_i_s
        self.band_eval_nside = None     #nside used for the computation of pix and beam discs
        self.band_fwhm_r = None         #they depend on the bands

    def read_dat_to_bunch(self, file_path):
        """
        Reads a .dat point source raw table and stores it in a Bunch object which is returned.
        """
        rows = []
        head = []
        with open(file_path, 'r') as file:
            # i=0
            for line in file:
                if line.split()[0].startswith("#"):
                    head.append(line.split()[1:])
                else:
                    rows.append(line.split())
        head = head[-1]
        rows = np.array(rows)       
        return Bunch(zip(head,[rows[:,i] for i in range(rows.shape[1])]))

    def compute_pix_beams(self, band_fwhm_r, band_nside, recompute=False):
        """
        Computes the beams values, based on nside and fwhm of the band, in map space around all the
        point sources and updates in-place pix_disc_idx_list, beam_disc_val_list and fwhm_r members.

        If recompute is True will always perform the computation otherwise it does so only if the
        beam lists are not initialized or if the band specs have changed.
        """
        if band_fwhm_r != self.band_fwhm_r \
        or band_nside != self.band_eval_nside \
        or self.pix_disc_idx_list is None \
        or self.beam_disc_val_list is None \
        or recompute:
            self.pix_disc_idx_list = []
            self.beam_disc_val_list = []
            self.band_fwhm_r = band_fwhm_r
            self.band_eval_nside = band_nside
            #compute and load the beams, these will reamain untouched. The FWHM depends on the band.
            for i in range(self.lonlat_arr.shape[0]):
                disc_pix_i_s = hp.query_disc(self.band_eval_nside, hp.ang2vec(self.lonlat_arr[i,0],
                        self.lonlat_arr[i,1], lonlat=True), get_gauss_beam_radius(self.band_fwhm_r))
                self.pix_disc_idx_list.append(disc_pix_i_s)
                beam_disc = gauss_beam(hp.rotator.angdist(self.lonlat_arr[i,:],
                            hp.pix2ang(self.band_eval_nside, disc_pix_i_s, lonlat=True),
                            lonlat=True), self.band_fwhm_r)
                self.beam_disc_val_list.append(beam_disc)
            return True
        else:
            return False

    def get_sed(self, nu:float):
        """
        Returns a list of sed's, one per `alpha_list`, evaluated at `nu`, with ref frequency `nu_ref`.
        Freq. are in GHz.
        """
        return (nu/self.nu_ref)**(self.alpha_arr - 2)
    
    def get_sky(self, nu:float, nside:int, fwhm:float=0.0):
        """
        Returns the sky component given by the point sources at a certain `nu` with a certain
        `nside` and `fwhm` smoothing.
        """
        self.compute_pix_beams(np.deg2rad(fwhm/60), nside)
        map = np.zeros((1, hp.nside2npix(nside)),
                       dtype=np.float64 if self.double_prec else np.float32)
        mJysr_to_uKRJ = (pysm3u.mJy / pysm3u.steradian).to(pysm3u.uK_RJ,
                                            equivalencies=pysm3u.cmb_equivalencies(nu*pysm3u.GHz))
        sed_s = self.get_sed(nu)
        _numba_proj2map(map[0,:], self.pix_disc_idx_list, self.beam_disc_val_list,
                        self._data[0,:],sed_s)
        map*=mJysr_to_uKRJ
        if fwhm == 0.0:
            pass
        else:
            map[0,:] = hp.smoothing(map[0,:], np.deg2rad(fwhm/60))
        return map

    def get_component_map(self, nside:int, fwhm:float=0.0):
        """
        Returns the map of the point sources component with a certain `nside` and `fwhm` smoothing.
        """
        self.compute_pix_beams(np.deg2rad(fwhm/60), nside)
        map = np.zeros((1, hp.nside2npix(nside)),
                       dtype=np.float64 if self.double_prec else np.float32)
        mJysr_to_uKRJ = (pysm3u.mJy / pysm3u.steradian).to(pysm3u.uK_RJ,
                                        equivalencies=pysm3u.cmb_equivalencies(self.nu*pysm3u.GHz))
        _numba_proj2map(map[0,:], self.pix_disc_idx_list, self.beam_disc_val_list, self._data[0,:])
        map*=mJysr_to_uKRJ
        if fwhm == 0.0:
            pass
        else:
            map[0,:] = hp.smoothing(map[0,:], np.deg2rad(fwhm/60))
        return map
    
    def _project_to_band_map(self, map:NDArray, nu:float):
        """
        Computes the point source contribution in uK_RJ for band's frequency and beam,
        and sums it to `map`.

        the `map` array should have shape [1,npix].
        """
        mJysr_to_uKRJ = (pysm3u.mJy / pysm3u.steradian).to(pysm3u.uK_RJ,
                                            equivalencies=pysm3u.cmb_equivalencies(nu*pysm3u.GHz))
        sed_s = self.get_sed(nu)

        _numba_proj2map(map[0,:], self.pix_disc_idx_list, self.beam_disc_val_list,
                        self._data[0,:], sed_s = sed_s)
        map*=mJysr_to_uKRJ
    
    def _eval_from_band_map(self, map, nu):
        """
        Computes the amplitude contribution from the local band to each point source, given `map`.

        All the contributions will be summed to the total proper amplitudes by the master node.

        the `map` array should have shape [1,npix].
        """
        mJysr_to_uKRJ = (pysm3u.mJy / pysm3u.steradian).to(pysm3u.uK_RJ,
                                            equivalencies=pysm3u.cmb_equivalencies(nu*pysm3u.GHz))
        sed_s = self.get_sed(nu)
        _numba_eval_from_map(map[0,:], self.pix_disc_idx_list,
                             self.beam_disc_val_list, self._data[0,:], sed_s = sed_s)
        self._data *= mJysr_to_uKRJ

    def project_comp_to_band(self, band:Band, nthreads: int = 1):
        """
        Project the point sources contribution to the given band in-place,
        summing its contribution to the alm array of the band object.

        NB: this function does not include the beam smoothing.
        """
        assert not band.is_pol, "Point sources component can only be projected to intensity band alms"
        band_fwhm_r, band_nside = np.deg2rad(band.fwhm/60.0), band.nside
        #if not initialized or if band's characteristics changed, recompute the arrays.
        self.compute_pix_beams(band_fwhm_r, band_nside)

        # the point-source correspondent of: M Y a
        ps_map = np.zeros((1,hp.nside2npix(band_nside)),   #empty band map
            dtype=(np.float32 if self.global_params.float_precision == "single" else np.float64))
        self._project_to_band_map(ps_map, band.nu)

        # Y^-1 M Y a
        map_to_alm(ps_map, band_nside, band.lmax, spin=0, out=band.alms, acc=True,
                   nthreads=nthreads)
        
        return band.alms

    def eval_comp_from_band(self, band:Band, nthreads: int = 1):
        """
        Evaluate the band's alm contribution to the point sources' amplitudes and stores it in the
        amp_s member, as well as returning it.

        All the contributions will be summed to the total proper amplitudes when reducing on the
        master node.

        NB: this function does not include the beam smoothing.
        """

        assert not band.is_pol, "Point sources comps can only be evaluated from intensity band alms"
        band_fwhm_r, band_nside = np.deg2rad(band.fwhm/60.0), band.nside
        #if not initialized or if band's characteristics changed, recompute the arrays.
        self.compute_pix_beams(band_fwhm_r, band_nside)

        # Y^-1^T B^T a
        band_map = map_to_alm_adjoint(band.alms, band.nside, band.lmax, spin=0, out=None,
                                      nthreads=nthreads)

        # M^T Y^-1 B^T a
        self._eval_from_band_map(band_map, band.nu) #updates amp_s in-place

        #return band_alm's contribution to point sources amplitudes
        return self._data
    
    def apply_Cl_prior_sqrt(self, alms):
        """
        In the case of point sources this is just a dummy, the input is simply returned.

        NB: point-source support in compsep is unfinished and awaiting review. In particular these
        components expose no ``alms`` attribute, so the ``comp.apply_Cl_prior_sqrt(comp.alms)``
        calls in the CG solver would not reach this dummy for a point source in the first place.
        """
        return alms

    def apply_Cl_prior_inv_sqrt(self, alms):
        """
        Dummy matching `apply_Cl_prior_sqrt`: point sources carry no C_l prior to invert.
        """
        return alms

    def __repr__(self):
        return f"Radio Source \n amps: {self._data}"
