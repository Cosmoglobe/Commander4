"""Point-source components: amplitudes living on discrete sky positions rather than in alms.

A point-source component stores one amplitude per source and paints it onto the sky through the
band beam, so its map/alm operations are unrelated to the diffuse machinery. The two numba kernels
at the top do that painting and its adjoint.
"""
import healpy as hp
import numpy as np
import pysm3.units as pysm3u
from numba import njit
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.data_models.band import Band
from commander4.sky.component import Component
from commander4.sky.beams import gauss_beam, get_gauss_beam_radius
from commander4.math_utils.sht import map_to_alm, map_to_alm_adjoint


@njit(fastmath=True)
def _numba_proj2map(skymap, pix_disc_idx_list, beam_disc_val_list, amps, sed_s=None):
    for src_i in range(len(pix_disc_idx_list)):
        skymap[pix_disc_idx_list[src_i]] += beam_disc_val_list[src_i] * amps[src_i]\
            * (sed_s[src_i] if sed_s is not None else 1)
    return skymap

@njit(fastmath=True, parallel=True)
def _numba_eval_from_map(map, pix_disc_idx_list, beam_disc_val_list, amps, sed_s=None):
    for src_i in range(len(pix_disc_idx_list)):
            amps[src_i] = np.sum(map[pix_disc_idx_list[src_i]] * beam_disc_val_list[src_i])\
                * (sed_s[src_i] if sed_s is not None else 1)
    return amps


class PointSourcesComponent(Component):
    """Base class for components whose amplitudes live on discrete sky positions.

    Intensity only: point sources are unpolarized in the current model.
    """

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
    """Radio point sources, each a power law with its own spectral index.

    Positions, flux amplitudes and spectral indices all come from the `template_path` table.
    """

    default_shortname = "radsources"
    # `alpha_arr` is deliberately absent from the SED parameter names: it is one spectral index per
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
        self.nu_ref = comp_params.nu_0  # Reference frequency (GHz)
        # Flux density to brightness temperature, evaluated once at `nu_ref` (C3's `getScale`).
        # This factor must NOT be re-evaluated at the band frequency: `get_sed` already carries the
        # full frequency dependence, including the nu^-2 of the RJ relation (hence its `alpha - 2`).
        # Applying the conversion at `nu` as well would scale the sources as nu^(alpha-4).
        self.mJysr_to_uKRJ = (pysm3u.mJy / pysm3u.steradian).to(pysm3u.uK_RJ,
                            equivalencies=pysm3u.cmb_equivalencies(self.nu_ref*pysm3u.GHz))
        ps_bunch = self.read_dat_to_bunch(comp_params.template_path)
        self._data = np.array(ps_bunch['I(mJy)'], dtype=np.float32).reshape((1,-1))  # Amplitudes
        self.alpha_arr = np.array(ps_bunch['alpha_I'], dtype=np.float32)  # Spectral indices
        self.lonlat_arr = np.array((ps_bunch['Glon(deg)'], ps_bunch['Glat(deg)']),
                                   dtype=np.float32).T  # Per-source coordinates
        del ps_bunch
        if self.alpha_arr.shape[0] != self.lonlat_arr.shape[0]\
        or self.alpha_arr.shape[0] != self._data.shape[1]:
            raise RuntimeError("Point Source tabulated data must be uniform in length.")

        # Beam discs, filled lazily by `compute_pix_beams` since they depend on the band.
        self.pix_disc_idx_list = None   # Per-source pixel indices of the disc around the source
        self.beam_disc_val_list = None  # Per-source beam values, one per pixel in that disc
        self.band_eval_nside = None     # nside the discs above were computed at
        self.band_fwhm_r = None         # Beam FWHM (radians) the discs above were computed at

    def read_dat_to_bunch(self, file_path):
        """ Reads a .dat point source raw table and stores it in a Bunch object which is returned.

        The last comment line before the data is taken as the column header.
        """
        rows = []
        head = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.split()[0].startswith("#"):
                    head.append(line.split()[1:])
                else:
                    rows.append(line.split())
        head = head[-1]
        rows = np.array(rows)       
        return Bunch(zip(head,[rows[:,i] for i in range(rows.shape[1])]))

    def compute_pix_beams(self, band_fwhm_r, band_nside, recompute=False):
        """ Computes the map-space beam values around every point source, for the given band nside
            and FWHM, updating pix_disc_idx_list, beam_disc_val_list and the band members in place.

        Each source's beam is normalized so that it integrates to unity over the pixels it covers,
        which keeps the source's total flux right at any resolution. Without it a beam narrower than
        a pixel loses essentially all of its flux, because `gauss_beam` is then evaluated only at
        pixel centres that all sit far out in the beam's tail.

        If `recompute` is True the computation always runs; otherwise it runs only when the beam
        lists are uninitialized or the band specs have changed.
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
            pixel_area = hp.nside2pixarea(band_nside)
            # Compute the beam disc for each source; these stay fixed until the band changes.
            for i in range(self.lonlat_arr.shape[0]):
                # `inclusive` keeps every pixel the disc touches, so a disc smaller than a pixel
                # still selects the pixels around the source rather than none at all.
                disc_pix_i_s = hp.query_disc(self.band_eval_nside, hp.ang2vec(self.lonlat_arr[i,0],
                        self.lonlat_arr[i,1], lonlat=True), get_gauss_beam_radius(self.band_fwhm_r),
                        inclusive=True)
                beam_disc = gauss_beam(hp.rotator.angdist(self.lonlat_arr[i,:],
                            hp.pix2ang(self.band_eval_nside, disc_pix_i_s, lonlat=True),
                            lonlat=True), self.band_fwhm_r)
                beam_integral = beam_disc.sum()*pixel_area
                if beam_integral > 0.0:
                    beam_disc = beam_disc/beam_integral
                else:
                    # Beam far below the pixel scale: every pixel centre sits in the far tail, so
                    # put the whole source in the single pixel containing it (the delta limit).
                    disc_pix_i_s = np.array([hp.ang2pix(self.band_eval_nside,
                            self.lonlat_arr[i,0], self.lonlat_arr[i,1], lonlat=True)])
                    beam_disc = np.array([1.0/pixel_area])
                self.pix_disc_idx_list.append(disc_pix_i_s)
                self.beam_disc_val_list.append(beam_disc)
            return True
        else:
            return False

    def get_sed(self, nu:float):
        """ Returns one SED value per source, evaluated at `nu` relative to `nu_ref` (both GHz).

        The -2 in the exponent converts the tabulated flux-density index to brightness temperature.
        """
        return (nu/self.nu_ref)**(self.alpha_arr - 2)

    def get_sky(self, nu:float, nside:int, fwhm:float=0.0):
        """ Returns the sky at frequency `nu` (GHz) from the point sources, at a certain `nside`,
            observed through a Gaussian beam of `fwhm` **radians**.

        The beam is applied by painting each source through it, so unlike a `DiffuseComponent` there
        is no separate smoothing step: `fwhm` selects the beam the sources are painted with. The
        unit matches `DiffuseComponent.get_sky`, because `SkyModel.get_sky_at_nu` calls both through
        the same interface and passes the band's `fwhm_rad`.
        """
        self.compute_pix_beams(fwhm, nside)
        map = np.zeros((1, hp.nside2npix(nside)),
                       dtype=np.float64 if self.double_prec else np.float32)
        _numba_proj2map(map[0,:], self.pix_disc_idx_list, self.beam_disc_val_list,
                        self._data[0,:], self.get_sed(nu))
        map *= self.mJysr_to_uKRJ
        return map

    def get_component_map(self, nside:int, fwhm:float=0.0):
        """ This component's *amplitude* map at a certain `nside` and `fwhm` (radians), in uK_RJ at
            `nu_ref`.

        No SED is applied, so the result is frequency-independent, which is why the mJy/sr to uK_RJ
        conversion is evaluated at the component's own reference frequency. That matches what a
        `DiffuseComponent` returns here: its alms are likewise stored in uK_RJ referenced to
        `nu_ref`, with `get_sed(nu)` carrying the amplitude to any other frequency.
        """
        self.compute_pix_beams(fwhm, nside)
        map = np.zeros((1, hp.nside2npix(nside)),
                       dtype=np.float64 if self.double_prec else np.float32)
        _numba_proj2map(map[0,:], self.pix_disc_idx_list, self.beam_disc_val_list, self._data[0,:])
        map *= self.mJysr_to_uKRJ
        return map
    
    def _project_to_band_map(self, map:NDArray, nu:float):
        """ Computes the point source contribution in uK_RJ for the band's frequency and beam, and
            sums it into `map`, which must have shape [1, npix].
        """
        _numba_proj2map(map[0,:], self.pix_disc_idx_list, self.beam_disc_val_list,
                        self._data[0,:], sed_s = self.get_sed(nu))
        map *= self.mJysr_to_uKRJ
    
    def _eval_from_band_map(self, map, nu):
        """ Computes the amplitude contribution from the local band to each point source, given
            `map`, which must have shape [1, npix].

        All the contributions will be summed to the total proper amplitudes by the master node.
        """
        _numba_eval_from_map(map[0,:], self.pix_disc_idx_list,
                             self.beam_disc_val_list, self._data[0,:], sed_s = self.get_sed(nu))
        self._data *= self.mJysr_to_uKRJ

    def project_comp_to_band(self, band:Band, nthreads: int = 1):
        """ Project the point sources contribution to the given band in-place, summing it into the
            alm array of the band object.

        NB: this function does not include the beam smoothing.
        """
        if band.is_pol:
            raise ValueError("Point-source components can only be projected to intensity bands.")
        band_fwhm_r, band_nside = np.deg2rad(band.fwhm/60.0), band.nside
        self.compute_pix_beams(band_fwhm_r, band_nside)  # No-op unless the band has changed.

        # The point-source equivalent of: M Y a
        ps_map = np.zeros((1,hp.nside2npix(band_nside)),   # Empty band map
            dtype=(np.float32 if self.global_params.float_precision == "single" else np.float64))
        self._project_to_band_map(ps_map, band.nu)

        # Y^-1 M Y a
        map_to_alm(ps_map, band_nside, band.lmax, spin=0, out=band.alms, acc=True,
                   nthreads=nthreads)
        
        return band.alms

    def eval_comp_from_band(self, band:Band, nthreads: int = 1):
        """ Evaluate the band's alm contribution to the point sources' amplitudes, storing it in
            `_data` as well as returning it.

        All the contributions will be summed to the total proper amplitudes when reducing on the
        master node.

        NB: this function does not include the beam smoothing.
        """
        if band.is_pol:
            raise ValueError("Point-source components can only be evaluated from intensity bands.")
        band_fwhm_r, band_nside = np.deg2rad(band.fwhm/60.0), band.nside
        self.compute_pix_beams(band_fwhm_r, band_nside)  # No-op unless the band has changed.

        # Y^-1^T B^T a
        band_map = map_to_alm_adjoint(band.alms, band.nside, band.lmax, spin=0, out=None,
                                      nthreads=nthreads)

        # M^T Y^-1 B^T a
        self._eval_from_band_map(band_map, band.nu)  # Updates self._data in place.

        return self._data

    def apply_Cl_prior_sqrt(self, alms):
        """ A dummy for point sources: the input is simply returned.

        NB: point-source support in compsep is unfinished and awaiting review. In particular these
        components expose no ``alms`` attribute, so the ``comp.apply_Cl_prior_sqrt(comp.alms)``
        calls in the CG solver would not reach this dummy for a point source in the first place.
        """
        return alms

    def apply_Cl_prior_inv_sqrt(self, alms):
        """ Dummy matching `apply_Cl_prior_sqrt`: point sources carry no C_l prior to invert.
        """
        return alms

    def __repr__(self):
        return f"Radio Source \n amps: {self._data}"
