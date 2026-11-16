"""Point-source components: amplitudes living on discrete sky positions rather than in alms.

A point-source component stores one amplitude per source and paints it onto the sky through the
band beam, so its map/alm operations are unrelated to the diffuse machinery.
"""
import h5py
import healpy as hp
import numpy as np
import pysm3.units as pysm3u
from pixell.bunch import Bunch

from commander4.data_models.band import Band
from commander4.sky.component import Component
from commander4.sky.beams import gauss_beam, get_gauss_beam_radius


class PointSourcesComponent(Component):
    """Base class for components whose amplitudes live on discrete sky positions.

    Intensity only: point sources are unpolarized in the current model.
    """

    DEFAULTS = {**Component.DEFAULTS, "shortname": "pnt"}
    legal_pols: tuple[str, ...] = ("I",)

    def write_chain(self, group: h5py.Group, band_frequencies: dict[str, float]) -> None:
        """Write the shared component sample (see `Component.write_chain`), plus the amplitudes.

        One amplitude per source. Positions and spectral indices come from the source catalogue,
        so they are not written.
        """
        super().write_chain(group, band_frequencies)
        group["source_amps"] = self.amps

    def read_chain(self, group: h5py.Group, restore_beta: bool) -> None:
        """Restore the shared sampled state (see `Component.read_chain`), plus the amplitudes.

        The catalogue must list the same sources, in the same order, as the run that wrote the
        chain.
        """
        super().read_chain(group, restore_beta)
        self.amps[:] = group["source_amps"][()]

class RadioSources(PointSourcesComponent):
    """Radio point sources, each a power law with its own spectral index.

    Positions, flux amplitudes and spectral indices all come from the `template_path` table.
    """

    # `template_path` is the table of the sources, and `nu_0` the reference frequency in GHz
    # (`self.nu_ref`).
    DEFAULTS = {**PointSourcesComponent.DEFAULTS, "shortname": "rdpnt",
                "template_path": None, "nu_0": None}
    amplitude_unit = "mJy"
    # `alpha_arr` is deliberately absent from the SED parameter names: it is one spectral index per
    # source, read from `template_path` alongside the source positions and amplitudes, so it belongs
    # with the template rather than in every iteration of the chain.
    sed_param_names = ("nu_ref",)

    def __init__(self, params: Bunch, comp_name: str, pol: str):
        """The component `comp_name`, with its sources read from the `template_path` table."""
        super().__init__(params, comp_name, pol)
        self.nu_ref = self.cfg.nu_0  # Reference frequency (GHz)
        # Flux density to brightness temperature, evaluated once at `nu_ref` (C3's `getScale`).
        # This factor must NOT be re-evaluated at the band frequency: `get_sed` already carries the
        # full frequency dependence, including the nu^-2 of the RJ relation (hence its `alpha - 2`).
        # Applying the conversion at `nu` as well would scale the sources as nu^(alpha-4).
        self.mJysr_to_uKRJ = (pysm3u.mJy / pysm3u.steradian).to(pysm3u.uK_RJ,
                            equivalencies=pysm3u.cmb_equivalencies(self.nu_ref*pysm3u.GHz))
        ps_bunch = self.read_dat_to_bunch(self.cfg.template_path)
        self.amps = np.array(ps_bunch['I(mJy)'], dtype=np.float32).reshape((1,-1))  # Amplitudes
        self.alpha_arr = np.array(ps_bunch['alpha_I'], dtype=np.float32)  # Spectral indices
        self.lonlat_arr = np.array((ps_bunch['Glon(deg)'], ps_bunch['Glat(deg)']),
                                   dtype=np.float32).T  # Per-source coordinates
        del ps_bunch
        if self.alpha_arr.shape[0] != self.lonlat_arr.shape[0]\
        or self.alpha_arr.shape[0] != self.amps.shape[1]:
            raise RuntimeError("Point Source tabulated data must be uniform in length.")

        # Ragged per-source beam discs, cached because they depend only on the band resolution and
        # beam. Keeping one array per source matches the natural representation from query_disc.
        self.pix_disc_idx_list = None
        self.beam_disc_val_list = None
        self.band_eval_nside = None  # nside the cached discs were computed at.
        self.band_fwhm_r = None      # Beam FWHM (radians) used for the cached discs.

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
            and FWHM, updating the per-source beam arrays and the band members in place.

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
            self.band_fwhm_r = band_fwhm_r
            self.band_eval_nside = band_nside
            pixel_area = hp.nside2pixarea(band_nside)
            self.pix_disc_idx_list = []
            self.beam_disc_val_list = []
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
                self.pix_disc_idx_list.append(np.asarray(disc_pix_i_s, dtype=np.int64))
                self.beam_disc_val_list.append(np.asarray(beam_disc, dtype=np.float64))
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

        The beam is applied by painting each source through it: `fwhm` selects the beam the sources
        are painted with. `SkyModel.get_sky_at_nu` adds this map, in pixel space, to the diffuse
        components it builds through `comp2band` (as C3's `compute_residual` adds point
        sources), so the unit is uK_RJ at `nu`, like theirs.
        """
        self.compute_pix_beams(fwhm, nside)
        map = np.zeros((1, hp.nside2npix(nside)),
                       dtype=np.float64 if self.double_prec else np.float32)
        scales = self.amps[0]*self.get_sed(nu)*self.mJysr_to_uKRJ
        for pixels, beam, scale in zip(self.pix_disc_idx_list, self.beam_disc_val_list, scales):
            map[0, pixels] += beam*scale  # Overlapping sources add up: the loop is sequential.
        return map

    def get_component_map(self, nside:int, fwhm:float=0.0):
        """ This component's *amplitude* map at a certain `nside` and `fwhm` (radians), in uK_RJ at
            `nu_ref`.

        This is the sky at `nu_ref`, where the SED is 1 for every source. That matches what a
        `DiffuseComponent` returns here: its alms are likewise stored in uK_RJ referenced to
        `nu_ref`, with `get_sed(nu)` carrying the amplitude to any other frequency.
        """
        return self.get_sky(self.nu_ref, nside, fwhm)
    
    def comp2band(self, band:Band, nthreads: int = 1):
        """ Add the point sources to the band's pixel map `band.map`.

        Each source is painted through the band beam, in uK_RJ at the band frequency. The sources
        are added in pixel space, as in C3, so they are not cut at the band lmax. `band.map` is
        allocated here if no component has written to it yet.
        """
        if band.pol == "QU":
            raise ValueError("Point-source components can only be projected to intensity bands.")
        band_fwhm_r, band_nside = np.deg2rad(band.fwhm/60.0), band.nside
        self.compute_pix_beams(band_fwhm_r, band_nside)  # No-op unless the band has changed.
        if band.map is None:
            band.map = np.zeros((1, hp.nside2npix(band_nside)), dtype=band.alms.real.dtype)
        # The unit factor goes into the per-source scale, not onto the map, because the map may
        # already hold other components.
        scales = self.amps[0]*self.get_sed(band.nu)*self.mJysr_to_uKRJ
        for pixels, beam, scale in zip(self.pix_disc_idx_list, self.beam_disc_val_list, scales):
            band.map[0, pixels] += beam*scale  # Overlapping sources add up: the loop is sequential.
        return band.map

    def band2comp(self, band:Band, nthreads: int = 1):
        """ The transpose of `comp2band`: reads the band's pixel map into the amplitudes, stored in
        `amps` and returned.

        All the contributions will be summed to the total amplitudes when reducing on the master.
        """
        if band.pol == "QU":
            raise ValueError("Point-source components can only be evaluated from intensity bands.")
        band_fwhm_r, band_nside = np.deg2rad(band.fwhm/60.0), band.nside
        self.compute_pix_beams(band_fwhm_r, band_nside)  # No-op unless the band has changed.
        scales = self.get_sed(band.nu)*self.mJysr_to_uKRJ
        for i, (pixels, beam) in enumerate(zip(self.pix_disc_idx_list, self.beam_disc_val_list)):
            self.amps[0, i] = np.dot(band.map[0, pixels], beam)*scales[i]
        return self.amps

    def apply_Cl_prior_sqrt(self, alms):
        """ A dummy for point sources (S = 1): the input is simply returned.

        Point sources have no amplitude prior (`Cl_prior_amplitude` is None), as in C3, where the
        per-source prior `P_x` is switched off. Point-source support in the CG is unfinished; see
        C4_issues.md, K4.
        """
        return alms

    def apply_Cl_prior_inv_sqrt(self, alms):
        """ Dummy matching `apply_Cl_prior_sqrt`: point sources carry no C_l prior to invert.
        """
        return alms

    def __repr__(self):
        return f"Radio Source \n amps: {self.amps}"
