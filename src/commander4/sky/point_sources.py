"""Point-source components: one amplitude per source, painted onto the sky through the band beam.
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
    """Base class of components with one amplitude per source. Intensity only."""

    legal_pols: tuple[str, ...] = ("I",)

    def write_chain(self, group: h5py.Group, band_frequencies: dict[str, float]) -> None:
        """As `Component.write_chain`, plus the amplitudes. Positions and spectral indices come
        from the catalogue, so they are not written."""
        super().write_chain(group, band_frequencies)
        group["source_amps"] = self.amps

    def read_chain(self, group: h5py.Group, restore_beta: bool) -> None:
        """As `Component.read_chain`, plus the amplitudes. The catalogue must list the same
        sources, in the same order, as the chain."""
        super().read_chain(group, restore_beta)
        self.amps[:] = group["source_amps"][()]


class RadioSources(PointSourcesComponent):
    """Radio sources, each a power law with its own spectral index.

    Positions, amplitudes (mJy at `nu_ref`, in GHz) and spectral indices come from the
    `template_path` table.
    """

    DEFAULTS = {**PointSourcesComponent.DEFAULTS, "template_path": None, "nu_ref": None}
    amplitude_unit = "mJy"
    # The spectral indices `alpha_arr` come from the table, so the chain does not store them.
    sed_param_names = ("nu_ref",)

    def __init__(self, params: Bunch, comp_name: str, pol: str):
        """The component `comp_name`, with its sources read from the `template_path` table."""
        super().__init__(params, comp_name, pol)
        # Flux density to RJ temperature at `nu_ref` only: `get_sed` already holds the full
        # frequency dependence, including the nu^-2 of the RJ relation. Converting at the band
        # frequency as well would scale the sources as nu^(alpha-4).
        self.mJysr_to_uKRJ = (pysm3u.mJy / pysm3u.steradian).to(pysm3u.uK_RJ,
                            equivalencies=pysm3u.cmb_equivalencies(self.nu_ref*pysm3u.GHz))
        ps_bunch = self.read_dat_to_bunch(self.cfg.template_path)
        self.amps = np.array([ps_bunch['I(mJy)']],
                             dtype=np.float64 if self.double_precision else np.float32)
        self.alpha_arr = np.array(ps_bunch['alpha_I'], dtype=np.float32)
        self.lonlat_arr = np.array((ps_bunch['Glon(deg)'], ps_bunch['Glat(deg)']),
                                   dtype=np.float32).T
        del ps_bunch
        if self.alpha_arr.shape[0] != self.lonlat_arr.shape[0]\
        or self.alpha_arr.shape[0] != self.amps.shape[1]:
            raise RuntimeError("Point Source tabulated data must be uniform in length.")

        # The pixels and beam values around each source, cached for the band nside and beam they
        # were computed for (see `compute_pix_beams`).
        self.pix_disc_idx_list = None
        self.beam_disc_val_list = None
        self.band_eval_nside = None
        self.band_fwhm_rad = None

    def read_dat_to_bunch(self, file_path):
        """Read a .dat source table into a Bunch of columns. The last comment line is the header.
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

    def compute_pix_beams(self, band_fwhm_rad, band_nside):
        """Compute the beam values in the pixels around each source, for a band of `band_nside`
        and `band_fwhm_rad`. Does nothing if they are already cached for that band.

        Each beam is normalized to integrate to 1 over its pixels, so a source keeps its flux at
        any resolution.
        """
        if band_fwhm_rad != self.band_fwhm_rad \
        or band_nside != self.band_eval_nside \
        or self.pix_disc_idx_list is None \
        or self.beam_disc_val_list is None:
            self.band_fwhm_rad = band_fwhm_rad
            self.band_eval_nside = band_nside
            pixel_area = hp.nside2pixarea(band_nside)
            self.pix_disc_idx_list = []
            self.beam_disc_val_list = []
            for i in range(self.lonlat_arr.shape[0]):
                beam_integral = 0.0
                if self.band_fwhm_rad > 0.0:  # A zero beam would divide by zero in `gauss_beam`.
                    # `inclusive` keeps every pixel the disc touches, so a disc smaller than a pixel
                    # still selects the pixels around the source rather than none at all.
                    disc_pix_i_s = hp.query_disc(self.band_eval_nside,
                            hp.ang2vec(self.lonlat_arr[i,0], self.lonlat_arr[i,1], lonlat=True),
                            get_gauss_beam_radius(self.band_fwhm_rad), inclusive=True)
                    beam_disc = gauss_beam(hp.rotator.angdist(self.lonlat_arr[i,:],
                                hp.pix2ang(self.band_eval_nside, disc_pix_i_s, lonlat=True),
                                lonlat=True), self.band_fwhm_rad)
                    beam_integral = beam_disc.sum()*pixel_area
                # Without the normalization, a beam narrower than a pixel would lose most of its
                # flux, as all pixel centres sit in its far tail.
                if beam_integral > 0.0:
                    beam_disc = beam_disc/beam_integral
                else:
                    # A zero beam, or one far below the pixel scale (every pixel centre sits in the
                    # far tail): put the whole source in the pixel containing it (the delta limit).
                    disc_pix_i_s = np.array([hp.ang2pix(self.band_eval_nside,
                            self.lonlat_arr[i,0], self.lonlat_arr[i,1], lonlat=True)])
                    beam_disc = np.array([1.0/pixel_area])
                self.pix_disc_idx_list.append(np.asarray(disc_pix_i_s, dtype=np.int64))
                self.beam_disc_val_list.append(np.asarray(beam_disc, dtype=np.float64))

    def get_sed(self, nu:float):
        """One SED value per source at `nu` (GHz), 1 at `nu_ref`."""
        # The -2 turns the flux-density index into an RJ-temperature index.
        return (nu/self.nu_ref)**(self.alpha_arr - 2)

    def get_sky(self, nu:float, nside:int, fwhm_rad:float=0.0):
        """The sky of the sources at `nu` (GHz) in uK_RJ, at `nside`, painted through a Gaussian
        beam of `fwhm_rad` radians."""
        self.compute_pix_beams(fwhm_rad, nside)
        map = np.zeros((1, hp.nside2npix(nside)),
                       dtype=np.float64 if self.double_precision else np.float32)
        scales = self.amps[0]*self.get_sed(nu)*self.mJysr_to_uKRJ
        for pixels, beam, scale in zip(self.pix_disc_idx_list, self.beam_disc_val_list, scales):
            map[0, pixels] += beam*scale  # Overlapping sources add up: the loop is sequential.
        return map

    def get_component_map(self, nside:int, fwhm_rad:float=0.0):
        """The amplitude map at `nside` and `fwhm_rad`: the sky at `nu_ref`, in uK_RJ, as for a
        diffuse component."""
        return self.get_sky(self.nu_ref, nside, fwhm_rad)

    def comp2band(self, band:Band, nthreads: int = 1):
        """Paint the sources into the band's pixel map `band.map` (allocated if None), in uK_RJ at
        the band frequency.

        They are added in pixel space, as in C3, so they are not cut at the band lmax.
        """
        if band.pol == "QU":
            raise ValueError("Point-source components can only be projected to intensity bands.")
        self.compute_pix_beams(band.fwhm_rad, band.nside)
        if band.map is None:
            band.map = np.zeros((1, hp.nside2npix(band.nside)), dtype=band.alms.real.dtype)
        # The unit factor goes into the per-source scale, not onto the map, because the map may
        # already hold other components.
        scales = self.amps[0]*self.get_sed(band.nu)*self.mJysr_to_uKRJ
        for pixels, beam, scale in zip(self.pix_disc_idx_list, self.beam_disc_val_list, scales):
            band.map[0, pixels] += beam*scale

    def band2comp(self, band:Band, nthreads: int = 1):
        """The transpose of `comp2band`: replace `amps` with the beam-weighted sums of `band.map`.
        """
        if band.pol == "QU":
            raise ValueError("Point-source components can only be evaluated from intensity bands.")
        self.compute_pix_beams(band.fwhm_rad, band.nside)
        scales = self.get_sed(band.nu)*self.mJysr_to_uKRJ
        for i, (pixels, beam) in enumerate(zip(self.pix_disc_idx_list, self.beam_disc_val_list)):
            self.amps[0, i] = np.dot(band.map[0, pixels], beam)*scales[i]

    def apply_Cl_prior_sqrt(self, alms):
        """Return `alms` unchanged: point sources have no prior (S = 1), as in C3, where the
        per-source prior `P_x` is off."""
        return alms

    def apply_Cl_prior_inv_sqrt(self, alms):
        """Return `alms` unchanged (see `apply_Cl_prior_sqrt`)."""
        return alms

    def __repr__(self):
        return f"Radio Source \n amps: {self.amps}"
