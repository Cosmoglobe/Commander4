"""`FarSidelobes`: the far-sidelobe pickup of each detector, projected into time-ordered data from
the Commander3 Planck far-sidelobe beams."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import h5py
import healpy as hp
import numpy as np
from ducc0 import totalconvolve
from mpi4py import MPI
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.math_utils.alm import alm_real2complex_commander3
from commander4.mpi.shared_memory import SharedArray
from commander4.parameters.schema import read_settings
from commander4.tod.step import TODStep


if TYPE_CHECKING:
    from commander4.data_models.tod_samples import TODSamples

logger = logging.getLogger(__name__)

BEAM_COMPONENTS = ("T", "E", "B")


class FarSidelobes(TODStep):
    """The far-sidelobe pickup of one band's detectors on one rank.

    Nothing here is sampled or written to the chain. The model of the pickup is one convolution
    cube per detector, built from the iteration's sky by `build` and released by `free` at the end
    of the iteration. In between, the mapmaking scan loop subtracts `get_projection` from the TOD.

    Attributes:
        cfg: The `tod_processing.far_beam_deconvolution` settings: whether and from which iteration
            the pickup is subtracted, the beam's harmonic limits `lmax` and `mmax` (Commander3 uses
            100 for both), the ducc0 accuracy `epsilon`, and `beam_norm`, the factor the sidelobe
            TOD is scaled by (Commander3 uses 2, found by comparison with LevelS).
        polangs: (ndet,) Polarization angle of each detector, which orients its beam.
        pols, nside, instrument_file, nthreads: The band's polarizations, map resolution and
            beam file, and the thread count of the convolution.
        plan, cubes, shared: The ducc0 convolution plan, and the per-detector cubes in memory
            shared by the ranks of a node, while a model is built.
    """

    def __init__(self, experiment_data: DetectorGroupTOD, params: Bunch, my_band: Bunch,
                 band_comm: MPI.Comm, chain: int):
        """Read and check the settings and, when the step is enabled, the band's detectors."""
        self.cfg = read_settings(params, "far_beam_deconvolution", ("tod_processing",),
                                 {"enabled": False, "from_iter": 1, "lmax": 100, "mmax": 100,
                                  "epsilon": 1.0e-4, "beam_norm": 2.0})
        if not self.cfg.enabled:
            return
        # The T beam always sees the intensity sky, so a band without it cannot be convolved.
        if experiment_data.pols not in ("I", "IQU"):
            raise ValueError(f"Sidelobe convolution needs an I or I/Q/U band, not "
                             f"{experiment_data.pols}.")
        self.pols = experiment_data.pols
        self.nside = experiment_data.nside
        self.nthreads = int(os.environ.get("OMP_NUM_THREADS", "1"))
        self.instrument_file = experiment_data.instrument_filepath
        # Scans can omit detectors. Walk every available detector-scan to fill arrays indexed by
        # det_idx_fullband, the stable position in the band's full detector list.
        self.polangs = np.full(experiment_data.ndet, np.nan)
        for _, det in experiment_data.iter_detector_scans():
            if det.polang is None:
                raise ValueError(f"Detector {det.name} has no polang; the sidelobe projection "
                                 f"needs it (the experiment's TOD reader must set it).")
            self.polangs[det.det_idx_fullband] = det.polang
        self.plan = None
        self.shared = None
        self.cubes = []


    def build(self, tod_samples: TODSamples, compsep_output_full: NDArray | None,
              node_comm: MPI.Comm) -> None:
        """Build one ducc0 convolution cube for each detector's T/E/B sidelobe beam (collective).

        Args:
            tod_samples: The band's sampled state, for its communicator and detector names.
            compsep_output_full: The band's full-sky sky map on rank 0, else None: (1, npix) for an
                intensity-only band, (3, npix) I/Q/U for a polarized one.
            node_comm: The band's ranks on this node, which share the cubes. `free` must be called
                on all of them once the cubes are no longer needed.
        """
        band_comm, detnames = tod_samples.band_comm, tod_samples.det_names
        # Each detector has its own beam. Commander3 forms one signal by summing the matching
        # sky/beam component pairs: T_sky*T_beam + E_sky*E_beam + B_sky*B_beam. An intensity-only
        # band has only the T sky, so only its T beam is used.
        sky_components = BEAM_COMPONENTS if self.pols == "IQU" else ("T",)
        with h5py.File(self.instrument_file, "r") as f:
            file_lmax = int(f[f"{detnames[0]}/sllmax"][0])
            file_mmax = int(f[f"{detnames[0]}/slmmax"][0])
            # Truncate to the configured limits without exceeding what the file contains.
            lmax = min(self.cfg.lmax, file_lmax)
            mmax = min(self.cfg.mmax, file_mmax, lmax)
            blms = []
            for detname in detnames:
                det_lmax = int(f[f"{detname}/sllmax"][0])
                det_mmax = int(f[f"{detname}/slmmax"][0])
                if (det_lmax, det_mmax) != (file_lmax, file_mmax):
                    raise ValueError(f"Sidelobe limits for {detname} are "
                                     f"(lmax={det_lmax}, mmax={det_mmax}); expected "
                                     f"(lmax={file_lmax}, mmax={file_mmax}).")
                # Check what components (T,E,B) are actually in the file:
                components = [name for name in sky_components if f"{detname}/sl/{name}" in f]
                if "T" not in components:  # Raise if T is not among them.
                    raise ValueError(f"No sidelobe T beam for {detname} in {self.instrument_file}.")
                beam_real = np.stack([f[f"{detname}/sl/{name}"][()] for name in components])
                # Commander3 stores complete l blocks consecutively. Truncating this leading slice
                # therefore keeps exactly the modes with l <= lmax for every component.
                beam_real = beam_real[..., :(lmax+1)**2]
                blm = alm_real2complex_commander3(beam_real, lmax, mmax)
                # The convolution is linear, so scaling the beam once is equivalent to scaling
                # every projected TOD, and far cheaper.
                blms.append((components, self.cfg.beam_norm * blm))

        # Only rank zero has the full-sky map. Transform there, then give every rank the small alm
        # array needed to build its local detector cubes. iter=0 matches Commander3's single,
        # non-iterative map-to-alm transform. healpy gives (3, nalm) T/E/B alms for an I/Q/U map,
        # but drops the axis of a (1, npix) intensity map and returns flat (nalm,) alms, which
        # atleast_2d turns back into (1, nalm).
        slm = (np.atleast_2d(hp.map2alm(compsep_output_full, lmax=lmax, iter=0,
                                        pol=self.pols == "IQU"))
               if band_comm.Get_rank() == 0 else None)
        slm = band_comm.bcast(slm, root=0)

        self.plan = totalconvolve.ConvolverPlan(lmax=lmax, kmax=mmax,
                                                epsilon=self.cfg.epsilon,
                                                nthreads=self.nthreads)
        # ducc0 decides a cube shape based on stuff like lmax, mmax, epsilon.
        cube_shape = (self.plan.Npsi(), self.plan.Ntheta(), self.plan.Nphi())
        if band_comm.Get_rank() == 0:
            gib = len(blms)*np.prod(cube_shape)*8/1024**3
            logger.verbose(f"Far beam: {len(blms)} cubes of shape {cube_shape} ({gib:.1f} GiB) "
                           f"per node, shared by {node_comm.Get_size()} ranks.")

        # The cubes are identical for every rank of the band, and far too large to hold per rank,
        # so one rank per node builds them and the rest read the same memory. Every rank of
        # node_comm must reach both this allocation and the matching free() below.
        self.shared = SharedArray(node_comm, (len(blms), *cube_shape),
                                  name="far-sidelobe convolution cubes")
        if self.shared.is_owner:
            for idet, (components, blm) in enumerate(blms):
                cube = self.shared.array[idet]
                # Before prepPsi, the first axis holds packed Fourier coefficients in the beam-
                # rotation angle: m=0, then real/imaginary planes for every m>0. prepPsi zero-pads
                # and transforms these 2*mmax+1 coefficient planes into Npsi angular samples used
                # by interpol().
                for mbeam in range(mmax+1):
                    start = 0 if mbeam == 0 else 2*mbeam - 1
                    stop = 1 if mbeam == 0 else 2*mbeam + 1
                    planes = cube[start:stop]
                    self.plan.getPlane(slm[0], blm[0], mbeam, planes)
                    # This ducc0 Python wrapper accepts only one component per getPlane call. Add E
                    # and B (when present) explicitly. The following prepPsi transform is linear,
                    # so summing here gives the same result as transforming and then summing
                    # separate component cubes. T is always the first beam component.
                    contribution = np.empty_like(planes)
                    for name, blm_component in zip(components[1:], blm[1:]):
                        isky = BEAM_COMPONENTS.index(name)  # row of the T/E/B sky alms
                        self.plan.getPlane(slm[isky], blm_component, mbeam, contribution)
                        planes += contribution
                self.plan.prepPsi(cube)
        self.shared.wait_until_filled()
        # Per-detector views into the shared buffer. Each is contiguous, and read-only on every
        # rank but the node owner. They are dropped by free(), which invalidates the memory.
        self.cubes = [self.shared.array[idet] for idet in range(len(blms))]


    def get_projection(self, pix: NDArray, psi: NDArray, idet: int) -> NDArray:
        """Far-sidelobe pickup for one detector-scan, in uK_RJ.

        Args:
            pix: HEALPix RING pixel per sample, at the band's evaluation nside.
            psi: Polarization angle per sample [rad], as stored in the data files.
            idet: Full-band detector column (`det_idx_fullband`), selecting this detector's
                sidelobe cube and polarization angle.
        """
        if pix.shape != psi.shape:
            raise ValueError(
                f"pix and psi must have the same shape, got {pix.shape} and {psi.shape}."
            )
        theta, phi = hp.pix2ang(self.nside, pix)
        # In Commander3, mbang is populated from the TOD file's polang field. A direct comparison
        # of the two convolution operators shows that ducc0 must receive psi-mbang.
        psi_beam = np.mod(psi - self.polangs[idet], 2*np.pi).astype(np.float64, copy=False)
        # C3 evaluated every fifth sample and interpolated the gaps to save time. ducc0 is fast
        # enough to evaluate every sample directly, avoiding that approximation.
        res = np.empty(theta.shape[0], dtype=np.float64)
        self.plan.interpol(self.cubes[idet], 0, 0, theta, phi, psi_beam, res)
        return res


    def free(self) -> None:
        """Release the shared cubes. Collective over the node communicator.

        Must be called once after every `build`: mpi4py leaves an unfreed MPI window in place until
        `MPI_Finalize`, so a model that is not freed leaks the whole allocation every Gibbs
        iteration. `get_projection` cannot be used afterwards.
        """
        self.cubes = []
        self.shared.free()
        self.shared = None
