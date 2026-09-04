"""Project the Commander3 Planck LFI far-sidelobe beams into time-ordered data."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import ClassVar, TYPE_CHECKING

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
from commander4.tod.step_config import StepConfig


if TYPE_CHECKING:
    from commander4.data_models.tod_samples import TODSamples

logger = logging.getLogger(__name__)

_BEAM_COMPONENTS = ("T", "E", "B")


@dataclass(frozen=True)
class FarBeamConfig(StepConfig):
    """Validated far-sidelobe deconvolution settings.

    The defaults reproduce Commander3: it truncates both the sky and the beam at l = m = 100
    before convolving, and scales the finished sidelobe TOD by two "by comparison with LevelS"
    (comm_tod_driver_mod.f90:233).
    """

    PARAMETER_NAME: ClassVar[str] = "far_beam_deconvolution"

    lmax: int = 100
    mmax: int = 100
    epsilon: float = 1.0e-4
    beam_norm: float = 2.0

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("lmax", "mmax"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{self.PARAMETER_NAME}.{name} must be a non-negative integer.")
        if self.mmax > self.lmax:
            raise ValueError(f"{self.PARAMETER_NAME}.mmax ({self.mmax}) cannot exceed lmax "
                             f"({self.lmax}); m is bounded by l.")
        # ducc0 rejects an accuracy outside this range, and both ends are far past anything useful.
        if not 1e-12 < self.epsilon < 1.0:
            raise ValueError(f"{self.PARAMETER_NAME}.epsilon must lie between 1e-12 and 1.")
        if not np.isfinite(self.beam_norm):
            raise ValueError(f"{self.PARAMETER_NAME}.beam_norm must be a finite number.")

    @classmethod
    def from_params(cls, params: Bunch, experiment_data: DetectorGroupTOD) -> FarBeamConfig:
        """Build the far-beam settings from their step block.

        An absent block is valid and gives the disabled defaults.
        """
        block = (params.tod_processing[cls.PARAMETER_NAME]
                 if cls.PARAMETER_NAME in params.tod_processing else Bunch())
        return cls._from_block(f"tod_processing.{cls.PARAMETER_NAME}", block)


class FarBeamProjector:
    """Precompute one polarized sidelobe-convolution cube per detector and project it to TOD."""

    def __init__(self, band_comm: MPI.Comm, node_comm: MPI.Comm,
                 experiment_data: DetectorGroupTOD, tod_samples: TODSamples,
                 compsep_output: NDArray, config: FarBeamConfig):
        self.config = config
        self.nside = experiment_data.nside
        self.nthreads = int(os.environ.get("OMP_NUM_THREADS", "1"))
        self.instrument_file = experiment_data.instrument_filepath
        self.detnames = tod_samples.det_names

        # Scans can omit detectors. Walk every available detector-scan to fill arrays indexed by
        # det_idx_fullband, the stable position in the band's full detector list.
        self.polangs = np.full(experiment_data.ndet, np.nan)
        for _, det in experiment_data.iter_detector_scans():
            self.polangs[det.det_idx_fullband] = det.polang

        self.construct_model(band_comm, node_comm, compsep_output)


    def construct_model(self, band_comm: MPI.Comm, node_comm: MPI.Comm,
                        compsep_output: NDArray) -> None:
        """Build one ducc0 convolution cube for each detector's T/E/B sidelobe beam."""
        # Each LFI detector has its own beam. Commander3 forms one signal by summing the matching
        # sky/beam component pairs: T_sky*T_beam + E_sky*E_beam + B_sky*B_beam.
        with h5py.File(self.instrument_file, "r") as f:
            file_lmax = int(f[f"{self.detnames[0]}/sllmax"][0])
            file_mmax = int(f[f"{self.detnames[0]}/slmmax"][0])
            # Truncate to the configured limits without exceeding what the file contains.
            lmax = min(self.config.lmax, file_lmax)
            mmax = min(self.config.mmax, file_mmax, lmax)
            blms = []
            for detname in self.detnames:
                det_lmax = int(f[f"{detname}/sllmax"][0])
                det_mmax = int(f[f"{detname}/slmmax"][0])
                if (det_lmax, det_mmax) != (file_lmax, file_mmax):
                    raise ValueError(f"Sidelobe limits for {detname} are "
                                     f"(lmax={det_lmax}, mmax={det_mmax}); expected "
                                     f"(lmax={file_lmax}, mmax={file_mmax}).")
                beam_real = np.stack([
                    f[f"{detname}/sl/{component}"][()] for component in _BEAM_COMPONENTS
                ])
                # Commander3 stores complete l blocks consecutively. Truncating this leading slice
                # therefore keeps exactly the modes with l <= lmax for all three components.
                beam_real = beam_real[..., :(lmax+1)**2]
                blm = alm_real2complex_commander3(beam_real, lmax, mmax)
                # The convolution is linear, so scaling the beam once is equivalent to scaling
                # every projected TOD, and far cheaper.
                blms.append(self.config.beam_norm * blm)

        # With sparse maps, only rank zero has a complete sky. Transform there, then give every rank
        # the small alm array needed to build its local detector cubes. iter=0 matches Commander3's
        # single, non-iterative map-to-alm transform.
        if compsep_output.ndim != 2 or compsep_output.shape[0] != len(_BEAM_COMPONENTS):
            raise ValueError("Polarized sidelobe convolution requires an I/Q/U sky map.")
        slm = (hp.map2alm(compsep_output, lmax=lmax, iter=0, pol=True)
               if band_comm.Get_rank() == 0 else None)
        slm = band_comm.bcast(slm, root=0)

        self.plan = totalconvolve.ConvolverPlan(lmax=lmax, kmax=mmax, epsilon=self.config.epsilon,
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
            for idet, blm in enumerate(blms):
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
                    # and B explicitly. The following prepPsi transform is linear, so summing here
                    # gives the same result as transforming and then summing three separate
                    # component cubes.
                    contribution = np.empty_like(planes)
                    for component in range(1, len(_BEAM_COMPONENTS)):
                        self.plan.getPlane(slm[component], blm[component], mbeam, contribution)
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

        Must be called once per constructed projector: mpi4py leaves an unfreed MPI window in
        place until `MPI_Finalize`, so letting the projector go out of scope instead leaks the
        whole allocation every Gibbs iteration. `get_projection` cannot be used afterwards.
        """
        self.cubes = []
        self.shared.free()
