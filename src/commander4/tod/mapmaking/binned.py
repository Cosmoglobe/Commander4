"""The binned mapmaker: accumulate P^T N^-1 d and P^T N^-1 P per pixel, then invert per pixel.

The simpler of the two mapmakers (see `mapmaking/cg.py` for the CG one). `Mapmaker`/`MapmakerIQU`
build the signal map, `WeightsMapmaker`/`WeightsMapmakerIQU` the inverse-variance weights that both
mapmakers need. `tod2map_bin` drives the whole per-band scan loop.
"""
import numpy as np
import ctypes as ct
from mpi4py import MPI
import logging
from numpy.typing import NDArray

from commander4.diagnostics.performance import log_memory, start_bench, stop_bench
from commander4.backend.ctypes_lib import load_cmdr4_ctypes_lib
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.detector_map import DetectorMap
from commander4.data_models.tod_samples import TODSamples
from commander4.tod.noise.sample_ncorr import sample_correlated_noise, log_corr_noise_stats,\
    CorrelatedNoiseConfig
from commander4.tod.noise.sigma0 import _estimate_standalone_sigma0
from commander4.tod.scan_diagnostics import _record_tod_diagnostics
from commander4.tod.view import TODView
from commander4.data_models.pixel_domain import PixelDomain
from commander4.tod.mapmaking.config import MapmakingConfig
from commander4.tod.mapmaking.output import finalize_band_maps
from commander4.tod.data_selection import DataSelectionConfig

logger = logging.getLogger(__name__)

class Mapmaker:
    """Scalar (temperature-only) mapmaker using binned TOD accumulation.

    Accumulates weighted TOD samples into a map, reduces across MPI ranks,
    and normalizes with a precomputed weights map. The internal accumulation
    is performed in float64; output dtype is controlled by `dtype`.

    The accumulation buffer lives on the rank's ``PixelDomain``: full-sky in the default mode, or
    only the locally-observed pixels in sparse mode. ``gather_map`` always produces a full-sky map
    on the master, so normalization and all downstream consumers are unchanged.
    """
    def __init__(self, map_comm:MPI.Comm, nside:int, dtype=np.float32,
                 pixel_domain:PixelDomain|None=None):
        self.map_comm = map_comm
        self.nside = nside
        self.npix = 12*nside**2
        self.dtype= dtype
        self.domain = pixel_domain if pixel_domain is not None else PixelDomain(map_comm, nside, "full")
        self._nloc = self.domain.n_local
        self._map_signal = np.zeros(self._nloc, dtype=np.float64)
        self._gathered_map = None
        self._finalized_map = None

        # Setting up Ctypes mapmaker
        self.maplib = load_cmdr4_ctypes_lib()
        ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_int64, ndim=1, flags="contiguous")
        ct_f64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=1, flags="contiguous")
        self.maplib.map_accumulator_f64.argtypes = [ct_f64_dim1, ct_f64_dim1, ct.c_double,
                                                    ct_i64_dim1, ct.c_int64]

    @property
    def final_map(self):
        if self.map_comm.Get_rank() == 0:
            if self._finalized_map is None:
                raise RuntimeError("Attempted to retrieve an unfinished map.")
        return self._finalized_map

    def accumulate_to_map(self, tod:NDArray, weights:NDArray, pix:NDArray, psi=None):
        """Accumulate weighted TOD samples into the local map buffer."""
        # Check that we are still in business, and haven't already called "gather_map".
        if self._map_signal is None:
            raise RuntimeError("Cannot accumulate into a finalized map.")
        ntod = tod.shape[0]
        tod_f64 = np.ascontiguousarray(tod, dtype=np.float64)
        weight_f64 = float(weights)
        pix = self.domain.to_local(pix)
        self.maplib.map_accumulator_f64(self._map_signal, tod_f64, weight_f64,
                                    pix.astype(np.int64, copy=False), ntod)

    def gather_map(self):
        """Reduce the local map buffers across MPI ranks into the full-sky root map."""
        self._gathered_map = self.domain.reduce_to_full(self._map_signal)
        self._map_signal = None  # Free memory and indicate that accumulation is done.

    def normalize_map(self, normalization_map):
        """Normalize the gathered map by the provided weights map."""
        if self.map_comm.Get_rank() == 0:
            norm_map = np.asarray(normalization_map, dtype=np.float64)
            mask = norm_map != 0
            result = np.zeros(self.npix, dtype=np.float64)
            # Only normalize where weights are non-zero to avoid division by zero.
            result[mask] = self._gathered_map[mask] / norm_map[mask]
            self._finalized_map = result.astype(self.dtype, copy=False)
            self._gathered_map = None



class WeightsMapmaker:
    """Scalar (temperature-only) weights mapmaker.

    Accumulates per-sample weights into a map, reduces across MPI ranks,
    and exposes the gathered weights map for normalization of Mapmaker.
    """
    def __init__(self, map_comm:MPI.Comm, nside:int, dtype=np.float32,
                 pixel_domain:PixelDomain|None=None):
        self.map_comm = map_comm
        self.nside = nside
        self.npix = 12*nside**2
        self.dtype= dtype
        self.domain = pixel_domain if pixel_domain is not None else PixelDomain(map_comm, nside, "full")
        self._nloc = self.domain.n_local
        self._map_signal = np.zeros(self._nloc, dtype=np.float64)
        self._gathered_map = None

        # Setting up Ctypes mapmaker
        self.maplib = load_cmdr4_ctypes_lib()
        ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_int64, ndim=1, flags="contiguous")
        ct_f64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=1, flags="contiguous")
        self.maplib.map_weight_accumulator_f64.argtypes = [ct_f64_dim1, ct.c_double, ct_i64_dim1,
                                   ct.c_int64]

    @property
    def final_map(self):
        if self.map_comm.Get_rank() == 0:
            if self._gathered_map is None:
                raise RuntimeError("Attempted to retrieve an unfinished weights map.")
        return self._gathered_map

    def accumulate_to_map(self, weight:NDArray, pix:NDArray, psi=None):
        """Accumulate per-sample weights into the local map buffer."""
        # Check that we are still in business, and haven't already called "gather_map".
        if self._map_signal is None:
            raise RuntimeError("Cannot accumulate into a finalized weights map.")
        ntod = pix.shape[0]
        weight_f64 = float(weight)
        pix = self.domain.to_local(pix)
        # The scalar weight kernel indexes map[pix] directly and takes no num_pix argument.
        self.maplib.map_weight_accumulator_f64(self._map_signal, weight_f64,
                                               pix.astype(np.int64, copy=False), ntod)

    def gather_map(self):
        """Reduce the local weights buffers across MPI ranks into the full-sky root map."""
        self._gathered_map = self.domain.reduce_to_full(self._map_signal)
        self._map_signal = None  # Free memory and indicate that accumulation is done.



class MapmakerIQU:
    """Binned polarized mapmaker solving per-pixel I,Q,U normal equations.

    This class accumulates the right-hand side of the mapmaking system (the
    weighted TOD projected into I,Q,U) across MPI tasks, gathers the result
    on the root rank, and solves the 3x3 system per pixel using the provided
    normalization map (A matrix) built by WeightsMapmakerIQU.

    Usage:
    - Use WeightsMapmakerIQU to accumulate the 6 unique elements of A and
        call `normalize_map()` there to produce RMS/covariance maps.
    - Use MapmakerIQU to accumulate signal maps, then call `normalize_map(A)`
        with the gathered A map to produce the finalized I,Q,U map.
    """
    def __init__(self, map_comm:MPI.Comm, nside:int, dtype=np.float32,
                 pixel_domain:PixelDomain|None=None):
        self.map_comm = map_comm
        self.nside = nside
        self.npix = 12*nside**2
        self.dtype= dtype
        self.domain = pixel_domain if pixel_domain is not None else PixelDomain(map_comm, nside, "full")
        self._nloc = self.domain.n_local
        self._map_signal = np.zeros((3, self._nloc), dtype=np.float64)
        self._gathered_map = None
        self._finalized_map = None
        self._has_gathered = False

        # Setting up Ctypes mapmaker
        self.maplib = load_cmdr4_ctypes_lib()
        ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_int64, ndim=1, flags="contiguous")
        ct_f64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=1, flags="contiguous")
        ct_f64_dim2 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=2, flags="contiguous")
        self.maplib.map_accumulator_IQU_f64.argtypes = [ct_f64_dim2, ct_f64_dim1, ct.c_double,
                                ct_i64_dim1, ct_f64_dim1, ct.c_int64,
                                ct.c_int64]
        self.maplib.map_accumulator_IQU_response_f64.argtypes = [ct_f64_dim2, ct_f64_dim1,
                                     ct.c_double, ct_i64_dim1,
                                     ct_f64_dim1, ct.c_double,
                                     ct.c_double, ct.c_int64,
                                     ct.c_int64]
        self.maplib.map_solve_IQU_f64.argtypes = [ct_f64_dim2, ct_f64_dim2, ct_f64_dim2,
                      ct.c_int64]

    @property
    def final_map(self):
        if self.map_comm.Get_rank() == 0:
            if self._finalized_map is None:
                raise RuntimeError("Attempted to read a map before it was finalized.")
        return self._finalized_map


    def accumulate_to_map(self, tod:NDArray, weights:NDArray, pix:NDArray, psi:NDArray,
                          response: NDArray | None = None):
        """Accumulate I,Q,U signal into the local map buffer."""
        # Check that we are still in business, and haven't already called "gather_map".
        if self._map_signal is None:
            raise RuntimeError("Cannot accumulate into a finalized map.")
        ntod = tod.shape[0]
        tod_f64 = np.ascontiguousarray(tod, dtype=np.float64)
        weight_f64 = float(weights)
        psi_f64 = np.ascontiguousarray(psi, dtype=np.float64)
        # The IQU kernels stride the (3, n) buffer by its pixel count, so num_pix must be n_local.
        pix_i64 = self.domain.to_local(pix).astype(np.int64, copy=False)
        if response is None:
            self.maplib.map_accumulator_IQU_f64(self._map_signal, tod_f64, weight_f64,
                                                pix_i64, psi_f64, ntod, self._nloc)
        else:
            response_I = float(response[0])
            response_QU = float(response[1])
            self.maplib.map_accumulator_IQU_response_f64(self._map_signal, tod_f64, weight_f64,
                                                         pix_i64, psi_f64, response_I,
                                                         response_QU, ntod, self._nloc)

    def accumulate_to_map_Python(self, tod:NDArray, weights:NDArray, pix:NDArray, psi:NDArray,
                                 response: NDArray | None = None):
        """Reference accumulator matching the ctypes IQU implementation."""
        # Reference implementation matching the ctypes IQU accumulator.
        if self._map_signal is None:
            raise RuntimeError("Cannot accumulate into a finalized map.")
        pix_idx = self.domain.to_local(pix).astype(np.int64, copy=False)
        w_tod = np.ascontiguousarray(tod, dtype=np.float64) * float(weights)
        ang = 2.0 * np.ascontiguousarray(psi, dtype=np.float64)
        c2 = np.cos(ang)
        s2 = np.sin(ang)
        if response is None:
            response_I, response_QU = 1.0, 1.0
        else:
            response_I = float(response[0])
            response_QU = float(response[1])
        np.add.at(self._map_signal[0], pix_idx, w_tod * response_I)
        np.add.at(self._map_signal[1], pix_idx, w_tod * response_QU * c2)
        np.add.at(self._map_signal[2], pix_idx, w_tod * response_QU * s2)

    def gather_map(self):
        """Reduce the local IQU buffers across MPI ranks into the full-sky root map."""
        self._gathered_map = self.domain.reduce_to_full(self._map_signal)
        self._map_signal = None  # Free memory and indicate that accumulation is done.
        self._has_gathered = True

    def normalize_map(self, normalization_map):
        """Solve the per-pixel 3x3 system using the provided A matrix."""
        if self.map_comm.Get_rank() == 0:
            if normalization_map.ndim != 2 or normalization_map.shape[0] != 6:
                raise ValueError("Normalization map must have shape (6, npix), got "
                                 f"{normalization_map.shape}.")
            if not self._has_gathered:
                raise RuntimeError("Cannot normalize a map before it is gathered.")
            norm_map = np.ascontiguousarray(normalization_map, dtype=np.float64)
            rhs_map = np.ascontiguousarray(self._gathered_map, dtype=np.float64)
            solved = np.zeros((3, self.npix), dtype=np.float64)
            self.maplib.map_solve_IQU_f64(solved, rhs_map, norm_map, self.npix)
            self._finalized_map = solved.astype(self.dtype, copy=False)
            self._gathered_map = None

    def normalize_map_Python(self, normalization_map):
        """Reference solver using NumPy for per-pixel normalization."""
        if self.map_comm.Get_rank() == 0:
            if normalization_map.ndim != 2 or normalization_map.shape[0] != 6:
                raise ValueError("Normalization map must have shape (6, npix), got "
                                 f"{normalization_map.shape}.")
            if not self._has_gathered:
                raise RuntimeError("Cannot normalize a map before it is gathered.")
            self._finalized_map = np.zeros((3, self.npix), dtype=self.dtype)
            A = np.zeros((self.npix, 3, 3), dtype=np.float64)
            A[:, 0, 0] = normalization_map[0]
            A[:, 0, 1] = normalization_map[1]
            A[:, 1, 0] = normalization_map[1]
            A[:, 0, 2] = normalization_map[2]
            A[:, 2, 0] = normalization_map[2]
            A[:, 1, 1] = normalization_map[3]
            A[:, 1, 2] = normalization_map[4]
            A[:, 2, 1] = normalization_map[4]
            A[:, 2, 2] = normalization_map[5]

            # Test whether the matrix is singular or ill-conditioned.
            det = np.linalg.det(A)
            diag_prod = A[:, 0, 0] * A[:, 1, 1] * A[:, 2, 2]
            eps = np.finfo(np.float64).eps
            # If a diagonal entry is 0 the matrix is singular, and if negative the matrix is not SPD
            # If the determinant is very small the matrix is ill-conditioned.
            mask = (diag_prod > 0) & (np.abs(det) > eps * diag_prod)

            # Ill-conditioned pixels stay as 0.0.
            if np.any(mask):
                rhs = self._gathered_map[:, mask].T[..., np.newaxis]
                sol = np.linalg.solve(A[mask], rhs)
                self._finalized_map[:, mask] = sol[..., 0].T.astype(self.dtype, copy=False)
            self._gathered_map = None


class WeightsMapmakerIQU:
    """Binned polarized weight/covariance mapmaker for I,Q,U.

    This class accumulates the left-hand side (A matrix) of the mapmaking
    system, storing the 6 unique elements per pixel. The gathered A map is
    then inverted per pixel to provide RMS/covariance information and used as
    normalization input for MapmakerIQU.

    Usage:
    - Call `accumulate_to_map()` for each scan to build the A elements.
    - Call `gather_map()` to reduce across MPI tasks.
    - Call `normalize_map()` to compute RMS maps and expose `final_cov_map`
        for MapmakerIQU normalization.
    """
    def __init__(self, map_comm:MPI.Comm, nside:int, dtype=np.float32,
                 pixel_domain:PixelDomain|None=None):
        self.map_comm = map_comm
        self.nside = nside
        self.npix = 12*nside**2
        self.dtype= dtype
        self.domain = pixel_domain if pixel_domain is not None else PixelDomain(map_comm, nside, "full")
        self._nloc = self.domain.n_local
        self._map_signal = np.zeros((6, self._nloc), dtype=np.float64)
        self._gathered_map = None
        self._finalized_rms_map = None
        self._has_gathered = False

        # Setting up Ctypes mapmaker
        self.maplib = load_cmdr4_ctypes_lib()
        ct_i64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_int64, ndim=1, flags="contiguous")
        ct_f64_dim1 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=1, flags="contiguous")
        ct_f64_dim2 = np.ctypeslib.ndpointer(dtype=ct.c_double, ndim=2, flags="contiguous")
        self.maplib.map_weight_accumulator_IQU_f64.argtypes = [ct_f64_dim2, ct.c_double,
                                        ct_i64_dim1, ct_f64_dim1, ct.c_int64, ct.c_int64]
        self.maplib.map_weight_accumulator_IQU_response_f64.argtypes = [ct_f64_dim2,
                                        ct.c_double, ct_i64_dim1, ct_f64_dim1, ct.c_double,
                                        ct.c_double, ct.c_int64, ct.c_int64]
        self.maplib.map_invdiag_IQU_f64.argtypes = [ct_f64_dim2, ct_f64_dim2, ct.c_int64]

    @property
    def final_rms_map(self):
        if self.map_comm.Get_rank() == 0:
            if self._finalized_rms_map is None:
                raise RuntimeError("Attempted to read an unfinished RMS map.")
        return self._finalized_rms_map
    
    @property
    def final_cov_map(self):
        if self.map_comm.Get_rank() == 0:
            if self._gathered_map is None:
                raise RuntimeError("Attempted to read an unfinished covariance map.")
        return self._gathered_map

    def accumulate_to_map(self, weight:float, pix:NDArray, psi:NDArray, response:NDArray | None = None):
        """Accumulate IQU weight/covariance elements into the local buffer."""
        # Check that we are still in business, and haven't already called "gather_map".
        if self._map_signal is None:
            raise RuntimeError("Cannot accumulate into a finalized weights map.")
        ntod = pix.shape[0]
        weight_f64 = float(weight)
        psi_f64 = np.ascontiguousarray(psi, dtype=np.float64)
        # The IQU kernels stride the (6, n) buffer by its pixel count, so num_pix must be n_local.
        pix_i64 = self.domain.to_local(pix).astype(np.int64, copy=False)
        if response is None:
            self.maplib.map_weight_accumulator_IQU_f64(self._map_signal, weight_f64,
                                                       pix_i64, psi_f64, ntod, self._nloc)
        else:
            response_I = float(response[0])
            response_QU = float(response[1])
            self.maplib.map_weight_accumulator_IQU_response_f64(self._map_signal, weight_f64,
                                                                pix_i64, psi_f64, response_I,
                                                                response_QU, ntod, self._nloc)

    def accumulate_to_map_Python(self, weight:float, pix:NDArray, psi:NDArray,
                                 response: NDArray | None = None):
        """Reference accumulator matching the ctypes IQU weights implementation."""
        # Reference implementation matching the ctypes IQU weight accumulator.
        if self._map_signal is None:
            raise RuntimeError("Cannot accumulate into a finalized weights map.")
        pix_idx = self.domain.to_local(pix).astype(np.int64, copy=False)
        ang = 2.0 * np.ascontiguousarray(psi, dtype=np.float64)
        c2 = np.cos(ang)
        s2 = np.sin(ang)
        weight_f64 = float(weight)
        if response is None:
            response_I, response_QU = 1.0, 1.0
        else:
            response_I = float(response[0])
            response_QU = float(response[1])
        np.add.at(self._map_signal[0], pix_idx, weight_f64 * response_I * response_I)
        np.add.at(self._map_signal[1], pix_idx, weight_f64 * response_I * response_QU * c2)
        np.add.at(self._map_signal[2], pix_idx, weight_f64 * response_I * response_QU * s2)
        np.add.at(self._map_signal[3], pix_idx, weight_f64 * response_QU * response_QU * c2 * c2)
        np.add.at(self._map_signal[4], pix_idx, weight_f64 * response_QU * response_QU * s2 * c2)
        np.add.at(self._map_signal[5], pix_idx, weight_f64 * response_QU * response_QU * s2 * s2)

    def gather_map(self):
        """Reduce the local IQU weight buffers across MPI ranks into the full-sky root map."""
        self._gathered_map = self.domain.reduce_to_full(self._map_signal)
        self._map_signal = None  # Free memory and indicate that accumulation is done.
        self._has_gathered = True

    def normalize_map(self):
        """Compute RMS maps from the per-pixel inverse covariance diagonals."""
        if self.map_comm.Get_rank() == 0:
            if not self._has_gathered:
                raise RuntimeError("Cannot normalize a weights map before it is gathered.")
            norm_map = np.ascontiguousarray(self._gathered_map, dtype=np.float64)
            rms = np.zeros((3, self.npix), dtype=np.float64)
            self.maplib.map_invdiag_IQU_f64(rms, norm_map, self.npix)
            self._finalized_rms_map = rms.astype(self.dtype, copy=False)

    def normalize_map_Python(self):
        """Reference RMS computation using NumPy inversion."""
        if self.map_comm.Get_rank() == 0:
            if not self._has_gathered:
                raise RuntimeError("Cannot normalize a weights map before it is gathered.")
            # Pixels which cannot be inverted keep RMS = inf (zero weight), as in the C++ solver.
            self._finalized_rms_map = np.full((3, self.npix), np.inf, dtype=self.dtype)
 
            # `A` matrix is float64 no matter what, to get very accurate inversion.
            A = np.zeros((self.npix, 3, 3), dtype=np.float64)
            A[:, 0, 0] = self._gathered_map[0]
            A[:, 0, 1] = self._gathered_map[1]
            A[:, 1, 0] = self._gathered_map[1]
            A[:, 0, 2] = self._gathered_map[2]
            A[:, 2, 0] = self._gathered_map[2]
            A[:, 1, 1] = self._gathered_map[3]
            A[:, 1, 2] = self._gathered_map[4]
            A[:, 2, 1] = self._gathered_map[4]
            A[:, 2, 2] = self._gathered_map[5]

            # Check if matrix is non-SPD, singular, or ill-conditioned.
            det = np.linalg.det(A)
            diag_prod = A[:, 0, 0] * A[:, 1, 1] * A[:, 2, 2]
            eps = np.finfo(np.float64).eps
            mask = (diag_prod > 0) & (np.abs(det) > eps * diag_prod)

            # If any of the above, RMS is set to inf.
            if np.any(mask):
                A_inv = np.linalg.inv(A[mask])
                diag = np.diagonal(A_inv, axis1=1, axis2=2)
                diag = np.where(diag >= 0, np.sqrt(diag), np.inf)
                self._finalized_rms_map[:, mask] = diag.T.astype(self.dtype, copy=False)


def tod2map_bin(band_comm: MPI.Comm, experiment_data: DetectorGroupTOD, compsep_output: NDArray,
                tod_samples: TODSamples, iteration: int,
                mapmaking: MapmakingConfig, correlated_noise: CorrelatedNoiseConfig,
                data_selection: DataSelectionConfig,
                ) -> tuple[dict[str, DetectorMap], dict[str, NDArray]]:
    """ Commander4 bin mapmaking. All ranks on the provided MPI communicator collaborates on creating
        the band maps (sky signal, inverse variance, possibly also aux maps like orbital dipole).
    Args:
        band_comm (Comm): The communicator consisting of all MPI ranks which holds TOD data that
                          should go into the same map.
        experiment_data (DetectorGroupTOD): TOD data class to be made into maps.
        compsep_output (NDArray): The sky model at our band. Not used, but written to chain file.
        tod_samples (TODSamples): Sampled TOD parameters, such as gain.
        iteration: Current Gibbs iteration.
        mapmaking: Validated mapmaking settings.
        correlated_noise: Validated correlated-noise settings.
        data_selection: Validated detector-scan selection settings.
    Output:
        Detector maps for component separation and maps selected for chain output.

    """
    start_bench("binned-mapmaker")
    corr_noise_active = correlated_noise.is_active(iteration)
    selection_active = data_selection.cuts_are_active(iteration, correlated_noise)
    pols = experiment_data.pols
    scan_view = TODView(experiment_data, tod_samples, compsep_output=compsep_output)
    # Optional per-experiment sparse map storage: each rank holds only its locally-observed pixels
    # rather than a full sky map. The band master still ends up with full-sky maps.
    domain = experiment_data.get_pixel_domain(scan_view, band_comm, mapmaking.sparse_maps)

    # Set up various mapmakers. Each aux map costs a full-sky map and a per-detector-scan
    # accumulation, so one is built only when the chain is going to hold it; `None` means "not
    # wanted this iteration" all the way through the scan loop and the finalization below.
    mapmaker_invvar = WeightsMapmakerIQU(band_comm, experiment_data.nside, pixel_domain=domain)
    mapmaker = MapmakerIQU(band_comm, experiment_data.nside, pixel_domain=domain)
    mapmaker_orbdipole = (MapmakerIQU(band_comm, experiment_data.nside, pixel_domain=domain)
                          if mapmaking.include_orbital_dipole_maps else None)
    mapmaker_res = (MapmakerIQU(band_comm, experiment_data.nside, pixel_domain=domain)
                    if mapmaking.include_residual_maps else None)
    mapmaker_ncorr = (MapmakerIQU(band_comm, experiment_data.nside, pixel_domain=domain)
                      if corr_noise_active and mapmaking.include_corr_noise_maps else None)
    # Hit counts are a plain per-pixel sample count, so they skip the IQU response/weighting the
    # mapmakers apply and are just accumulated with np.bincount into the local pixel buffer.
    nhit_local = np.zeros(domain.n_local) if mapmaking.include_hit_maps else None
    if corr_noise_active:
        sampled_params = []
        residuals = []
        niters = []
        num_failed_convergences_ncorr = 0
        num_too_high_var_ncorr = 0
        worst_residual_ncorr = 0
    stop_bench("binned-mapmaker")

    ### MAIN SCAN LOOP ###
    for view in scan_view.iter_focused(accepted_only=True):
        start_bench("binned-mapmaker")
        good_data_mask = view.get_mask(proc_mask=False)
        pix, psi = view.pix, view.psi
        pix_masked = pix[good_data_mask]
        psi_masked = psi[good_data_mask]
        response = view.det_response
        gain = view.get_gain()
        stop_bench("binned-mapmaker", increment_count=False)

        ### DATA-SELECTION VETO 1 (too little unflagged data).
        good_frac = good_data_mask.mean()
        tod_samples.good_fraction[view.iscan, view.idet] = good_frac
        if selection_active and good_frac < data_selection.min_good_fraction:
            tod_samples.accept[view.iscan, view.idet] = False
            continue

        ### CORRELATED NOISE / SIGMA0 SAMPLING (first, so the weights below use the new sigma0) ###
        n_corr_est = None
        if corr_noise_active:
            start_bench("ncorr-sampling")
            sky_subtracted_TOD = view.get_tod(
                subtract=(("sky", TODView._ALL_GAIN_TERMS),
                          ("orbital_dipole", TODView._ALL_GAIN_TERMS)),
            )
            res = sample_correlated_noise(
                sky_subtracted_TOD, view.get_mask(proc_mask_type="ncorr"),
                np.array(view.noise_params, copy=True),
                experiment_data.noise_model, view.fsamp, cg_err_tol=correlated_noise.cg.err_tol,
                cg_max_iter=correlated_noise.cg.max_iter,
                sample_params=correlated_noise.sample_psd_params,
                sample_sigma0=correlated_noise.sample_sigma0,
                sigma0_method=correlated_noise.sigma0_method,
                nomono=correlated_noise.nomono,
                onlymono=correlated_noise.onlymono,
                sigma0_dec=correlated_noise.sigma0_decimation,
                psd_fit_nu_min=correlated_noise.psd_fit_nu_min,
                psd_fit_nu_max=correlated_noise.psd_fit_nu_max,
                psd_bin=correlated_noise.psd_bin)
            n_corr_est = res.n_corr
            tod_samples.noise_params[view.iscan, view.idet, :] = res.noise_params
            tod_samples.ncorr_cg_residual[view.iscan, view.idet] = res.residual
            tod_samples.ncorr_cg_niter[view.iscan, view.idet] = res.niter
            tod_samples.ncorr_converged[view.iscan, view.idet] = res.converged and not res.high_var
            if correlated_noise.sample_psd_params:
                sampled_params.append(np.array(res.noise_params, copy=True))
            if not res.converged:
                num_failed_convergences_ncorr += 1
            if res.high_var:
                num_too_high_var_ncorr += 1
            worst_residual_ncorr = max(worst_residual_ncorr, res.residual)
            residuals.append(res.residual)
            niters.append(res.niter)
            stop_bench("ncorr-sampling")
        elif correlated_noise.sample_sigma0:
            # No correlated noise this iteration: estimate sigma0 here, at the same point in the
            # chain (after gain) as the n_corr-coupled estimate, instead of a separate pre-gain pass.
            tod_samples.noise_params[view.iscan, view.idet, 0] = _estimate_standalone_sigma0(
                view, correlated_noise.sigma0_method)

        residual_tod = _record_tod_diagnostics(tod_samples, view.iscan, view.idet, view, n_corr_est)

        ### DATA-SELECTION VETO 2 (catastrophic chi^2)
        if selection_active:
            z = tod_samples.chisq_z[view.iscan, view.idet]
            if not (np.isfinite(z) and abs(z) <= data_selection.chisq_abs_threshold):
                tod_samples.accept[view.iscan, view.idet] = False
                continue

        start_bench("binned-mapmaker")
        # Retrieve the new sigma0 for this det-scan, sampled above.
        sigma0 = view.sigma0
        # sigma0 is in detector-units, transform into uK_RJ by dividing it by the gain.
        inv_var = (gain/sigma0)**2
        mapmaker_invvar.accumulate_to_map(inv_var, pix_masked, psi_masked, response=response)

        ### ORBITAL DIPOLE ###
        d_sky = view.get_tod(subtract=(("orbital_dipole", TODView._ALL_GAIN_TERMS),))

        # If we're doing ncorr, accumulate to map and subtract from sky TOD.
        if mapmaker_ncorr is not None:
            mapmaker_ncorr.accumulate_to_map(
                (n_corr_est[good_data_mask]/gain).astype(np.float32, copy=False),
                inv_var, pix_masked, psi_masked, response=response)
        if corr_noise_active:
            d_sky -= n_corr_est

        d_sky_masked = d_sky[good_data_mask]
        mapmaker.accumulate_to_map(d_sky_masked/gain, inv_var, pix_masked, psi_masked, response=response)
        if mapmaker_orbdipole is not None:
            # The dipole TOD is cached on the view, so `get_tod` above already paid for it.
            sky_orb_dipole = view.get_orbital_dipole_tod()
            mapmaker_orbdipole.accumulate_to_map(sky_orb_dipole[good_data_mask], inv_var,
                                                 pix_masked, psi_masked, response=response)

        ### RESIDUAL AND HIT MAPS ###
        # `residual_tod` is the detector-unit noise residual `_record_tod_diagnostics` already
        # built (sky model, orbital dipole and n_corr all subtracted); dividing by the gain puts it
        # in uK_RJ, like the signal map. Masked like the signal map, so its numerator counts the
        # same good samples as the shared `map_cov` denominator.
        if mapmaker_res is not None:
            mapmaker_res.accumulate_to_map(residual_tod[good_data_mask]/gain, inv_var,
                                           pix_masked, psi_masked, response=response)
        if nhit_local is not None:
            nhit_local += np.bincount(domain.to_local(pix_masked), minlength=domain.n_local)
        stop_bench("binned-mapmaker", increment_count=False)
    if corr_noise_active:
        log_memory("ncorr-sampling")

    ### PRINT NOISE SAMPLING STATS ###
    if corr_noise_active:
        log_corr_noise_stats(band_comm, experiment_data,
                             sampled_params, residuals, niters, num_failed_convergences_ncorr,
                             num_too_high_var_ncorr, worst_residual_ncorr,
                             sum(len(s.detectors) for s in experiment_data.scans),
                             tod_samples.chain, iteration)


    start_bench("binned-mapmaker")
    ### GATHER AND NORMALIZE MAPS ###
    # Finalize the inverse-variance map (now accumulated with this iteration's sigma0) before reading
    # its rms/cov, which normalize the signal and every aux map below.
    mapmaker_invvar.gather_map()
    mapmaker_invvar.normalize_map()
    mapmaker.gather_map()
    map_rms = mapmaker_invvar.final_rms_map
    map_cov = mapmaker_invvar.final_cov_map
    mapmaker.normalize_map(map_cov)
    map_signal = mapmaker.final_map

    def finalize_aux(aux_mapmaker: MapmakerIQU | None) -> NDArray | None:
        """Gather and normalize one aux map against the shared cov; `None` if it was not built."""
        if aux_mapmaker is None:
            return None
        aux_mapmaker.gather_map()
        aux_mapmaker.normalize_map(map_cov)
        return aux_mapmaker.final_map

    map_orbdipole = finalize_aux(mapmaker_orbdipole)
    map_corrnoise = finalize_aux(mapmaker_ncorr)
    map_residual = finalize_aux(mapmaker_res)
    map_nhit = None
    if nhit_local is not None:
        nhit_full = domain.reduce_to_full(nhit_local)
        if nhit_full is not None:
            map_nhit = np.round(nhit_full).astype(np.int64)
    stop_bench("binned-mapmaker", increment_count=False)
    log_memory("binned-mapmaker")

    ### FINAL CLEANUP ON MASTER RANK ###
    detmap_dict_out = {}
    maps_to_file = {}
    if band_comm.Get_rank() == 0:
        detmap_dict_out, maps_to_file = finalize_band_maps(
            map_signal, map_rms, pols, experiment_data, mapmaking, tod_samples, compsep_output,
            map_orbdipole=map_orbdipole, map_corrnoise=map_corrnoise,
            map_residual=map_residual, map_nhit=map_nhit, map_cov=map_cov)

    return detmap_dict_out, maps_to_file
