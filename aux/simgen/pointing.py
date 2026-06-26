"""Pointing strategies (swappable).

A pointing strategy maps a contiguous block of TOD samples -- identified by its absolute sample
offset within the mission and its length -- to sky pointing for a band's shared boresight:

    compute(sample_offset, ntod) -> PointingChunk(theta, phi, psi, vsun)

``theta``/``phi`` are HEALPix angles (radians, Galactic), ``psi`` is the boresight polarization
(scan-direction) angle (radians), and ``vsun`` is the spacecraft orbital velocity (m/s, Galactic)
at the chunk midpoint, used for the orbital dipole. Per-detector pointing is obtained later by
adding each detector's polarization-angle offset to ``psi`` (the shared-boresight model).

Add a new strategy by subclassing ``PointingStrategy`` and registering it in
``POINTING_STRATEGIES``.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import healpy as hp
from numpy.typing import NDArray
from pixell.bunch import Bunch

from simgen.config import bget

logger = logging.getLogger(__name__)


@dataclass
class PointingChunk:
    theta: NDArray[np.floating]   # co-latitude [rad], Galactic
    phi: NDArray[np.floating]     # longitude [rad], Galactic
    psi: NDArray[np.floating]     # boresight polarization angle [rad]
    vsun: NDArray[np.floating]    # orbital velocity at midpoint (3,) [m/s], Galactic


def scan_bearing(theta: NDArray, phi: NDArray) -> NDArray:
    """Initial bearing along the scan path, used as the boresight polarization angle.

    The bearing from each sample to the next is computed on the sphere (lon=phi, lat=pi/2-theta);
    the final sample repeats the previous bearing. This gives a smooth, deterministic polarization
    reference tied to the scan direction.
    """
    lat = np.pi / 2.0 - theta
    dlon = np.diff(phi)
    y = np.sin(dlon) * np.cos(lat[1:])
    x = np.cos(lat[:-1]) * np.sin(lat[1:]) - np.sin(lat[:-1]) * np.cos(lat[1:]) * np.cos(dlon)
    bearing = np.arctan2(y, x)
    if bearing.size == 0:
        return np.zeros_like(theta)
    return np.concatenate([bearing, bearing[-1:]]).astype(np.float32)


class PointingStrategy(ABC):
    # If True, the pipeline calls ``compute`` once per detector, passing that detector's focal-plane
    # offset as ``det_offset`` so the strategy can give detectors distinct *spatial* pointing (e.g.
    # RasterScan). The default False keeps the efficient shared-boresight path used by the satellite
    # strategies, where detectors differ only by their polarization-angle offset.
    per_detector_pointing: bool = False

    def __init__(self, params: Bunch, fsamp: float):
        self.params = params
        self.fsamp = fsamp

    def samples_per_scan(self, scan_duration_sec: float, fsamp: float) -> int:
        """Number of samples in one scan. Default: from the scan duration, rounded to even.

        Strategies whose natural scan unit is not a fixed wall-clock duration (e.g. RasterScan, one
        scan = one full patch fill) override this.
        """
        n = int(round(scan_duration_sec * fsamp))
        n -= n % 2          # keep even for clean real FFTs in the noise model
        return max(n, 2)

    @abstractmethod
    def compute(self, sample_offset: int, ntod: int,
                det_offset: tuple[float, float] = (0.0, 0.0)) -> PointingChunk:
        """Pointing for ``ntod`` samples starting at absolute index ``sample_offset``.

        ``det_offset`` is a per-detector focal-plane offset ``(xi, eta)`` in radians; strategies
        with ``per_detector_pointing = True`` use it to spatially offset detectors, others ignore it.
        """


class PlanckScan(PointingStrategy):
    """Analytic Planck-like satellite scanning law (port of ``aux/Planck_pointing_sim.py``).

    The spin axis points near the anti-Sun direction, precesses about it, and the line of sight
    sweeps a cone of half-angle ``los_angle`` at ``spin_rate`` rpm. Pointing and the relativistic
    orbital velocity are computed in the ecliptic frame and rotated to Galactic coordinates. Unlike
    the original, this evaluates an arbitrary time window so scans can be distributed across ranks.
    """
    # Physical / scanning constants.
    C_LIGHT = 299792458.0
    V_ORBITAL_SPEED = 30000.0     # [m/s]
    COARSE_STEP_SEC = 600         # Sun position is sampled coarsely then interpolated.

    def __init__(self, params: Bunch, fsamp: float):
        super().__init__(params, fsamp)
        self.los_angle = np.deg2rad(bget(params, "los_angle", 85.0))
        self.spin_tilt = np.deg2rad(bget(params, "spin_angle_tilt", 7.5))
        self.spin_rate = bget(params, "spin_rate", 1.00165345964511)  # [rpm]
        self.mission_start = bget(params, "mission_start", "2009-08-13T00:00:00")
        # Ecliptic->Galactic rotation matrix, built once (astropy import is local to keep the module
        # importable without astropy where only other strategies are used).
        from astropy.coordinates import SkyCoord
        ecl_basis = SkyCoord(x=[1, 0, 0], y=[0, 1, 0], z=[0, 0, 1],
                             representation_type='cartesian', frame='geocentrictrueecliptic')
        self._ecl_to_gal = ecl_basis.transform_to('galactic').cartesian.xyz.value

    def compute(self, sample_offset: int, ntod: int,
                det_offset: tuple[float, float] = (0.0, 0.0)) -> PointingChunk:
        from astropy.coordinates import get_sun
        from astropy.time import Time
        import astropy.units as u

        # Absolute mission time (seconds) for each sample in this chunk.
        t_abs = (sample_offset + np.arange(ntod, dtype=np.float64)) / self.fsamp
        start_time = Time(self.mission_start)

        # Anti-Sun longitude on a coarse grid, then linearly interpolated to all samples (matching
        # the original implementation; interpolating cos/sin avoids the 2*pi wrap).
        t_coarse = np.arange(t_abs[0], t_abs[-1] + self.COARSE_STEP_SEC, self.COARSE_STEP_SEC)
        sun_coarse = get_sun(start_time + t_coarse * u.s).transform_to('geocentrictrueecliptic')
        anti_sun_lon_coarse = sun_coarse.lon.rad + np.pi
        cos_i = np.interp(t_abs, t_coarse, np.cos(anti_sun_lon_coarse))
        sin_i = np.interp(t_abs, t_coarse, np.sin(anti_sun_lon_coarse))
        norm = np.sqrt(cos_i**2 + sin_i**2)
        anti_sun_lon = np.arctan2(sin_i / norm, cos_i / norm)

        # Anti-Sun coordinate basis.
        a_vec = np.vstack([np.cos(anti_sun_lon), np.sin(anti_sun_lon), np.zeros(ntod)]).T
        b_vec = np.vstack([-np.sin(anti_sun_lon), np.cos(anti_sun_lon), np.zeros(ntod)]).T
        c_vec = np.array([0., 0., 1.])

        # Spin axis precesses about the anti-Sun direction with a 6-month period.
        six_months = 182.625 * 24 * 3600
        prec_phase = 2 * np.pi * t_abs / six_months
        s_vec = (np.cos(self.spin_tilt) * a_vec
                 + np.sin(self.spin_tilt) * (np.cos(prec_phase)[:, None] * b_vec
                                             + np.sin(prec_phase)[:, None] * c_vec))
        u_vec = np.cross(s_vec, c_vec)
        v_vec = np.cross(s_vec, u_vec)

        # Line of sight sweeps a cone about the spin axis at the spin rate.
        spin_phase = (2 * np.pi * self.spin_rate / 60.0) * t_abs
        z_ecl = (np.cos(self.los_angle) * s_vec
                 + np.sin(self.los_angle) * (np.cos(spin_phase)[:, None] * u_vec
                                             + np.sin(spin_phase)[:, None] * v_vec))

        # Orbital velocity (anti-Sun tangential direction b_vec) in the ecliptic, then Galactic.
        v_orb_ecl = self.V_ORBITAL_SPEED * b_vec
        z_gal = z_ecl @ self._ecl_to_gal.T
        v_orb_gal = v_orb_ecl @ self._ecl_to_gal.T

        theta, phi = hp.vec2ang(z_gal)
        psi = scan_bearing(theta, phi)
        vsun = v_orb_gal[ntod // 2].astype(np.float32)
        return PointingChunk(theta.astype(np.float32), phi.astype(np.float32), psi, vsun)


class FilePointing(PointingStrategy):
    """Load precomputed pointing from an HDF5 file (e.g. a downsampled real scan history).

    The file is expected to hold flat per-sample arrays; each chunk reads the slice
    ``[sample_offset : sample_offset+ntod]``. Dataset names are configurable; ``psi`` defaults to the
    scan bearing and ``vsun`` to zeros (no orbital dipole) when absent.
    """
    def __init__(self, params: Bunch, fsamp: float):
        super().__init__(params, fsamp)
        import h5py
        self._f = h5py.File(params.file_path, "r")
        self.theta_key = bget(params, "theta_key", "theta")
        self.phi_key = bget(params, "phi_key", "phi")
        self.psi_key = bget(params, "psi_key", "psi")
        self.vsun_key = bget(params, "vsun_key", "orbital_dir_vec")
        self._warned_vsun = False

    def compute(self, sample_offset: int, ntod: int,
                det_offset: tuple[float, float] = (0.0, 0.0)) -> PointingChunk:
        sl = slice(sample_offset, sample_offset + ntod)
        theta = self._f[self.theta_key][sl].astype(np.float32)
        phi = self._f[self.phi_key][sl].astype(np.float32)
        if self.psi_key in self._f:
            psi = self._f[self.psi_key][sl].astype(np.float32)
        else:
            psi = scan_bearing(theta, phi)
        if self.vsun_key in self._f:
            vsun = self._f[self.vsun_key][sample_offset + ntod // 2].astype(np.float32)
        else:
            if not self._warned_vsun:
                logger.warning("FilePointing: no '%s' dataset; orbital dipole disabled (vsun=0).",
                               self.vsun_key)
                self._warned_vsun = True
            vsun = np.zeros(3, dtype=np.float32)
        return PointingChunk(theta, phi, psi, vsun)


class RasterScan(PointingStrategy):
    """Synthetic raster scan over a small sky patch.

    The boresight sweeps the patch along the longitude ("along-scan") direction at constant rate,
    teleports back to the start of the row when it reaches the end, steps by one row in the
    perpendicular ("cross-scan") latitude direction, and wraps back to the first row after the last,
    so the whole patch is covered. One scan is exactly one full fill of the patch
    (``n_rows * samples_per_row`` samples, overriding the duration-based scan length), and each
    successive scan re-covers the patch from the first row -- so ``scan_duration_sec`` is ignored
    for this strategy. Sample ``i`` maps deterministically to a row and a fractional along-scan
    position, so scans distribute trivially across ranks.

    This strategy sets ``per_detector_pointing = True``: each detector is pointed at the patch
    *shifted by its focal-plane offset* (``det_offset``, radians), so the band's detectors trace
    mutually offset tracks across the patch (see ``Detector.fp_offset`` / ``fp_offset_deg``). There is
    no orbital motion, so ``vsun = 0`` (orbital dipole disabled); the polarization angle sweeps
    0..pi along each row.

    Params (under ``simulation.pointing``):
        patch_center_deg: [lon0, lat0] Galactic centre of the patch (default [0, 0]).
        patch_size_deg:   [width, height] angular extent (default [10, 10]).
        n_rows:           number of cross-scan rows covering the patch height (default 50).
        samples_per_row:  samples to traverse one along-scan row (default 200).
    """
    per_detector_pointing = True

    def __init__(self, params: Bunch, fsamp: float):
        super().__init__(params, fsamp)
        self.lon0, self.lat0 = bget(params, "patch_center_deg", [0.0, 0.0])
        self.width, self.height = bget(params, "patch_size_deg", [10.0, 10.0])
        self.n_rows = int(bget(params, "n_rows", 50))
        self.samples_per_row = int(bget(params, "samples_per_row", 200))

    def samples_per_scan(self, scan_duration_sec: float, fsamp: float) -> int:
        """One scan covers the whole patch exactly once: every row traversed once."""
        return self.n_rows * self.samples_per_row

    def compute(self, sample_offset: int, ntod: int,
                det_offset: tuple[float, float] = (0.0, 0.0)) -> PointingChunk:
        i = sample_offset + np.arange(ntod, dtype=np.int64)
        # Row index (cross-scan) and fractional position along the row (along-scan). The row index
        # wraps modulo n_rows so coverage repeats once the patch is filled.
        row = (i // self.samples_per_row) % self.n_rows
        frac_x = (i % self.samples_per_row) / self.samples_per_row          # [0, 1)
        frac_y = row / max(self.n_rows - 1, 1)                              # [0, 1]
        x = (frac_x - 0.5) * self.width                                     # along-scan, centered
        y = (frac_y - 0.5) * self.height                                    # cross-scan, centered
        # Per-detector focal-plane offset (radians -> degrees), shifting the patch position.
        dxi, deta = np.rad2deg(det_offset[0]), np.rad2deg(det_offset[1])
        lat = self.lat0 + y + deta
        # Divide the along-scan offset by cos(lat0) so a given width spans a roughly constant angle.
        lon = self.lon0 + (x + dxi) / np.cos(np.deg2rad(self.lat0))
        theta = np.deg2rad(90.0 - lat).astype(np.float32)
        phi = np.deg2rad(lon % 360.0).astype(np.float32)
        psi = (np.pi * frac_x).astype(np.float32)        # sweep the pol angle along each row
        vsun = np.zeros(3, dtype=np.float32)             # no orbital motion in a raster scan
        return PointingChunk(theta, phi, psi, vsun)


POINTING_STRATEGIES: dict[str, type[PointingStrategy]] = {
    "planck_scan": PlanckScan,
    "file": FilePointing,
    "raster": RasterScan,
}


def make_pointing(params: Bunch, fsamp: float) -> PointingStrategy:
    """Instantiate the pointing strategy named by ``params.strategy``."""
    name = params.strategy
    if name not in POINTING_STRATEGIES:
        raise ValueError(f"Unknown pointing strategy {name!r}. "
                         f"Available: {sorted(POINTING_STRATEGIES)}.")
    return POINTING_STRATEGIES[name](params, fsamp)
