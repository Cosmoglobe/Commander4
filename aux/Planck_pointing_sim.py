import numpy as np
import healpy as hp
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import get_sun, SkyCoord
from scipy.interpolate import CubicSpline
from tqdm import trange


def get_Planck_pointing(sample_rate = 32.5015421, duration_days = 16436):
    """ Function for simulating the Planck pointing strategy and returning data necessary for TOD simulations.
        Args:
            sample_rate: Sampling rate in Hz.
            duration_days: Number of Planck mission days to simulate.
        Returns:
            theta_arr: [Ntime,] array of theta angle pointing values for every time sample.
            phi_arr: [Ntime,] array of corresponding phi angles.
            LOS_arr: [Ntime,3] array with satellite pointing in a solar coordinate system.
            orb_dir_arr: [Ntime,3] array with satellite orbital directions, in the same coordinate system as above.
            dipole_arr: [Ntime,] array with the orbital CMB dipole pickup, in Kelvin.
    """
    # -- 1. Define Simulation Parameters
    SAMPLE_RATE_HZ = sample_rate
    DURATION_DAYS = duration_days

    LOS_ANGLE_DEG = 85.0
    SPIN_AXIS_TILT_DEG = 7.5
    SPIN_RATE_RPM = 1.00165345964511
    BATCH_DURATION_DAYS = 1

    # Dipole calculation parameters
    T_CMB = 2.72548  # K
    C_LIGHT = 299792458.0 # m/s
    V_ORBITAL_SPEED = 30000.0 # m/s, approximate orbital speed of Planck

    # -- 2. Set up Main Loop and Total Hit Map --
    print(f"Total duration: {DURATION_DAYS} days, processing in {BATCH_DURATION_DAYS}-day batches.")

    num_batches = int(np.ceil(DURATION_DAYS / BATCH_DURATION_DAYS))

    theta_arr = []
    phi_arr = []
    dipole_arr = []
    LOS_arr = []
    orb_dir_arr = []

    for i in trange(num_batches):
        
        start_day = i * BATCH_DURATION_DAYS
        start_time_offset = start_day * 24 * 3600 * u.s
        batch_start_time = Time('2009-08-13T00:00:00') + start_time_offset

        batch_duration_sec = BATCH_DURATION_DAYS * 24 * 3600
        num_samples = int(batch_duration_sec * SAMPLE_RATE_HZ)
        t_seconds = np.linspace(0, batch_duration_sec, num_samples, endpoint=False)

        # -- Get Angles --
        COARSE_STEP_SEC = 600
        t_coarse = np.arange(0, batch_duration_sec + COARSE_STEP_SEC, COARSE_STEP_SEC)
        obs_times_coarse = batch_start_time + t_coarse * u.s
        
        sun_coords_coarse = get_sun(obs_times_coarse).transform_to('geocentrictrueecliptic')
        anti_sun_lon_coarse = sun_coords_coarse.lon.rad + np.pi
        
        cos_spline = CubicSpline(t_coarse, np.cos(anti_sun_lon_coarse))
        sin_spline = CubicSpline(t_coarse, np.sin(anti_sun_lon_coarse))
        cos_interp = cos_spline(t_seconds)
        sin_interp = sin_spline(t_seconds)
        norm = np.sqrt(cos_interp**2 + sin_interp**2)
        anti_sun_lon_rad = np.arctan2(sin_interp / norm, cos_interp / norm)
        
        spin_phase_rad = (2 * np.pi * SPIN_RATE_RPM / 60.0) * (start_day * 24 * 3600 + t_seconds)
        six_months_sec = 182.625 * 24 * 3600
        precession_phase_rad = 2 * np.pi * (start_day * 24 * 3600 + t_seconds) / six_months_sec

        # -- Generate Pointing Vectors --
        los_angle_rad = np.deg2rad(LOS_ANGLE_DEG)
        tilt_angle_rad = np.deg2rad(SPIN_AXIS_TILT_DEG)

        a_vec = np.vstack([np.cos(anti_sun_lon_rad), np.sin(anti_sun_lon_rad), np.zeros_like(anti_sun_lon_rad)]).T  # "anti-sun" vector
        b_vec = np.vstack([-np.sin(anti_sun_lon_rad), np.cos(anti_sun_lon_rad), np.zeros_like(anti_sun_lon_rad)]).T  # Orbital direction vector
        c_vec = np.array([0., 0., 1.])  # NCP vector

        # Spin axis vector.
        s_vec = (np.cos(tilt_angle_rad) * a_vec + 
                np.sin(tilt_angle_rad) * (np.cos(precession_phase_rad)[:, np.newaxis] * b_vec + 
                                        np.sin(precession_phase_rad)[:, np.newaxis] * c_vec))
        
        u_vec = np.cross(s_vec, c_vec)
        u_vec /= np.linalg.norm(u_vec, axis=1)[:, np.newaxis]
        v_vec = np.cross(s_vec, u_vec)

        # Line-of-sight vector (direction satellite is currently looking).
        z_vec_ecliptic = (np.cos(los_angle_rad) * s_vec + 
                        np.sin(los_angle_rad) * (np.cos(spin_phase_rad)[:, np.newaxis] * u_vec + 
                                                np.sin(spin_phase_rad)[:, np.newaxis] * v_vec))
        
        # -- Generate Orbital Dipole --        
        # 1. Define orbital velocity vector (v_orbital)
        # The direction is orthogonal to the anti-Sun vector in the ecliptic plane (b_vec)
        v_orbital_vec = V_ORBITAL_SPEED * b_vec

        # 2. Calculate beta_vec = v_orbital / c and gamma
        beta_vec = v_orbital_vec / C_LIGHT
        # For a constant speed, beta_mag_sq and gamma are constant
        beta_mag_sq = (V_ORBITAL_SPEED / C_LIGHT)**2
        gamma = 1.0 / np.sqrt(1.0 - beta_mag_sq)

        # 3. Calculate the dot product of beta_vec and the pointing vector x
        dot_product = np.sum(beta_vec * z_vec_ecliptic, axis=1)

        # 4. Apply Equation 7 to get the dipole amplitude in Kelvin
        orbital_dipole_amplitude = T_CMB * ( (1.0 / (gamma * (1.0 - dot_product))) - 1.0 )

        # -- Transform data --
        ecl_basis = SkyCoord(x=[1,0,0], y=[0,1,0], z=[0,0,1], representation_type='cartesian', frame='geocentrictrueecliptic')
        ecl_to_gal_matrix = ecl_basis.transform_to('galactic').cartesian.xyz.value
        z_vec_galactic = z_vec_ecliptic @ ecl_to_gal_matrix.T
        
        theta, phi = hp.vec2ang(z_vec_galactic)
        theta_arr.append(theta)
        phi_arr.append(phi)
        dipole_arr.append(orbital_dipole_amplitude)
        LOS_arr.append(z_vec_ecliptic)
        orb_dir_arr.append(v_orbital_vec)

    theta_arr = np.concatenate(theta_arr)
    phi_arr = np.concatenate(phi_arr)
    dipole_arr = np.concatenate(dipole_arr)
    LOS_arr = np.concatenate(LOS_arr)
    orb_dir_arr = np.concatenate(orb_dir_arr)

    return theta_arr, phi_arr, LOS_arr, orb_dir_arr, dipole_arr