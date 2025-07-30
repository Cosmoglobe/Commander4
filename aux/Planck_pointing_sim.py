import numpy as np
import healpy as hp
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import get_sun, SkyCoord, GeocentricTrueEcliptic, Galactic
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from mpi4py import MPI

def get_Planck_pointing(ntod, sample_rate = 32.5015421):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # -- 1. Define Simulation Parameters
    SAMPLE_RATE_HZ = sample_rate
    DURATION_DAYS = ntod / (sample_rate * 3600 * 24)
    LOS_ANGLE_DEG = 85.0
    SPIN_AXIS_TILT_DEG = 7.5
    SPIN_RATE_RPM = 1.00165345964511
    BATCH_DURATION_DAYS = 1
    T_CMB = 2.72548
    C_LIGHT = 299792458.0
    V_ORBITAL_SPEED = 30000.0
    samples_per_batch = int(BATCH_DURATION_DAYS * 24 * 3600 * SAMPLE_RATE_HZ)

    # ---------------- HELPER FUNCTION FOR A SINGLE BATCH ----------------
    def calculate_batch(batch_idx):
        """Performs all calculations for a single batch index."""
        start_day = batch_idx * BATCH_DURATION_DAYS
        start_time_offset = start_day * 24 * 3600 * u.s
        batch_start_time = Time('2009-08-13T00:00:00') + start_time_offset
        batch_duration_sec = BATCH_DURATION_DAYS * 24 * 3600
        num_samples = int(batch_duration_sec * SAMPLE_RATE_HZ)
        t_seconds = np.linspace(0, batch_duration_sec, num_samples, endpoint=False)
        COARSE_STEP_SEC = 600
        t_coarse = np.arange(0, batch_duration_sec + COARSE_STEP_SEC, COARSE_STEP_SEC)
        obs_times_coarse = batch_start_time + t_coarse * u.s
        sun_coords_coarse = get_sun(obs_times_coarse).transform_to('geocentrictrueecliptic')
        anti_sun_lon_coarse = sun_coords_coarse.lon.rad + np.pi
        cos_interp = np.interp(t_seconds, t_coarse, np.cos(anti_sun_lon_coarse))
        sin_interp = np.interp(t_seconds, t_coarse, np.sin(anti_sun_lon_coarse))
        norm = np.sqrt(cos_interp**2 + sin_interp**2)
        anti_sun_lon_rad = np.arctan2(sin_interp / norm, cos_interp / norm)
        spin_phase_rad = (2 * np.pi * SPIN_RATE_RPM / 60.0) * (start_day * 24 * 3600 + t_seconds)
        six_months_sec = 182.625 * 24 * 3600
        precession_phase_rad = 2 * np.pi * (start_day * 24 * 3600 + t_seconds) / six_months_sec
        los_angle_rad = np.deg2rad(LOS_ANGLE_DEG)
        tilt_angle_rad = np.deg2rad(SPIN_AXIS_TILT_DEG)
        a_vec = np.vstack([np.cos(anti_sun_lon_rad), np.sin(anti_sun_lon_rad), np.zeros_like(anti_sun_lon_rad)]).T
        b_vec = np.vstack([-np.sin(anti_sun_lon_rad), np.cos(anti_sun_lon_rad), np.zeros_like(anti_sun_lon_rad)]).T
        c_vec = np.array([0., 0., 1.])
        s_vec = (np.cos(tilt_angle_rad) * a_vec + np.sin(tilt_angle_rad) * (np.cos(precession_phase_rad)[:, np.newaxis] * b_vec + np.sin(precession_phase_rad)[:, np.newaxis] * c_vec))
        u_vec = np.cross(s_vec, c_vec)
        u_vec /= np.linalg.norm(u_vec, axis=1)[:, np.newaxis]
        v_vec = np.cross(s_vec, u_vec)
        z_vec_ecliptic = (np.cos(los_angle_rad) * s_vec + np.sin(los_angle_rad) * (np.cos(spin_phase_rad)[:, np.newaxis] * u_vec + np.sin(spin_phase_rad)[:, np.newaxis] * v_vec))
        v_orbital_vec = V_ORBITAL_SPEED * b_vec
        beta_vec = v_orbital_vec / C_LIGHT
        beta_mag_sq = (V_ORBITAL_SPEED / C_LIGHT)**2
        gamma = 1.0 / np.sqrt(1.0 - beta_mag_sq)
        dot_product = np.sum(beta_vec * z_vec_ecliptic, axis=1)
        orbital_dipole_amplitude = T_CMB * ((1.0 / (gamma * (1.0 - dot_product))) - 1.0)
        ecl_basis = SkyCoord(x=[1,0,0], y=[0,1,0], z=[0,0,1], representation_type='cartesian', frame='geocentrictrueecliptic')
        ecl_to_gal_matrix = ecl_basis.transform_to('galactic').cartesian.xyz.value
        z_vec_galactic = z_vec_ecliptic @ ecl_to_gal_matrix.T
        theta, phi = hp.vec2ang(z_vec_galactic)
        return theta, phi, z_vec_ecliptic, v_orbital_vec, orbital_dipole_amplitude
    # --------------------------------------------------------------------

    # -- Master Logic (Rank 0) --
    if rank == 0:
        num_batches = int(np.ceil(DURATION_DAYS / BATCH_DURATION_DAYS))
        
        # Pre-allocate memory for the final arrays
        theta_arr = np.empty(ntod, dtype=np.float64)
        phi_arr = np.empty(ntod, dtype=np.float64)
        dipole_arr = np.empty(ntod, dtype=np.float64)
        LOS_arr = np.empty((ntod, 3), dtype=np.float64)
        orb_dir_arr = np.empty((ntod, 3), dtype=np.float64)

        # Handle the two cases: parallel (size > 1) or serial (size == 1)
        if size > 1:
            # Create a list of jobs for the workers
            all_batches = np.arange(num_batches)
            # Split the jobs among the worker processes (size - 1 of them)
            worker_chunks = np.array_split(all_batches, size - 1)
            # The master (rank 0) gets an empty list of jobs
            chunks_for_ranks = [[]] + worker_chunks
        else: # Running on a single process
            print("Running in serial mode on 1 process.")
            chunks_for_ranks = [np.arange(num_batches)]
        
        # Scatter the jobs
        my_batches = comm.scatter(chunks_for_ranks, root=0)

        # Master process receives results from all workers
        print(f"Master (Rank 0) receiving results from {size - 1} workers...")
        for _ in tqdm(range(num_batches), desc="Receiving Batches"):
            batch_idx, theta, phi, z_vec, v_orb, dipole = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            
            # Calculate the correct slice and place data into the full arrays
            start_idx = batch_idx * samples_per_batch
            end_idx = start_idx + len(theta)
            if end_idx > ntod:
                end_idx = ntod
                # Truncate data if it's from the final, partial batch
                theta, phi, dipole, z_vec, v_orb = (arr[:end_idx-start_idx] for arr in (theta, phi, dipole, z_vec, v_orb))

            theta_arr[start_idx:end_idx] = theta
            phi_arr[start_idx:end_idx] = phi
            dipole_arr[start_idx:end_idx] = dipole
            LOS_arr[start_idx:end_idx, :] = z_vec
            orb_dir_arr[start_idx:end_idx, :] = v_orb

    # -- Worker Logic (Ranks > 0) --
    else: # if rank > 0:
        my_batches = comm.scatter(None, root=0) # Receive the scattered jobs
        if len(my_batches) > 0:
            print(f"Worker (Rank {rank}) processing {len(my_batches)} batches (from {my_batches[0]} to {my_batches[-1]}).")
        
        for batch_idx in my_batches:
            results = calculate_batch(batch_idx)
            comm.send((batch_idx, *results), dest=0, tag=rank)

    # All processes wait here until Rank 0 has finished its work
    comm.Barrier()

    if rank == 0:
        return theta_arr, phi_arr, LOS_arr, orb_dir_arr, dipole_arr
    else:
        return None, None, None, None, None