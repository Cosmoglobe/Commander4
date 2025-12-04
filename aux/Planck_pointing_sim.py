import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord, get_sun
from astropy.time import Time
from mpi4py import MPI
from tqdm import tqdm


def get_Planck_pointing(ntod, sample_rate = 32.5015421,
                        los_angle=85.0, spin_angle_tilt=7.5,
                        spin_rate=1.00165345964511,
                        batch_duration=1):
    """
    Generate Planck-like pointing and orbital dipole data using MPI for parallel processing.
    
    Parameters
    ----------
    ntod : int
        Total number of time-ordered data samples to simulate.
    sample_rate : float, optional
        Sampling rate in Hz. Default is 32.5015421 Hz.
    los_angle : float, optional
        Line-of-sight angle in degrees. Default is 85.0 degrees.
    spin_angle_tilt : float, optional
        Spin axis tilt angle in degrees. Default is 7.5 degrees.
    spin_rate : float, optional
        Spin rate in revolutions per minute (RPM). Default is 1.00165345964511 RPM.
    batch_duration : float, optional
        Duration of each batch in days for processing. Default is 1 day.

    Returns
    -------
    theta_arr : np.ndarray
        Array of theta angles in radians for each sample.
    phi_arr : np.ndarray
        Array of phi angles in radians for each sample.
    orb_dir_arr : np.ndarray
        Array of shape (ntod, 3) containing the orbital direction vectors.
    dipole_arr : np.ndarray
        Array of orbital dipole amplitudes in Kelvin for each sample.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # constants
    T_CMB = 2.72548
    C_LIGHT = 299792458.0
    V_ORBITAL_SPEED = 30000.0
    # computational variables
    COARSE_STEP_SEC = 600

    # determine simulation parameters from given inputs
    duration = ntod / (sample_rate * 3600 * 24)
    samples_per_batch = int(batch_duration * 24 * 3600 * sample_rate)

    # ---------------- HELPER FUNCTION FOR A SINGLE BATCH ----------------
    def calculate_batch(batch_idx):
        """
        Performs all calculations for a single batch index (corresponding to batch_duration).
        """
        # calculate start time of batch
        start_day = batch_idx * batch_duration
        start_time_offset = start_day * 24 * 3600 * u.s
        batch_start_time = Time('2009-08-13T00:00:00') + start_time_offset

        # generate time array for the batch
        batch_duration_sec = batch_duration * 24 * 3600 # seconds
        n_samples = int(batch_duration_sec * sample_rate)
        t_seconds = np.linspace(0, batch_duration_sec, n_samples, endpoint=False)
        t_coarse = np.arange(0, batch_duration_sec + COARSE_STEP_SEC, COARSE_STEP_SEC)

        # generate (anti-)sun positions at coarse time intervals
        obs_times_coarse = batch_start_time + t_coarse * u.s
        sun_coords_coarse = get_sun(obs_times_coarse).transform_to('geocentrictrueecliptic')
        anti_sun_lon_coarse = sun_coords_coarse.lon.rad + np.pi

        # interpolate (anti-)sun positions to fine time intervals
        cos_interp = np.interp(t_seconds, t_coarse, np.cos(anti_sun_lon_coarse))
        sin_interp = np.interp(t_seconds, t_coarse, np.sin(anti_sun_lon_coarse))
        norm = np.sqrt(cos_interp**2 + sin_interp**2)
        anti_sun_lon = np.arctan2(sin_interp / norm, cos_interp / norm)
        
        # generate basis for anti-sun coordinate system
        anti_sun_cos = np.cos(anti_sun_lon)
        anti_sun_sin = np.sin(anti_sun_lon)
        a_vec = np.vstack([anti_sun_cos, anti_sun_sin, np.zeros_like(anti_sun_lon)]).T
        b_vec = np.vstack([-anti_sun_sin, anti_sun_cos, np.zeros_like(anti_sun_lon)]).T
        c_vec = np.array([0., 0., 1.])

        # generate basis spin
        tilt_angle = np.deg2rad(spin_angle_tilt) # radians
        six_months = 182.625 * 24 * 3600 # seconds
        precession_phase = 2 * np.pi * (start_day * 24 * 3600 + t_seconds) / six_months # radians
        s_vec = (np.cos(tilt_angle) * a_vec +
                 np.sin(tilt_angle) * (np.cos(precession_phase)[:, np.newaxis] * b_vec +
                                       np.sin(precession_phase)[:, np.newaxis] * c_vec))
        u_vec = np.cross(s_vec, c_vec)
        v_vec = np.cross(s_vec, u_vec)

        # calculate line-of-sight vector in ecliptic coordinates
        spin_phase = (2 * np.pi * spin_rate / 60.0) * (start_day * 24 * 3600 + t_seconds) # radians
        los_angle_rad = np.deg2rad(los_angle) # radians
        los_cos = np.cos(los_angle_rad)
        los_sin = np.sin(los_angle_rad)
        spin_cos = np.cos(spin_phase)[:, np.newaxis]
        spin_sin = np.sin(spin_phase)[:, np.newaxis]
        z_vec_ecliptic = (los_cos * s_vec + los_sin * (spin_cos * u_vec + spin_sin * v_vec))

        # calculate orbital dipole parameters
        v_orbital_vec = V_ORBITAL_SPEED * b_vec
        beta_vec = v_orbital_vec / C_LIGHT
        beta_mag_sq = (V_ORBITAL_SPEED / C_LIGHT)**2
        gamma = 1.0 / np.sqrt(1.0 - beta_mag_sq)
        dot_product = np.sum(beta_vec * z_vec_ecliptic, axis=1)
        orbital_dipole_amplitude = T_CMB * ((1.0 / (gamma * (1.0 - dot_product))) - 1.0)

        # transform pointing and velocity vectors to galactic coordinates
        ecl_basis = SkyCoord(x=[1,0,0], y=[0,1,0], z=[0,0,1],
                             representation_type='cartesian', frame='geocentrictrueecliptic')
        ecl_to_gal_matrix = ecl_basis.transform_to('galactic').cartesian.xyz.value
        z_vec_galactic = z_vec_ecliptic @ ecl_to_gal_matrix.T
        v_orbital_vec_galactic = v_orbital_vec @ ecl_to_gal_matrix.T

        # convert pointing vectors to spherical coordinates
        theta, phi = hp.vec2ang(z_vec_galactic)
        return theta, phi, v_orbital_vec_galactic, orbital_dipole_amplitude
    # --------------------------------------------------------------------

    # -- Master Logic (Rank 0) --
    if rank == 0:
        num_batches = int(np.ceil(duration / batch_duration))
        
        # Pre-allocate memory for the final arrays
        theta_arr = np.empty(ntod, dtype=np.float32)
        phi_arr = np.empty(ntod, dtype=np.float32)
        dipole_arr = np.empty(ntod, dtype=np.float32)
        orb_dir_arr = np.empty((ntod, 3), dtype=np.float32)

        # Create a list of jobs for the workers
        all_batches = np.arange(num_batches)
        # Split the jobs among the worker processes (size - 1 of them)
        worker_chunks = np.array_split(all_batches, size - 1)
        # The master (rank 0) gets an empty list of jobs
        chunks_for_ranks = [[]] + worker_chunks
        
        # Scatter the jobs
        my_batches = comm.scatter(chunks_for_ranks, root=0)

        # Master process receives results from all workers
        print(f"Master (Rank 0) receiving results from {size - 1} workers...")
        for _ in tqdm(range(num_batches), desc="Receiving Batches"):
            batch_idx, theta, phi, v_orb, dipole = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            
            # Calculate the correct slice and place data into the full arrays
            start_idx = batch_idx * samples_per_batch
            end_idx = start_idx + len(theta)
            if end_idx > ntod:
                end_idx = ntod
                # Truncate data if it's from the final, partial batch
                theta, phi, dipole, v_orb = (arr[:end_idx-start_idx] for arr in (theta, phi, dipole, v_orb))

            theta_arr[start_idx:end_idx] = theta
            phi_arr[start_idx:end_idx] = phi
            dipole_arr[start_idx:end_idx] = dipole
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
        return theta_arr, phi_arr, orb_dir_arr, dipole_arr
    else:
        return None, None, None, None
    
if __name__ == "__main__":
    # Example usage
    ntod = 100000  # Total number of samples
    theta, phi, orb_dir, dipole = get_Planck_pointing(ntod)
    if MPI.COMM_WORLD.Get_rank() == 0:
        pix = hp.ang2pix(2048, theta, phi)
        hit_map = np.bincount(pix, minlength=hp.nside2npix(2048))
        hp.mollview(hit_map, title="Hit Map", unit="Hits")
        plt.savefig("hit_map.png")
