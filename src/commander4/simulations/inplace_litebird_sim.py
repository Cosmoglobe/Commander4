"""Replacing TOD read from disk with a simulated sky, in place, at read time.

Enabled by a band's ``replace_tod_with_sim``: the real pointing and flags are kept, but the TOD is
regenerated from PySM3/CAMB skies plus noise. Useful for validating the pipeline against a known
truth on a real scan strategy. The standalone ``simgen`` package is the more general
alternative, writing simulated scan files rather than patching them at read time.
"""
import numpy as np
from copy import deepcopy
import camb
import healpy as hp
import pysm3
import pysm3.units as u
import ducc0
from numpy.typing import NDArray
from pixell.bunch import Bunch
from scipy.fft import rfftfreq, rfft, irfft
import gc
from mpi4py import MPI

from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.sky.diffuse_components import ThermalDust, Synchrotron, FreeFree,\
        SpinningDust
from commander4.diagnostics.performance import benchmark, bench_summary, start_bench,\
                                               stop_bench, log_memory, increment_count, bench_reset
from commander4.tod.sky_projection import project_sky_to_tod
from commander4.units import T_CMB, unit_factor


def _scalar_nu_ref(comp_params):
    """Resolve a component's `nu_ref` (scalar or [nu_I, nu_QU] list) to a single scalar (the I
    value), since the in-place simulator uses one reference frequency per component."""
    nu_ref = comp_params.nu_ref
    return nu_ref[0] if isinstance(nu_ref, (list, tuple)) else nu_ref


def generate_cmb(freq, fwhm, units, nside, lmax, params):
    H0 = 67.5
    ombh2 = 0.022
    omch2 = 0.122
    mnu = 0.06
    omk = 0
    tau = 0.06
    As = 2.e-9
    ns = 0.965
    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau, As=As, 
                           ns=ns, halofit_version='mead', lmax=lmax)

    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
    totCL=powers['total']

    ell = np.arange(lmax+1)
    Cl = totCL[ell,0]
    Cl_EE = totCL[ell,1]
    Cl_BB = totCL[ell,2]
    Cl_TE = totCL[ell,3]

    Cls = np.array([Cl, Cl_EE, Cl_BB, Cl_TE])

    np.random.seed(0)
    anisotropy_alms = hp.synalm(Cls, lmax=lmax, new=True)

    # --- Calculate the Solar Dipole alms ---
    # A pure dipole only has l=1 components. We set these directly.
    dipole_alms = np.zeros_like(anisotropy_alms)
    
    # [cite_start]Dipole parameters from the BEYONDPLANCK analysis [cite: 5314, 2003]
    dipole_amplitude_uK = 3362.7
    dipole_glon_deg = 264.11
    dipole_glat_deg = 48.279

    # Convert direction to spherical coordinates (theta, phi) in radians
    theta = np.deg2rad(90.0 - dipole_glat_deg)
    phi = np.deg2rad(dipole_glon_deg)

    # Calculate a_1m coefficients for a dipole in direction (theta, phi)
    # a_lm = A * Y_lm*(theta, phi) * sqrt(4pi / (2l+1)) which for l=1 is
    amp_norm = dipole_amplitude_uK * np.sqrt(4 * np.pi / 3)
    
    # a_1,-1, a_1,0, a_1,1
    # Note: healpy expects (T, E, B) alms, we only need T for the dipole.
    dipole_alms[0, hp.Alm.getidx(lmax, 1, 0)] = amp_norm*np.cos(theta)
    dipole_alms[0, hp.Alm.getidx(lmax, 1, 1)] = -amp_norm*np.sin(theta)*np.exp(-1j*phi)/np.sqrt(2)

    total_alms = anisotropy_alms + dipole_alms

    smooth_alms = np.zeros((3, hp.Alm.getsize(lmax)), dtype=np.complex128)
    smooth_alms = hp.smoothalm(total_alms, fwhm=fwhm)
    cmb = np.zeros(12*nside**2, dtype=np.float32)
    cmb = hp.alm2map(total_alms, nside, pixwin=False)

    cmb_smooth = np.zeros((3, 12*nside**2), dtype=np.float32)
    cmb_smooth = hp.alm2map(smooth_alms, nside, pixwin=False)
    cmb = cmb * u.uK_CMB
    cmb = cmb.to(units, equivalencies=u.cmb_equivalencies(freq*u.GHz))
    cmb_smooth = cmb_smooth * u.uK_CMB
    cmb_smooth = cmb_smooth.to(units, equivalencies=u.cmb_equivalencies(freq*u.GHz))

    return cmb_smooth.value


def generate_thermal_dust(freq, fwhm, unit, nside, params):
    nu_dust = _scalar_nu_ref(params.components.ThermalDust.params)

    dust_params = deepcopy(params.components.ThermalDust.params)
    dust_params.nu_ref = nu_dust  # single reference frequency for the in-place sim
    dust_params.polarized = True
    dust = ThermalDust(dust_params, params, comp_name="ThermalDust")

    # d0 = constant beta 1.54 and T = 20. The emission at nu_ref is the component amplitude, so it
    # is generated in the amplitude unit, and `get_sed` carries it to the band's `unit`.
    dust_sky = pysm3.Sky(nside=min(1024,nside), preset_strings=["d0"],
                         output_unit=dust.amplitude_unit)
    dust_ref = dust_sky.get_emission(nu_dust*u.GHz).value
    dust_s = hp.smoothing(dust_ref, fwhm=fwhm)*dust.get_sed(freq, unit)
    return hp.ud_grade(dust_s, nside)


def generate_sync(freq, fwhm, unit, nside, params):
    nu_sync = _scalar_nu_ref(params.components.Synchrotron.params)

    sync_params = deepcopy(params.components.Synchrotron.params)
    sync_params.nu_ref = nu_sync  # single reference frequency for the in-place sim
    sync_params.polarized = True
    sync = Synchrotron(sync_params, params, comp_name="Synchrotron")

    # s5 = const beta -3.1, generated in the amplitude unit as for the dust above.
    sync_sky = pysm3.Sky(nside=min(1024,nside), preset_strings=["s5"],
                         output_unit=sync.amplitude_unit)
    sync_ref = sync_sky.get_emission(nu_sync*u.GHz).value
    sync_s = hp.smoothing(sync_ref, fwhm=fwhm)*sync.get_sed(freq, unit)
    return hp.ud_grade(sync_s, nside)


def generate_ff(freq, fwhm, unit, nside, params):
    nu_ff = _scalar_nu_ref(params.components.FreeFree.params)

    ff_params = deepcopy(params.components.FreeFree.params)
    ff_params.nu_ref = nu_ff  # single reference frequency for the in-place sim
    ff_params.polarized = False
    ff = FreeFree(ff_params, params, comp_name="FreeFree")

    ff_sky = pysm3.Sky(nside=min(1024,nside), preset_strings=["f1"], output_unit=ff.amplitude_unit)
    ff_ref = ff_sky.get_emission(nu_ff*u.GHz).value
    ff_s = hp.smoothing(ff_ref, fwhm=fwhm)*ff.get_sed(freq, unit)
    return hp.ud_grade(ff_s, nside)


def generate_spdust(freq, fwhm, unit, nside, params):
    nu_spdust = params.nu_ref_sync

    spdust_params = deepcopy(params.components.SpinningDust.params)
    spdust_params.polarized = False
    spdust = SpinningDust(spdust_params, params, comp_name="SpinningDust")

    spdust_sky = pysm3.Sky(nside=min(1024,nside), preset_strings=["a1"],
                           output_unit=spdust.amplitude_unit)
    spdust_ref = spdust_sky.get_emission(nu_spdust*u.GHz).value
    spdust_s = hp.smoothing(spdust_ref, fwhm=fwhm)*spdust.get_sed(freq, unit)
    return hp.ud_grade(spdust_s, nside)


C_LIGHT = 299792458.0  # m/s
def get_orbital_dipole(det: DetectorTOD, pix: NDArray[np.integer], freq: float,
                       unit: str) -> NDArray:
    orb_vel_vec = det.orbital_velocity_m_per_s
    if orb_vel_vec is None:
        raise ValueError("Read-time orbital-dipole simulation requires an orbital velocity.")
    # The orbital dipole is unpolarized, so only the intensity response of the detector matters.
    resp_I, _ = det.response_I_P
    if resp_I == 0.0:
        return np.zeros(pix.shape, dtype=np.float32)
    # pointing_vec = hp.pix2vec(det.nside, pix)
    geom = ducc0.healpix.Healpix_Base(det.nside, "RING")
    pointing_vec = geom.pix2vec(pix)

    v_orbital_speed = np.linalg.norm(orb_vel_vec, axis=-1)

    beta_vec = orb_vel_vec / C_LIGHT
    beta_mag_sq = (v_orbital_speed / C_LIGHT)**2
    gamma = 1.0 / np.sqrt(1.0 - beta_mag_sq)
    dot_product = np.sum(beta_vec * pointing_vec, axis=1)
    orbital_dipole_amplitude = T_CMB * ((1.0 / (gamma * (1.0 - dot_product))) - 1.0)

    # We have calculated the orbital dipole in units of CMB Kelvin; convert it to the band's unit.
    return orbital_dipole_amplitude*unit_factor(freq, "K_CMB", unit)*resp_I



def replace_tod_with_sim(band_comm: MPI.Comm, detector_data: DetectorGroupTOD, band_params: Bunch,
                         params: Bunch, sim_params: Bunch) -> DetectorGroupTOD:
    """Replace the band's TOD with a simulated sky, orbital dipole and noise, at gain 1.

    Everything is generated in the band's own `band_unit`, so a gain of 1 (detector unit per
    band_unit) reproduces the simulated TOD.
    """
    nside = detector_data.nside
    npix = 12*nside**2
    fwhm = np.deg2rad(detector_data.fwhm/60.0)
    freq = detector_data.nu
    unit = detector_data.unit

    # Hard-coded noise parameters
    alpha_ncorr = sim_params.corr_noise_alpha
    fknee_ncorr = sim_params.corr_noise_fknee

    # sigma0_rts is quoted in uK_CMB*sqrt(s). Convert it to the band unit, and from per-root-second
    # to per-sample RMS.
    sigma0_persamp = unit_factor(freq, "uK_CMB", unit)*band_params.sigma0_rts \
        * np.sqrt(band_params.fsamp)

    start_bench("sky")
    comps_sum_smoothed = np.zeros((3, npix), dtype=np.float32)
    if band_comm.Get_rank() == 0:
        if sim_params.include_CMB:
            comps_sum_smoothed += generate_cmb(freq, fwhm, u.Unit(unit), nside, 3*nside, params)
            gc.collect()
        if sim_params.include_ThermalDust:
            comps_sum_smoothed += generate_thermal_dust(freq, fwhm, unit, nside, params)
            gc.collect()
        if sim_params.include_Synchrotron:
            comps_sum_smoothed += generate_sync(freq, fwhm, unit, nside, params)
            gc.collect()
        if sim_params.include_FreeFree:
            comps_sum_smoothed += generate_ff(freq, fwhm, unit, nside, params)
            gc.collect()
    stop_bench("sky")

    start_bench("bcast")
    band_comm.Bcast(comps_sum_smoothed, root=0)
    stop_bench("bcast")

    for scan in detector_data.scans:
        for det in scan.detectors:
            start_bench("orbdip")
            pix, psi = det.get_pix_psi()
            ntod = det.tod.size
            det.tod[:] = project_sky_to_tod(comps_sum_smoothed, pix, psi, det.response_I_P)
            if sim_params.include_OrbitalDipole:
                det.tod[:] += get_orbital_dipole(det, pix, freq, unit)
            stop_bench("orbdip")

            start_bench("noise")
            # Create some white noise.
            noise = np.random.normal(0, sigma0_persamp, ntod)

            if sim_params.include_corr_noise:
                # 1/f power spectrum, without sigma0**2 factor (which is already in the data).
                PS_freqs = rfftfreq(ntod, 1.0/band_params.fsamp)
                PS_freqs[0] = 0.5*PS_freqs[1]  # Add some DC power while avoiding divide by 0.
                PS = 1.0 + (PS_freqs/fknee_ncorr)**alpha_ncorr
                # Morph the shape of the noise power spectrum to be 1/f + white noise.
                det.tod[:] += irfft(rfft(noise)*np.sqrt(PS))
                del(PS_freqs, PS, noise)
                gc.collect()
            stop_bench("noise")

    return detector_data
