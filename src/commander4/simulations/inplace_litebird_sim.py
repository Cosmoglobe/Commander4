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

from commander4.data_models.detector_TOD import DetectorTOD
from commander4.data_models.scan_TOD import ScanTOD
from commander4.sky_models.component import ThermalDust, Synchrotron, FreeFree


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


def generate_thermal_dust(freq, fwhm, units, nside, params):
    nu_dust = params.components.ThermalDust.params.nu0

    #d0 = constant beta 1.54 and T = 20
    dust = pysm3.Sky(nside=min(1024,nside), preset_strings=["d0"], output_unit=units)
    dust_ref = dust.get_emission(nu_dust*u.GHz)
    dust_ref_smoothed = hp.smoothing(dust_ref, fwhm=fwhm)*dust_ref.unit

    dust_params = deepcopy(params.components.ThermalDust.params)
    dust_params.polarized = True

    dust = ThermalDust(dust_params, params)
    dust_s = dust_ref_smoothed*dust.get_sed(freq)
    dust_s = hp.ud_grade(dust_s.value, nside)*dust_s.unit

    return dust_s.value


def generate_sync(freq, fwhm, units, nside, params):
    nu_sync = params.components.Synchrotron.params.nu0

    # s5 = const beta -3.1
    sync = pysm3.Sky(nside=min(1024,nside), preset_strings=["s5"], output_unit=units)
    sync_ref = sync.get_emission(nu_sync*u.GHz)
    sync_ref_smoothed = hp.smoothing(sync_ref, fwhm=fwhm)*sync_ref.unit

    sync_params = deepcopy(params.components.Synchrotron.params)
    sync_params.polarized = True

    sync = Synchrotron(sync_params, params)
    sync_s = sync_ref_smoothed*sync.get_sed(freq)
    sync_s = hp.ud_grade(sync_s.value, nside)*sync_s.unit

    return sync_s.value


def generate_ff(freq, fwhm, units, nside, params):
    nu_ff = params.components.FreeFree.params.nu0

    ff = pysm3.Sky(nside=min(1024,nside), preset_strings=["f1"], output_unit=units)
    ff_ref = ff.get_emission(nu_ff*u.GHz)
    ff_ref_smoothed = hp.smoothing(ff_ref, fwhm=fwhm)*ff_ref.unit

    ff_params = deepcopy(params.components.FreeFree.params)
    ff_params.polarized = False

    ff = FreeFree(ff_params, params)
    ff_s = ff_ref_smoothed*ff.get_sed(freq)
    ff_s = hp.ud_grade(ff_s.value, nside)*ff_s.unit

    return ff_s.value


def generate_spdust(freq, fwhm, units, nside, params):
    nu_spdust = params.nu_ref_sync

    spdust = pysm3.Sky(nside=min(1024,nside), preset_strings=["a1"], output_unit=units)
    spdust_ref = spdust.get_emission(nu_spdust*u.GHz)
    spdust_ref_smoothed = hp.smoothing(spdust_ref, fwhm=fwhm)

    spdust_params = deepcopy(params.components.SpinningDust.params)
    spdust_params.polarized = False

    spdust = Synchrotron(spdust_params, params)
    spdust_s = spdust_ref_smoothed*spdust.get_sed(freq)
    spdust_s = hp.ud_grade(spdust_s.value, nside)*spdust_s.unit

    return spdust_s.value


T_CMB = 2.72548  # K_CMB
C_LIGHT = 299792458.0  # m/s
def get_orbital_dipole(scan: ScanTOD, pix: NDArray[np.integer], freq: float, units) -> NDArray:
    orb_vel_vec = scan.orb_dir_vec  # Satellite velocity vector relative to sun.
    # pointing_vec = hp.pix2vec(scan.nside, scan.pix)
    geom = ducc0.healpix.Healpix_Base(scan.nside, "RING")
    pointing_vec = geom.pix2vec(pix)

    v_orbital_speed = np.linalg.norm(orb_vel_vec, axis=-1)

    beta_vec = orb_vel_vec / C_LIGHT
    beta_mag_sq = (v_orbital_speed / C_LIGHT)**2
    gamma = 1.0 / np.sqrt(1.0 - beta_mag_sq)
    dot_product = np.sum(beta_vec * pointing_vec, axis=1)
    orbital_dipole_amplitude = T_CMB * ((1.0 / (gamma * (1.0 - dot_product))) - 1.0)

    # We have calculated the orbital dipole in units of CMB Kelvin.
    # Find the conversion factor from K_CMB to the units expected by the code (typically uK_RJ).
    KCMB_to_uKRJ = (1.0 * u.K_CMB).to(units, equivalencies=u.cmb_equivalencies(freq * u.GHz)).value

    return orbital_dipole_amplitude * KCMB_to_uKRJ



def replace_tod_with_sim(detector_data: DetectorTOD, band_params: Bunch, params: Bunch):
    nside = detector_data.nside
    npix = 12*nside**2
    fwhm = np.deg2rad(detector_data.fwhm/60.0)
    freq = detector_data.nu
    units = u.uK_RJ

    # Hard-coded noise parameters
    alpha_ncorr = -1.0
    fknee_ncorr = 0.1

    KCMB_to_KRJ = (1.0 * u.K_CMB).to(u.K_RJ, equivalencies=u.cmb_equivalencies(freq * u.GHz)).value
    # Convert per-root-second RMS to per-sample RMS.
    sigma0_persamp = KCMB_to_KRJ*band_params.sigma0_rts*np.sqrt(band_params.fsamp)

    comps_sum_smoothed = np.zeros((3, npix), dtype=np.float32)
    comps_sum_smoothed += generate_thermal_dust(freq, fwhm, units, nside, params)
    comps_sum_smoothed += generate_sync(freq, fwhm, units, nside, params)
    comps_sum_smoothed += generate_ff(freq, fwhm, units, nside, params)
    # comps_sum_smoothed += generate_spdust(freq, fwhm, units, nside, 3*nside, params)
    cmb = generate_cmb(freq, fwhm, units, nside, 3*nside, params)
    comps_sum_smoothed += cmb

    I, Q, U = comps_sum_smoothed
    for scan in detector_data.scans:
        pix = scan.pix
        psi = scan.psi
        ntod = pix.size
        scan._tod = np.zeros(ntod, dtype=np.float32)
        scan._tod[:] = I[pix] + Q[pix]*np.cos(2*psi) + U[pix]*np.sin(2*psi)
        scan._tod[:] += get_orbital_dipole(scan, pix, freq, units)

        # Create some white noise.
        noise = np.random.normal(0, sigma0_persamp, ntod)
        # 1/f power spectrum, without sigma0**2 factor (which is already in the data).
        PS_freqs = rfftfreq(ntod, 1.0/band_params.fsamp)
        PS_freqs[0] = 0.5*PS_freqs[1]  # Add some DC power while avoiding divide by 0.
        PS = 1.0 + (PS_freqs/fknee_ncorr)**alpha_ncorr
        # Morph the shape of the noise power spectrum to be 1/f + white noise.
        noise = irfft(rfft(noise)*np.sqrt(PS))

        scan._tod[:] += noise

    return detector_data