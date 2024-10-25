# TOD simulation script

# Data = CMB + noise
#   Nside  = 2048
#   Lmax   = 6000
#   FWHM   = 10 arcmin
#   sigma0 = 30µK
#
# Scanning strategy = visit each pixel in order; repeat 9 times, such that final noise is 10µK/pix
#
# Split in chunks with 2^22 samples each (except for last one) = ~109 files, total of ~4GB
# 

import numpy as np
import healpy as hp
import pysm3
import pysm3.units as u
from commander_tod import commander_tod
import matplotlib.pyplot as plt
import h5py
from  mpi4py import MPI

import camb
from camb import model, initialpower

from astropy.modeling.physical_models import BlackBody
import paramfile_sim as param
from save_sim_to_h5 import save_to_h5_file

def mixmat_d(nu, nu_0, beta, T):
    bb = BlackBody(temperature=T*u.K)
    M = (nu/nu_0)**beta
    M *= bb(nu*u.GHz)/bb(nu_0*u.GHz)
    return M

def mixmat_s(nu, nu_0, beta):
    M = (nu/nu_0)**beta
    return M

# initiliazing MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# reading in main parameters
nside = param.NSIDE
lmax = 3*nside-1
npix = 12*nside**2
chunk_size = npix//40
if chunk_size % 2 != 0:
    chunk_size += 1
fwhm_arcmin = param.FWHM
fwhm = fwhm_arcmin*u.arcmin
sigma_fac = param.SIGMA_SCALE

sigma0s = np.array(param.SIGMA0)*sigma_fac*u.uK_CMB
freqs = np.array(param.FREQ)


def generate_cmb():
    pars = camb.set_params(H0=param.H0, ombh2=param.ombh2, omch2=param.omch2, 
                           mnu=param.mnu, omk=param.omk, tau=param.tau, As=param.As, 
                           ns=param.ns, halofit_version='mead', lmax=lmax)

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
    alms = hp.synalm(Cls, lmax=lmax, new=True)
    cmb = hp.alm2map(alms, nside, pixwin=False)
    hp.write_map(param.OUTPUT_FOLDER + "true_sky_cmb_{0}.fits".format(nside), cmb, overwrite=True)
    cmb_s = hp.smoothing(cmb, fwhm=fwhm.to('rad').value) * u.uK_CMB
    cmb_s = [cmb_s.to(u.MJy/u.sr, equivalencies=u.cmb_equivalencies(f*u.GHz)) for f in freqs]
    return np.array(cmb_s)


def generate_thermal_dust():
    nu_dust = param.nu_ref_dust
    beta = param.beta_dust
    T = param.T_dust

    dust = pysm3.Sky(nside=1024, preset_strings=["d0"]) #d0 = constant beta 1.54 and T = 20
    dust_ref = dust.get_emission(nu_dust*u.GHz).to(u.MJy/u.sr, equivalencies=u.cmb_equivalencies(nu_dust*u.GHz))
    dust_ref_smoothed = hp.smoothing(dust_ref, fwhm=fwhm.to('rad').value)*dust_ref.unit
    hp.write_map(param.OUTPUT_FOLDER + "true_sky_dust_{0}.fits".format(1024), dust_ref_smoothed, overwrite=True)

    dust_s = [dust_ref_smoothed*mixmat_d(f, nu_dust, beta, T) for f in freqs]
    #dust_s = [d.to(u.MJy/u.sr, equivalencies=u.cmb_equivalencies(f*u.GHz)) for d,f in zip(dust_s,freqs)]
    dust_s = [hp.ud_grade(d.value, nside)*d.unit for d in dust_s]
    return np.array(dust_s)


def generate_sync():
    nu_sync = param.nu_ref_sync
    beta_s = param.beta_sync

    sync = pysm3.Sky(nside=1024, preset_strings=["s5"]) # s5 = const beta -3.1
    sync_ref = sync.get_emission(nu_sync*u.GHz).to(u.MJy/u.sr, equivalencies=u.cmb_equivalencies(nu_sync*u.GHz))
    sync_ref_smoothed = hp.smoothing(sync_ref, fwhm=fwhm.to('rad').value)*sync_ref.unit
    hp.write_map(param.OUTPUT_FOLDER + "true_sky_sync_{0}.fits".format(1024), sync_ref_smoothed, overwrite=True)

    sync_s = [sync_ref_smoothed*mixmat_s(f, nu_sync, beta_s) for f in freqs]
    #sync_s = [d.to(u.MJy/u.sr, equivalencies=u.cmb_equivalencies(f*u.GHz)) for d,f in zip(sync_s,freqs)]
    sync_s = [hp.ud_grade(d.value, nside)*d.unit for d in sync_s]
    return np.array(sync_s)


def get_pointing():
    if param.POINTING_PATH is None:
        pix = np.arange(ntod) % npix
        return pix.astype('int32')

    with h5py.File(param.POINTING_PATH, 'r') as file:
        pix = file['pix'][()]
        file.close()

    if param.NTOD is None:
        ntod = len(pix)
    else:
        indc = np.linspace(0, len(pix)-1, param.NTOD, dtype=int)
        pix = pix[indc]

    if param.NSIDE != 2048:
        vec = hp.pix2vec(2048, pix.astype(int))
        pix = hp.vec2pix(param.NSIDE, vec[0], vec[1], vec[2])

    return pix.astype('int32')


def sim_noise(sigma0):
    # white noise + 1/f noise
    n_chunks = param.NTOD // chunk_size
    f_samp = param.SAMP_FREQ
    f_chunk = f_samp / chunk_size

    noise = np.random.randn(ntod)*sigma0
    f = np.fft.rfftfreq(chunk_size, d = 1/f_samp)
    sel = (f >= f_chunk)

    b = n_chunks-1
    perrank = b//size
    comm.Barrier()
    for i in range(rank*perrank, (rank+1)*perrank):
        Fx = np.fft.rfft(noise[i*chunk_size:(i+1)*chunk_size])
        Fx[sel] = Fx[sel]*(1 + 1/f[sel])
        Fx[f < f_chunk] = Fx[sel][0]
        chunk_noise = np.fft.irfft(Fx)
        noise[i*chunk_size:(i+1)*chunk_size] = chunk_noise

    if rank == 0:
        total = np.zeros(ntod)
    else:
        total = None

    comm.Barrier()
    comm.Reduce(noise, total, op=MPI.SUM, root=0)

    if rank == 0:
        for i in range(size*perrank, b):
            Fx = np.fft.rfft(noise[i*chunk_size:(i+1)*chunk_size])
            Fx[sel] = Fx[sel]*(1 + 1/f[sel])
            Fx[f < f_chunk] = Fx[sel][0]
            chunk_noise = np.fft.irfft(Fx)
            total[i*chunk_size:(i+1)*chunk_size] = chunk_noise

    
        if param.NTOD % chunk_size != 0:
            Fx = np.fft.rfft(noise[n_chunks*chunk_size:])
            f = np.fft.rfftfreq(ntod-n_chunks*chunk_size, d = 1/f_samp)
            sel = (f >= f_chunk)
            Fx[sel] = Fx[sel]*(1 + 1/f[sel])
            Fx[f < f_chunk] = Fx[sel][0]
            chunk_noise = np.fft.irfft(Fx)
            total[n_chunks*chunk_size:] = chunk_noise

    return total

        
if size >= 3:
    comm.Barrier()
    mp_c = np.zeros((len(freqs), 3, npix))
    if rank == 0:
        mp_c = generate_thermal_dust()
    if rank == 1:
        mp_c = generate_sync()
    if rank == 2:
        mp_c = generate_cmb()
        print(mp_c.shape)

    if rank == 0:
        m_s = np.zeros((len(freqs), 3, npix))
    else:
        m_s = None

    comm.Barrier()
    comm.Reduce(mp_c, m_s, op=MPI.SUM, root=0)
    if rank == 0:
        print(m_s.shape)

#dust_s = generate_thermal_dust()
#sync_s = generate_sync()
#cmb_s = generate_cmb()

repeat = 50
ntod = param.NTOD

if rank == 0:
    pix = get_pointing()
    psi = np.repeat(np.arange(repeat)*np.pi/repeat, npix)
    ds = []

for i in range(len(freqs)):
    #m_s = cmb_s[i] + dust_s[i] + sync_s[i]

    if rank == 0:
        I,Q,U = m_s[i]
        if param.pol:
            d = I[pix] + Q[pix]*np.cos(2*psi) + U[pix]*np.sin(2*psi)
        else:
            d = I[pix]
        #d = d.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freqs[i]*u.GHz))

    
    noise = sim_noise(sigma0s[i].value)

    if rank == 0:    
        ds += [(d + noise).astype('float32')]


if rank == 0:
    for i in range(len(freqs)):
        hp.write_map(param.OUTPUT_FOLDER + "true_sky_full_{0}_{1}.fits".format(nside, freqs[i]), m_s[i], overwrite=True)
    save_to_h5_file(ds, pix, psi)
