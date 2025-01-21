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
from tqdm import trange
import time
import sys
import os

import camb
from camb import model, initialpower

from astropy.modeling.physical_models import BlackBody
import paramfile_sim as params
from save_sim_to_h5 import save_to_h5_file


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) # Add the parent directory of this file, which is the Commander 4 root directory, to PATH, so that we can import packages from e.g. src/.

from src.python.model.component import ThermalDust, Synchrotron


def generate_cmb(freqs, fwhm, units, nside, lmax):
    pars = camb.set_params(H0=params.H0, ombh2=params.ombh2, omch2=params.omch2, 
                           mnu=params.mnu, omk=params.omk, tau=params.tau, As=params.As, 
                           ns=params.ns, halofit_version='mead', lmax=lmax)

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
    hp.write_map(params.OUTPUT_FOLDER + f"true_sky_cmb_{nside}.fits", cmb, overwrite=True)
    cmb_s = hp.smoothing(cmb, fwhm=fwhm.to('rad').value) * u.uK_CMB
    cmb_s = [cmb_s.to(units, equivalencies=u.cmb_equivalencies(f*u.GHz)) for f in freqs]
    return np.array(cmb_s)


def generate_thermal_dust(freqs, fwhm, units, nside):
    nu_dust = params.nu_ref_dust
    beta = params.beta_dust
    T = params.T_dust

    dust = pysm3.Sky(nside=1024, preset_strings=["d0"], output_unit=units) #d0 = constant beta 1.54 and T = 20
    dust_ref = dust.get_emission(nu_dust*u.GHz)#.to(u.MJy/u.sr, equivalencies=u.cmb_equivalencies(nu_dust*u.GHz))
    dust_ref_smoothed = hp.smoothing(dust_ref, fwhm=fwhm.to('rad').value)*dust_ref.unit
    hp.write_map(params.OUTPUT_FOLDER + f"true_sky_dust_{nside}.fits", dust_ref_smoothed, overwrite=True)

    dust = ThermalDust()
    dust_s = [dust_ref_smoothed*dust.get_sed(f) for f in freqs]
    #dust_s = [d.to(u.MJy/u.sr, equivalencies=u.cmb_equivalencies(f*u.GHz)) for d,f in zip(dust_s,freqs)]
    dust_s = [hp.ud_grade(d.value, nside)*d.unit for d in dust_s]
    return np.array(dust_s)


def generate_sync(freqs, fwhm, units, nside):
    nu_sync = params.nu_ref_sync
    beta_s = params.beta_sync

    sync = pysm3.Sky(nside=1024, preset_strings=["s5"], output_unit=units) # s5 = const beta -3.1
    sync_ref = sync.get_emission(nu_sync*u.GHz)#.to(u.MJy/u.sr, equivalencies=u.cmb_equivalencies(nu_sync*u.GHz))
    sync_ref_smoothed = hp.smoothing(sync_ref, fwhm=fwhm.to('rad').value)*sync_ref.unit
    hp.write_map(params.OUTPUT_FOLDER + f"true_sky_sync_{nside}.fits", sync_ref_smoothed, overwrite=True)

    sync = Synchrotron()
    sync_s = [sync_ref_smoothed*sync.get_sed(f) for f in freqs]
    #sync_s = [d.to(u.MJy/u.sr, equivalencies=u.cmb_equivalencies(f*u.GHz)) for d,f in zip(sync_s,freqs)]
    sync_s = [hp.ud_grade(d.value, nside)*d.unit for d in sync_s]
    return np.array(sync_s)


def get_pointing(npix):
    if params.POINTING_PATH is None:
        pix = np.arange(ntod) % npix
        return pix.astype('int32')

    with h5py.File(params.POINTING_PATH, 'r') as file:
        pix = file['pix'][()]
        file.close()

    ntod = params.NTOD
    indc = np.linspace(0, len(pix)-1, params.NTOD, dtype=int)
    pix = pix[indc]

    if params.NSIDE != 2048:
        vec = hp.pix2vec(2048, pix.astype(int))
        pix = hp.vec2pix(params.NSIDE, vec[0], vec[1], vec[2])

    return pix.astype('int32')


def sim_noise(sigma0, chunk_size, with_corr_noise):
    # white noise + 1/f noise
    ntod = params.NTOD
    n_chunks = ntod // chunk_size
    f_samp = params.SAMP_FREQ
    f_chunk = f_samp / chunk_size

    noise = np.random.randn(ntod)*sigma0
    if not with_corr_noise:
        return noise

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
        for i in trange(size*perrank, b):
            Fx = np.fft.rfft(noise[i*chunk_size:(i+1)*chunk_size])
            Fx[sel] = Fx[sel]*(1 + 1/f[sel])
            Fx[f < f_chunk] = Fx[sel][0]
            chunk_noise = np.fft.irfft(Fx)
            total[i*chunk_size:(i+1)*chunk_size] = chunk_noise

    
        if params.NTOD % chunk_size != 0:
            Fx = np.fft.rfft(noise[n_chunks*chunk_size:])
            f = np.fft.rfftfreq(ntod-n_chunks*chunk_size, d = 1/f_samp)
            sel = (f >= f_chunk)
            Fx[sel] = Fx[sel]*(1 + 1/f[sel])
            Fx[f < f_chunk] = Fx[sel][0]
            chunk_noise = np.fft.irfft(Fx)
            total[n_chunks*chunk_size:] = chunk_noise

    return total


if __name__ == "__main__":

    # initiliazing MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()


    # reading in main parameters
    nside = params.NSIDE
    lmax = 3*nside-1
    npix = 12*nside**2
    chunk_size = npix//40
    if chunk_size % 2 != 0:
        chunk_size += 1
    fwhm_arcmin = params.FWHM
    fwhm = fwhm_arcmin*u.arcmin
    sigma_fac = params.SIGMA_SCALE
    unit = params.unit
    if unit == 'MJ/sr':
        units = u.MJy/u.sr
    elif unit == 'uK_RJ':
        units = u.uK_RJ

    sigma0s = np.array(params.SIGMA0)*sigma_fac*u.uK_RJ
    freqs = np.array(params.FREQ)


    if size < 3:
        raise ValueError("Please run this script with at least 3 MPI tasks.")

    comm.Barrier()
    mp_c = np.zeros((len(freqs), 3, npix))
    if rank == 0 and "dust" in params.components:
        t0 = time.time()
        print(f"Rank 0 generating thermal dust")
        mp_c = generate_thermal_dust(freqs, fwhm, units, nside)
        for i in range(len(freqs)):
            hp.mollview(mp_c[i,0], title=f"True thermal dust at {freqs[i]:.2f}GHz")
            plt.savefig(params.OUTPUT_FOLDER + f"true_thermal_dust_{nside}_{freqs[i]}.png")
            plt.close()
        print(f"Rank 0 finished thermal dust in {time.time()-t0:.1f}s.")
    if rank == 1 and "sync" in params.components:
        t0 = time.time()
        print(f"Rank 1 generating synchrotron")
        mp_c = generate_sync(freqs, fwhm, units, nside)
        for i in range(len(freqs)):
            hp.mollview(mp_c[i,0], title=f"True synchrotron at {freqs[i]:.2f}GHz")
            plt.savefig(params.OUTPUT_FOLDER + f"true_synchrotron_{nside}_{freqs[i]}.png")
            plt.close()
        print(f"Rank 1 finished synchrotron in {time.time()-t0:.1f}s.")
    if rank == 2 and "CMB" in params.components:
        t0 = time.time()
        print(f"Rank 2 generating CMB")
        mp_c = generate_cmb(freqs, fwhm, units, nside, lmax)
        for i in range(len(freqs)):
            hp.mollview(mp_c[i,0], title=f"True CMB at {freqs[i]:.2f}GHz")
            plt.savefig(params.OUTPUT_FOLDER + f"true_CMB_{nside}_{freqs[i]}.png")
            plt.close()
        print(f"Rank 2 finished CMB in {time.time()-t0:.1f}s.")

    if rank == 0:
        m_s = np.zeros((len(freqs), 3, npix))
    else:
        m_s = None

    comm.Barrier()
    comm.Reduce(mp_c, m_s, op=MPI.SUM, root=0)

    repeat = 50
    ntod = params.NTOD

    if rank == 0:
        t0 = time.time()
        print(f"Rank 1 calculating pointing")
        pix = get_pointing(npix)
        psi = np.repeat(np.arange(repeat)*np.pi/repeat, npix)
        ds = []
        print(f"Rank 1 finished calculating pointing in {time.time()-t0:.1f}s.")

    for i in range(len(freqs)):
        #m_s = cmb_s[i] + dust_s[i] + sync_s[i]

        if rank == 0:
            t0 = time.time()
            print(f"Rank 1 calculating sky signal")
            I,Q,U = m_s[i]
            if params.pol:
                d = I[pix] + Q[pix]*np.cos(2*psi) + U[pix]*np.sin(2*psi)
            else:
                d = I[pix]
            print(f"Rank 1 finished calculating sky signal in {time.time()-t0:.1f}s.")
            #d = d.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freqs[i]*u.GHz))

        if rank == 0:
            t0 = time.time()
            print(f"All ranks starting noise simulations.")

        with_corr_noise = "corr_noise" in params.components
        noise = sim_noise(sigma0s[i].value, chunk_size, with_corr_noise)

        if rank == 0:    
            print(f"Finished noise simulations in {time.time()-t0:.1f}s.")
            ds += [(d + noise).astype('float32')]


    if rank == 0:
        t0 = time.time()
        print(f"Rank 0 writing simulation to file.")
        for i in range(len(freqs)):
            hp.write_map(params.OUTPUT_FOLDER + f"true_sky_full_{nside}_{freqs[i]}.fits", m_s[i], overwrite=True)

        save_to_h5_file(ds, pix, psi)
        print(f"Rank 0 finished writing to file in {time.time()-t0:.1f}s.")