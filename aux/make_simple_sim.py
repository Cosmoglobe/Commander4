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
from traceback import print_exc

import camb
from camb import model, initialpower

from astropy.modeling.physical_models import BlackBody
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
    cmb_us = cmb * u.uK_CMB
    cmb_us = np.array([cmb_us.to(units, equivalencies=u.cmb_equivalencies(f*u.GHz)) for f in freqs], dtype=np.float32)
    
    cmb_s = [hp.smoothing(cmb, fwhm=fwhm[iband].to('rad').value) * u.uK_CMB for iband in range(len(freqs))]
    cmb_s = np.array([cmb_s[i].to(units, equivalencies=u.cmb_equivalencies(freqs[i]*u.GHz)) for i in range(len(freqs))], dtype=np.float32)

    if params.write_fits:
        hp.write_map(params.OUTPUT_FOLDER + f"true_sky_cmb_{nside}.fits", cmb, overwrite=True)
        for i in range(len(freqs)):
            hp.write_map(params.OUTPUT_FOLDER + f"true_sky_cmb_smoothed_{nside}_b{fwhm[i].value:.0f}.fits", cmb_s[i], overwrite=True)

    if params.make_plots:
        for i in range(len(freqs)):
            hp.mollview(cmb_us[i,0], title=f"True CMB at {freqs[i]:.2f}GHz")
            plt.savefig(params.OUTPUT_FOLDER + f"true_CMB_{nside}_{freqs[i]}.png")
            plt.close()
        for i in range(len(freqs)):
            hp.mollview(cmb_s[i,0], title=f"True smoothed CMB at {freqs[i]:.2f}GHz")
            plt.savefig(params.OUTPUT_FOLDER + f"true_CMB_smoothed_{nside}_{freqs[i]}_b{fwhm[i].value:.0f}.png")
            plt.close()
    return cmb_s


def generate_thermal_dust(freqs, fwhm, units, nside):
    nu_dust = params.nu_ref_dust
    beta = params.beta_dust
    T = params.T_dust

    dust = pysm3.Sky(nside=min(1024,nside), preset_strings=["d0"], output_unit=units) #d0 = constant beta 1.54 and T = 20
    dust_ref = dust.get_emission(nu_dust*u.GHz)#.to(u.MJy/u.sr, equivalencies=u.cmb_equivalencies(nu_dust*u.GHz))
    dust_ref_smoothed = []
    for iband in range(len(freqs)):
        dust_ref_smoothed.append(hp.smoothing(dust_ref, fwhm=fwhm[iband].to('rad').value)*dust_ref.unit)
    if params.write_fits:
        hp.write_map(params.OUTPUT_FOLDER + f"true_sky_dust_{nside}.fits", dust_ref, overwrite=True)
        for iband in range(len(freqs)):
            hp.write_map(params.OUTPUT_FOLDER + f"true_sky_dust_smooth_{nside}_b{fwhm[iband].value:.0f}.fits", dust_ref_smoothed[-1], overwrite=True)

    dust = ThermalDust()
    dust_us = [dust_ref*dust.get_sed(f) for f in freqs]
    dust_us = np.array([hp.ud_grade(d.value, nside)*d.unit for d in dust_us], dtype=np.float32)
    dust_s = [dust_ref_smoothed[i]*dust.get_sed(freqs[i]) for i in range(len(freqs))]
    dust_s = np.array([hp.ud_grade(d.value, nside)*d.unit for d in dust_s], dtype=np.float32)

    if params.make_plots:
        for i in range(len(freqs)):
            hp.mollview(dust_us[i,0], title=f"True thermal dust at {freqs[i]:.2f}GHz")
            plt.savefig(params.OUTPUT_FOLDER + f"true_thermal_dust_{nside}_{freqs[i]}.png")
            plt.close()
        for i in range(len(freqs)):
            hp.mollview(dust_s[i,0], title=f"True smoothed thermal dust at {freqs[i]:.2f}GHz")
            plt.savefig(params.OUTPUT_FOLDER + f"true_thermal_dust_smoothed_{nside}_{freqs[i]}_b{fwhm[i].value:.0f}.png")
            plt.close()

    return dust_s


def generate_sync(freqs, fwhm, units, nside):
    nu_sync = params.nu_ref_sync
    beta_s = params.beta_sync

    sync = pysm3.Sky(nside=min(1024,nside), preset_strings=["s5"], output_unit=units) # s5 = const beta -3.1
    sync_ref = sync.get_emission(nu_sync*u.GHz)#.to(u.MJy/u.sr, equivalencies=u.cmb_equivalencies(nu_sync*u.GHz))
    sync_ref_smoothed = []
    for iband in range(len(freqs)):
        sync_ref_smoothed.append(hp.smoothing(sync_ref, fwhm=fwhm[iband].to('rad').value)*sync_ref.unit)
    if params.write_fits:
        hp.write_map(params.OUTPUT_FOLDER + f"true_sky_sync_smoothed_{nside}_b{fwhm[iband].value:.0f}.fits", sync_ref_smoothed[-1], overwrite=True)
        for iband in range(len(freqs)):
            hp.write_map(params.OUTPUT_FOLDER + f"true_sky_sync_{nside}.fits", sync_ref, overwrite=True)

    sync = Synchrotron()
    sync_us = [sync_ref*sync.get_sed(f) for f in freqs]
    sync_us = np.array([hp.ud_grade(d.value, nside)*d.unit for d in sync_us], dtype=np.float32)
    sync_s = [sync_ref_smoothed[i]*sync.get_sed(freqs[i]) for i in range(len(freqs))]
    sync_s = np.array([hp.ud_grade(d.value, nside)*d.unit for d in sync_s], dtype=np.float32)

    if params.make_plots:
        for i in range(len(freqs)):
            hp.mollview(sync_us[i,0], title=f"True synchrotron at {freqs[i]:.2f}GHz")
            plt.savefig(params.OUTPUT_FOLDER + f"true_synchrotron_{nside}_{freqs[i]}.png")
            plt.close()
        for i in range(len(freqs)):
            hp.mollview(sync_s[i,0], title=f"True smoothed synchrotron at {freqs[i]:.2f}GHz")
            plt.savefig(params.OUTPUT_FOLDER + f"true_synchrotron_smoothed_{nside}_{freqs[i]}_b{fwhm[i].value:.0f}.png")
            plt.close()

    return sync_s


def get_pointing(npix):
    ntod = params.NTOD
    if params.POINTING_PATH is None:
        pix = np.arange(ntod) % npix
        return pix.astype('int32')

    with h5py.File(params.POINTING_PATH, 'r') as file:
        tot_file_len = file['pix'].shape[0]
        pix = file['pix'][:ntod]

    print(f"Reading {ntod} out of {tot_file_len} points from pointing file ({100*ntod/tot_file_len:.1f}%)")
    # indc = np.linspace(0, len(pix)-1, params.NTOD, dtype=int)
    # pix = pix[indc]

    assert pix.shape[0] == ntod, f"Parameter file ntod {ntod} does not match pixel length {pix.shape[0]}, likely because desired length is longer than entire pointing file."

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
            chunk_noise = np.fft.irfft(Fx, n=total[n_chunks*chunk_size:].shape[0])
            total[n_chunks*chunk_size:] = chunk_noise

    return total


def main():

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
    if rank == 0 and "dust" in params.components:
        t0 = time.time()
        print(f"Rank 0 generating thermal dust")
        comp_smoothed = generate_thermal_dust(freqs, fwhm, units, nside)
        print(f"Rank 0 finished thermal dust in {time.time()-t0:.1f}s.")
    elif rank == 1 and "sync" in params.components:
        t0 = time.time()
        print(f"Rank 1 generating synchrotron")
        comp_smoothed = generate_sync(freqs, fwhm, units, nside)
        print(f"Rank 1 finished synchrotron in {time.time()-t0:.1f}s.")
    elif rank == 2 and "CMB" in params.components:
        t0 = time.time()
        print(f"Rank 2 generating CMB")
        comp_smoothed = generate_cmb(freqs, fwhm, units, nside, lmax)
        print(f"Rank 2 finished CMB in {time.time()-t0:.1f}s.")

    if rank == 0:
        comps_sum_smoothed = np.zeros((len(freqs), 3, npix))
        
        # Add own component if it was computed
        if "dust" in params.components:
            comps_sum_smoothed += comp_smoothed
        
        # Receive from other computing ranks
        if "sync" in params.components:
            temp = np.empty((len(freqs), 3, npix))
            comm.Recv(temp, source=1, tag=1)
            comps_sum_smoothed += temp
        
        if "CMB" in params.components:
            temp = np.empty((len(freqs), 3, npix))
            comm.Recv(temp, source=2, tag=2)
            comps_sum_smoothed += temp

    elif rank == 1 and "sync" in params.components:
        comm.Send(comp_smoothed, dest=0, tag=1)

    elif rank == 2 and "CMB" in params.components:
        comm.Send(comp_smoothed, dest=0, tag=2)

    repeat = params.NTOD//npix+1
    ntod = params.NTOD

    if rank == 0:
        t0 = time.time()
        print(f"Rank 1 calculating pointing")
        pix = get_pointing(npix)
        psi = np.repeat(np.arange(repeat)*np.pi/repeat, npix)
        psi = psi[:ntod]
        signal_tod = []
        noise_tod = []
        print(f"Rank 1 finished calculating pointing in {time.time()-t0:.1f}s.")

        noise_map = np.zeros((len(freqs), npix))

    for i in range(len(freqs)):

        if rank == 0:
            t0 = time.time()
            print(f"Rank 1 calculating sky signal")
            I,Q,U = comps_sum_smoothed[i]
            if params.pol:
                d = I[pix] + Q[pix]*np.cos(2*psi) + U[pix]*np.sin(2*psi)
            else:
                d = I[pix]
            print(f"Rank 1 finished calculating sky signal in {time.time()-t0:.1f}s.")

        if rank == 0:
            t0 = time.time()
            print(f"All ranks starting noise simulations.")

        with_corr_noise = "corr_noise" in params.components
        noise = sim_noise(sigma0s[i].value, chunk_size, with_corr_noise)

        if rank == 0:    
            print(f"Finished noise simulations in {time.time()-t0:.1f}s.")
            signal_tod.append((d + noise).astype('float32'))
            noise_tod.append(noise.astype('float32'))

            noise_map[i] = np.bincount(pix, weights=noise, minlength=npix)
            hitmap = np.bincount(pix, minlength=npix)
            assert (hitmap > 0).all(), f"{np.sum(hitmap == 0)} out of {hitmap.shape[0]} pixels were never hit by the scanning strategy."
            if (hitmap <= 10).any():
                print(f"Warning: {np.sum(hitmap <= 10)} out of {hitmap.shape[0]} pixels were hit 10 or fewer times by scanning strategy.")
            else:
                print(f"Lowest pixel hit count is {np.min(hitmap)}.")
            noise_map[i] /= hitmap

            assert signal_tod[-1].shape == psi.shape, f"Shape of simulated TOD {signal_tod[-1].shape} differs from generated psi {psi.shape}"


    if rank == 0:
        t0 = time.time()
        print(f"Rank 0 writing simulation to file.")
        if params.make_plots:
            for i in range(len(freqs)):
                hp.write_map(params.OUTPUT_FOLDER + f"true_sky_full_{nside}_{freqs[i]}.fits", comps_sum_smoothed[i], overwrite=True)
                
                hp.mollview(comps_sum_smoothed[i,0], title=f"Full 'clean' sky smoothed {freqs[i]:.2f}GHz")
                plt.savefig(params.OUTPUT_FOLDER + f"sky_smoothed_{nside}_{freqs[i]}_b{fwhm[i].value:.0f}.png")
                plt.close()

                hp.mollview(comps_sum_smoothed[i,0]+noise_map[i], title=f"Full observed sky smoothed {freqs[i]:.2f}GHz")
                plt.savefig(params.OUTPUT_FOLDER + f"sky_smoothed_{nside}_{freqs[i]}_b{fwhm[i].value:.0f}.png")
                plt.close()

                hp.mollview(noise_map[i], title=f"Noise {freqs[i]:.2f}GHz")
                plt.savefig(params.OUTPUT_FOLDER + f"noise_{nside}_{freqs[i]}_b{fwhm[i].value:.0f}.png")
                plt.close()

        save_to_h5_file(signal_tod, pix, psi, fname=f'tod_sim_{params.NSIDE}_s{params.SIGMA_SCALE}_b{params.FWHM[0]:.0f}')
        save_to_h5_file(noise_tod, pix, psi, fname=f'noise_sim_{params.NSIDE}_s{params.SIGMA_SCALE}_b{params.FWHM[0]:.0f}')
        print(f"Rank 0 finished writing to file in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    # initiliazing MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    from parse_params import params, params_dict
    try:
        main()
    except Exception as error:
        print_exc()  # Print the full exception raise, including trace-back.
        print(f">>>>>>>> Error encountered on rank {MPI.COMM_WORLD.Get_rank()}, calling MPI abort.")
        MPI.COMM_WORLD.Abort()