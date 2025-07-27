# TOD simulation script

import numpy as np
import healpy as hp
import pysm3
import pysm3.units as u
import matplotlib.pyplot as plt
import h5py
from  mpi4py import MPI
import time
import sys
import os
from traceback import print_exc
from pixell.bunch import Bunch
from Planck_pointing_sim import get_Planck_pointing
import camb

module_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(module_root_path) # Add the parent directory of this file, which is the Commander 4 root directory, to PATH, so that we can import packages from e.g. src/.

from save_sim_to_h5 import save_to_h5_file
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
    t0 = time.time()
    alms = hp.synalm(Cls, lmax=lmax, new=True)
    print(f"Finished CMB synALMs in {time.time()-t0:.1f}s."); t0 = time.time()
    smooth_alms = np.zeros((len(freqs), 3, hp.Alm.getsize(lmax)), dtype=np.complex128)
    for i in range(len(freqs)):
        smooth_alms[i] = hp.smoothalm(alms, fwhm=fwhm[i].to('rad').value)
    print(f"Finished CMB smoothing in {time.time()-t0:.1f}s."); t0 = time.time()
    cmb = np.zeros(12*nside**2, dtype=np.float32)
    cmb = hp.alm2map(alms, nside, pixwin=False)
    cmb_smooth = np.zeros((len(freqs), 3, 12*nside**2), dtype=np.float32)
    for i in range(len(freqs)):
        cmb_smooth[i] = hp.alm2map(smooth_alms[i], nside, pixwin=False)
    print(f"Finished CMB alm2map in {time.time()-t0:.1f}s."); t0 = time.time()
    cmb = cmb * u.uK_CMB
    cmb = np.array([cmb.to(units, equivalencies=u.cmb_equivalencies(f*u.GHz)) for f in freqs], dtype=np.float32)
    cmb_smooth = cmb_smooth * u.uK_CMB
    cmb_smooth = np.array([cmb_smooth[i].to(units, equivalencies=u.cmb_equivalencies(f*u.GHz)) for i, f in enumerate(freqs)], dtype=np.float32)
    
    print(f"Finished CMB casting to right units in {time.time()-t0:.1f}s."); t0 = time.time()

    if params.write_fits:
        hp.write_map(params.OUTPUT_FOLDER + f"true_sky_cmb_{nside}.fits", cmb[0], overwrite=True)
        for i in range(len(freqs)):
            hp.write_map(params.OUTPUT_FOLDER + f"true_sky_cmb_smoothed_{nside}_b{fwhm[i].value:.0f}.fits", cmb_smooth[i], overwrite=True)

    if params.make_plots:
        for i in range(len(freqs)):
            hp.mollview(cmb[i,0], title=f"True CMB at {freqs[i]:.2f}GHz")
            plt.savefig(params.OUTPUT_FOLDER + f"true_CMB_{nside}_{freqs[i]}.png")
            plt.close()
        for i in range(len(freqs)):
            hp.mollview(cmb_smooth[i,0], title=f"True smoothed CMB at {freqs[i]:.2f}GHz")
            plt.savefig(params.OUTPUT_FOLDER + f"true_CMB_smoothed_{nside}_{freqs[i]}_b{fwhm[i].value:.0f}.png")
            plt.close()
    return cmb_smooth


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

    dust_params = Bunch({"beta": beta, "T": T, "nu0": nu_dust, "lmax": "full"})
    dust = ThermalDust(dust_params)
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

    sync_params = Bunch({"beta": beta_s, "nu0": nu_sync, "lmax": "full"})
    sync = Synchrotron(sync_params)
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
    theta, phi, LOS, orb_dir, dipole = get_Planck_pointing(params.NTOD, params.f_samp)
    if rank == 0:
        pix = hp.ang2pix(params.NSIDE, theta, phi)
        return pix, LOS, orb_dir, dipole
    else:
        return None, None, None, None
    
    ### Old code for getting pointing, which relied on reading pointing indices from a file
    # ntod = params.NTOD
    # if params.POINTING_PATH is None:
    #     pix = np.arange(ntod) % npix
    #     return pix.astype('int32')

    # with h5py.File(params.POINTING_PATH, 'r') as file:
    #     tot_file_len = file['pix'].shape[0]
    #     pix = file['pix'][:ntod].astype('int32')

    # print(f"Reading {ntod} out of {tot_file_len} points from pointing file ({100*ntod/tot_file_len:.1f}%)")
    # indc = np.linspace(0, len(pix)-1, params.NTOD, dtype=int)
    # pix = pix[indc]

    # assert pix.shape[0] == ntod, f"Parameter file ntod {ntod} does not match pixel length {pix.shape[0]}, likely because desired length is longer than entire pointing file."

    # if params.NSIDE > 2048:
    #     print("Warning: NSIDE is larger than 2048, which is the resolution of the loaded pointing files.")
    #     np.random.seed(42)
    #     ang = hp.pix2ang(2048, pix)
    #     ang1 = ang[0] + np.random.uniform(-0.025, 0.025, ang[0].shape)
    #     ang2 = ang[1] + np.random.uniform(-0.025, 0.025, ang[1].shape)
    #     ang1 = np.clip(ang1, 0, np.pi)
    #     ang2 = np.clip(ang2, 0, 2*np.pi)
    #     pix = hp.ang2pix(params.NSIDE, ang1, ang2)
    # else:
    #     theta, phi = hp.pix2ang(2048, pix)
    #     pix = hp.ang2pix(params.NSIDE, theta, phi)

    # return pix



def sim_noise(sigma0, chunk_size, with_corr_noise):
    # white noise + 1/f noise
    ntod = params.NTOD
    if not with_corr_noise:
        if rank==0:
            return np.random.randn(ntod)*sigma0
        else:
            return None

    n_chunks = ntod // chunk_size
    f_samp = params.SAMP_FREQ

    noise_full = np.zeros(ntod, dtype='float32')

    f = np.fft.rfftfreq(chunk_size, d = 1/f_samp)

    noisePS = np.zeros_like(f)
    # noisePS[1:] = sigma0**2*(1 + (f[1:]/params.NOISE_FKNEE)**params.NOISE_ALPHA)
    noisePS[1:] = sigma0**2*((f[1:]/params.NOISE_FKNEE)**params.NOISE_ALPHA)

    b = n_chunks
    perrank = b//size
    comm.Barrier()
    if rank != 0:
        for i in range((rank-1)*perrank, rank*perrank):
            Fx = np.fft.rfft(np.random.randn(chunk_size))
            Fx *= np.sqrt(noisePS)
            noise_segment = np.fft.irfft(Fx).astype('float32')
            comm.Send(noise_segment, dest=0, tag=rank)
    else:
        for irank in range(1, size):
            for i in range((irank-1)*perrank, irank*perrank):
                temp = np.zeros(chunk_size, dtype=np.float32)
                comm.Recv(temp, source=irank, tag=irank)
                noise_full[i*chunk_size:(i+1)*chunk_size] = temp

    if rank == 0:
        for i in range((size-1)*perrank, b):
            Fx = np.fft.rfft(np.random.randn(chunk_size))
            Fx *= np.sqrt(noisePS)
            noise_segment = np.fft.irfft(Fx)
            noise_full[i*chunk_size:(i+1)*chunk_size] = noise_segment

        if params.NTOD % chunk_size != 0:
            Fx = np.fft.rfft(np.random.randn(noise_full[n_chunks*chunk_size:].shape[0]))
            f = np.fft.rfftfreq(ntod-n_chunks*chunk_size, d = 1/f_samp)
            noisePS = np.zeros_like(f)
            # noisePS[1:] = sigma0**2*(1 + (f[1:]/params.NOISE_FKNEE)**params.NOISE_ALPHA)
            noisePS[1:] = sigma0**2*((f[1:]/params.NOISE_FKNEE)**params.NOISE_ALPHA)
            Fx *= np.sqrt(noisePS)
            noise_segment = np.fft.irfft(Fx, n=noise_full[n_chunks*chunk_size:].shape[0])
            noise_full[n_chunks*chunk_size:] = noise_segment
        
        ### Sanity checks to see if everything went as planned ###
        num_zeros = np.sum(noise_full == 0)
        if num_zeros > 0:
            print(f"ERROR: There are {num_zeros} 0-valued elements in noise-simulation of size {noise_full.size}")

    return noise_full


def main():
    os.makedirs(params.OUTPUT_FOLDER, exist_ok=True)

    # reading in main parameters
    nside = params.NSIDE
    lmax = 3*nside-1
    npix = 12*nside**2
    chunk_size = int(3600*params.SAMP_FREQ) # 1 hour  # npix//40
    if chunk_size % 2 != 0:
        chunk_size += 1
    n_chunks = params.NTOD // chunk_size
    if rank == 0:
        print(f'Number of scans is {n_chunks}')
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
        if "dust" in params.components:
            comps_sum_smoothed = comp_smoothed
        else:
            comps_sum_smoothed = np.zeros((len(freqs), 3, npix), dtype=np.float32)
        
        # Receive from other computing ranks
        if "sync" in params.components:
            temp = np.empty((npix), dtype=np.float32)
            for ifreq in range(len(freqs)):
                for ipol in range(3):
                    comm.Recv(temp, source=1, tag=1)
                    comps_sum_smoothed[ifreq,ipol] += temp
        
        if "CMB" in params.components:
            temp = np.empty((npix), dtype=np.float32)
            for ifreq in range(len(freqs)):
                for ipol in range(3):
                    comm.Recv(temp, source=2, tag=2)
                    comps_sum_smoothed[ifreq,ipol] += temp

    elif rank == 1 and "sync" in params.components:
        for ifreq in range(len(freqs)):
            for ipol in range(3):
                comm.Send(comp_smoothed[ifreq,ipol], dest=0, tag=1)
        del(comp_smoothed)

    elif rank == 2 and "CMB" in params.components:
        for ifreq in range(len(freqs)):
            for ipol in range(3):
                comm.Send(comp_smoothed[ifreq,ipol], dest=0, tag=2)
        del(comp_smoothed)

    repeat = params.NTOD//npix+1
    ntod = params.NTOD

    t0 = time.time()
    pix, LOS, orb_dir, dipole = get_pointing(npix)
    if rank == 0:
        psi = np.repeat(np.arange(repeat)*np.pi/repeat, npix)
        psi = psi[:ntod]
        signal_tod = []
        corr_noise_tod = []
        white_noise_tod = []
        print(f"Finished calculating pointing in {time.time()-t0:.1f}s.")

        observed_map = np.zeros((len(freqs), npix))
        white_noise_map = np.zeros((len(freqs), npix))
        corr_noise_map = np.zeros((len(freqs), npix))
        hit_map = np.zeros((len(freqs), npix), dtype=int)
        inv_var_map = np.zeros((len(freqs), npix))

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
        corr_noise = sim_noise(sigma0s[i].value, chunk_size, with_corr_noise)

        if rank == 0:
            white_noise = np.random.normal(0, sigma0s[i].value, ntod)
    
            print(f"Finished noise simulations in {time.time()-t0:.1f}s.")
            t0 = time.time()

            # Convert dipole to units specified in parameter file.
            unit = params.unit
            if unit == 'MJ/sr':
                units = u.MJy/u.sr
            elif unit == 'uK_RJ':
                units = u.uK_RJ
            dipole_myfreq = dipole.copy()
            dipole_myfreq = dipole_myfreq * u.uK_CMB * 1e6
            dipole_myfreq = dipole_myfreq.to(units, equivalencies=u.cmb_equivalencies(freqs[i]*u.GHz))

            # Add together signal. Sky components get a gain term added to them.
            gain = params.g0
            signal_tod.append((gain*(d + dipole_myfreq.value) + white_noise + corr_noise).astype('float32'))
            white_noise_tod.append(white_noise.astype('float32'))
            corr_noise_tod.append(corr_noise.astype('float32'))

            observed_map[i] = np.bincount(pix, weights=signal_tod[-1], minlength=npix)
            white_noise_map[i] = np.bincount(pix, weights=white_noise, minlength=npix)
            corr_noise_map[i] = np.bincount(pix, weights=corr_noise, minlength=npix)
            hit_map[i] = np.bincount(pix, minlength=npix)
            assert (hit_map[i] > 0).all(), f"{np.sum(hit_map[i] == 0)} out of {hit_map[i].shape[0]} pixels were never hit by the scanning strategy."
            if (hit_map[i] <= 10).any():
                print(f"Warning: {np.sum(hit_map[i] <= 10)} out of {hit_map[i].shape[0]} pixels were hit 10 or fewer times by scanning strategy.")
            print(f"Lowest pixel hit count is {np.min(hit_map[i])}.")
            white_noise_map[i] /= hit_map[i]
            corr_noise_map[i] /= hit_map[i]
            inv_var_map[i] = hit_map[i]/sigma0s[i].value**2
            assert signal_tod[-1].shape == psi.shape, f"Shape of simulated TOD {signal_tod[-1].shape} differs from generated psi {psi.shape}"


    if rank == 0:
        t0 = time.time()
        print(f"Rank 0 writing simulation to file.")
        if params.write_fits:
            for i in range(len(freqs)):
                hp.write_map(params.OUTPUT_FOLDER + f"true_sky_full_{nside}_{freqs[i]}.fits", comps_sum_smoothed[i], overwrite=True)
                hp.write_map(params.OUTPUT_FOLDER + f"rms_map_{nside}_{freqs[i]}.fits", 1.0/np.sqrt(inv_var_map[i]), overwrite=True)
        if params.make_plots:
            for i in range(len(freqs)):
                hp.mollview(1.0/np.sqrt(inv_var_map[i]), title=f"Uncertainty {freqs[i]:.2f}GHz")
                plt.savefig(params.OUTPUT_FOLDER + f"rms_{nside}_{freqs[i]}_b{fwhm[i].value:.0f}.png")
                plt.close()
                hp.mollview(comps_sum_smoothed[i,0], title=f"Full 'clean' sky smoothed {freqs[i]:.2f}GHz")
                plt.savefig(params.OUTPUT_FOLDER + f"sky_smoothed_{nside}_{freqs[i]}_b{fwhm[i].value:.0f}.png")
                plt.close()

                hp.mollview(comps_sum_smoothed[i,0]+white_noise_map[i]+corr_noise_map[i], title=f"Full observed sky smoothed {freqs[i]:.2f}GHz")
                plt.savefig(params.OUTPUT_FOLDER + f"observed_sky_smoothed_{nside}_{freqs[i]}_b{fwhm[i].value:.0f}.png")
                plt.close()

                hp.mollview(white_noise_map[i], title=f"White noise {freqs[i]:.2f}GHz", cmap="RdBu_r", min=np.percentile(white_noise_map[i],2), max=np.percentile(white_noise_map[i],98))
                plt.savefig(params.OUTPUT_FOLDER + f"noise_white_{nside}_{freqs[i]}_b{fwhm[i].value:.0f}.png")
                plt.close()

                hp.mollview(corr_noise_map[i], title=f"Corr noise {freqs[i]:.2f}GHz", cmap="RdBu_r", min=np.percentile(corr_noise_map[i],2), max=np.percentile(corr_noise_map[i],98))
                plt.savefig(params.OUTPUT_FOLDER + f"noise_corr_{nside}_{freqs[i]}_b{fwhm[i].value:.0f}.png")
                plt.close()

                hp.mollview(observed_map[i], title=f"Actual observations of the sky at {freqs[i]:.2f}GHz", cmap="RdBu_r", min=np.percentile(observed_map[i],2), max=np.percentile(observed_map[i],98))
                plt.savefig(params.OUTPUT_FOLDER + f"actual_observations_{nside}_{freqs[i]}_b{fwhm[i].value:.0f}.png")
                plt.close()

            hp.mollview(hit_map[0], title=f"Hits", norm="log")
            plt.savefig(params.OUTPUT_FOLDER + f"hits_{nside}_b{fwhm[i].value:.0f}.png")
            plt.close()

        save_to_h5_file(signal_tod, pix, psi, LOS, orb_dir, fname=f'tod_sim_{params.NSIDE}_s{params.SIGMA_SCALE}_b{params.FWHM[0]:.0f}')
        save_to_h5_file(white_noise_tod, pix, psi, fname=f'white_noise_sim_{params.NSIDE}_s{params.SIGMA_SCALE}_b{params.FWHM[0]:.0f}')
        save_to_h5_file(corr_noise_tod, pix, psi, fname=f'corr_noise_sim_{params.NSIDE}_s{params.SIGMA_SCALE}_b{params.FWHM[0]:.0f}')
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