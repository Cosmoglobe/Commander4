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

import camb
from camb import model, initialpower

from astropy.modeling.physical_models import BlackBody


def mixmat_d(nu, nu_0, beta, T):
    bb = BlackBody(temperature=T*u.K)
    M = (nu/nu_0)**beta
    M *= bb(nu*u.GHz)/bb(nu_0*u.GHz)
    return M

def mixmat_s(nu, nu_0, beta):
    M = (nu/nu_0)**beta
    return M


nside = 64 #8192#256
lmax = 3*nside-1
fwhm_arcmin = 20
fwhm = fwhm_arcmin*u.arcmin
sigma_fac = 1.0

sigma0s = np.array([100, 80, 30, 100, 200])*sigma_fac*u.uK_CMB
freqs = np.array([30, 100, 353, 545, 857])



pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06,
                       As=2e-9, ns=0.965, halofit_version='mead', lmax=lmax)

results = camb.get_results(pars)
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
totCL=powers['total']

ell = np.arange(lmax+1)
Cl = totCL[ell,0]
Cl_EE = totCL[ell,1]
Cl_BB = totCL[ell,2]
Cl_TE = totCL[ell,3]

Cls = np.array([Cl, Cl_EE, Cl_BB, Cl_TE])


nu_dust = 857

#dust = pysm3.Sky(nside=1024, preset_strings=["d1"])
#dust_maps = [dust.get_emission(f*u.GHz) for f in freqs]
#dust_s = [pysm3.apply_smoothing_and_coord_transform(d, fwhm=fwhm) for d in dust_maps]
#dust_s = [d.to(u.MJy/u.sr, equivalencies=u.cmb_equivalencies(f*u.GHz)) for d, f in zip(dust_s, freqs)]
#dust_s = [hp.ud_grade(d.value, nside)*d.unit for d in dust_s]


beta = 1.5
T = 20

beta_s = -3

dust = pysm3.Sky(nside=1024, preset_strings=["d1"])
dust_857 = dust.get_emission(857*u.GHz).to(u.MJy/u.sr, equivalencies=u.cmb_equivalencies(857*u.GHz))
dust_857_s = hp.smoothing(dust_857, fwhm=fwhm.to('rad').value)*dust_857.unit
dust_s = [dust_857_s*mixmat_d(f, 857, beta, T) for f in freqs]
dust_s = [d.to(u.MJy/u.sr, equivalencies=u.cmb_equivalencies(f*u.GHz)) for d,f in zip(dust_s,freqs)]
dust_s = [hp.ud_grade(d.value, nside)*d.unit for d in dust_s]



sync = pysm3.Sky(nside=1024, preset_strings=["d1"])
sync_23 = sync.get_emission(23*u.GHz).to(u.MJy/u.sr, equivalencies=u.cmb_equivalencies(23*u.GHz))
sync_23_s = hp.smoothing(sync_23, fwhm=fwhm.to('rad').value)*sync_23.unit
sync_s = [sync_23_s*mixmat_s(f, 23, beta_s) for f in freqs]
sync_s = [d.to(u.MJy/u.sr, equivalencies=u.cmb_equivalencies(f*u.GHz)) for d,f in zip(sync_s,freqs)]
sync_s = [hp.ud_grade(d.value, nside)*d.unit for d in sync_s]


npix = 12*nside**2

chunk_size = npix//40

np.random.seed(0)
alms = hp.synalm(Cls, lmax=3*nside-1, new=True)
cmb   = hp.alm2map(alms, nside, pixwin=False)
hp.write_map(f"true_sky_cmb_{nside}.fits", cmb, overwrite=True)

cmb_s = hp.smoothing(cmb, fwhm=fwhm.to('rad').value)

cmb_s = cmb_s * u.uK_CMB


repeat = 50
ntod = repeat*npix

pix = np.arange(ntod) % npix
psi = np.repeat(np.arange(repeat)*np.pi/repeat, npix)

ds = []
for i in range(len(freqs)):
    cmb_freq = cmb_s.to(u.MJy/u.sr, equivalencies=u.cmb_equivalencies(freqs[i]*u.GHz))
    m_s = cmb_freq + dust_s[i] + sync_s[i]
    I,Q,U = m_s
    d = I[pix] + Q[pix]*np.cos(2*psi) + U[pix]*np.sin(2*psi)
    d = d.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freqs[i]*u.GHz))
    ds.append((d + np.random.randn(ntod)*sigma0s[i]).astype('float32').value)

#hp.write_map(f"true_sky_dust857_s_{nside}.fits", dust_857_s, overwrite=True)

pix = pix.astype('int32')


n_chunks = ntod // chunk_size
print(f'Number of scans is {n_chunks}')



output_path = '.'
version = 'hello'
comm_tod = commander_tod(output_path, "", version, overwrite=True)

hdf_filename = f'tod_example_{nside}_s{sigma_fac}_b{fwhm_arcmin}_dust'

COMMON_GROUP = "/common"
HUFFMAN_COMPRESSION = ["huffman", {"dictNum": 1}]

comm_tod.init_file(freq=hdf_filename, od="", mode="w")
#
## nside
comm_tod.add_field(COMMON_GROUP + "/nside", [nside])

for pid in range(n_chunks):
    pid_label = f'{pid+1:06}'
    pid_common_group = pid_label + "/common"
    for i, freq in enumerate(freqs):
        pid_data_group = f'{pid_label}/{freq:04}'

        comm_tod.add_field(pid_common_group + "/ntod", [chunk_size])


        tod_chunk_i = ds[i][pid*chunk_size : (pid+1)*chunk_size]
        pix_chunk_i =   pix[pid*chunk_size : (pid+1)*chunk_size]
        psi_chunk_i =   psi[pid*chunk_size : (pid+1)*chunk_size]

        comm_tod.add_field(pid_data_group + "/tod", tod_chunk_i)
        comm_tod.add_field(pid_data_group + "/pix", pix_chunk_i)
        comm_tod.add_field(pid_data_group + "/psi", psi_chunk_i)
    comm_tod.finalize_chunk(pid+1)

if (ntod//chunk_size != ntod/chunk_size):
    pid = n_chunks
    pid_label = f'{pid+1:06}'
    pid_common_group = pid_label + "/common"
    for i, freq in enumerate(freqs):
        pid_data_group = f'{pid_label}/{freq:04}'
    
        tod_chunk_i = ds[i][pid*chunk_size : ]
        pix_chunk_i = pix[pid*chunk_size : ]
        psi_chunk_i = psi[pid*chunk_size : ]
        comm_tod.add_field(pid_common_group + "/ntod", [len(tod_chunk_i)])
        comm_tod.add_field(pid_data_group + "/tod", tod_chunk_i)
        comm_tod.add_field(pid_data_group + "/pix", pix_chunk_i)
        comm_tod.add_field(pid_data_group + "/psi", psi_chunk_i)
    comm_tod.finalize_chunk(pid+1)


comm_tod.finalize_file()
