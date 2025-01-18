# PARAMETER FILE TO RUN A SIMULATED 
# DATA CREATION

OUTPUT_FOLDER = 'sim_data_cmb_dust/'
POINTING_PATH = '/mn/stornext/d5/data/artemba/other/extracted_pointing_no_skip_the_whole_survey_shifted.h5'

# general map characteristics
NSIDE = 256
NTOD = 12*256**2 * 50
FWHM = 20 # [arcmin]
FREQ = [30, 100, 353, 545, 857] # [GHz]
pol = False # include (Q,U) polarization
unit = 'uK_RJ' #'MJ/sr' # or 'uK_RJ'

# cosmology parameters
H0 = 67.5
ombh2 = 0.022
omch2 = 0.122
mnu = 0.06
omk = 0
tau = 0.06
As = 2.e-9
ns = 0.965

# components = ["CMB", "dust", "sync", "corr_noise"]  # Remove components from this list to remove them from simulation.
components = ["CMB", "dust"]

# thermal dust parameters
nu_ref_dust = 857 # [GHz]
beta_dust = 1.54 # check the preset
T_dust = 20 # [K]

# synchrotron parameters
nu_ref_sync = 23 # [GHz]
beta_sync = -3.1 # check the preset

# noise parameters
SIGMA0 = [100, 80, 30, 100, 200] # tod level white noise [uK_RJ] 
SIGMA_SCALE = 1.0 # scale noise level
SAMP_FREQ = 180 # sampling frequency [Hz]

