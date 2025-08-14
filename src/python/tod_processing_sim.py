import h5py
import logging 
from pixell.bunch import Bunch
from src.python.data_models.detector_TOD import DetectorTOD
from src.python.data_models.scan_samples import ScanSamples
from src.python.data_models.detector_samples import DetectorSamples
from src.python.data_models.scan_TOD import ScanTOD

def read_TOD_sim_data(h5_filename: str, my_band: Bunch, params: Bunch, scan_idx_start: int, scan_idx_stop: int) -> DetectorTOD:
    logger = logging.getLogger(__name__)
    scanlist = []
    band_formatted = f"{my_band.freq:04d}"
    with h5py.File(h5_filename) as f:
        for iscan in range(scan_idx_start, scan_idx_stop):
            try:
                tod = f[f"{iscan+1:06}/{band_formatted}/tod"][()]
                theta = f[f"{iscan+1:06}/{band_formatted}/theta"][()]
                phi = f[f"{iscan+1:06}/{band_formatted}/phi"][()]
                psi = f[f"{iscan+1:06}/{band_formatted}/psi"][()]
                Ntod = tod.size
                # We assume the orbital velocity to be constant over the duration of a scan, so we read the half-way point.
                # This orbital velociy is relative to the Sun, but is in **Galactic coordinates**, to be easily compatible with other quantities.
                orb_dir_vec = f[f"{iscan+1:06}/{band_formatted}/orbital_dir_vec"][Ntod//2]
            except KeyError:
                logger.exception(f"{iscan+1:06}/{band_formatted}")
                raise KeyError
            scanlist.append(ScanTOD(tod, theta, phi, psi, 0., iscan))
            scanlist[-1].orb_dir_vec = orb_dir_vec
            scanlist[-1].fsamp = my_band.fsamp

    scansample_list = []
    for iscan in range(scan_idx_start, scan_idx_stop):
        scansample_list.append(ScanSamples())
        scansample_list[-1].time_dep_rel_gain_est = 0.0
        scansample_list[-1].rel_gain_est = my_band.rel_gain_est
        scansample_list[-1].gain_est = my_band.rel_gain_est + params.initial_g0
    det_samples = DetectorSamples(scansample_list)
    det_samples.detector_id = my_band.detector_id
    det_samples.g0_est = params.initial_g0
    
    det_static = DetectorTOD(scanlist, float(my_band.freq), my_band.fwhm, my_band.nside)
    det_static.detector_id = my_band.detector_id
    return det_static, det_samples
