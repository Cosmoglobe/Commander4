import numpy as np
import sys
import healpy as hp
sys.path.append("../src/python/")
from data_models import DetectorTOD , ScanTOD
from utils import single_det_mapmaker_python, single_det_mapmaker

def test_single_det_mapmaker():
    nscans = 11
    ntod = 251
    nside = 16
    npix = hp.nside2npix(nside)
    scanlist = []
    for iscan in range(nscans):
        scan = ScanTOD(value=np.random.normal(1.5, 2.5, ntod),
                           theta=np.random.uniform(0, np.pi, ntod),
                           phi=np.random.uniform(-np.pi, np.pi, ntod),
                           psi=np.zeros(ntod),
                           startTime=0.0)
        scanlist.append(scan)
    detector = DetectorTOD(scanlist=scanlist, nu=120.0)
    skymap = np.random.normal(0.5, 0.1, npix)

    detmap_signal_python, detmap_rms_python = single_det_mapmaker_python(detector, skymap)
    detmap_signal, detmap_rms = single_det_mapmaker_python(detector, skymap)

    assert np.allclose(detmap_signal_python, detmap_signal)
    assert np.allclose(detmap_rms_python, detmap_rms)
