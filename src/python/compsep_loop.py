import numpy as np
from mpi4py import MPI
import time
import h5py
import healpy as hp
from data import SimpleScan, SimpleDetector, SimpleDetectorGroup, SimpleBand, TodProcData
from data import SimpleDetectorMap, SimpleDetectorGroupMap, SimpleBandMap
from model.component import CMB, ThermalDust
from model.sky_model import SkyModel
import matplotlib.pyplot as plt

class Compsep2TodprocData:
    pass

class Tod2CompsepData:
    pass

def amplitude_sampling_per_pix(map_sky, map_rms, freqs) -> np.array:
    ncomp = 2
    nband, npix = map_sky.shape
    comp_maps = np.zeros((ncomp, npix))
    A = np.zeros((ncomp, ncomp, npix))
    M = np.zeros((nband, ncomp))
    cmb = CMB()
    dust = ThermalDust()
    for i in range(npix):
        M[:,0] = cmb.get_sed(freqs)
        M[:,1] = dust.get_sed(freqs)
        x = M.T.dot((1/map_rms[:,i]**2*map_sky[:,i]))
        x += M.T.dot(np.random.randn(nband)/map_rms[:,i])
        A = (M.T.dot(np.diag(1/map_rms[:,i]**2)).dot(M))
        try:
            comp_maps[:,i] = np.linalg.solve(A, x)
        except np.linalg.LinAlgError:
            comp_maps[:,i] = 0
    return comp_maps


# Component separation loop
def compsep_loop(comm, tod_master: int):
    # am I the master of the compsep communicator?
    master = comm.Get_rank() == 0
    if master:
        print("Compsep: loop started")

    # Initialization for all component separattion tasks goes here


    # we wait for new jobs until we get a stop signal
    while True:
        # check for simulation end
        stop = MPI.COMM_WORLD.recv(source=tod_master) if master else False
        stop = comm.bcast(stop, root=0)
        if stop:
            if master:
                print("Compsep: stop requested; exiting")
            return
        if master:
            print("Compsep: new job obtained")

        # get next data set for component separation
        data, iter, chain = MPI.COMM_WORLD.recv(source=tod_master) if master else None 
        # Broadcast te data to all tasks, or do anything else that's appropriate
        data = comm.bcast(data, root=0)
        if master:
            print("Compsep: data obtained. Working on it ...")

        # do stuff with data
        # time.sleep(1)
        # result = 2*data  # dummy

        # Assuming "data" is a nested list of shape (detgroups, dets, bands)
        signal_maps = []
        rms_maps = []
        # band_freqs = []
        for i_band in range(len(data)):
            band = data[i_band]
            for i_detgrp in range(len(band)):
                # detector_group = band.detgrplist[i_detgrp]
                detector_group = band[i_detgrp]
                for i_det in range(len(detector_group)):
                    print(i_band, i_detgrp, i_det)
                    # detector = detector_group.detlist[i_det]
                    detector = detector_group[i_det]
                    # signal_maps.append(band.map_sky)
                    # rms_maps.append(band.map_rms)
                    signal_maps.append(detector[0])
                    rms_maps.append(detector[1])
                    hp.mollview(signal_maps[-1])
                    plt.savefig(f"map_test_{i_band}_{i_det}.png")
                    hp.mollview(rms_maps[-1])
                    plt.savefig(f"rms_test_{i_band}_{i_det}.png")
        signal_maps = np.array(signal_maps)
        rms_maps = np.array(rms_maps)
        band_freqs = np.arange(1, signal_maps.shape[0]+1)
        comp_maps = amplitude_sampling_per_pix(signal_maps, rms_maps, band_freqs)

        component_types = [CMB, ThermalDust]
        component_list = []
        for i, component_type in enumerate(component_types):
            component = component_type()
            component.component_map = comp_maps[i]
            component_list.append(component)

        sky_model = SkyModel(component_list)

        band_maps = []
        for i_band in range(len(data)):
            band = data[i_band]
            detector_group_maps = []
            for i_detgrp in range(len(band)):
                detector_group = band[i_detgrp]
                detector_maps = []
                for i_det in range(len(detector_group)):
                    detector = detector_group[i_det]
                    detector_map = sky_model.get_sky_at_nu(1.0, 12*64**2)
                    detector_maps.append(detector_map)
                    hp.mollview(detector_map)
                    plt.savefig(f"skymap_test_{i_band}_{i_det}.png")
                detector_group_maps.append(detector_maps)
            band_maps.append(detector_group_maps)

        # band_maps = []
        # for i_band in range(len(data)):
        #     band = data[i_band]
        #     detector_group_maps = []
        #     for i_detgrp in range(len(band)):
        #         # detector_group = band.detgrplist[i_detgrp]
        #         detector_group = band[i_detgrp]
        #         detector_maps = []
        #         for i_det in range(len(detector_group)):
        #             # detector = detector_group.detlist[i_det]
        #             detector = detector_group[i_det]
        #             # detector_map = SimpleDetectorMap(sky_model.get_sky_at_nu(detector.nu))
        #             detector_map = SimpleDetectorMap(sky_model.get_sky_at_nu(1.0, 64), None)
        #             detector_maps.append(detector_map)
        #         detector_group_maps.append(detector_maps)
        #     band_maps.append(detector_group_maps)
        # out_data = SimpleBandMap(band_maps)
        




        # assemble result on master, via reduce, gather, whatever ...
        # send result
# compsep result is a data structure _per detector_, describing which sky this
# detector would see. Probably best described as a set of a_lm

        if master:
            MPI.COMM_WORLD.send(band_maps, dest=tod_master)
            print("Compsep: results sent back")

