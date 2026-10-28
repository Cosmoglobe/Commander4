"""Builds TOD sampling steps and their surroundings for tests, without a parameter file or data
files."""
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI
from pixell.bunch import Bunch

from commander4.tod.data_selection import DataSelection
from commander4.tod.glitches.sampling import Glitches
from commander4.tod.hfi_demodulation import HFIModulation
from commander4.tod.jumps.sampling import Jumps
from commander4.tod.noise.sampling import Noise
from commander4.tod.scan_diagnostics import TODDiagnostics
from commander4.tod.sidelobe_deconvolve import FarSidelobes


def make_step(step_class: type, nscans: int = 1, ndet: int = 1, settings: dict | None = None,
              experiment: dict | None = None, **include):
    """`step_class` built for band BAND of experiment EXP, whose rank holds `nscans` scans of
    `ndet` detectors.

    `settings` holds `tod_processing` blocks (e.g. ``{"jump_detection": {"enabled": True}}``),
    `experiment` the experiment's own settings (e.g. ``{"jump_bitmask": 4}``), and `include` sets
    `output.chains.include` (e.g. ``residual_tods=True``). The band has no data, so a step that
    starts from the data files, like `Noise`, is built with its settings but without start values.
    It is not an HFI band, so `HFIModulation` is built switched off.
    """
    experiment_data = SimpleNamespace(nscans=nscans, ndet=ndet, experiment_name="EXP",
                                      band_name="BAND", hfi_demodulation=False, pols="I", nside=1,
                                      instrument_filepath=None,
                                      noise_model=SimpleNamespace(npar=1),
                                      iter_detector_scans=lambda: iter(()))
    params = Bunch(tod_processing=Bunch(**(settings or {})),
                   experiments=Bunch(EXP=Bunch(**(experiment or {}), bands=Bunch(BAND=Bunch()))),
                   output=Bunch(chains=Bunch(include=Bunch(**include))))
    return step_class(experiment_data, params, None, MPI.COMM_SELF, 1)


def fake_tod_samples(band, sigma0: float, abs_gain: float = 1.0,
                     corr_noise: dict | None = None) -> SimpleNamespace:
    """A stand-in for `TODSamples` of `band`, a one-scan band on a single rank, for the mapmakers.

    Every detector has the gain `abs_gain` and the white-noise level `sigma0`. The steps have their
    default settings, except that `corr_noise` sets the noise settings, and sigma0 is not sampled
    unless it says so.
    """
    ndet = band.ndet
    noise = make_step(Noise, 1, ndet,
                      settings={"corr_noise": {"sample_sigma0": False, **(corr_noise or {})}})
    noise.noise_params = np.full((1, ndet, 1), sigma0)
    return SimpleNamespace(
        experiment_data=band, band_comm=MPI.COMM_SELF, chain=1, noise=noise,
        gain=SimpleNamespace(abs_gain=abs_gain, rel_gain=np.zeros(ndet),
                             temporal_gain=np.zeros((1, ndet))),
        selection=make_step(DataSelection, 1, ndet), jumps=make_step(Jumps, 1, ndet),
        hfi=make_step(HFIModulation, 1, ndet), glitches=make_step(Glitches, 1, ndet),
        far_sidelobes=make_step(FarSidelobes, 1, ndet),
        diagnostics=make_step(TODDiagnostics, 1, ndet))


def mapmaking_params(band, mapmaker: str, cg: dict | None = None, **include) -> Bunch:
    """The part of a parameter file the mapmakers read, for `band`.

    The `mapmaker` is set for all bands, `cg` gives the CG limits (default 20 iterations to a
    tolerance of 1e-12), there is one TOD thread and no smoothing, and `include` switches on maps
    for the chain (e.g. ``hit_maps=True``); the others are off.
    """
    include = {"orbital_dipole_maps": False, "corr_noise_maps": False, "sky_model_maps": False,
               **include}
    cg = {"max_iter": 20, "err_tol": 1e-12} if cg is None else cg
    return Bunch(
        tod_processing=Bunch(mapmaker=mapmaker, cg_mapmaker=Bunch(**cg)),
        experiments=Bunch(**{band.experiment_name: Bunch(bands=Bunch(**{band.band_name: Bunch()}))}),
        resources=Bunch(tod=Bunch(num_threads=1)),
        compsep=Bunch(common_res_fwhm=0.0),
        output=Bunch(chains=Bunch(include=Bunch(**include))))
