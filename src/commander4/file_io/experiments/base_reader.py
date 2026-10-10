"""`TODReader`: reads one band's scans from Commander-format HDF5 scan files.

Every experiment reader is this class or a subclass of it. A subclass passes the properties of its
experiment's file format (flag bitmask, compression, noise priors, ...) to `TODReader.__init__`. It
overrides one of the steps below only if its files need logic that the base class does not have:

    read()                    the scan loop; calls the steps below for every scan and detector
      read_scan_header()      the values shared by all detectors of one scan
      read_detector()         one detector-scan as a `DetectorTOD` (None if the detector is absent)
        read_pointing()       its pixel and polarization-angle pointing
      keep_detector()         the cuts that need only the detector-scan itself
      drop_level_outliers()   the cut that compares each detector-scan with the rest of the band

The data-quality thresholds of the cuts come only from the parameter file, never from the code.
The functions below the class are the reader's helpers: noise priors, FFT sizes, processing masks.

The file layout, one file per scan (also what ``simgen`` writes)::

    common/{fsamp, nside, npsi, det, polang, ...}            once per file
    <pid>/common/{ntod, hufftree, huffsymb, vsun, time, ...}  once per scan
    <pid>/<detector>/{tod or ztod, pix, psi, flag, scalars}   once per detector-scan

where ``<pid>`` is the scan ID as six digits. Entries that only some experiments carry are read
when present.
"""
import gc
import logging

import ducc0.fft
import h5py
import healpy as hp
import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.scan_tod import ScanTOD
from commander4.data_models.detector_group_tod import DetectorGroupTOD
from commander4.data_models.pointing import (PixelPointing, ScanBoresightPointing,
                                             DetectorBoresightPointing)
from commander4.parameters.schema import resolve_param
from commander4.tod.noise.psd import NoisePSD, NoisePSDOof
from commander4.math_utils.transfer_func import _tau_sec

logger = logging.getLogger(__name__)

# Flag bits marking unusable samples in the standard format. Matches `GOOD_SCAN_BITMASK` in
# simgen/writers.py, which documents the contract a writer of this format must satisfy.
STANDARD_BAD_DATA_BITMASK = 6111232

# Minimum polarization efficiency before the detector is treated as unpolarized.
# Current only applies to HFI, where the "unpolarized" bolometers are still quoted at a few
# percent polarization. The measurement is considered inaccurate, so we treat them as unpolarized.
UNPOLARIZED_POLEFF_CUTOFF = 0.2


class TODReader:
    """Read one band's scans from Commander-format HDF5 scan files.

    The defaults describe the standard format (``experiment_id: general``). Construction and
    `read` are collective over ``band_comm``.

    Args:
        band_comm: The band's MPI communicator.
        experiment, band: The experiment and band parameter blocks.
        det_names: Detector names in full-band order; a detector's position here is its
            ``det_idx_fullband`` column in the dense per-detector sample arrays.
        params: The full parameter file.
        noise_model: The band's noise PSD model, with priors suited to the instrument. The
            parameter file's noise priors and fit range are applied on top. None means
            `NoisePSDOof` with its own defaults.
        gain_factor: Multiplies the first of each detector's file ``scalars``, the initial gain.
            The standard format stores the gain in micro-units, hence 1e-6.
        bad_data_bitmask: Flag bits that mark a sample unusable; None cuts no samples. The
            parameter file's ``bad_data_bitmask`` overrides it.
        tod_is_compressed: Read the Huffman-compressed ``ztod`` instead of the plain ``tod``. The
            parameter file's ``tod_is_compressed`` overrides it.
        boresight_pointing: The files store one boresight path per scan plus each detector's
            focal-plane offset, instead of each detector's pixels and angles.
        pol_eff_from_instrument_file: Take each detector's polarization response from
            ``<detector>/polEff`` (in percent) in the experiment's ``instrument_file``.
        hfi_demodulation: The TOD holds alternating Planck HFI modulation half-cycles.

    Two data-quality thresholds are read from the parameter file's experiment block:
    ``min_unmasked_fraction`` (default 0) for `keep_detector`, and ``max_rms_ratio`` (default
    None, which disables the cut) for `drop_level_outliers`.
    """
    def __init__(self, band_comm: MPI.Comm, experiment: Bunch, band: Bunch, det_names: list[str],
                 params: Bunch, *, noise_model: NoisePSD | None = None, gain_factor: float = 1e-6,
                 bad_data_bitmask: int | None = STANDARD_BAD_DATA_BITMASK,
                 tod_is_compressed: bool = False, boresight_pointing: bool = False,
                 pol_eff_from_instrument_file: bool = False, hfi_demodulation: bool = False):
        self.band_comm = band_comm
        self.experiment = experiment
        self.band = band
        self.det_names = det_names
        self.gain_factor = gain_factor
        self.bad_data_bitmask = getattr(experiment, "bad_data_bitmask", bad_data_bitmask)
        self.tod_is_compressed = getattr(experiment, "tod_is_compressed", tod_is_compressed)
        self.min_unmasked_fraction = float(getattr(experiment, "min_unmasked_fraction", 0.0))
        self.max_rms_ratio = getattr(experiment, "max_rms_ratio", None)
        self.boresight_pointing = boresight_pointing
        self.hfi_demodulation = hfi_demodulation
        # A TOD that the dispatcher replaces with a simulation after reading may be zeros on disk.
        self.tod_is_simulated = getattr(experiment, "replace_tod_with_sim", False)

        self.default_mask, self.specific_masks = read_processing_masks(band_comm, band)
        self.noise_model = NoisePSDOof() if noise_model is None else noise_model
        apply_noise_priors(self.noise_model, params, experiment._name, band._name)
        apply_noise_fit_range(self.noise_model, params)

        # Polarization response [I, P] per detector name from the instrument file; empty if unused.
        self.instrument_response_I_P = {}
        if pol_eff_from_instrument_file:
            pol_eff = np.empty(len(det_names), dtype=np.float64)
            if band_comm.Get_rank() == 0:
                with h5py.File(experiment.instrument_file, "r") as instrument:
                    for idet, det_name in enumerate(det_names):
                        # Instrument files store polEff in percent; the response is a fraction.
                        pol_eff[idet] = float(instrument[f"{det_name}/polEff"][()].item()) / 100.0
                unpolarized = pol_eff < UNPOLARIZED_POLEFF_CUTOFF
                if unpolarized.any():
                    names = ", ".join(name for name, cut in zip(det_names, unpolarized) if cut)
                    logger.info(f"Band {band._name}: {names} have polEff below "
                                f"{100*UNPOLARIZED_POLEFF_CUTOFF:.0f}% and are treated as "
                                "intensity-only.")
                    pol_eff[unpolarized] = 0.0
            band_comm.Bcast(pol_eff, root=0)
            self.instrument_response_I_P = {name: (1.0, eff)
                                            for name, eff in zip(det_names, pol_eff)}


    def read(self, scan_ids: list[int], paths: list[str]) -> DetectorGroupTOD:
        """Read the given scans and keep the detector-scans that pass `keep_detector`.

        A scan whose detectors are all dropped is left out, so the band may hold fewer scans than
        were asked for.

        Args:
            scan_ids: The scan IDs to read, in time order.
            paths: The file holding each scan.
        """
        scans = []
        fsamp = 0.0
        for iscan, (scan_id, path) in enumerate(zip(scan_ids, paths)):
            with h5py.File(path, "r") as f:
                header = self.read_scan_header(f, scan_id)
                if header is None:
                    # Some converted files hold empty placeholder scans (about a fifth of the HFI
                    # 143 GHz filelist). They show in the summary as scans not read.
                    logger.debug(f"Band {self.band._name}: {path} has no data for scan "
                                 f"{scan_id}; the scan is skipped.")
                    continue
                fsamp = header.fsamp
                detectors = []
                for idet, det_name in enumerate(self.det_names):
                    det = self.read_detector(f, header, idet, det_name)
                    if det is not None and self.keep_detector(det):
                        detectors.append(det)
            if len(detectors) > 0:
                scans.append(ScanTOD(detectors, header.start_time, scan_id))
            if self.band_comm.Get_rank() == 0 and iscan % max(1, len(scan_ids)//5) == 0:
                logger.debug(f"Reading scans from disk, progress on master rank of band "
                             f"{self.band._name}: {iscan}/{len(scan_ids)}")
            if iscan % 10 == 0:
                gc.collect()
        if self.max_rms_ratio is not None and not self.tod_is_simulated:
            scans = self.drop_level_outliers(scans)

        # Ranks that read no scan get the sample rate from the others.
        fsamp = self.band_comm.allreduce(fsamp, op=MPI.MAX)
        return DetectorGroupTOD(scans, self.experiment._name, self.band._name,
                                self.band.eval_nside, self.band.freq, self.band.fwhm, fsamp,
                                len(self.det_names), self.band.polarization, self.noise_model,
                                tf_tau_sec=_tau_sec(self.band),
                                instrument_filepath=getattr(self.experiment, "instrument_file",
                                                            None),
                                hfi_demodulation=self.hfi_demodulation)


    def read_scan_header(self, f: h5py.File, scan_id: int) -> Bunch | None:
        """Read the values shared by all detectors of one scan.

        Returns:
            A `Bunch` of the scan's values, or None if the file has no ``ntod`` for this scan.
            Per-detector values from the file (``polang``, ``response_I_P``) are dicts keyed by
            detector name, because the file's detector order need not be the band's.
        """
        pid = f"{scan_id:06d}"
        if f"{pid}/common/ntod" not in f:
            return None
        common = f["common"]
        scan_common = f[f"{pid}/common"]
        ntod = int(scan_common["ntod"][()].item())
        header = Bunch(pid=pid, ntod=ntod,
                       # TODO(fft sizes): use a size chosen for the whole band from a pre-pass.
                       ntod_fft=find_good_fourier_size(ntod),
                       fsamp=float(common["fsamp"][()].item()),
                       # The flags are always Huffman-compressed, so this tree is always present.
                       huffman_tree=scan_common["hufftree"][()],
                       huffman_symbols=scan_common["huffsymb"][()],
                       huffman_tree2=None, huffman_symbols2=None, data_nside=None, npsi=None,
                       vsun=np.zeros(3), start_time=0.0, polang={}, response_I_P={})
        # A second tree decodes Huffman-compressed TODs.
        if "hufftree2" in scan_common:
            header.huffman_tree2 = scan_common["hufftree2"][()]
            header.huffman_symbols2 = scan_common["huffsymb2"][()]
        if "nside" in common:
            header.data_nside = int(common["nside"][()].item())
        if "npsi" in common:
            header.npsi = int(common["npsi"][()].item())
        if "vsun" in scan_common:
            header.vsun = scan_common["vsun"][()]
        if "time" in scan_common:
            header.start_time = float(scan_common["time"][0])  # MJD
        if "det" in common:
            file_det_names = [name.strip() for name in common["det"].asstr()[()].split(",")]
            if "polang" in common:
                header.polang = dict(zip(file_det_names, common["polang"][()].tolist()))
            if "resp" in common:
                header.response_I_P = dict(zip(file_det_names, common["resp"][()]))
        # The instrument file's response takes precedence over the scan file's.
        header.response_I_P.update(self.instrument_response_I_P)

        if self.boresight_pointing:
            # One boresight path for the whole scan, rotated per detector by its focal-plane
            # offset and polarization angle (both listed in the file's detector order).
            header.file_det_index = {name: i for i, name in enumerate(file_det_names)}
            header.scan_pointing = ScanBoresightPointing(
                header.start_time, float(scan_common["time_end"][0]), ntod, common["site"][()],
                scan_common["bore"][()], common["detoff"][()], common["polang"][()],
                self.band.eval_nside, header.ntod_fft)
        return header


    def read_pointing(self, f: h5py.File, header: Bunch, idet: int,
                      det_name: str) -> PixelPointing | DetectorBoresightPointing:
        """Return one detector's pointing for this scan.

        An intensity-only band gets a zero psi, because its files need not store one.
        """
        if self.boresight_pointing:
            return DetectorBoresightPointing(header.scan_pointing, header.file_det_index[det_name])
        group = f[f"{header.pid}/{det_name}"]
        pix = group["pix"][()]
        if pix.ndim == 2:  # Some simulations store shape (1, ntod).
            pix = pix[0]
        if "QU" in self.band.polarization:
            psi = group["psi"][()]
            if psi.ndim == 2:
                psi = psi[0]
        else:
            psi = np.zeros(header.ntod_fft, dtype=np.float32)
        return PixelPointing(pix, psi, header.huffman_tree, header.huffman_symbols, header.npsi,
                             self.band.eval_nside, header.data_nside, header.ntod, header.ntod_fft)


    def read_detector(self, f: h5py.File, header: Bunch, idet: int,
                      det_name: str) -> DetectorTOD | None:
        """Read one detector-scan, or return None if the detector is absent from this scan."""
        if det_name not in f[header.pid]:
            return None
        group = f[f"{header.pid}/{det_name}"]
        if self.tod_is_compressed:
            tod = group["ztod"][()]
        else:
            # TODO(buffers, read_direct): read straight into a slice of one per-rank buffer.
            tod = group["tod"][:header.ntod_fft].astype(np.float32, copy=False)
        init_scalars = None
        if "scalars" in group:
            init_scalars = group["scalars"][()]  # [gain, sigma0, fknee, alpha]
            init_scalars[0] *= self.gain_factor
        return DetectorTOD(
            name=det_name,
            det_idx_fullband=idet,
            tod=tod,
            pointing=self.read_pointing(f, header, idet, det_name),
            sampling_rate_hz=header.fsamp,
            orbital_velocity_m_per_s=header.vsun,
            huffman_tree=header.huffman_tree,
            huffman_symbols=header.huffman_symbols,
            huffman_tree2=header.huffman_tree2,
            huffman_symbols2=header.huffman_symbols2,
            default_proc_mask=self.default_mask,
            specific_proc_masks=self.specific_masks,
            flag_encoded=group["flag"][()],
            bad_data_bitmask=self.bad_data_bitmask,
            init_scalars=init_scalars,
            tod_is_compressed=self.tod_is_compressed,
            response_I_P=header.response_I_P.get(det_name),
            polang=header.polang.get(det_name),
        )


    def keep_detector(self, det: DetectorTOD) -> bool:
        """Apply the cuts that need only the detector-scan itself.

        A detector-scan is always dropped when it is unusable: it has no unflagged samples, or its
        TOD is non-finite or all zero. It is also dropped when less than ``min_unmasked_fraction``
        of its samples are unflagged. A dropped detector-scan never reaches the samplers, unlike the
        per-iteration vetoes in `tod/data_selection.py`, which only flag. The TOD-value checks are
        skipped when the TOD will be replaced by a simulation.
        """
        # With no unflagged samples there is no measurable white-noise level: sigma0 comes out
        # inf, the correlated-noise CG divides by zero, and the resulting NaN spreads through the
        # band-wide gain sums.
        good_fraction = det.good_data_mask.mean()
        if good_fraction == 0 or good_fraction < self.min_unmasked_fraction:
            return False
        if self.tod_is_simulated:
            return True
        tod = det.tod
        return bool(np.isfinite(tod).all() and tod.any())


    def drop_level_outliers(self, scans: list[ScanTOD]) -> list[ScanTOD]:
        """Drop detector-scans whose TOD level is far above their detector's usual level.

        The level is the RMS of the unflagged samples, offset included. A detector-scan is dropped
        when its level is above ``max_rms_ratio`` times the median level of the same detector over
        the whole band. This catches broken detector-scans, such as a large constant offset,
        without a threshold in detector units. Collective over ``band_comm``.

        Returns:
            The scans with those detector-scans removed; a scan left with no detectors is removed.
        """
        detectors = [det for scan in scans for det in scan.detectors]
        det_idx = np.array([det.det_idx_fullband for det in detectors], dtype=np.int64)
        rms = np.array([np.sqrt(np.mean(np.square(det.tod[det.good_data_mask], dtype=np.float64)))
                        for det in detectors], dtype=np.float64)
        # Each detector's median level needs its detector-scans from every rank.
        all_det_idx = np.concatenate(self.band_comm.allgather(det_idx))
        all_rms = np.concatenate(self.band_comm.allgather(rms))
        median_rms = np.full(len(self.det_names), np.inf)
        for idet in np.unique(all_det_idx):
            median_rms[idet] = np.median(all_rms[all_det_idx == idet])
        keep = rms <= self.max_rms_ratio * median_rms[det_idx]

        ndropped = self.band_comm.allreduce(int(np.sum(~keep)), op=MPI.SUM)
        if self.band_comm.Get_rank() == 0 and ndropped > 0:
            logger.info(f"Band {self.band._name}: dropped {ndropped} detector-scans whose TOD RMS "
                        f"is more than {self.max_rms_ratio} times their detector's median.")
        # `keep` follows the order of `detectors`: scan by scan, detector by detector.
        kept_scans = []
        start = 0
        for scan in scans:
            stop = start + len(scan.detectors)
            scan.detectors = [det for det, ok in zip(scan.detectors, keep[start:stop]) if ok]
            start = stop
            if len(scan.detectors) > 0:
                kept_scans.append(scan)
        return kept_scans


def _resolve_noise_prior_block(params: Bunch, key: str, expname: str, bandname: str,
                               param_names: tuple[str, ...], model_name: str) -> dict | None:
    """One noise-prior block from the band, else the experiment, else ``tod_processing``.

    Returns None when no scope sets it. Every key is checked against the model's parameter names,
    because a misspelled name would otherwise silently leave the default in force, and a parameter
    stuck at a wrong default looks exactly like a converged one in the chain.
    """
    block = resolve_param(params, key, (f"experiments.{expname}.bands.{bandname}",
                                        f"experiments.{expname}", "tod_processing"), default=None)
    if block is None:
        return None
    for name in block:
        if name not in param_names:
            raise ValueError(f"{key!r} for band {bandname!r} names {name!r}, which is not a "
                             f"parameter of {model_name}: {list(param_names)}.")
    return block


def apply_noise_priors(noise_model: NoisePSD, params: Bunch, expname: str, bandname: str) -> None:
    """Override the noise model's prior defaults with anything the parameter file specifies.

    Two optional blocks, each a mapping from noise-parameter name to its setting, taken from the
    band block, else the experiment block, else ``tod_processing``. Only the named parameters are
    changed; the rest keep the instrument-appropriate defaults the reader built the model with::

        noise_prior_bounds:          # hard [lo, hi] limits (C3's p_uni)
          fknee: [0.01, 100.0]
          alpha: [-4.5, -0.5]
        noise_prior:                 # informative [mean, rms] (C3's p_active); optional
          fknee: [10.0, 0.5]         # rms in *decades* for log-normal parameters such as fknee
          alpha: [-2.7, 0.3]

    The bounds are the endpoints of the grid the PSD sampler draws on, so a true value outside them
    cannot be recovered: the sample pins against the nearest edge instead. The informative prior
    multiplies the likelihood along that grid (see `NoisePSD.log_prior`); an rms of ``.inf`` leaves
    it uninformative, and an rms ``<= 0`` holds the parameter fixed at its current value entirely.

    Args:
        noise_model: The model to modify in place.
        expname, bandname: Keys of this band's experiment and band blocks in `params`.
    """
    param_names = noise_model.param_names
    model_name = type(noise_model).__name__
    bounds = _resolve_noise_prior_block(params, "noise_prior_bounds", expname, bandname,
                                        param_names, model_name)
    prior = _resolve_noise_prior_block(params, "noise_prior", expname, bandname,
                                       param_names, model_name)
    if bounds is None and prior is None:
        return
    for name, limits in (bounds or {}).items():
        noise_model.P_uni[param_names.index(name)] = limits
    for name, (mean, rms) in (prior or {}).items():
        noise_model.P_active[param_names.index(name)] = (mean, rms)
    # `sampled` is spelled out because an rms of <= 0 switching a parameter off is easy to miss in
    # the [mean, rms] pairs, and a silently unsampled parameter looks exactly like a converged one.
    logger.info(f"Band {bandname}: noise priors overridden from the parameter file. "
                f"bounds={dict(zip(param_names, noise_model.P_uni.tolist()))}, "
                f"[mean, rms]={dict(zip(param_names, noise_model.P_active.tolist()))}, "
                f"sampled={[n for i, n in enumerate(param_names) if noise_model.is_sampled(i)]}.")


def apply_noise_fit_range(noise_model: NoisePSD, params: Bunch) -> None:
    """Apply configured frequency limits to the model, keeping unspecified reader defaults.

    The shared limits in ``tod_processing.corr_noise`` apply to every PSD parameter except sigma0.
    The model's ``nu_fit`` array is the sole source of frequency limits during sampling.
    """
    for column, key in enumerate(("psd_fit_nu_min", "psd_fit_nu_max")):
        value = resolve_param(params, key, ("tod_processing.corr_noise",), default=None,
                              raise_on_missing_scope=False)
        if value is not None:
            noise_model.nu_fit[1:, column] = value


def find_good_fourier_size(ntod: int) -> int:
    """Return the largest fast real-FFT size that is at most ``ntod``.

    ``ducc0.fft.good_size(n, True)`` returns the first fast real-FFT size at or above ``n``. Walking
    downward until it returns the candidate itself finds the closest fast size at or below the scan
    length without a machine-specific timing table.
    """
    if ntod < 2:
        raise ValueError("TOD length must be at least 2 to select an FFT size.")

    candidate = ntod
    while ducc0.fft.good_size(candidate, True) != candidate:
        candidate -= 1
    return candidate


def read_processing_masks(band_comm: MPI.Comm,
                          band_params: Bunch) -> tuple[NDArray | None, dict[str, NDArray]]:
    """Read a band's default and named processing-mask maps once and broadcast them.

    Args:
        band_comm: The band's MPI communicator; only rank 0 touches the filesystem.
        band_params: The band's parameter block (``processing_mask`` and/or ``processing_masks``).

    Returns:
        ``(default_mask, named_masks)``: the default boolean HEALPix map (or ``None`` if the band
        defines none) and a dict of named boolean HEALPix maps (empty if none are defined). Maps are
        kept at their native nside; ``TODView`` handles any nside mismatch with the pointing.
    """
    default_mask = None
    named_masks: dict[str, NDArray] = {}
    if band_comm.Get_rank() == 0:
        filename = getattr(band_params, "processing_mask", None)
        if filename is not None:
            default_mask = hp.read_map(filename, field=0, dtype=bool)
        for name in getattr(band_params, "processing_masks", []) or []:
            named_masks[name] = hp.read_map(band_params.processing_masks[name], field=0, dtype=bool)
    # bcast returns the broadcast object (it does not fill in place), so capture the return value.
    default_mask = band_comm.bcast(default_mask, root=0)
    named_masks = band_comm.bcast(named_masks, root=0)
    return default_mask, named_masks
