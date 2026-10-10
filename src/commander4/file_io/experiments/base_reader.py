"""`TODReader`: reads one band's scans from Commander-format HDF5 scan files.

Every experiment reader is this class or a subclass of it. A subclass passes the properties of its
experiment's file format (gain units, pointing layout, noise priors, ...) to `TODReader.__init__`.
It overrides one of the steps below only if its files need logic that the base class does not have:

    read()                    the scan loop; calls the steps below for every scan and detector
      read_scan_header()      the values shared by all detectors of one scan
      read_detector()         one detector-scan as a `DetectorTOD` (None if the detector is absent)
        read_pointing()       its pixel and polarization-angle pointing
      keep_detector()         the cuts that need only the detector-scan itself
      drop_level_outliers()   the cut that compares each detector-scan with the rest of the band

The data-quality settings (the flag bitmask and the cut thresholds) come only from the parameter
file, never from the code. The functions below the class are the reader's helpers.

The file layout, one file per scan (also what ``simgen`` writes)::

    common/{fsamp, nside, npsi, det, polang, ...}            once per file
    <pid>/common/{ntod, hufftree, huffsymb, vsun, time, ...}  once per scan
    <pid>/<detector>/{tod or ztod, pix, psi, flag, scalars}   once per detector-scan

where ``<pid>`` is the scan ID as six digits. Entries that only some experiments carry are read
when present. A detector's TOD is Huffman-compressed when it is stored as ``ztod`` rather than
``tod``; it is then decoded with the scan's second tree (``hufftree2``, ``huffsymb2``).
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
        params, experiment, band: The full parameter file, and this band's experiment and band
            blocks. The band's ``detectors`` set the full-band detector order: a detector's
            position there is its ``det_idx_fullband``.
        noise_model: The band's noise PSD model, with priors suited to the instrument. The
            parameter file's noise settings are applied on top. None means `NoisePSDOof` with its
            own defaults.
        gain_factor: Multiplies the first of each detector's file ``scalars``, the initial gain.
            The standard format stores the gain in micro-units, hence 1e-6.
        boresight_pointing: The files store one boresight path per scan plus each detector's
            focal-plane offset, instead of each detector's pixels and angles.
        pol_eff_from_instrument_file: Take each detector's polarization response from
            ``<detector>/polEff`` (in percent) in the experiment's ``instrument_file``.
        hfi_demodulation: The TOD holds alternating Planck HFI modulation half-cycles.

    Read from the experiment block of the parameter file: ``bad_data_bitmask`` (required);
    ``min_unmasked_fraction`` (default 0) for `keep_detector`; ``max_rms_ratio`` (default None,
    which disables the cut) for `drop_level_outliers`; and ``replace_tod_with_sim``.
    """
    def __init__(self, band_comm: MPI.Comm, params: Bunch, experiment: Bunch, band: Bunch, *,
                 noise_model: NoisePSD | None = None, gain_factor: float = 1e-6,
                 boresight_pointing: bool = False, pol_eff_from_instrument_file: bool = False,
                 hfi_demodulation: bool = False):
        self.band_comm = band_comm
        self.experiment = experiment
        self.band = band
        self.det_names = list(band.detectors)
        self.gain_factor = gain_factor
        self.boresight_pointing = boresight_pointing
        self.hfi_demodulation = hfi_demodulation
        scope = (f"experiments.{experiment._name}",)
        self.bad_data_bitmask = resolve_param(params, "bad_data_bitmask", scope, legal_types=int)
        self.min_unmasked_fraction = resolve_param(params, "min_unmasked_fraction", scope,
                                                   default=0.0)
        self.max_rms_ratio = resolve_param(params, "max_rms_ratio", scope, default=None)
        # A TOD that the dispatcher replaces with a simulation after reading may be zeros on disk.
        self.tod_is_simulated = resolve_param(params, "replace_tod_with_sim", scope, default=False)

        self.default_mask, self.specific_masks = read_processing_masks(band_comm, band)
        self.noise_model = NoisePSDOof() if noise_model is None else noise_model
        self.noise_model.apply_param_file(params, experiment._name, band._name)

        # (ndet,) polarization efficiency per detector from the instrument file, or None if unused.
        self.pol_eff = None
        if pol_eff_from_instrument_file:
            self.pol_eff = np.empty(len(self.det_names), dtype=np.float64)
            if band_comm.Get_rank() == 0:
                with h5py.File(experiment.instrument_file, "r") as instrument:
                    for idet, det_name in enumerate(self.det_names):
                        # Instrument files store polEff in percent; the response is a fraction.
                        self.pol_eff[idet] = float(instrument[f"{det_name}/polEff"][()].item())/100
                unpolarized = self.pol_eff < UNPOLARIZED_POLEFF_CUTOFF
                if unpolarized.any():
                    names = ", ".join(np.array(self.det_names)[unpolarized])
                    logger.info(f"Band {band._name}: {names} have polEff below "
                                f"{100*UNPOLARIZED_POLEFF_CUTOFF:.0f}% and are treated as "
                                "intensity-only.")
                    self.pol_eff[unpolarized] = 0.0
            band_comm.Bcast(self.pol_eff, root=0)


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
            A `Bunch` of the scan's values, or None if the file has no ``ntod`` for this scan. The
            per-detector values (``polang``, ``response_I_P``, ``file_idx``) are arrays in the
            band's detector order, indexed by ``det_idx_fullband``.
        """
        pid = f"{scan_id:06d}"
        if f"{pid}/common/ntod" not in f:
            return None
        common = f["common"]
        scan_common = f[f"{pid}/common"]
        ntod = int(scan_common["ntod"][()].item())
        ndet = len(self.det_names)
        header = Bunch(pid=pid, ntod=ntod,
                       # TODO(fft sizes): use a size chosen for the whole band from a pre-pass.
                       ntod_fft=find_good_fourier_size(ntod),
                       fsamp=float(common["fsamp"][()].item()),
                       # The flags are always Huffman-compressed, so this tree is always present.
                       huffman_tree=scan_common["hufftree"][()],
                       huffman_symbols=scan_common["huffsymb"][()],
                       huffman_tree2=None, huffman_symbols2=None, data_nside=None, npsi=None,
                       vsun=np.zeros(3), start_time=0.0,
                       polang=np.full(ndet, np.nan),  # NaN: the file gives no angle
                       response_I_P=np.ones((ndet, 2)))
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
            # The file lists its detectors in its own order, which need not be the band's.
            # `file_idx[idet]` is band detector idet's position in the file (-1: not in the file).
            file_det_names = [name.strip() for name in common["det"].asstr()[()].split(",")]
            position = {name: i for i, name in enumerate(file_det_names)}
            header.file_idx = np.array([position.get(name, -1) for name in self.det_names])
            in_file = header.file_idx >= 0
            if "polang" in common:
                header.polang[in_file] = common["polang"][()][header.file_idx[in_file]]
            if "resp" in common:
                header.response_I_P[in_file] = common["resp"][()][header.file_idx[in_file]]
        # The instrument file's polarization efficiency takes precedence over the scan file's.
        if self.pol_eff is not None:
            header.response_I_P[:, 0] = 1.0
            header.response_I_P[:, 1] = self.pol_eff

        if self.boresight_pointing:
            # One boresight path for the whole scan, rotated per detector by its focal-plane
            # offset and polarization angle (both listed in the file's detector order).
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
            return DetectorBoresightPointing(header.scan_pointing, header.file_idx[idet])
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
        tod_is_compressed = "ztod" in group
        if tod_is_compressed:
            tod = group["ztod"][()]
        else:
            # TODO(buffers, read_direct): read straight into a slice of one per-rank buffer.
            tod = group["tod"][:header.ntod_fft].astype(np.float32, copy=False)
        init_scalars = None
        if "scalars" in group:
            init_scalars = group["scalars"][()]  # [gain, sigma0, fknee, alpha]
            init_scalars[0] *= self.gain_factor
        polang = header.polang[idet]
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
            tod_is_compressed=tod_is_compressed,
            response_I_P=header.response_I_P[idet],
            # DetectorTOD marks a missing angle with None, which the sidelobe model checks for.
            polang=None if np.isnan(polang) else float(polang),
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
