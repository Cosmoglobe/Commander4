"""TOD reader that spawns many synthetic detectors from one simulated pointing.

``experiment_id: litebird_sim_spawndetectors``. Reads a single detector's pointing from a
``litebird_sim`` file and reuses it for every detector in the band, so a scaling test can be run
with an arbitrary detector count without simulating each one's pointing.
"""
import h5py
import numpy as np
from mpi4py import MPI
from pixell.bunch import Bunch

import commander4.compression.huffman as huffman
from commander4.data_models.detector_tod import DetectorTOD
from commander4.data_models.pointing import PixelPointing
from commander4.file_io.experiments.base_reader import TODReader

# The detector in the files whose pointing every spawned detector reuses.
SOURCE_DETECTOR = "001_000_002_60A_166_T"
# The spawned detectors take these polarization-angle offsets in turn, so together they see Q and U.
PSI_OFFSETS = np.deg2rad(np.array([0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]))
# Number of bins the spawned detectors' psi is digitized into for Huffman compression.
NPSI = 4096


class SpawnDetectorsReader(TODReader):
    """Spawn every band detector from the pointing of `SOURCE_DETECTOR`, for MPI scaling tests.

    Each spawned detector gets the shared pointing rotated by its own psi offset, and a zero TOD for
    the in-place simulation (``replace_tod_with_sim: true``) to fill. Every detector is therefore
    present in every scan. The files' pointing must be stored uncompressed.
    """
    def __init__(self, band_comm: MPI.Comm, experiment: Bunch, band: Bunch, det_names: list[str],
                 params: Bunch):
        if getattr(experiment, "pix_is_compressed", False) or getattr(experiment,
                                                                      "psi_is_compressed", False):
            raise NotImplementedError("Compressed data not yet implemented in litebird injection "
                                      "sims.")
        super().__init__(band_comm, experiment, band, det_names, params)


    def read_scan_header(self, f: h5py.File, scan_id: int) -> Bunch | None:
        """Add the shared pointing and a block of zero TODs to the standard scan header."""
        header = super().read_scan_header(f, scan_id)
        if header is None:
            return None
        # The shared pointing is read once per scan, not once per detector, to spare the disks.
        group = f[f"{header.pid}/{SOURCE_DETECTOR}"]
        # reshape(-1) drops the leading axis of simulations that store shape (1, ntod).
        header.pix = group["pix"][()].reshape(-1)[:header.ntod_fft].astype(np.int64, copy=False)
        header.psi = group["psi"][()].reshape(-1)[:header.ntod_fft].astype(np.float32, copy=False)
        header.tods = np.zeros((len(self.det_names), header.ntod_fft), dtype=np.float32)
        return header


    def read_detector(self, f: h5py.File, header: Bunch, idet: int, det_name: str) -> DetectorTOD:
        """Spawn one detector: the shared pointing with this detector's psi offset, and a zero TOD.

        The pointing is Huffman-compressed per detector, because each psi offset changes the
        symbols. This is cheap compared with the rest of the read.
        """
        psi = header.psi.copy()
        psi += PSI_OFFSETS[idet % PSI_OFFSETS.size]
        psi = huffman.preproc_digitize_and_diff(psi, NPSI)
        pix = huffman.preproc_diff(header.pix)
        tree, symbols, sym_codes, sym_lengths = huffman.build_huffman_tree([pix, psi])
        pointing = PixelPointing(huffman.huffman_compress_array(pix, sym_codes, sym_lengths),
                                 huffman.huffman_compress_array(psi, sym_codes, sym_lengths),
                                 tree, symbols, NPSI, self.band.eval_nside, header.data_nside,
                                 header.ntod_fft, header.ntod_fft)
        return DetectorTOD(
            name=det_name,
            det_idx_fullband=idet,
            tod=header.tods[idet],
            pointing=pointing,
            sampling_rate_hz=header.fsamp,
            orbital_velocity_m_per_s=header.vsun,
            huffman_tree=tree,
            huffman_symbols=symbols,
            default_proc_mask=self.default_mask,
            specific_proc_masks=self.specific_masks,
        )
