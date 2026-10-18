"""`TODStep`: what every TOD sampling step has in common.

A sampling step is one class, in its own module, that holds a group of sampled quantities, the
methods that sample them, and its share of the band chain file. `FarSidelobes` (a model built
from each iteration's sky) and `TODDiagnostics` (output for inspection) sample nothing, but are
built and handled the same way.

The module also holds the two objects of a step's chain I/O: `ChainArrays`, what a step hands
over for the chain file, and `SavedBandSample`, what it reads back from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import h5py
import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray
from pixell.bunch import Bunch

from commander4.data_models.detector_group_tod import DetectorGroupTOD


@dataclass
class ChainArrays:
    """One step's share of the band chain file on this rank, by dataset name.

    Attributes:
        per_scan: Arrays with this rank's scans as the first axis. `TODSamples` gathers them over
            the band's ranks and sorts them by scan ID.
        per_band: Arrays that are identical on every rank of the band (per band or per detector).
            The band master writes them as they are.
        per_detector_scan: Arrays whose length differs between detector-scans, such as events or
            full TODs, as ``{group: {(iscan, idet): {name: array}}}``. They are written as
            ``<group>/<scan ID>/<detector name>/<name>``, and detector-scans with nothing to write
            are left out. Each group is gathered collectively, so a step must give the same groups
            on every rank, also where a rank has nothing to write.
    """
    per_scan: dict[str, NDArray] = field(default_factory=dict)
    per_band: dict[str, NDArray] = field(default_factory=dict)
    per_detector_scan: dict[str, dict[tuple[int, int], dict[str, NDArray]]] = \
        field(default_factory=dict)


class SavedBandSample:
    """One rank's part of a band chain file: the arrays cut to this rank's scans and detectors.

    Scans are matched by scan ID and detectors by name, so the MPI layout and the detector order
    may differ from the run that wrote the file, but every scan and detector of this run must be
    in it.

    Attributes:
        band_unit: The unit the file quotes its gains and maps in.
    """

    def __init__(self, f: h5py.File, scan_ids: NDArray, det_names: list[str], band_name: str):
        self._file = f
        # The file's row of each of this rank's scans, and the file's column of each detector.
        row_of_scan = {scan_id: row for row, scan_id in enumerate(f["scan_ids"][()])}
        column_of_det = {name: col for col, name in enumerate(f["det_names"].asstr()[()])}
        try:
            self._rows = np.array([row_of_scan[scan_id] for scan_id in scan_ids], dtype=np.int64)
            self._columns = [column_of_det[name] for name in det_names]
        except KeyError as error:
            raise ValueError(f"Scan or detector {error} of band {band_name} is not in "
                             f"{f.filename!r}.") from error
        self.band_unit = f["metadata/band_unit"].asstr()[()]


    def per_scan(self, name: str) -> NDArray:
        """A per-scan dataset, (nscans, ndet, ...) in this rank's scan and detector order."""
        # h5py reads a list of rows only in increasing order, so the rows are read sorted and then
        # put back in this rank's scan order.
        order = np.argsort(self._rows)
        sorted_values = self._file[name][self._rows[order]]
        values = np.empty_like(sorted_values)
        values[order] = sorted_values
        return values[:, self._columns]


    def per_detector(self, name: str) -> NDArray:
        """A per-detector dataset, (ndet, ...) in this rank's detector order."""
        return self._file[name][()][self._columns]


    def band(self, name: str) -> NDArray:
        """A per-band dataset, as stored."""
        return self._file[name][()]


class TODStep:
    """Base class of a TOD sampling step: its settings, its sampled state, the methods that sample
    it, and its share of the band chain file.

    `TODSamples` creates every step alike, before the Gibbs loop, and calls `chain_arrays` and
    `read_chain` on each for the chain file; `process_tod` calls `start_iteration` on each before
    the mapmaking scan loop. The defaults below do nothing, so a step overrides only what it uses.

    The constructor reads the step's settings into ``self.cfg`` (`parameters.schema.read_settings`)
    and checks them, so a bad setting fails at startup. A step's docstring lists the quantities the
    chain samples under "Sampled", and everything else it holds under "Attributes".

    The sampling methods are called explicitly, in the Gibbs order, by whoever decides from
    ``step.cfg`` whether they run in this iteration:
      * before mapmaking, from `process_tod`: ``step.sample...(tod_samples, sky)``, with ``sky``
        the band's sky model at this rank's pixels (plus the iteration, if the step logs it);
      * once per detector-scan, from the scan loop of `tod2map`: ``step.<verb>(view, ...)``;
      * on every TOD request, from `TODView.get_tod`: the step's correction of the TOD, such as
        `HFIModulation.demodulate`.
    A sampling method reaches the band's communicator and static data as ``tod_samples.band_comm``
    and ``tod_samples.experiment_data``, and the other steps as ``tod_samples.<step>``.

    Every step exists for every band. A step that can be switched off as a whole (including one
    only some experiments use, like HFI demodulation) has ``cfg.enabled``; when that is False, its
    constructor returns right after reading the settings, so it holds only ``cfg``, its users check
    ``cfg.enabled`` first, and it writes and reads nothing. Gain, noise and data selection always
    build their state, since it is used even when their sampling is off.

    Adding a step:
      1. Subclass `TODStep` in the step's own module. `__init__` reads and checks its settings,
         and, if the step is enabled, allocates its arrays and gives them their start values.
      2. Override `chain_arrays` and `read_chain` if it has anything to write or to restore.
      3. Add one line for it in `TODSamples.__init__`.
      4. Call its sampling methods from their place in `process_tod` or `tod2map`.
      5. If it changes the TOD, apply it in `TODView.get_tod`, at its place in the data model,
         and add its name to `TODView.MODELS`, so that its own sampler can `keep` it.
    """

    def __init__(self, experiment_data: DetectorGroupTOD, params: Bunch, my_band: Bunch,
                 band_comm: MPI.Comm, chain: int):
        """Read the step's settings and allocate its state, for one band on one rank and one chain.

        Each step is created once per chain, so a warning about the settings is given for chain 1
        only (and on the band master).

        Args:
            experiment_data: The band's static TOD on this rank (shapes, detectors, noise model).
            params: The full parameter file.
            my_band: The band's block of the parameter file.
            band_comm: The ranks that hold the band's scans.
            chain: The Gibbs chain (1 or 2) the state belongs to.
        """


    def chain_arrays(self) -> ChainArrays:
        """This rank's share of the band chain file. The default writes nothing."""
        return ChainArrays()


    def read_chain(self, saved: SavedBandSample) -> None:
        """Continue from a saved sample, by reading back the restart state. The default has none.

        Only what the next iteration cannot do without is read back; diagnostics are not.
        """


    def start_iteration(self) -> None:
        """Forget what the previous iteration's scan loop recorded. The default records nothing."""
