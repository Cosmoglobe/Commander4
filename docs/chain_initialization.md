# Starting and continuing chains

A run restarts from its ordinary chain files. There is no separate checkpoint. The behaviour
follows Commander3 wherever C4's design allows it.

## Parameters

| C4 (`gibbs` block) | C3 | Meaning |
|---|---|---|
| `num_iterations` | `NUM_GIBBS_ITER` | Total length of the chains. An appended run samples only the iterations still missing, and stops at once if none are. |
| `chain_status: new` | `CHAIN_STATUS = new` | Start at iteration 1. Existing chain files are overwritten as the run reaches them, with a warning (C3 deletes the old chain file). |
| `chain_status: append` | `CHAIN_STATUS = append` | Continue the chains in `output.dir` after their newest complete iteration. If there is none, start new chains, so the same parameter file can be resubmitted until the chains are done. |
| `init_chain: "<dir>"` | `INIT_CHAIN = file:iter` | Start new chains from another run's newest complete iteration; `"<dir>:<iter>"` picks one. Used when no append happens. |
| component `init_from` | `COMP_INIT_FROM_HDF` | A FITS map or a `chains_compsep/` file for one component. Beats `init_chain`, but not an appended chain's own saved state. `init_from: null` keeps the parameter-file start. |

Seeds are set per (chain, iteration, rank), so an appended run with the same number of ranks uses
the same random numbers as an unbroken run.

## What is saved and restored

Each class writes and reads its own state, as C3's `dumpToHDF` / `initHDF` pairs do:

- The TOD sampling steps (one band file per band and chain). Each step is a subclass of
  `TODStep` (`tod/step.py`) that holds its own state, hands its arrays over in `chain_arrays` (as a
  `ChainArrays`: per scan, per band, and per detector-scan for events and debug TODs) and reads its
  restart state back in `read_chain`: `Gain` (`abs_gain`, `detrel_gain`, `temporal_gain`), `Noise`
  (`noise_params`), `DataSelection` (`accept`) and, for HFI, `HFIModulation`
  (`modulation_phase`, `baselines`). `TODSamples` holds these objects and does the MPI part for all
  of them alike: `gather_chain_arrays` collects every step's arrays on the band master, and
  `read_chain` hands every step a `SavedBandSample`, which reads only this rank's scans. Scans are
  matched by scan ID and detectors by name, so the number of ranks may change. Gains are converted
  with the unit stored in the file.
- `Component.write_chain` / `Component.read_chain` and their subclass overrides (one CompSep file
  per chain): the amplitudes (`alms` or `source_amps`), the beam they carry (`amp_fwhm_arcmin`),
  and the spectral parameters this run samples.

The block that `read_chain` reads is always written; everything else in a chain file (maps,
diagnostics, debug output) stays optional. Jumps and glitches are not read back: they are found
again from the data, as in C3.

**Sampled, not fixed, spectral parameters.** Fixed parameters always come from the parameter
file, so an edit there takes effect with `init_chain`. C3 instead restores every spectral
parameter. Which parameters are sampled is decided by explicit code in each class's `read_chain`:
`beta` is restored when an MCMC sampling group names the component in its `beta` block
(`init_compsep_processing` passes this as `restore_beta`). A name list cannot express this in
general. The `FullDust` model (branch `update_dust`) samples only the amplitudes in row 1 of its
spline-node table, carries pieces that are `None` unless configured, and must rebuild its spline
after a restore. A component with such state overrides `read_chain`, calls `super()`, and
restores and rebuilds its own state.

## Which iteration a run starts from

`file_io/chain_restart.py` decides this once, on the world master, from file names alone:

1. The parts this run reads: one per enabled TOD band, plus the sky if CompSep runs.
2. A saved iteration is complete when every part the source run wrote exists for every chain it
   wrote. The newest complete iteration is used, or the one `init_chain` names.
3. A chain the source run never wrote (normally chain 2, since `output.chains.write: [1]` saves
   disk) starts from the state of a written chain. Its own history was never saved, so nothing
   is lost.
4. A part the source run never wrote at all, such as a band added since, or the sky of a TOD-only
   run, starts from the parameter file, with a warning.

Chain files are written as `<name>.tmp` and renamed once complete. A file with its final name is
therefore always complete, and C3's "drop the last sample, it may be corrupt" is not needed.

Thinning (`output.chains.interval.bands` / `.compsep`) limits where a chain can restart, as C3's
`THINNING_FACTOR` does: only iterations with all required files are complete.

## The initial exchange

CompSep sends one initial sky per chain, and the TOD side receives a sky on every step. Before,
chain 2's first TOD step reused chain 1's initial sky, which is wrong once the two chains start
from different saved states.

## Not covered

- C3's per-band (`BAND_TOD_INIT_FROM_HDF`) and instrument-only (`INIT_INSTRUMENT_FROM_HDF`)
  sources.
- Starting a run whose detectors are a subset of the saved run's: each detector keeps its saved
  gain, but the relative gains are not re-centred to the subset's zero-sum constraint.
- A TOD-only run's fixed sky is rebuilt from the parameter file, not read from a chain.
