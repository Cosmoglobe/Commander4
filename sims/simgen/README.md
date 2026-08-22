# simgen — modular TOD simulator for Commander4

`simgen` generates simulated per-detector time-ordered data (TOD) in the **`litebird_sim`** HDF5
format, directly readable by the main program (no Commander4 changes needed). It replaces the older
`aux/make_simple_sim.py` with a modular design whose five extension points — **pointing strategies,
sky components, noise models, bolometer transfer functions, TOD modifiers** — are each a small
registry of swappable classes.

## Running

Commander4 must be importable (`import commander4`) for the component SED classes and the Huffman
codec. Run from the `sims/` directory so the `simgen` package is on the path:

```bash
cd sims
mpirun -n 4 python -m simgen -p simgen/params/example_param.yml
```

Work (`band × scan`) is split across MPI ranks; rank 0 builds the sky maps and broadcasts them, each
rank writes its own scan files, and rank 0 writes a per-band `filelist.txt`.

## Output layout

```
<output_dir>/<BandName>/scan_<pid:06>.h5   # one file per scan, litebird format
<output_dir>/<BandName>/filelist.txt       # pid -> file, consumed by the C4 reader
<output_dir>/simgen_params.yml             # resolved parameters (provenance)
```

Each scan file holds `common/{nside,fsamp,npsi}`, `<pid>/common/{ntod,hufftree,huffsymb,vsun}`, and
per detector `<pid>/<det>/{tod, pix, psi, flag}` where `pix`/`psi`/`flag` are Huffman-compressed
with one shared per-scan tree.

Set the optional `simulation.debug_output_dir` to also write diagnostic products outside the TOD
directory:

```
<debug_output_dir>/<BandName>_diagnostics.h5
<debug_output_dir>/<BandName>_sky_{I,Q,U}.png
<debug_output_dir>/<BandName>_hits.png
<debug_output_dir>/<BandName>_rms_white_noise_{I,Q,U}.png
<debug_output_dir>/<BandName>_noise_{I,Q,U}.png
<debug_output_dir>/truth_<ComponentName>.fits
```

Each diagnostic HDF5 file contains band metadata plus `maps/sky` (beam-smoothed input sky at the
band's evaluation `nside`), `maps/hits` (all detector samples), `maps/inv_white_noise` (the packed
upper triangle of the per-pixel pointing normal matrix), `maps/rms_white_noise`, and `maps/noise`.
The RMS is the square root of the diagonal of the inverse normal matrix, so it accounts for
polarization-angle coverage and is `NaN` where a Stokes component is unconstrained. It is the same
quantity as Commander4's own `map_rms`, and the two agree on a run's own simulation. It includes
only the configured detector
white-noise level: correlated noise, orbital dipole, transfer functions, and TOD modifiers are
intentionally not propagated into this diagnostic uncertainty. `maps/noise` is the actual simulated
noise realization binned with that same nominal normal matrix after TOD modifiers, including the
configured cross-talk.

### Component truth maps

`truth_<ComponentName>.fits` is one HEALPix IQU map per enabled component holding that component's
input **amplitude**: the *unsmoothed* sky at the component's own reference frequency, in the run's
`general.units`, at `general.nside`. That is exactly the quantity a Commander4 `DiffuseComponent`
carries in its alms — beams and the per-band SED are applied by Commander4's mixing operator, so
neither is baked in — which makes the file usable, unchanged, as that component's

- **`init_from`**: start the Gibbs chain at the truth, to isolate one sampler from component
  separation converging at the same time;
- **`amp_prior_mean_map`**: the mean μ of the Gaussian amplitude prior a ~ N(μ, S);
- **reference for scoring**: what a recovered component map has to be compared against.

Declare `units: "uK_RJ"` (or whatever `general.units` was) and the matching `nu_ref` on the
Commander4 side; the reference frequency and unit are also recorded in the FITS header as `NU_REF`
and `BUNIT`. All three Stokes rows are always written — Commander4 infers a map's polarization
content from its shape, not its column names — so a component with no polarized emission gets
zeros. Set `simulation.write_component_truth_maps: false` to skip them (they are one full-sky map
per component, which is worth avoiding at large `nside`).

The defining property, checked in `tests/test_simgen_truth_maps.py`, is
`band_map(band) == beam(truth_map) * get_sed(band.freq)`, and it holds to float32 rounding when the
beam is applied in the same space the component used.

One caveat when *verifying* against a band map: going back through a map → alm step (which is what
Commander4 does when it reads the file, and what applying a beam in alm space needs) costs about
`1e-3` in relative RMS at `l` near `3*nside-1`, where the HEALPix analysis stops being invertible —
measured at `2-6e-4` for these components through Commander4's own LSMR inverse SHT, and confined
to `l > 100` (below `l = 50` it is `3e-7`). The amplitude in the file is exact; that error is the
spherical-harmonic transform's, and Commander4 incurs the same on any `init_from` map.

## Parameter file

A YAML file (see [params/example_param.yml](params/example_param.yml)) reusing the main program's conventions:

- `general`: `nside`, `units` (TOD unit, `uK_RJ`), `float_precision`, `seed`, `output_dir`.
- `components`: **the same block shape as a Commander4 param file** — each enabled component is
  realized by the matching `commander4.sky.component` class for its SED. Diffuse foregrounds
  take a `template:` block (`{source: pysm3, preset: ...}` or `{source: fits, path: ...}`); the CMB
  is a CAMB realization (optional `solar_dipole`).
- `simulation`: `nscans`, `scan_duration_sec`, `npsi`, `orbital_dipole`, `pointing`, `noise`,
  `modifiers`, `compress` (default `true`), optional `debug_output_dir`, and
  `write_component_truth_maps` (default `true`, only acted on when `debug_output_dir` is set; see
  [Component truth maps](#component-truth-maps)). With `compress: false`,
  `pix`/`psi` are written as plain `int32`/`float32` arrays instead of Huffman payloads
  (the `general` reader reads either transparently). `flag` is always Huffman-compressed because
  that reader unconditionally decodes it.
- `experiments → bands → detectors`: per-band `freq`, `fwhm`, `fsamp`, `eval_nside`, `data_nside`,
  `sigma0`/`sigma0_rts`, `polarization`, optional `crosstalk`, and a `detectors` dict (inline or via
  `!inc <file>.yml`, exactly as the main param files do). Each detector may set `psi_offset_deg`
  (polarization-angle offset), `fp_offset_deg: [xi, eta]` (focal-plane offset), `gain`, and a
  bolometer time constant `tau_ms`/`tau_sec` (or a `transfer_function` block; see below).
- `simulation.transfer_function`: run-wide default **bolometer transfer function** applied to every
  detector's signal, overridable per detector. `{enabled: true, tau_ms: 10.0}` gives a single-pole
  low-pass `H(f) = 1/(1 + 2πi f τ)` (the per-detector time constant, ~10 ms for Planck HFI); a
  `poles:` list of `{amp, tau_ms}` gives a multi-pole (HFI "LFER"-style) response instead. `H` is
  DC-normalized (`H(0)=1`), so it changes only the temporal shape (scan-direction lag/smearing), not
  the calibration. A detector opts out with `transfer_function: {enabled: false}`. This is the
  time-domain convolution the Commander4 CG mapmaker's `T_omega` operator is built to deconvolve.

## Feature-test parameter files

Besides the two `example_*` files, `params/` holds a set of small simulations each built to exercise
one part of the main program, together with a matching Commander4 parameter file in
[`params/sims/`](../../params/sims/) that turns that feature on and starts it away from the injected
truth. See [`params/sims/README.md`](../../params/sims/README.md) for how to run a pair.

| simgen parameter file | Commander4 parameter file | Feature under test |
|---|---|---|
| `params/param_gain.yml`      | `params/sims/simparam_gain.yml`         | `abs_gain` / `rel_gain` calibration (orbital dipole + sky) |
| `params/param_corrnoise.yml` | `params/sims/simparam_corrnoise.yml`    | `corr_noise`: n_corr, PSD parameters, sigma0 |
| `params/param_compsep.yml`   | `params/sims/simparam_compsep_CG.yml`   | CG amplitude sampling, C(l) prior, fluctuations |
| `params/param_compsep.yml`   | `params/sims/simparam_compsep_MCMC.yml` | MH spectral-index sampling |
| `params/param_patch.yml`     | `params/sims/simparam_patch.yml`        | partial sky: `sparse_maps`, prior-driven unobserved pixels |
| `params/param_transfunc.yml` | `params/sims/simparam_transfunc.yml`    | bolometer transfer function: the damage from *not* modelling it |

The transfer-function pair is the odd one out: it is a copy of the `param_compsep.yml` pair with
only the bolometer response switched on, so the two are run together and differenced. It measures a
gap rather than a recovery — Commander4 never sets the CG mapmaker's `T_omega`, so nothing on the
analysis side deconvolves the response. See the two files' headers.

Two simulation-side conventions these files depend on:

- **Keep `orbital_dipole: true` for any satellite pointing strategy.** The main program always
  subtracts an orbital dipole reconstructed from the `vsun` stored in the scan files, so a
  simulation that stores a non-zero `vsun` but omits the dipole from the TOD is inconsistent with
  its own analysis. `raster` stores `vsun = 0`, so there the flag is free.
- **Gains and initial noise parameters may come from either the file or the parameter file.**
  The `general` reader reads the per-detector `scalars` (gain, sigma0, fknee, alpha), so a band whose
  `detectors:` entries are empty starts the chain from the simulated truth. A `gain:` on a detector
  or an `initial_noise_params:` on the band still takes priority, which is how a recovery test
  starts the chain deliberately away from the truth.

## Built-in plugins

**Pointing** (`simulation.pointing.strategy`):
- `planck_scan` — analytic Planck-like satellite scan (anti-Sun spin + precession + orbital dipole).
  `anti_sun_period_days` sets the large-scale sweep period (default `365.25`; full-sky coverage
  takes roughly half this period). `precession_period_days` and `spin_angle_tilt` separately set
  the period and radius of the spin-axis wobble around the anti-Sun direction. Shortening
  `anti_sun_period_days` accelerates coverage while retaining the physical $30\,\mathrm{km\,s^{-1}}$
  orbital-dipole amplitude.
- `file` — load precomputed pointing from an HDF5 file (configurable dataset names).
- `raster` — sweep a small patch row-by-row: traverse along longitude, teleport back at the end of
  each row, step one row in latitude, and wrap after the last row. **One scan is exactly one full
  fill of the patch** (`n_rows * samples_per_row` samples; `scan_duration_sec` is ignored), and each
  subsequent scan repeats the coverage. Params: `patch_center_deg`, `patch_size_deg`, `n_rows`,
  `samples_per_row`. The raster uses **per-detector pointing**, so each detector is pointed at the
  patch shifted by its `fp_offset_deg` and the band's detectors trace mutually offset tracks. See
  [params/example_raster_param.yml](params/example_raster_param.yml).

The satellite strategies (`planck_scan`, `file`) use a shared boresight: detectors differ only by
`psi_offset_deg` (`fp_offset_deg` is applied only by strategies with `per_detector_pointing = True`).

**Sky components** (`components.<name>.component_class`): `CMB`, `ThermalDust`, `Synchrotron`,
`FreeFree`, `SpinningDust` (PySM3/FITS template × Commander4 SED), and `GriddedPointSources` —
equal-amplitude point sources on a regular `nlon × nlat` (lon, lat) grid over
`lon_range_deg × lat_range_deg`, intensity-only, frequency-flat by default (optional `beta`/`nu_ref`
power-law SED). Pair it with the `raster` strategy to image a patch of identical sources.

## Extending

Each extension point is a base class + a name→class registry; add a class and register it.

| Capability        | File           | Base class        | Registry                    |
|-------------------|----------------|-------------------|-----------------------------|
| Pointing strategy | `pointing.py`  | `PointingStrategy`| `POINTING_STRATEGIES`       |
| Sky component     | `sky.py`       | `SkyComponent`    | `_COMPONENT_BUILDERS`       |
| Noise model       | `noise.py`     | `NoiseModel`      | (`make_noise_model`)        |
| Transfer function | `transfer.py`  | `TransferFunction`| (`make_detector_transfer`)  |
| TOD modifier      | `modifiers.py` | `TODModifier`     | `MODIFIERS`                 |

- **Pointing**: implement `compute(sample_offset, ntod, det_offset=(0,0)) -> PointingChunk(theta,
  phi, psi, vsun)`. Set `per_detector_pointing = True` to have the pipeline call `compute` per
  detector with its `fp_offset` (for spatial detector offsets); otherwise the boresight is shared.
- **Sky component**: implement `band_map(band) -> (npol, npix_eval)` in `uK_RJ`; reuse a C4 SED via
  `get_sed`. Also implement `truth_map(nside) -> (3, npix)`, the unsmoothed amplitude at the
  component's `nu_ref`, and expose that `nu_ref` as an attribute; the base class raises unless
  `write_component_truth_maps` is off.
- **Noise model**: implement `realize(ntod, fsamp, sigma0, rng) -> ndarray`.
- **Transfer function**: implement `response(freqs_hz) -> complex ndarray` (the DC-normalized filter
  `H(f)`); the base class's `apply(signal, fsamp)` does the FFT convolution. Wire it into
  `make_detector_transfer` to build it from the parameter file.
- **TOD modifier**: implement `apply(tod[ndet, ntod], band, ctx) -> ndarray` (e.g. cross-talk).

## Consuming the output in Commander4

In a Commander4 parameter file, set the experiment to read the generated files:

```yaml
experiments:
  SimSat:
    is_sim: true
    experiment_id: "general"
    replace_tod_with_sim: false        # use the TODs in the files (do not overwrite with the in-place sim)
    Fourier_times_path: "<existing FFT_times .npy>"
    bands:
      Band30GHz:
        filelist: "<output_dir>/Band30GHz/filelist.txt"
        ...
```

Scans of length ≤ 10 000 or ≥ 400 000 samples make `find_good_Fourier_time` a no-op, so the
FFT-times file content is irrelevant (the file must still load).

## Tests

Run with `pytest`:

- `tests/test_simgen_pointing.py` — pointing strategies (pure numpy).
- `tests/test_simgen_transfer.py` — the transfer-function filter math and its pipeline wiring.
- `tests/test_simgen_diagnostics.py` — the binned diagnostic maps.
- `tests/test_simgen_truth_maps.py` — the component amplitude convention, and the truth map's round
  trip through Commander4's init-map reader.
- `tests/test_simgen_pipeline_e2e.py` — a real end-to-end `pipeline.run`, written to HDF5 and read
  back.

The sky and end-to-end tests need `camb`, `pysm3`, `ducc0`, `astropy` and the compiled
`commander4.backend.utils` extension importable in the environment; each skips itself when
its dependencies are missing.
