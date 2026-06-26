# simgen — modular TOD simulator for Commander4

`simgen` generates simulated per-detector time-ordered data (TOD) in the **`litebird_sim`** HDF5
format, directly readable by the main program (no Commander4 changes needed). It replaces the older
`aux/make_simple_sim.py` with a modular design whose four extension points — **pointing strategies,
sky components, noise models, TOD modifiers** — are each a small registry of swappable classes.

## Running

Commander4 must be importable (`import commander4`) for the component SED classes and the Huffman
codec. Run from the `aux/` directory so the `simgen` package is on the path:

```bash
cd aux
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

## Parameter file

A YAML file (see [params/example_param.yml](params/example_param.yml)) reusing the main program's conventions:

- `general`: `nside`, `units` (TOD unit, `uK_RJ`), `CG_float_precision`, `seed`, `output_dir`.
- `components`: **the same block shape as a Commander4 param file** — each enabled component is
  realized by the matching `commander4.sky_models.component` class for its SED. Diffuse foregrounds
  take a `template:` block (`{source: pysm3, preset: ...}` or `{source: fits, path: ...}`); the CMB
  is a CAMB realization (optional `solar_dipole`).
- `simulation`: `nscans`, `scan_duration_sec`, `npsi`, `orbital_dipole`, `pointing`, `noise`,
  `modifiers`, and `compress` (default `true`). With `compress: false`, `pix`/`psi` are written as
  plain `int32`/`float32` arrays instead of Huffman payloads (`tod_reader_litebird_sim` reads either
  transparently). `flag` is always Huffman-compressed because that reader unconditionally decodes it.
- `experiments → bands → detectors`: per-band `freq`, `fwhm`, `fsamp`, `eval_nside`, `data_nside`,
  `sigma0`/`sigma0_rts`, `polarization`, optional `crosstalk`, and a `detectors` dict (inline or via
  `!inc <file>.yml`, exactly as the main param files do). Each detector may set `psi_offset_deg`
  (polarization-angle offset) and `fp_offset_deg: [xi, eta]` (focal-plane offset).

## Built-in plugins

**Pointing** (`simulation.pointing.strategy`):
- `planck_scan` — analytic Planck-like satellite scan (anti-Sun spin + precession + orbital dipole).
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

| Capability        | File           | Base class        | Registry              |
|-------------------|----------------|-------------------|-----------------------|
| Pointing strategy | `pointing.py`  | `PointingStrategy`| `POINTING_STRATEGIES` |
| Sky component     | `sky.py`       | `SkyComponent`    | `_COMPONENT_BUILDERS` |
| Noise model       | `noise.py`     | `NoiseModel`      | (`make_noise_model`)  |
| TOD modifier      | `modifiers.py` | `TODModifier`     | `MODIFIERS`           |

- **Pointing**: implement `compute(sample_offset, ntod, det_offset=(0,0)) -> PointingChunk(theta,
  phi, psi, vsun)`. Set `per_detector_pointing = True` to have the pipeline call `compute` per
  detector with its `fp_offset` (for spatial detector offsets); otherwise the boresight is shared.
- **Sky component**: implement `band_map(band) -> (npol, npix_eval)` in `uK_RJ`; reuse a C4 SED via
  `get_sed`.
- **Noise model**: implement `realize(ntod, fsamp, sigma0, rng) -> ndarray`.
- **TOD modifier**: implement `apply(tod[ndet, ntod], band, ctx) -> ndarray` (e.g. cross-talk).

## Consuming the output in Commander4

In a Commander4 parameter file, set the experiment to read the generated files:

```yaml
experiments:
  SimSat:
    is_sim: true
    experiment_id: "litebird_sim"
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

`tests/test_simgen.py` (run with `pytest`) covers the Huffman pointing round-trip through the writer,
SED consistency with the C4 component classes, a small MPI end-to-end run, and reader integration via
`tod_reader_litebird_sim`. The sky/reader tests need `ducc0`, `pysm3` and the compiled
`commander4.cmdr4_support.utils` extension importable in the environment.
