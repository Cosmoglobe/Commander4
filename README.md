# 1. Setup before use (at the ITA clusters)
If you are at the ITA cluster (the "owls") load the following modules before installing or running Commander4:
```bash
module load intel/oneapi
module load mpi/latest
module load compiler/latest  # Only necessary for developers.
```
Then, make sure you have a sensible Python setup. At ITA, I recommend using the interpreter located at `/astro/local/mamba/envs/py313/bin/python`. You can hijack the interpreter without tying yourself to the Mamba ecosystem by simply calling it directly (e.g. put `alias python313="/astro/local/mamba/envs/py313/bin/python"` in your `~/.profile`).

If not using the Mamba environment, you should set up a Python virtual environment. This can be done as
```bash
python313 -m venv ../.com4_venv  # or just `python`, depending on your setup.
source ../.com4_venv/bin/activate
```

**Optional:** Commander4 heavily utilizes `ducc0`, which will be installed automatically, but if you want maximum performance from `ducc0` install `ducc0` from source yourself, e.g.:
```bash
pip install --no-binary ducc0 ducc0
```

# 2. Installation
### 2.1 Installation for users
If you are not intending to edit Commander4, you can install it by cloning the repository, and doing a pip install.
```bash
git clone --recurse-submodules git@github.com:Cosmoglobe/Commander4.git
cd Commander4
pip install .
```
You are now ready to run Commander4 (see further down).

If you have already cloned the repo and forgot to add the `--recursive-submodules`, you can run
```bash
git submodule init
git submodule update
```

### 2.2 Installation for developers
If you intend to edit Commander4, you must first have the build tools installed:
```bash
pip install scikit-build-core cmake pybind11 pybind11-stubgen numpy
```
Then, clone the repo (and submodules), and perform a so-called *editable* PIP install:
```bash
git clone --recurse-submodules git@github.com:Cosmoglobe/Commander4.git
cd Commander4
pip install -v -e . --no-build-isolation
```
The editable install (`-e`) will tell PIP and scikit-build-core/CMake that the installation should point back to the source location, meaning that **you can edit Python files and run Commander4 without re-installing**. The `--no-build-isolation` helps ensure the build uses your environment (useful on HPC systems), which is why you have to manually pip install build dependencies first. The verbose install (`-v`) will show the more specific build steps, as well as compilation warnings, which would otherwise be hidden.

Note that if you edit non-Python files (C/C++) you must re-install for changes to take effect.

Native (ctypes) helper code is built into a single shared library installed as `commander4/_libs/cmdr4_ctypes.so`.
To add new ctypes-exposed C/C++ code, add a new `.cpp` file under `src/lib_cpp/ctypes/` and re-install.

# 3. Running Commander4
Commander4 has to be run with MPI, and a parameter file has to be indicated using the `-p` argument. Example usage:
```
mpirun -n 15 commander4 -p params/param_default.yml
```
To quickly check how many MPI ranks you must use for any given parameter file, you can run `c4-validate-params path/to/params.yml`.

Note that Commander4 cannot be run as a standalone script (e.g. python src/commander4/cli.py). It must be installed, and is then run as a binary. Note also that the binary should be called directly, and running `python commander4` will not work.

### 3.1 Parameter file
A parameter file is seven top-level blocks, each named after the part of the program that reads them: `gibbs`, `resources`, `output`, `components`, `experiments`, `tod_processing` and `compsep`.


The MPI task counts are **derived**, not stated: the TOD total is the sum of the per-band `num_tasks` over enabled bands of enabled experiments, and component separation takes one task per enabled `compsep.bands` view (one for I, one for QU). Commander4 reports the total it needs, and `mpirun -n` must match it. `compsep.enabled: false` runs TOD-only, allocating no compsep ranks.

### 3.2 Output
A run writes everything below the single directory named by `output.dir`, which it creates:
```YAML
<output_dir>/logs/              # log file (output.logging.file.filename) and cProfile dumps
<output_dir>/chains_bands/      # per-band TOD samples and output maps
<output_dir>/chains_compsep/    # sky components and the fit against the band maps
<output_dir>/plots/             # figures
```

`output.logging.file.filename` is a bare file name, not a path: it is always placed in the logs directory. The subdirectory names are defined in [`src/commander4/file_io/paths.py`](src/commander4/file_io/paths.py), which is also what the plotting and standalone tools read, so those take the same `<output_dir>` as their argument.

#### Why two chains
Everything is HDF5, one file per Gibbs sample, named `..._chain<CC>_iter<NNNN>.h5`. There are two chains rather than one because the two halves of the program are **disjoint MPI rank sets** (see §4.1): the band chain is written by each band's master on the TOD side, and the compsep chain by the master on the component-separation side, one schedule step later. Merging them would need parallel HDF5 and a barrier between two sides that otherwise only exchange maps. The band chain is per band, the compsep chain per run, and matching `(chain, iter)` labels describe the same Gibbs sample.

Both carry `metadata/datetime` and `metadata/parameter_file_as_string`, the verbatim parameter file, which is what lets the standalone tools rebuild a run's configuration from its output alone.

Which iterations are written is set by `output.chains.write` (a list of chain numbers) and `output.chains.interval`, one entry per output: `bands` and `compsep` thin a whole file, while `maps` thins only the `maps/` group inside the band file, since those maps are far larger than anything beside them. Datasets marked *(opt)* below appear only when the matching `output.chains.include` flag is set.

#### `chains_bands/<experiment>_<band>_chain<CC>_iter<NNNN>.h5`
Per-scan sampled quantities at the top level, output maps under `maps/`. `NSC` is the band's total scan count, `ND` its detector count, `NPAR` its noise-model parameter count. Gains are written in the band's `band_unit`; maps are brightnesses in the same unit.
```YAML
metadata/band_unit             # thermodynamic unit of the gains and maps below
metadata/map_fwhm_arcmin       # beam of maps/observed_sky and maps/rms
scan_ids           (NSC,)      # int64 scan IDs; the row order of every per-scan array
det_names          (ND,)       # detector names; the column order of every per-detector array
scan_start_time    (NSC,)      # scan start time (C3 'MJD'); 0.0 if the reader supplies no time
orbital_velocity   (NSC,ND,3)  # spacecraft velocity [m/s], what anchors the absolute calibration
abs_gain           scalar      # absolute gain, one per band
detrel_gain        (ND,)       # relative gain offset, one per detector (zero-sum)
temporal_gain      (NSC,ND)    # per-scan gain variation about abs+rel
gain_prior         (ND,3)      # (sigma0, fknee, alpha) of the temporal-gain Wiener prior
noise_params       (NSC,ND,NPAR)  # noise PSD parameters, sigma0 first (sigma0,fknee,alpha for normal oof model).
present            (NSC,ND)    # int8: this detector has data in this scan
accept             (NSC,ND)    # int8: data-quality flag (present data that is not rejected)
good_fraction      (NSC,ND)    # unflagged fraction of samples
chisq_z            (NSC,ND)    # white-noise chi^2 z-score; ~N(0,1) for clean data, NaN if unevaluated
ncorr_cg_residual  (NSC,ND)    # final relative CG residual of the correlated-noise draw
ncorr_cg_niter     (NSC,ND)    # int32 CG iterations; 0 when the stationary fallback was used
ncorr_converged    (NSC,ND)    # int8: 1 accepted, 0 failed, -1 no n_corr drawn
tod_ps_freqs       (NSC,ND,100)   # log-binned frequency axis shared by the four spectra below
tod_ps_raw         (NSC,ND,100)   # binned PSD of the raw TOD
tod_ps_ncorr       (NSC,ND,100)   # ... of the correlated-noise realization
tod_ps_ncorrsub    (NSC,ND,100)   # ... of the TOD with only the correlated noise removed
tod_ps_residual    (NSC,ND,100)   # ... of the residual (sky, dipole and n_corr all removed)
jump_counts        (NSC,ND)    # jumps found per detector-scan; indexes the two ragged arrays below
jump_locations     (M,)        # sample index of each jump, concatenated scan-major
jump_offsets       (M,)        # amplitude of each jump
ncorr_tod_lengths  (NSC,ND)    # (opt, DEBUG) length of each full n_corr TOD in the flat array below
ncorr_tod_flat     (sum,)      # (opt, DEBUG) every n_corr TOD concatenated; very large

maps/observed_sky  (3,npix)    # the solved sky map (I, Q, U)
maps/rms           (3,npix)    # per-pixel white-noise rms; inf where unobserved
maps/skymodel      (3,npix)    # (opt) sky model this iteration was processed against
maps/res           (3,npix)    # (opt) binned residual: data minus sky, dipole and n_corr
maps/orbdipole     (3,npix)    # (opt) binned orbital dipole
maps/corrnoise     (3,npix)    # (opt) binned correlated noise
maps/nhit          (npix,)     # (opt) int64 count of unflagged samples per pixel
maps/cov           (6,npix)    # (opt) the 6 unique elements of P^T N^-1 P (II,IQ,IU,QQ,QU,UU)
```

#### `chains_compsep/chain<CC>_iter<NNNN>.h5`
One file per iteration holding every component, plus how well they fit the band maps. `<sn>` is a component's `shortname`, `<view>` a `<band>_<I|QU>` pair.
```YAML
comps/<sn>/alms          (npol,nalm)  # complex amplitudes in uK_RJ at nu_ref (T/E/B rows)
comps/<sn>/source_amps   (1,nsrc)     # point-source components carry this instead of alms
comps/<sn>/sigma_l       (npol,lmax+1)  # realized power spectrum of those alms
comps/<sn>/lmax          scalar       # band limit of the alms
comps/<sn>/comp_name     str          # class name, e.g. ThermalDust
comps/<sn>/shortname     str          # the <sn> used above
comps/<sn>/defined_pol   str          # polarization the component is defined in
comps/<sn>/eval_pol      str          # polarization it was evaluated in
comps/<sn>/amp_fwhm_arcmin  scalar    # beam already in the amplitudes; 0 for the CG solver
comps/<sn>/sed/<param>                # each SED parameter, including sampled ones (beta)
comps/<sn>/Cl_prior/<param>           # the C(l) prior's model parameters; absent if the prior is off
comps/<sn>/mixing/<band> scalar       # this component's SED evaluated at that band

chi2/total               scalar       # all-band sum of the whitened residual squared
chi2/ndof                scalar       # observed pixels summed over all bands and polarizations
chi2/reduced             scalar       # total / ndof
chi2/z                   scalar       # (total - ndof) / sqrt(2 ndof); ~N(0,1) for a good fit
chi2/bands/<view>/{chi2,ndof,reduced,nu}   # the same, per band and polarization
chi2/map                 (3,npix)     # (opt) z^2 summed into output.chains.nside_chisq pixels
residuals/<view>         (npol,npix)  # (opt) full-resolution data minus model
amplitude_groups/<group>/<I|QU>/{n_iter,cg_residuals}   # CG convergence, per polarization
mcmc/<group>/{numstep,n_accept,accept_rate}             # Metropolis-Hastings acceptance
mcmc/<group>/params/<comp>            # the accepted parameter value per component
```

`gibbs.init_from_chain` restarts from a chain: the TOD side reads the per-scan quantities out of a `chains_bands/` file, and each component's `init_from` reads its alms out of a `chains_compsep/` one.


# 4. Development / Contributing
### 4.1 Code layout
An overview the most important directories and files. This is not exhaustive.

```
src/commander4/
  cli.py               # Entry point. Splits MPI into a TOD side and a compsep side, runs the Gibbs loop.
  units.py             # Brightness-unit conversions (uK_RJ / uK_CMB / MJy/sr).
  polarization.py      # The I / QU / IQU vocabulary shared by both sides.

  tod/                 # === TOD SIDE: one Gibbs iteration over time-ordered data ===
    processing.py      #   Drives the iteration: gain, jumps, correlated noise, mapmaking, data selection.
    view.py            #   TODView: the read interface to one detector-scan and every TOD derived from it.
    gain.py            #   Gain sampling (absolute, relative, temporal).
    noise/             #   Correlated-noise realizations, sigma0 estimation, PSD models and their priors.
    mapmaking/         #   binned.py (per-pixel inversion) and cg.py (iterative, deconvolves a transfer function).

  compsep/             # === COMPSEP SIDE: solving for component amplitudes and spectral parameters ===
    processing.py      #   Drives the iteration and validates the sampling groups.
    cg_solver.py       #   The global CG amplitude solve; perpix_solver.py is the common-resolution alternative.
    mcmc.py            #   Metropolis-Hastings machinery; spectral_index.py is its one concrete user.
    preconditioners.py #   Preconditioners for the compsep CG (the mapmaker's live in tod/mapmaking/).

  sky/                 # The sky model that compsep solves for and the TOD side subtracts.
    component.py       #   Component base class; diffuse_components.py and point_sources.py are the families.
    comp_list.py       #   CompList: a list of components usable as a single vector by the CG driver.

  data_models/         # The containers the two sides pass around (band TOD, band maps, pointing, samples).
  file_io/             # Everything that touches disk: chain writing, map reading, and:
    experiments/       #   One TOD reader per experiment, each module named after its `experiment_id`.
  parameters/          # Parameter-file parsing (parse.py) and scoped lookup + validation (schema.py).
  mpi/                 # Communicator setup, and the transfer of maps and sky models between the two sides.
  math_utils/          # SHTs, alm helpers, FFTs, in-place array arithmetic.
  diagnostics/         # Logging, performance profiling, and the plots written alongside a chain.
  standalone_tools/    # Command-line tools installed alongside commander4 (e.g. c4-plot-chain).
  backend/             # Loader for the compiled C/C++ backend built from src/lib_cpp/.

sims/simgen/           # Standalone TOD simulator. Writes scan files read back with experiment_id: "general".
params/                # Parameter files, grouped by instrument.
tests/                 # pytest suite; run with `pytest` from the repository root.
notes/                 # Design notes.
```

### 4.2 Output, logs and error handling

Commander4 uses Python's logging system instead of `print`. A record has the form `time - rank - module - level - message`. Note that the `module` tells you what file the message came from. For example, the following message comes from `src/commander4/data_models/tod_samples.py` (the `src/commander4/` is implicit):
```bash
16:32:06 - rank   1 - data_models.tod_samples - INFO - Band Band44GHz, chain 2: starting fresh Gibbs chain.
```

The console handler writes to `stderr`, which the file handler writes to `<output_dir>/logs/<filename>`. Console and file levels are configured independently under `output.logging`. Each level also includes every level below it in this table:
| Level | Contents |
|---|---|
| `DEBUG` | Developer and per-rank diagnostics. |
| `VERBOSE` | Solver progress, per-band details and performance reports. |
| `INFO` | Scientific results from individual sampling steps. |
| `SUMMARY` | Startup information and one compact result per Gibbs iteration. |
| `WARNING` | Suspicious conditions that should be checked. |
| `ERROR` | An operation failed, but an explicit fallback lets the run continue. |
| `CRITICAL` | The run cannot continue and will abort. |

Commander4 raises explicit exceptions such as `ValueError` or `RuntimeError`, and never uses `assert` outside of tests. The errors are caught and hanled by the logging system: The top-level MPI boundary catches an unhandled exception and atomically selects one rank to report it. That rank prints one `CRITICAL` record with the full Python traceback, writes the same traceback to `logs/fatal-<run-id>.log`, flushes the handlers, and calls `MPI_Abort`. Other failing ranks stay silent, preventing thousands of duplicate messages and files. If the output directory is unavailable, the traceback falls back to `stderr`.

Python's `faulthandler` is also enabled. It can print minimal thread stacks for native faults such as segmentation faults that cannot become Python exceptions. Forced termination such as `SIGKILL`, out-of-memory killing, or node loss cannot be reported reliably by the application; use the Slurm job state and launcher output for those failures.

### 4.3 Git workflow
1. Make sure you are on main (`git checkout main`) and up to date (`git pull`).
2. Create a new local branch (`git checkout -b dev-compsep`).
3. Make commits from small self-contained changes to the code. The individual commits should not break the code, but should otherwise be as limited in scope as possible.
4. After each commit, push to remote. First such commit must specify upstream (`git push --set-upstream origin dev-compsep`).
5. Create a pull request from your branch to main whenever you have made a meaningful self-contained change to the code. This could be as small as a single bug fix, or a larger new feature, but it should ideally not be so large that it contains several completely unrelated updates to the code.
6. If the pull request is small and unlikely to break anything or affect others, simply merge it yourself.
7. If you are not immediately planning to keep developing the same features on the same branch, it is best to check out to main (`git checkout main`) and delete your local branch (`git branch -d dev-compsep`) (you can always re-branch with the exact same name later). The exception is if you intend to keep working on the same features in the code, that depends on the new changes you made.
8. If you are the reviewer of a pull request, always delete the merged branch immediately after merging. There will be a prompt for this on GitHub.

### 4.4 Python style guidelines
Commander 4 does not strictly adhere to a specific style guideline, and you are encouraged to use common sense. You are generally recommended to follow PEP8 (https://peps.python.org/pep-0008/) style guidelines, with the following clarifications and exceptions:

#### Line length
Commander 4 uses a maximum line length of 100 characters.

#### Line breaks
You are generally encouraged to avoid unnecessary line breaks, unless you feel it strongly adds to the readability of the code.
```Python
my_sum = the_first_value + some_other_value + a_third_value  # Better
my_sum = the_first_value\  # Worse
       + some_other_value\
       + a_third_value

result = some_func(arg1, arg2)  # Better
result = some_func(arg1,  # Worse
                   arg2)
```

#### Function arguments
You are generally encouraged not to line-break between every function argument, unless you feel it is necessary.
```Python
def my_very_long_function_name(argument1: NDArray, argument2: NDArray, argument3: NDArray,
                               argument4: float = 10.0, argument5: int = 1):
    return argument1 + argument2
```

#### Name capitalization
Classes should use PascalCase capitalization, while functions and files should be lower-case.
```Python
class MyClass:
    ...
def my_func():
    ...
```
All filenames should be lowercase, e.g. `tod_processing.py` not `TOD_processing.py`.
#### Internal class methods
Class methods that are only used by other class methods, and never by any external actor, should start with an underscore (_)
```Python
class MyClass:
    def _calculate_something_internal(self):
        ...
    def solve(self):
        self._calculate_something_internal()
```

#### Type hints
Functions should normally have type hints for all their function arguments and return type.
```Python
from numpy.typing import NDArray

def my_pow_func(array: NDArray, pow: float) -> NDArray:
    return array**pow
```

# 5. Misc

### 5.1 Standalone tools
Commander4 comes with some tools that all follow the `c4-[tool-name]` pattern. These are:
```bash
c4-validate-params path/to/param.yml  # Gives you some info about the param-file, including how many MPI ranks it needs.

c4-diff-params path/to/param1.yml path/to/param2.yml  # Prints the difference between two parameter files.

c4-plot-chain path/to/chain-dir/  # Creates plots from the data in a provided chain directory.

c4-cmb-realizations  path/to/chain-dir/  # Generate constrained CMB realizations from chain (non yet fully functional).

c4-generate-stubs  # Manually re-generate the stubs that are automatically during a build (very niche).
```
