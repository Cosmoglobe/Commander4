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

**Temporary fix:** `pysm3` and `toml` do not yet officially support Numpy 2.0, and you must hackily install them without triggering a downgrade to Numpy 1.x, as following:
```
pip install toml pysm3 --no-deps
```
This will be resolved in the next `pysm3` release.

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
<output_dir>/chains_tod/        # per-band TOD sample chains
<output_dir>/chains_compsep/    # component amplitude chains
<output_dir>/chains_datamaps/   # per-band output maps
<output_dir>/plots/             # figures
```

`output.logging.file.filename` is a bare file name, not a path: it is always placed in the logs directory. The subdirectory names are defined in [`src/commander4/file_io/paths.py`](src/commander4/file_io/paths.py), which is also what the plotting and standalone tools read, so those take the same `<output_dir>` as their argument.

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

### 5.1 Nanobind backend
If you want to build the extension with nanobind instead of pybind11:
```bash
pip install -e ".[nanobind]" --no-build-isolation
CMDR4_USE_NANOBIND=1 pip install -e . --no-build-isolation
```

### 5.2 Regenerate type stubs
The repository includes checked-in `.pyi` files for the compiled extension. If you change the C++ API and want to regenerate stubs:
Stub files are generated automatically during the build (mirroring the previous Meson setup).

If you want to regenerate stubs manually:
```bash
commander4-generate-stubs
```
