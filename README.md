# Commander4

## Setup before use (at the ITA clusters)
If you are at the ITA clusters, before installing or running Commander4, load the following modules
```bash
module load intel/oneapi
module load mpi/latest
module load compiler/latest  # Only necessary for developers.
```
Then, make sure you have a sensible Python setup. At ITA, I recommend using the interpreter located at `/astro/local/mamba/envs/py313/bin/python`. You can hijack the interpreter without tying yourself to the Mamba ecosystem by simply calling it directly (e.g. put `alias python313="/astro/local/mamba/envs/py313/bin/python"` in your `~/.profile`).

If not using the Mamba environment, you should set up a Python virtual environment. This can be done as
```bash
python -m venv ../.com4_venv
source ../.com4_venv/bin/activate
```
## Installation for users
If you are not intending to edit Commander4, you can install it by cloning the repository, and doing a pip install.
```bash
git clone --recurse-submodules git@github.com:Cosmoglobe/Commander4.git
cd Commander4
python -m pip install .
```
You are now ready to run Commander4 (see further down).

If you have already cloned the repo and forgot to add the `--recursive-submodules`, you can run
```bash
git submodule init
git submodule update
```

Note: Commander4 depends on the `ducc0` Python package from pip by default.

If you are developing and need unreleased upstream changes or want maximum performance, install `ducc0` from source yourself, e.g.:
```bash
python -m pip install --no-binary ducc0 ducc0
```
or install from a checkout (e.g. from the `external/ducc0` submodule).

## Installation for developers
If you intend to edit Commander4, you must first have the build tools installed:
```bash
python -m pip install scikit-build-core pybind11 pybind11-stubgen numpy
```
Then, clone the repo (and submodules), and perform a so-called *editable* PIP install:
```bash
git clone --recurse-submodules git@github.com:Cosmoglobe/Commander4.git
cd Commander4
python -m pip install -e . --no-build-isolation
```
The editable install (`-e`) will tell PIP and scikit-build-core/CMake that the installation should point back to the source location, meaning that **you can edit Python files and run Commander4 without re-installing**. The `--no-build-isolation` helps ensure the build uses your environment (useful on HPC systems), which is why you have to manually pip install build dependencies first.

Note that if you edit non-Python files (C/C++) you must re-install for changes to take effect.

Native (ctypes) helper code is built into a single shared library installed as `commander4/_libs/cmdr4_ctypes.so`.
To add new ctypes-exposed C/C++ code, add a new `.cpp` file under `src/lib_cpp/ctypes/` and re-install.

### Optional: nanobind backend
If you want to build the extension with nanobind instead of pybind11:
```bash
python -m pip install -e ".[nanobind]" --no-build-isolation
CMDR4_USE_NANOBIND=1 python -m pip install -e . --no-build-isolation
```

### Optional: regenerate type stubs
The repository includes checked-in `.pyi` files for the compiled extension. If you change the C++ API and want to regenerate stubs:
Stub files are generated automatically during the build (mirroring the previous Meson setup).

If you want to regenerate stubs manually:
```bash
commander4-generate-stubs
```

## Running Commander4
Commander4 has to be run with MPI, and a parameter file has to be indicated using the `-p` argument. Example usage:
```
mpirun -n 15 commander4 -p params/param_default.yml
```
(The `-u` makes stdout unbuffered, which I have found to be necessary in order to make MPI programs print properly to the terminal).

Note that Commander4 cannot be run as a standalone script (e.g. python src/commander4/cli.py). It must be installed, and is then run as a binary. Note also that the binary should be called directly, and running `python commander4` will not work.


## Development / Contributing
### Git workflow
1. Make sure you are on main (`git checkout main`) and up to date (`git pull`).
2. Create a new local branch (`git checkout -b dev-compsep`).
3. Make commits from small self-contained changes to the code. The individual commits should not break the code, but should otherwise be as limited in scope as possible.
4. After each commit, push to remote. First such commit must specify upstream (`git push --set-upstream origin dev-compsep`).
5. Create a pull request from your branch to main whenever you have made a meaningful self-contained change to the code. This could be as small as a single bug fix, or a larger new feature, but it should ideally not be so large that it contains several completely unrelated updates to the code.
6. If the pull request is small and unlikely to break anything or affect others, simply merge it yourself.
7. If you are not immediately planning to keep developing the same features on the same branch, it is best to check out to main (`git checkout main`) and delete your local branch (`git branch -d dev-compsep`) (you can always re-branch with the exact same name later). The exception is if you intend to keep working on the same features in the code, that depends on the new changes you made.
8. If you are the reviewer of a pull request, always delete the merged branch immediately after merging. There will be a prompt for this on GitHub.

### Python style guidelines
Commander 4 does not strictly adhere to a specific style guideline, and you are encouraged to use common sense. You are generally recommended to follow PEP8 (https://peps.python.org/pep-0008/) style guidelines, with the following clarifications and exceptions:

#### Line length
Commander 4 uses a maximum line length of 100 characters.

#### Line breaks
You are generally encouraged to avoid unnecessary line breaks, unless you feel it strongly adds to the readability of the code.
```Python
my_sum = the_first_value + some_other_value + a_third_value  # Correct
my_sum = the_first_value\  # Incorrect
       + some_other_value\
       + a_third_value
```

#### Function arguments
You are generally encouraged to not line-break for the first function argument, and the keep multi-line arguments aligned.
```Python
def my_very_long_function_name(argument1, argument2, argument3,
                               argument4, argument5):
    return argument1 + argument2
```

#### Name capitalization
Classes should use PascalCase capitalization, while functions should be lower-case.
```Python
class MyClass:
    ...
def my_func():
    ...
```

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
Functions should have type hints for all their function arguments and return type. Beyond this, type hints are optional.
```Python
from numpy.typing import NDArray

def my_pow_func(array: NDArray, pow: float) -> NDArray:
    return array**pow
```
