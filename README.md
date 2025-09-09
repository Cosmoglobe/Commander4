# Commander4

## 1. How to install

### TL;DR (assumes you are on the ITA cluster)
```bash
module use --append /mn/stornext/u3/jonassl/.modules  # Make my modules avaiable to you. You can skip this if you are not on the ITA cluster or you have reasonable modules loaded already.
module load Commander4  # Load the Commander4 module.
git clone --recurse-submodules git@github.com:Cosmoglobe/Commander4.git  # Clone the repo
cd Commander4
python -m venv ../.com4_venv  # (recommended) Create a new python virtual enviroment.
source ../.com4_venv/bin/activate
make all  # Will install needed Python packages and compile some local libraries.
mpirun -n 6 python3 -u src/python/commander4.py -p src/python/params/param_Planck+WMAP_n128_mapsonly_perpix.yml  # Example test run.
```

### 1.1 Initializing the submodules
Commander4 pulls in the ducc0 sources to make developing of C++ helper functions easier.

To install this submodule directly when cloning the Commander4 repository you can do
```bash
git clone --recurse-submodules git@github.com:Cosmoglobe/Commander4.git
```
If you have already cloned Commander 4,  the easiest way is to go to the Commender4 directory and then do
```bash
git submodule init
git submodule update
```

### 1.2 Load relevant modules (optional, for the ITA cluster)
The easiest way of making sure you have the modules you need loaded is to load my Commander 4 module:
```bash
module use --append /mn/stornext/u3/jonassl/.modules
module load Commander4
```

### 1.3 Set up a Python virtual enviroment (optional, recommended)
It's a good idea to create a virtual Python enviroment, so that you can install exactly the packages you need for Commander 4 without mixing it with other installations.
```bash
python -m venv ../.com4_venv
```
This enviroment can then be activated with:
```bash
source ../.com4_venv/bin/activate
```
And de-activated with
```bash
deactivate
```

### 1.4 Run makefile
There is a relatively simple makefile that installs necessary Python packages and compiles relevant code. Make sure you are in the `Commander4` directory, and run:
```bash
make all
```


## 2. How to run
Commander4 has to be run with MPI, and a parameter file has to be indicated using the `-p` argument. Example usage:
```
mpirun -n 15 python -u src/python/commander4.py -p params/param_default.yml
```
(The `-u` makes stdout unbuffered, which I have found to be necessary in order to make MPI programs print properly to the terminal).

## 3. Development / Contributing
### 3.1 Git workflow
1. Make sure you are on main (`git checkout main`) and up to date (`git pull`).
2. Create a new local branch (`git checkout -b dev-compsep`).
3. Make commits from small self-contained changes to the code. The individual commits should not break the code, but should otherwise be as limited in scope as possible.
4. After each commit, push to remote. First such commit must specify upstream (`git push --set-upstream origin dev-compsep`).
5. Create a pull request from your branch to main whenever you have made a meaningful self-contained change to the code. This could be as small as a single bug fix, or a larger new feature, but it should ideally not be so large that it contains several completely unrelated updates to the code.
6. If the pull request is small and unlikely to break anything or affect others, simply merge it yourself.
7. If you are not immediately planning to keep developing the same features on the same branch, it is best to check out to main (`git checkout main`) and delete your local branch (`git branch -d dev-compsep`) (you can always re-branch with the exact same name later). The exception is if you intend to keep working on the same features in the code, that depends on the new changes you made.
8. If you are the reviewer of a pull request, always delete the merged branch immediately after merging. There will be a prompt for this on GitHub.

### 3.2 Python style guidelines
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
