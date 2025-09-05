# Commander4

## 1. How to install
### 1.1 Initializing the submodules
Commander4 pulls in the ducc0 sources to make developing of C++ helper functions easier.

To install this submodule directly when cloning the Commander4 repository you can do
```bash
git clone --recurse-submodules <Commander4 repo URL>
```
If you have already cloned Commander 4,  the easiest way is to go to the Commender4 directory and then do
```bash
git submodule init
git submodule update
```

### 1.2 Compiling pybind11 libraries
The code depends on C++ code interfaced with `pybind11`. This is installed as a local pip package. To perform the installation, navigate into the `cmdr4_support` directory, and run a pip install (NB: the `.` is necessary).
```bash
cd cmdr4_support
CC=gcc CXX=g++ pip3 install -v .
```
The `CC=gcc CXX=g++` tells it to use these specific compilers during installation (it might try to use another, incompatible, compiler).

### 1.3 Compiling Ctypes libraries
The code currently also depends on C++ code interfaces with Ctypes. This code is located in the `src/cpp/` directory. There should be a file named `src/cpp/compilation.sh` which contains the compilation procedure for all C++ files:
```bash
cd src/cpp
bash compilation.sh
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
