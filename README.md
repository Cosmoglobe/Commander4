# Commander4

## How to run
Commander4 has to be run with MPI, and a parameter file has to be indicated using the `-p` argument. Example usage:
```
mpirun -n 15 python -u src/python/commander4.py -p params/param_default.yml
```
(The `-u` makes stdout unbuffered, which I have found to be necessary in order to make MPI programs print properly to the terminal).

## Compiling Ctypes libraries
The code depends on C++ Ctypes libraries which are located in the `src/cpp/` directory. There should be a file named `src/cpp/compilation.sh` which contains the compilation procedure for all C++ files.