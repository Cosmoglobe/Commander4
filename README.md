# Commander4

## 1. How to install
### Initializing the submodules
Commander4 pulls in the ducc0 sources to make developing of C++ helper functions easier.

To install this submodule directly when cloning the Commander4 repository you can do
```
git clone --recurse-submodules <Commander4 repo URL>
```
If you have already cloned Commander 4,  the easiest way is to go to the Commender4 directory and then do
```
git submodule init
git submodule update
```

### Compiling Ctypes libraries
The code depends on C++ Ctypes libraries which are located in the `src/cpp/` directory. There should be a file named `src/cpp/compilation.sh` which contains the compilation procedure for all C++ files.

## 2. How to run
Commander4 has to be run with MPI, and a parameter file has to be indicated using the `-p` argument. Example usage:
```
mpirun -n 15 python -u src/python/commander4.py -p params/param_default.yml
```
(The `-u` makes stdout unbuffered, which I have found to be necessary in order to make MPI programs print properly to the terminal).

## 3. Development / Contributing
### Git workflow
1. Make sure you are on main (`git checkout main`) and up to date (`git pull`).
2. Create a new local branch (`git checkout -b dev-compsep`).
3. Make commits from small self-contained changes to the code. The individual commits should not break the code, but should otherwise be as limited in scope as possible.
4. After each commit, push to remote. First such commit must specify upstream (`git push --set-upstream origin dev-compsep`).
5. Create a pull request from your branch to main whenever you have made a meaningful self-contained change to the code. This could be as small as a single bug fix, or a larger new feature, but it should ideally not be so large that it contains several completely unrelated updates to the code.
6. If the pull request is small and unlikely to break anything or affect others, simply merge it yourself.
7. If you are not immediately planning to keep developing the same features on the same branch, it is best to check out to main (`git checkout main`) and delete your local branch (`git branch -d dev-compsep`) (you can always re-branch with the exact same name later). The exception is if you intend to keep working on the same features in the code, that depends on the new changes you made.
8. If you are the reviewer of a pull request, always delete the merged branch immediately after merging. There will be a prompt for this on GitHub.
