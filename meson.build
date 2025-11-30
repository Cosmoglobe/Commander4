# This is the root Meson file that sets up the general build environment, and then calls
# `src/commander4/meson.build`, which handles the specific Python and C++ building.

# --- Project info ---Z
project('commander4', 'cpp',
  version : '0.1.0',
  default_options : [
    'warning_level=2',  # Standard warnings (-Wall -Wextra).
    'cpp_std=c++17',
    'buildtype=release',  # Optimized build (-O3). 'debug' would be -O0 -g.
    'b_ndebug=if-release'  # Disables assertions in release mode
  ]
)

# --- Dependencies ---
py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# Run a bash command to find pybind11 includes
pybind11_inc = run_command(py, ['-c', 'import pybind11; print(pybind11.get_include())'], check : true).stdout().strip()

# --- Recurse ---
# Tells meson to look in this subdir for further build files.
subdir('src/commander4')