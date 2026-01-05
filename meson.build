# Root Meson file. See also `src/commnader4/meson.build`

### Project info ###
project('commander4', 'cpp',
  version : '0.1.0',
  default_options : ['warning_level=2', 'cpp_std=c++17', 'buildtype=release']
)

### Dependencies ###
py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# Find pybind11
pybind11_inc = run_command(py, ['-c', 'import pybind11; print(pybind11.get_include())'],
                           check : true).stdout().strip()

### 1. Python Source Handling ###
# We delegate to src/commnader4/meson.build.
# This fixes editable installs by allowing Meson to map the source tree correctly.
subdir('src/commander4')

### Submodule Check ###
ducc0_check = run_command(py, '-c', 
  'import os; print(os.path.exists("external/ducc0/src/ducc0/infra/mav.cc"))', 
  check: true
).stdout().strip()

if ducc0_check != 'True'
  error('ducc0 submodule missing! Run: git submodule update --init --recursive')
endif

### 2. C++ Extension ###
cc = meson.get_compiler('cpp')
incdir = include_directories('.', 'src/lib_cpp', 'external/ducc0/src')

cpp_sources = [
  'src/lib_cpp/cmdr4_support.cc',
  'external/ducc0/src/ducc0/infra/string_utils.cc',
  'external/ducc0/src/ducc0/infra/threading.cc',
  'external/ducc0/src/ducc0/infra/mav.cc',
]

extra_args = [
  '-march=native', '-ffast-math', '-Wfatal-errors', '-Wfloat-conversion',
  '-DPKGNAME=_cmdr4_backend', '-DPKGVERSION=0.1.0'
]

extension = py.extension_module(
  '_cmdr4_backend',
  cpp_sources,
  include_directories : [incdir, include_directories(pybind11_inc)],
  dependencies : [py_dep],
  cpp_args : extra_args,
  install : true,
  subdir : 'commander4' # Installs the .so file into the package
)

# --- STUB GENERATION ---
# The rest of this file contains a hacky way of having Meson create `.pyi` stub files.
# These files contain "empty" copies (arguments + docstrings) of the compiled shared library
# Pybind11 functions. This enables language servers like Pycharm to understand what is inside
# these files, as Pycharm only perform static analysis, and can't read compiled .so files.
custom_target(
  'generate_stubs_strict',
  input : extension,
  output : 'stubs.stamp',
  command : [
    py, '-c',
    '''
import sys, os, subprocess

# 1. Setup
so_file_path = sys.argv[1]
source_root = sys.argv[2]
stamp_file = sys.argv[3]

# 2. Env: Add build dir to path so we can import the raw .so
env = os.environ.copy()
env["PYTHONPATH"] = os.path.dirname(so_file_path) + os.pathsep + env.get("PYTHONPATH", "")

# 3. Run Stubgen
# Fix: Target "_cmdr4_backend" directly, bypassing the package import.
cmd = [
    sys.executable, "-m", "pybind11_stubgen", 
    "_cmdr4_backend", 
    "--output-dir", os.path.join(source_root, "src", "commander4"), 
    "--root-suffix", ""
]

print("Regenerating stubs...")
subprocess.run(cmd, env=env, check=True) # strict check=True

# 4. Stamp
with open(stamp_file, 'w') as f:
    f.write("Done.")
    ''',
    extension.full_path(),
    meson.project_source_root(),
    '@OUTPUT@'
  ],
  build_by_default : true,
  install : false
)